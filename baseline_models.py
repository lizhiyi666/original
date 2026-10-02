"""Explicit energy guidance and trained partial-order CFG; no implicit fallbacks."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from constraint_projection import ConstraintProjection
from discrete_diffusion.diffusion_transformer import DiffusionTransformer, log_onehot_to_index


class EnergyGuidance:
    def __init__(self, dd, temperature=1.0, scale=10.0, last_k=40, frequency=4):
        if not math.isfinite(scale) or scale < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid energy guidance settings')
        if last_k < 1 or frequency < 1:
            raise ValueError('Invalid guidance window')
        self.scale, self.last_k, self.frequency = scale, last_k, frequency
        self.projector = ConstraintProjection(dd.num_classes, dd.type_classes, dd.num_spectial,
            use_gumbel_softmax=False, gumbel_temperature=temperature, verbose=False)
        self.stats = dict(calls=0, probes=0, zero_probes=0, gradient_norm_sum=0.0)

    def apply(self, values, positions, constraints, step):
        if self.scale == 0 or step is None or step >= self.last_k or step % self.frequency:
            return values
        if not torch.isfinite(values).all():
            raise FloatingPointError('Non-finite guidance input')
        p = self.projector
        if constraints and isinstance(constraints[0], list):
            a,b,mask = p.compile_batched_constraints(constraints, values.device)
        else:
            a,b = p._compile_constraints(constraints, values.device)
            mask = None
        if a is None:
            return values
        effective = p.effective_constraint_mask(values,a,b,positions,mask)
        if not effective.any():
            return values
        with torch.no_grad():
            ho,he = p.compute_hard_constraint_violation_optimized(values,a,b,positions,effective)
            unmet = (((ho>0)|(he>0)) & effective).any(1)
        with torch.enable_grad():
            logits = values.detach().clone().requires_grad_(True)
            order,exist,_ = p.compute_constraint_violation_optimized(logits,a,b,positions,effective)
            energy = (order + 5.0*exist).sum()
            if not torch.isfinite(energy):
                raise FloatingPointError('Non-finite guidance energy')
            grad = torch.autograd.grad(energy,logits)[0]
        if not torch.isfinite(grad).all():
            raise FloatingPointError('Non-finite guidance gradient')
        updated = values-self.scale*grad
        if not torch.isfinite(updated).all():
            raise FloatingPointError('Non-finite guidance update before clamp')
        norms = grad.flatten(1).norm(dim=1)[unmet]
        self.stats['calls'] += 1
        self.stats['probes'] += norms.numel()
        self.stats['zero_probes'] += int((norms==0).sum())
        self.stats['gradient_norm_sum'] += float(norms.sum())
        return updated.clamp(-70,0).detach()


def configure_energy(dd, temperature=1.0, scale=10.0, last_k=40, frequency=4):
    dd.use_constraint_projection = False
    dd.use_guidance_baseline = True
    dd.energy_guidance = EnergyGuidance(dd,temperature,scale,last_k,frequency)
    dd.projection_call_count = 0


class POCFGDiffusion(DiffusionTransformer):
    """Background conditions remain present when the partial-order condition drops."""
    def __init__(self, po_drop_probability=0.1, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= po_drop_probability <= 1:
            raise ValueError('Invalid PO dropout probability')
        self.po_encoder = nn.Sequential(nn.Linear(self.type_classes**2,256),nn.ReLU(),nn.Linear(256,256))
        self.po_null = nn.Parameter(torch.zeros(256))
        self.po_drop_probability = po_drop_probability
        self.cfg_scale = 1.0
        self.cfg_trained = False
        self.reset_cfg_stats()

    def reset_cfg_stats(self):
        self.cfg_stats = dict(conditional_rows=0,null_rows=0,calls=0,active_rows=0,max_branch_difference=0.0)

    def po_condition(self, background, batch, mode):
        matrix = getattr(batch,'po_matrix',None)
        if matrix is None or matrix.shape != (background.shape[0], self.type_classes,self.type_classes):
            raise ValueError('CFG requires an aligned per-sample PO matrix')
        if not torch.isfinite(matrix).all():
            raise FloatingPointError('Non-finite CFG condition')
        active = matrix.bool().flatten(1).any(1)
        keep = active.clone()
        if mode == 'null':
            keep.zero_()
        elif mode == 'train':
            keep &= torch.rand(background.shape[0],device=background.device) >= self.po_drop_probability
            self.cfg_stats['conditional_rows'] += int(keep.sum())
            self.cfg_stats['null_rows'] += int((~keep).sum())
        elif mode != 'conditional':
            raise ValueError('Unknown CFG condition mode')
        encoded = self.po_encoder(matrix.float().flatten(1))
        encoded = torch.where(keep[:,None],encoded,self.po_null[None,:])
        return background + encoded[:,None,:]*batch.mask[:,:,None], active

    def raw_cfg_logits(self, log_x_t, background, t, batch):
        tokens = log_onehot_to_index(log_x_t)
        if self.training:
            condition,_ = self.po_condition(background,batch,'train')
            return self.transformer(tokens,condition,t,batch)
        if not self.cfg_trained:
            raise RuntimeError('CFG checkpoint has not completed conditional/null training')
        conditional,active = self.po_condition(background,batch,'conditional')
        null,_ = self.po_condition(background,batch,'null')
        lc = self.transformer(tokens,conditional,t,batch)
        lu = self.transformer(tokens,null,t,batch)
        if not torch.isfinite(lc).all() or not torch.isfinite(lu).all():
            raise FloatingPointError('Non-finite CFG branch logits')
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0:
            raise ValueError('Invalid CFG scale')
        out = lu if self.cfg_scale == 0 else lc if self.cfg_scale == 1 else lu+self.cfg_scale*(lc-lu)
        if not torch.isfinite(out).all():
            raise FloatingPointError('Non-finite CFG combined logits')
        self.cfg_stats['calls'] += 1
        self.cfg_stats['active_rows'] += int(active.sum())
        if active.any():
            self.cfg_stats['max_branch_difference'] = max(self.cfg_stats['max_branch_difference'],
                                                        float((lc-lu)[active].abs().max()))
        return out

    def _prediction(self, log_x_t, cond_emb, t, batch, truncation_k=None):
        out = self.raw_cfg_logits(log_x_t,cond_emb,t,batch)
        log_pred = F.log_softmax(out.double(),dim=1).float()
        if truncation_k is not None:
            val,ind = log_pred.topk(k=truncation_k,dim=1)
            log_pred = torch.full_like(log_pred,-70).scatter(1,ind,val)
        zeros = torch.zeros(log_x_t.shape[0],2,log_x_t.shape[2],device=log_x_t.device,dtype=log_x_t.dtype)-70
        return torch.cat((log_pred,zeros),dim=1).clamp(-70,0)

    def predict_start(self, log_x_t, cond_emb, t, batch):
        return self._prediction(log_x_t,cond_emb,t,batch)

    def predict_start_with_truncate(self, log_x_t, cond_emb, t, batch, truncation_k=15):
        return self._prediction(log_x_t,cond_emb,t,batch,truncation_k)
