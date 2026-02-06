#update matrix for larger dataset
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from torch.nn import TransformerEncoderLayer, TransformerEncoder 
from discrete_diffusion.conditional_attention import Transformer
from datamodule import Batch
from einops import rearrange
from constraint_projection import ConstraintProjection, parse_po_matrix_to_constraints

eps = 1e-8



def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.to(t.device).gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)



def alpha_schedule(time_step, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999, type_classes=9, poi_classes=381):

    sep=5
    sep_1=sep-1
    att= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.0001 - 0.99999) + 0.99999,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.000009- 0.00009) + 0.00009))
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]

    att1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.9999 - 0.99999) + 0.99999,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.000009- 0.9999) + 0.9999))
    att1 = np.concatenate(([1], att1))
    at1 = att1[1:]/att1[:-1]

    ctt= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.00009 - 0.000009) + 0.000009,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.9999- 0.0001) + 0.0001))
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct

    ctt1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.00009 - 0.000009) + 0.000009,
    np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.9998- 0.00009) + 0.00009))
    ctt1 = np.concatenate(([0], ctt1)) 
    one_minus_ctt1 = 1 - ctt1 
    one_minus_ct1 = one_minus_ctt1[1:] / one_minus_ctt1[:-1]
    ct1 = 1-one_minus_ct1 

    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    att1 = np.concatenate((att1[1:], [1]))
    ctt1 = np.concatenate((ctt1[1:], [0]))
    btt1 = (1-att1-ctt1) / type_classes
    btt2 = (1-att-ctt)

    bt1 = (1-at1-ct1) / type_classes 
    btt2 = np.concatenate(([0], btt2))
    one_minus_btt2 = 1 - btt2
    one_minus_bt = one_minus_btt2[1:] / one_minus_btt2[:-1]
    bt = 1-one_minus_bt
    btt2 = (1-att-ctt)/poi_classes

    bt=np.concatenate((bt[:time_step*sep_1//sep],at1[time_step*sep_1//sep:]/poi_classes))
    at=np.concatenate((at[:time_step*sep_1//sep],(1-ct-bt*poi_classes)[time_step*sep_1//sep:])).clip(min=1e-30)
    ct=np.concatenate(((1-at-bt)[:time_step*sep_1//sep],ct[time_step*sep_1//sep:])).clip(min=1e-30)

    return at,at1, bt,bt1, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 


class ConditionEmbeddingModel(nn.Module):
    def __init__(
        self,
        cond_token_num = 200, # total number of all of condition tokens
        emb_dims = 256,
        num_condition_types = 6,
        max_position_embeddings = 3000
    ):
        super().__init__()
        self.token_num = cond_token_num
        self.emb_dims = emb_dims
        self.num_condition_types = num_condition_types + 1 # hour in day are discretized as additional condition
        self.max_position_embeddings = max_position_embeddings
        self.encoder = nn.Embedding(self.token_num, self.emb_dims)
        self.input_up_proj =  nn.Sequential(
            nn.Linear(self.num_condition_types * self.emb_dims, self.emb_dims),
            nn.ReLU(),
            nn.Linear(self.emb_dims, self.emb_dims)
        )
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.emb_dims)
        self.register_buffer("position_ids", torch.arange(self.max_position_embeddings).expand((1, -1)))
        encoder_layer = TransformerEncoderLayer(d_model=self.emb_dims, nhead=4, batch_first=True)
        self.condition_transformers = TransformerEncoder(encoder_layer, num_layers=3)


    def forward(self, batch):
        seq_length = batch.time.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        time_embeddings = self.encoder(batch.time.long()+1)
        condition1_embeddings = self.encoder(batch.condition1)
        condition2_embeddings = self.encoder(batch.condition2)
        condition3_embeddings = self.encoder(batch.condition3)
        condition4_embeddings = self.encoder(batch.condition4)
        condition5_embeddings = self.encoder(batch.condition5)
        condition6_embeddings = self.encoder(batch.condition6)

        condition_embeddings = self.input_up_proj(torch.cat([time_embeddings,condition1_embeddings,condition2_embeddings,\
            condition3_embeddings,condition4_embeddings,condition5_embeddings,condition6_embeddings],dim=-1))
        condition_embeddings = self.position_embeddings(position_ids) + condition_embeddings
        encoded_conditions = self.condition_transformers(condition_embeddings)
        return encoded_conditions



class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        diffusion_step=200,
        alpha_init_type='alpha1',
        num_condition_types=6,
        type_classes=9,
        poi_classes=3477,
        num_spectial=4,
        num_classes=None,
        use_constraint_projection=False,  # [新增] 是否使用约束投影
        projection_tau=0.0,
        projection_lambda=0.0,          # λinit
        #projection_alm_iters=10,        # 保持兼容，但实际外层内层由下方参数控制
        projection_eta=1.0,             # η
        projection_mu=1.0,              # μinit
        projection_frequency=10,
        use_gumbel_softmax=True,
        gumbel_temperature=1.0,
        projection_last_k_steps: int = 60,
        projection_mu_max: float = 1000.0,      # μmax
        projection_outer_iters: int = 10,     # outer_itermax
        projection_inner_iters: int = 10,      # inner_itermax
        projection_mu_alpha: float = 2.0,       # α 放大系数
        projection_delta_tol: float = 0.25,     # δ 容忍
        projection_existence_weight: float = 0.02,
    ):
        super().__init__()  

        self.schedule_type=alpha_init_type
        self.amp = False
        self.num_condition_types = num_condition_types

        self.num_classes = type_classes+poi_classes+num_spectial+2 
        self.type_classes = type_classes 
        self.num_spectial = num_spectial 
        self.poi_classes = poi_classes 
        self.transformer = Transformer(tgt_vocab_size=self.num_classes,num_spectial=self.num_spectial,type_classes=self.type_classes,poi_classes=self.poi_classes)
        self.loss_type = 'vb_stochastic'
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.use_gumbel_softmax = use_gumbel_softmax
        self.gumbel_temperature = gumbel_temperature
        self.projection_last_k_steps = projection_last_k_steps
        self.projection_existence_weight=projection_existence_weight

        # [新增] 约束投影模块
        self.use_constraint_projection = use_constraint_projection
        self.projection_frequency = projection_frequency  # [新增] 投影频率
        if self.use_constraint_projection:
            self.constraint_projector = ConstraintProjection(
                num_classes=self.num_classes,
                type_classes=self.type_classes,
                num_spectial=self.num_spectial,
                tau=projection_tau,
                lambda_init=projection_lambda,
                mu_init=projection_mu,
                mu_alpha=projection_mu_alpha,
                mu_max=projection_mu_max,
                outer_iterations=projection_outer_iters,
                inner_iterations=projection_inner_iters,
                eta=projection_eta,
                delta_tol=projection_delta_tol,
                device='cuda',
                use_gumbel_softmax=self.use_gumbel_softmax,
                gumbel_temperature=self.gumbel_temperature,
                projection_existence_weight=projection_existence_weight, 
            )

        at,at1, bt,bt1, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 = alpha_schedule(self.num_timesteps, type_classes=self.type_classes, poi_classes = self.poi_classes)

 
        at1 = torch.tensor(at1.astype('float64'))
        bt1 = torch.tensor(bt1.astype('float64'))
        ct1 = torch.tensor(ct1.astype('float64'))
        log_at1 = torch.log(at1).clamp(-70,0)
        log_bt1 = torch.log(bt1).clamp(-70,0)
        log_ct1 = torch.log(ct1).clamp(-70,0)

        att1 = torch.tensor(att1.astype('float64'))
        btt1 = torch.tensor(btt1.astype('float64'))
        ctt1 = torch.tensor(ctt1.astype('float64'))
        log_cumprod_at1 = torch.log(att1).clamp(-70,0)
        log_cumprod_bt1 = torch.log(btt1).clamp(-70,0)
        log_cumprod_ct1 = torch.log(ctt1).clamp(-70,0) 

        log_1_min_ct1 = log_1_min_a(log_ct1) # torch.log(1 - a.exp() + 1e-40)
        log_1_min_cumprod_ct1 = log_1_min_a(log_cumprod_ct1)
        assert log_add_exp(log_ct1, log_1_min_ct1).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct1, log_1_min_cumprod_ct1).abs().sum().item() < 1.e-5
        self.register_buffer('log_ct1', log_ct1.float())
        self.register_buffer('log_bt1', log_bt1.float())
        self.register_buffer('log_at1', log_at1.float())
        self.register_buffer('log_cumprod_at1', log_cumprod_at1.float())
        self.register_buffer('log_cumprod_bt1', log_cumprod_bt1.float())
        self.register_buffer('log_cumprod_ct1', log_cumprod_ct1.float())
        self.register_buffer('log_1_min_ct1', log_1_min_ct1.float())
        self.register_buffer('log_1_min_cumprod_ct1', log_1_min_cumprod_ct1.float())

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at).clamp(-70,0)
        log_bt = torch.log(bt).clamp(-70,0)
        log_ct = torch.log(ct).clamp(-70,0)
        att = torch.tensor(att.astype('float64'))
        btt2 = torch.tensor(btt2.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att).clamp(-70,0)
        log_cumprod_bt = torch.log(btt2).clamp(-70,0)
        log_cumprod_ct = torch.log(ctt).clamp(-70,0)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        
        self.zero_vector = None

        self.condition_encoder = ConditionEmbeddingModel(num_condition_types=self.num_condition_types)

    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t, batch):         # q(xt|xt_1) [-2] mask_category [-1] mask_poi
        B,V,L=log_x_t.shape
        t = t.unsqueeze(1).repeat(1,L)
        log_x_start = rearrange(log_x_t, 'b v l -> b l v')

        log_x_start_category = log_x_start[batch.category_mask.bool()]
        t_tmp = t[batch.category_mask.bool()]

        selected_range = log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes]

        log_ct1 = extract(self.log_ct1, t_tmp, selected_range.shape)         # ct~
        log_1_min_ct1 = extract(self.log_1_min_ct1, t_tmp, selected_range.shape)       # 1-ct~

        selected_range = selected_range + log_1_min_ct1
        log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes] = selected_range
        log_x_start_category = torch.cat([log_x_start_category[:,:-2],log_add_exp(log_x_start_category[:,-2:-1],log_ct1), log_x_start_category[:,-1:]],dim=-1)
        log_x_start[batch.category_mask.bool()] = log_x_start_category

        log_x_start_poi = log_x_start[batch.poi_mask.bool()]
        selected_range = log_x_start_poi[:,self.num_spectial+self.type_classes:-2]
        t_tmp = t[batch.poi_mask.bool()]
        log_at = extract(self.log_at, t_tmp, selected_range.shape) 
        log_bt = extract(self.log_bt, t_tmp, selected_range.shape)             # bt
        log_ct = extract(self.log_ct, t_tmp, selected_range.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t_tmp, selected_range.shape)          # 1-ct

        selected_range = log_add_exp(selected_range +log_at, log_bt)
        log_x_start_poi[:,self.num_spectial+self.type_classes:-2] = selected_range
        log_x_start_poi = torch.cat([log_x_start_poi[:,:-1],log_add_exp(log_x_start_poi[:,-1:]+log_1_min_ct, log_ct)],dim=-1)
        log_x_start[batch.poi_mask.bool()] = log_x_start_poi

        log_probs = rearrange(log_x_start, 'b l v -> b v l')

        return log_probs

    def q_pred(self, log_x_start, t, batch):           # q(xt|x0)
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        B,V,L=log_x_start.shape
        t = t.unsqueeze(1).repeat(1,L)

        log_x_start = rearrange(log_x_start, 'b v l -> b l v')

        log_x_start_category = log_x_start[batch.category_mask.bool()]
        selected_range = log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes]
        t_tmp = t[batch.category_mask.bool()]

        log_cumprod_ct1 = extract(self.log_cumprod_ct1, t_tmp, selected_range.shape)         # ct~
        log_1_min_cumprod_ct1 = extract(self.log_1_min_cumprod_ct1, t_tmp, selected_range.shape)       # 1-ct~


        selected_range = selected_range + log_1_min_cumprod_ct1
        log_x_start_category[:,self.num_spectial:self.num_spectial+self.type_classes] = selected_range
        log_x_start_category = torch.cat([log_x_start_category[:,:-2],log_add_exp(log_x_start_category[:,-2:-1],log_cumprod_ct1), log_x_start_category[:,-1:]],dim=-1)
        log_x_start[batch.category_mask.bool()] = log_x_start_category

        log_x_start_poi = log_x_start[batch.poi_mask.bool()]
        selected_range = log_x_start_poi[:,self.num_spectial+self.type_classes:-2]
        t_tmp = t[batch.poi_mask.bool()]

        log_cumprod_at = extract(self.log_cumprod_at, t_tmp, selected_range.shape)
        log_cumprod_bt = extract(self.log_cumprod_bt, t_tmp, selected_range.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t_tmp, selected_range.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t_tmp, selected_range.shape)       # 1-ct~

        selected_range = log_add_exp(selected_range +log_cumprod_at, log_cumprod_bt)
        log_x_start_poi[:,self.num_spectial+self.type_classes:-2] = selected_range
        log_x_start_poi = torch.cat([log_x_start_poi[:,:-1],log_add_exp(log_x_start_poi[:,-1:]+log_1_min_cumprod_ct, log_cumprod_ct)],dim=-1)
        log_x_start[batch.poi_mask.bool()] = log_x_start_poi


        log_probs = rearrange(log_x_start, 'b l v -> b v l')
            
        return log_probs

    def predict_start(self, log_x_t, cond_emb, t, batch):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        if self.amp == True:
            with autocast():
                out = self.transformer(x_t, cond_emb, t, batch)
        else:
            out = self.transformer(x_t, cond_emb, t, batch)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-2
        assert out.size()[2:] == x_t.size()[1:]
        
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]

        zero_vector = torch.zeros(batch_size, 2, log_x_t.size(2)).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred
    
    def predict_start_with_truncate(self, log_x_t, cond_emb, t, batch, truncation_k=15):  # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        if self.amp == True:
            with autocast():
                out = self.transformer(x_t, cond_emb, t, batch)
        else:
            out = self.transformer(x_t, cond_emb, t, batch)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-2
        assert out.size()[2:] == x_t.size()[1:]
        
        log_pred = F.log_softmax(out.double(), dim=1).float()

        val, ind = log_pred.topk(k=truncation_k, dim=1)
        probs = torch.full_like(log_pred, -70)
        log_pred = probs.scatter_(1, ind, val)

        batch_size = log_x_t.size()[0]
        zero_vector = torch.zeros(batch_size, 2, log_x_t.size(2)).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred


    
    def q_posterior(self, log_x_start, log_x_t, t, batch):            

        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(log_x_start, t - 1, batch)
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_emb, t, batch):             # if x0, first p(x0|xt), then sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start_with_truncate(log_x, cond_emb, t, batch)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t, batch=batch)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_emb, t, batch)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    '''@torch.no_grad()
    def p_sample(self, log_x, cond_emb,  t, batch, po_constraints=None, diffusion_index=None):               # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob, log_x_recon = self.p_pred(log_x, cond_emb, t, batch)
        
        # [新增] 应用约束投影（如果启用且有约束）
        # 根据 projection_frequency 决定是否在当前步骤应用投影
        should_apply_projection = False
        if self.use_constraint_projection and po_constraints is not None:
            if diffusion_index is not None:
                # 当 diffusion_index % projection_frequency == 0 时应用投影
                should_apply_projection = (diffusion_index % self.projection_frequency == 0)
            else:
                # 如果没有提供 diffusion_index，则每步都应用（向后兼容）
                should_apply_projection = True
                
        if should_apply_projection:
            # 只在类别采样时应用投影
            # 投影应用于模型预测的分布
            model_log_prob = self.constraint_projector.apply_projection_to_category_positions(
                model_log_prob,
                po_constraints,
                batch.category_mask
            )
        
        # Gumbel sample
        out = self.log_sample_categorical(model_log_prob)
        return out'''
    @torch.no_grad()
    def p_sample(self, log_x, cond_emb, t, batch, po_constraints=None, diffusion_index=None):
        model_log_prob, log_x_recon = self.p_pred(log_x, cond_emb, t, batch)

        # 根据 projection_frequency 决定是否应用投影
        # 只在后期 step 投影（速度优先）
        last_k_steps = self.projection_last_k_steps

        should_apply_projection = False
        if self.use_constraint_projection and po_constraints is not None:
            if diffusion_index is not None:
                should_apply_projection = (
                    (diffusion_index % self.projection_frequency == 0)
                    and (diffusion_index < last_k_steps)
                )
            else:
                should_apply_projection = True

        # [关键] 默认不投影时 after=before，避免未定义变量
        model_log_prob_after = model_log_prob

        if should_apply_projection:
            # 只打印一次，避免刷屏
            if getattr(self, "debug_constraint_projection", False) and not getattr(self, "_debug_projection_printed", False):
                self._debug_projection_printed = True
                print("[DEBUG][projection] CALLED in p_sample")
                print(f"[DEBUG][projection] diffusion_index={diffusion_index}, projection_frequency={self.projection_frequency}")
                print(f"[DEBUG][projection] model_log_prob.shape={tuple(model_log_prob.shape)}")
                print(f"[DEBUG][projection] category_mask.sum={batch.category_mask.sum().item() if hasattr(batch,'category_mask') and batch.category_mask is not None else None}")
                print(f"[DEBUG] Projection triggered at step {diffusion_index}")
                
            # viol debug：只打印一次（很耗时）
            debug_viol = getattr(self, "debug_constraint_projection", False) and not getattr(self, "_debug_viol_printed", False)
            if debug_viol:
                self._debug_viol_printed = True

            # 判断是否为 per-sample constraints（list-of-list）
            is_per_sample = (
                isinstance(po_constraints, list)
                and len(po_constraints) > 0
                and isinstance(po_constraints[0], list)
            )
            if debug_viol:
                print(f"[DEBUG] Constraint Type: {'Per-Sample (List of Lists)' if is_per_sample else 'Shared (List of Tuples)'}")
            
            # 1. 编译约束矩阵 (Compiling Constraint Matrices)
            W_A, W_B, c_mask = None, None, None
            
            if is_per_sample:
                # Case A: 每条样本约束不同 -> W_A, W_B shape: [B, V_type, K_max]
                # 这是一个新函数，需要在 ConstraintProjection 中实现 (见上文)
                W_A, W_B, c_mask = self.constraint_projector.compile_batched_constraints(
                    po_constraints, device=model_log_prob.device
                )
            else:
                # Case B: 全 Batch 共享约束 -> W_A, W_B shape: [V_type, K]
                # 这是上一轮建议的 compile_constraints 函数
                W_A, W_B = self.constraint_projector._compile_constraints(
                    po_constraints, device=model_log_prob.device
                )

            # 2. 执行 Debug (Before Projection)
            if debug_viol and W_A is not None:
                with torch.no_grad():
                    # 复用优化后的计算函数，速度极快
                    viol_before, _ = self.constraint_projector.compute_constraint_violation_optimized(
                        model_log_prob, W_A, W_B,  batch.category_mask,constraint_mask=c_mask, gumbel_noise=None
                    )
                    # 打印 Batch 中第一个样本的违规，以及平均违规
                    print(f"[DEBUG][projection] viol_before[0]={viol_before[0].item():.6f}, mean={viol_before.mean().item():.6f}")

            # 3. 执行投影 (Project)
            # 无论 W_A 是 2D 还是 3D，project_with_matrices 内部的 matmul 都能自动广播处理
            if W_A is not None:
                model_log_prob_after = self.constraint_projector.project_with_matrices(
                    model_log_prob,
                    W_A, W_B,
                    batch.category_mask,
                    constraint_mask=c_mask,
                )
            else:
                model_log_prob_after = model_log_prob

            # 4. 执行 Debug (After Projection)
            if debug_viol and W_A is not None:
                with torch.no_grad():
                    viol_after, _ = self.constraint_projector.compute_constraint_violation_optimized(
                        model_log_prob_after, W_A, W_B, batch.category_mask, constraint_mask=c_mask, gumbel_noise=None
                    )
                    print(f"[DEBUG][projection]  viol_after[0]={viol_after[0].item():.6f}, mean={viol_after.mean().item():.6f}")
                    delta = (viol_before - viol_after).mean().item()
                    print(f"[DEBUG][projection]  delta_mean={delta:.6f}")
            

        # 用 after（不投影时等于 before）
        model_log_prob = model_log_prob_after

        def apply_classifier_guidance(
            model_log_prob: torch.Tensor,   # [B, V, L] 扩散模型的 log p(x_{t-1}|x_t)
            log_x_t: torch.Tensor,          # [B, V, L] 当前噪声状态
            t: torch.Tensor,                # [B] 时间步
            classifier,                     # ConstraintClassifier 实例
            category_mask: torch.Tensor,    # [B, L]
            guidance_scale: float = 1.0,    # s: guidance 强度
        ) -> torch.Tensor:
            """
            应用 Classifier-Based Guidance。
            
            修改后的采样分布:
                log p_guided(x_{t-1}) = log p(x_{t-1}|x_t) + s * ∇_{log_x_t} log p(y=1|x_t, t)
            
            Args:
                model_log_prob: 扩散模型预测的转移分布 [B, V, L]
                log_x_t: 当前状态的 log-onehot [B, V, L]
                t: 时间步 [B]
                classifier: 训练好的约束分类器
                category_mask: [B, L] category 位置 mask
                guidance_scale: 引导强度
            
            Returns:
                guided_log_prob: 引导后的 log 分布 [B, V, L]
            """
            if classifier is None or guidance_scale == 0:
                return model_log_prob
            
            # 计算分类器的梯度
            gradient = classifier.get_log_prob_gradient(log_x_t, t, category_mask)  # [B, V, L]
            
            # 应用引导: log p + s * grad
            guided_log_prob = model_log_prob + guidance_scale * gradient
            
            # 重新归一化（确保仍然是有效的 log 概率）
            guided_log_prob = torch.clamp(guided_log_prob, -70, 0)
            
            return guided_log_prob

        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t, batch):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, batch)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def get_category_logits(self, batch):
        """
        专门为偏序 Loss 提供预测的 Logits (预测的 x_start 分布)
        """
        b, device = batch.batch_size, batch.device
        
        # 1. 准备数据 (复用 training_losses 的逻辑)
        x_start = batch.checkin_sequences
        
        # 2. 随机采样时间步 t
        # 使用 'importance' 采样，与训练保持一致，这样计算的梯度最合理
        t, pt = self.sample_time(b, device, 'importance')
        
        # 3. 将离散数据转换为 One-hot 并加噪得到 x_t
        # 注意：这里假设 index_to_log_onehot 是文件内可访问的函数
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t, batch=batch)

        # 4. 获取条件编码
        cond_emb = self.condition_encoder(batch)

        # 5. 模型预测 x_0 的 Logits
        # log_x0_recon 的形状通常是 [Batch, Num_Classes, Seq_Len]
        log_x0_recon = self.predict_start(log_xt, cond_emb, t=t, batch=batch)
        
        # 注意：training_losses 里有一句 log_x0_recon.transpose(1, 2)
        # 这说明原始的 log_x0_recon 是 [B, C, L] 格式。
        # 我们的 PartialOrderLoss 正好期望 [B, C, L] 格式作为输入。
        # 所以这里不需要转置，直接返回即可。
        
        return log_x0_recon

    @property
    def device(self):
        return self.transformer.to_logits[-1].weight.device ##todo

    def training_losses(
        self,
        batch,
        is_train=True
        ):
        b, device = batch.batch_size, batch.device
        assert self.loss_type == 'vb_stochastic'
        x_start = batch.checkin_sequences
        t, pt = self.sample_time(b, device, 'importance')
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t, batch=batch) #gt x_t #use matrix

        cond_emb = self.condition_encoder(batch)

        log_x0_recon = self.predict_start(log_xt, cond_emb, t=t, batch=batch) 
        log_x0_recon = log_x0_recon.transpose(1, 2)

        losses = {}
        loss = F.cross_entropy(log_x0_recon.reshape(-1, log_x0_recon.shape[-1]), x_start.flatten(),ignore_index=3, reduce=False)
        loss = loss.reshape(x_start.size(0), -1)
        losses['loss'] = torch.mean(loss, -1)
        return losses['loss'], log_x0_recon.transpose(1, 2)

    def sample_fast(
            self,
            batch,
            content_token = None,
            **kwargs):
        B, L = batch.batch_size, batch.content_len

        device = self.log_at.device

        batch.device = device

        #cond_emb = self.condition_encoder(batch)
        print("tau before cond_encoder:", batch.tau.shape, "time:", batch.time.shape)
        cond_emb = self.condition_encoder(batch)
        print("tau after  cond_encoder:", batch.tau.shape, "time:", batch.time.shape)
        
        mask_poi=self.num_classes-1
        mask_cat=self.num_classes-2
        
        bottom=torch.tensor([2],device=device)
        input=torch.ones(B,L,dtype=torch.int64,device=device) *3 ##padding

        for i in range(B):
            seq_len = batch.unpadded_length[i]
            head=torch.tensor([0] + [mask_cat] * seq_len ,device=device)
            body=torch.tensor([1] + [mask_poi] * seq_len ,device=device)

            tmp=torch.cat([head,body,bottom],dim=-1)
            input[i][:len(tmp)]=tmp

        log_z = index_to_log_onehot(input,self.num_classes)
        start_step = self.num_timesteps
        
        # [新增] 从 batch 中提取偏序约束
        '''po_constraints = None
        if self.use_constraint_projection and hasattr(batch, 'po_matrix') and batch.po_matrix is not None:
            # 假设 batch 中所有样本共享同一个 po_matrix
            # po_matrix shape: [B, C, C] 或 [C, C]
            po_matrix = batch.po_matrix[0] if batch.po_matrix.dim() == 3 else batch.po_matrix
            po_constraints = parse_po_matrix_to_constraints(po_matrix.to(device))'''
        # [新增] 从 batch 中提取偏序约束（支持每条样本独立 po_matrix）
        po_constraints = None
        if self.use_constraint_projection and hasattr(batch, "po_matrix") and batch.po_matrix is not None:
            pm = batch.po_matrix
            # debug：只打印一次
            if getattr(self, "debug_constraint_projection", False) and not getattr(self, "_debug_po_printed", False):
                self._debug_po_printed = True
                print("[DEBUG][projection] sample_fast: batch has po_matrix=True")
                print("[DEBUG][projection] po_matrix.dim()=", pm.dim(), "shape=", tuple(pm.shape))

            if pm.dim() == 2:
                # 全 batch 共用
                po_constraints = parse_po_matrix_to_constraints(pm.to(device))
            elif pm.dim() == 3:
                # 每条样本一个
                po_constraints = [parse_po_matrix_to_constraints(pm[i].to(device)) for i in range(pm.shape[0])]
            else:
                raise ValueError(f"Unexpected po_matrix.dim()={pm.dim()}, expected 2 or 3")

        with torch.no_grad():
            for diffusion_index in range(start_step - 1, -1, -1):
                t = torch.full((B,), diffusion_index, device=device, dtype=torch.long)
                log_z = self.p_sample(log_z, cond_emb, t, batch, po_constraints=po_constraints, diffusion_index=diffusion_index)  # log_z is log_onehot

        content_token = log_onehot_to_index(log_z)

        # content_token: [B, content_len]
        content_len = content_token.shape[1]
        device = content_token.device

        category_mask = torch.zeros((B, content_len), device=device, dtype=torch.int64)
        poi_mask = torch.zeros((B, content_len), device=device, dtype=torch.int64)

        # 对每条序列单独写入 mask（每条 L_i 不同）
        for i in range(B):
            seq_len_i = int(batch.unpadded_length[i].item())
            # content layout: [0] + cat(seq_len_i) + [1] + poi(seq_len_i) + [2]
            # cat positions: 1 ... seq_len_i
            category_mask[i, 1:1+seq_len_i] = 1
            # poi positions: (seq_len_i+2) ... (seq_len_i+2+seq_len_i-1)
            poi_start = seq_len_i + 2
            poi_mask[i, poi_start:poi_start+seq_len_i] = 1

        print("[DEBUG][shape] time", batch.time.shape)
        print("[DEBUG][shape] mask", batch.mask.shape)
        print("[DEBUG][shape] tau ", batch.tau.shape)
        print("[DEBUG][shape] cond1", batch.condition1.shape)
        print("[DEBUG][shape] unpadded_length", batch.unpadded_length.shape, batch.unpadded_length.max().item())
        print("[DEBUG][shape] content_token", content_token.shape)

        return Batch(
            time=batch.time,
            condition1=batch.condition1, 
            condition2=batch.condition2,
            condition3=batch.condition3,
            condition4=batch.condition4,
            condition5=batch.condition5,
            condition6=batch.condition6,
            condition1_indicator=batch.condition1_indicator,
            condition2_indicator=batch.condition2_indicator,
            condition3_indicator=batch.condition3_indicator,
            condition4_indicator=batch.condition4_indicator,
            condition5_indicator=batch.condition5_indicator,
            condition6_indicator=batch.condition6_indicator,
            mask=batch.mask, 
            tmax=batch.tmax,
            checkin_sequences=content_token,
            category_mask=category_mask,
            poi_mask=poi_mask,
            tau=batch.tau,
            unpadded_length=batch.unpadded_length
        )
