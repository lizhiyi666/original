"""One metric schema for method and ablation tables; no privileged Full bolding."""
import csv
import json
from pathlib import Path
import statistics

from experiment_io import atomic_json
from evaluations.statistical_metrics import EVALUATION_VERSION, STATISTICAL_NAMES, require_evaluation_version

JSD=tuple((name,name,'min') for name in STATISTICAL_NAMES)
CONSTRAINTS=(('strict_ovr','OVR_strict','min'),('ovr_skip','OVR_skip','min'),
             ('pair_coverage','Pair coverage','max'),('category_coverage','Category coverage','max'),
             ('Unsat','Unsat','min'))
METHODS=(('no_projection','JointGen'),('postswap','PostSwap'),('energy','EnergyGuide'),
         ('cfg','PO-CFG (s=1)'),('full','PCDG'))
ABLATIONS=(('no_projection','No Projection'),('no_existence','No Existence'),('no_order','No Order'),
           ('no_kl','No KL'),('fixed_multipliers','Fixed Multipliers'),('no_gumbel','No Gumbel Noise'),('full','Full'))
DISTANCE_ABLATIONS = ABLATIONS[:-1] + (('no_distance_kl','No Distance KL'), ABLATIONS[-1])
DATASETS=('Istanbul_PO1_OOD','NewYork_PO1_OOD')
SEEDS=(135398,135399,135400)


def key(dataset,seed,variant):
    return f'{dataset}/{seed}/{variant}'


def aggregate(records,dataset,variant,metric):
    values=[records[key(dataset,s,variant)]['metrics'][metric] for s in SEEDS]
    if any(v is None for v in values):
        return dict(mean=None,sd=None,n=3,defined=sum(v is not None for v in values))
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),n=3,defined=3)


def panel_ready(records,dataset,rows):
    return all(key(dataset,s,v) in records for v,_ in rows for s in SEEDS)


def best_rows(stats,metric,direction):
    values={v:columns[metric]['mean'] for v,columns in stats.items() if columns[metric]['mean'] is not None}
    if not values: return set()
    target=(min if direction=='min' else max)(values.values())
    return {v for v,value in values.items() if value==target}


def tex_escape(value):
    return value.replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def table(records,rows,metrics,percent=False,title=''):
    for record in records.values():
        require_evaluation_version(record['metrics'])
    md=[]; tex=[]; summaries={}; ready=[]
    header=['Method']+[name+(' ↑' if direction=='max' else ' ↓') for _,name,direction in metrics]
    for dataset in DATASETS:
        if not panel_ready(records,dataset,rows): continue
        ready.append(dataset)
        stats={v:{m:aggregate(records,dataset,v,m) for m,_,_ in metrics} for v,_ in rows}
        summaries[dataset]=stats
        winners={m:best_rows(stats,m,direction) for m,_,direction in metrics}
        md.extend([f'### {dataset}','','| '+' | '.join(header)+' |',
                   '|---|'+'---:|'*len(metrics)])
        tex.extend([rf'\multicolumn{{{len(metrics)+1}}}{{l}}{{\textbf{{{tex_escape(dataset)}}}}} \\',
                    r'\midrule','Method & '+' & '.join(tex_escape(n)+
                    (r' $\uparrow$' if d=='max' else r' $\downarrow$') for _,n,d in metrics)+r' \\',r'\midrule'])
        for variant,label in rows:
            mc=[]; tc=[]
            for metric,_,_ in metrics:
                a=stats[variant][metric]
                if a['mean'] is None:
                    mc.append('—'); tc.append(r'\textemdash{}'); continue
                digits=2 if percent else 4
                factor=100 if percent else 1
                left=f"{a['mean']*factor:.{digits}f}"; right=f"{a['sd']*factor:.{digits}f}"
                best=variant in winners[metric]
                value=f'{left} ± {right}'
                mc.append('**'+value+'**' if best else value)
                math=left+r'\pm '+right
                tc.append('$'+(r'\mathbf{'+math+'}' if best else math)+'$')
            md.append('| '+label+' | '+' | '.join(mc)+' |')
            tex.append(tex_escape(label)+' & '+' & '.join(tc)+r' \\')
        md.append(''); tex.append(r'\midrule')
    if not ready:
        return '尚无完成全部三种子的城市面板。\n','% Table not ready; no placeholder numerical results.\n',summaries
    tex[-1]=r'\bottomrule'
    intro=[r'% Requires booktabs. Cells are mean +/- sample SD over three sampling seeds.',
           r'\begin{table*}[t]',r'\centering',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',
           r'\caption{'+tex_escape(title)+(r' (percent)' if percent else '')+'}',
           r'\begin{tabular}{l'+'r'*len(metrics)+'}',r'\toprule']
    tail=[r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{\textwidth}\footnotesize',
          r'Mean $\pm$ sample standard deviation over three sampling seeds. Bold denotes the best unrounded mean within each dataset, not statistical significance. Undefined values are shown as a dash.',
          r'\end{minipage}',r'\end{table*}']
    return '\n'.join(md),'\n'.join(intro+tex+tail)+'\n',summaries


def render(out,records,manifest,complete=False):
    for record in records.values():
        require_evaluation_version(record['metrics'])
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    declared = manifest.get('variants', {})
    distance_protocol = (manifest.get('projection_revision') == 'pcdg-distance-v2'
                         or 'no_distance_kl' in declared
                         or any(k.endswith('/no_distance_kl') for k in records))
    ablations = DISTANCE_ABLATIONS if distance_protocol else ABLATIONS
    target_count = len({v for v,_ in (*METHODS,*ablations)}) * len(DATASETS) * len(SEEDS)
    if manifest.get('total_unique_results', target_count) != target_count:
        raise ValueError('Declared result count disagrees with table protocol')
    allowed = {key(d, s, v) for d in DATASETS for s in SEEDS for v, _ in (*METHODS, *ablations)}
    if not set(records).issubset(allowed) or (complete and set(records) != allowed):
        raise ValueError('Unexpected or incomplete final result registry')
    atomic_json(out/'evaluation-schema.json',dict(evaluation_version=EVALUATION_VERSION,metrics=STATISTICAL_NAMES))
    plans=(('methods-jsd',METHODS,JSD,False,'Method comparison: distributional similarity'),
           ('methods-constraints',METHODS,CONSTRAINTS,True,'Method comparison: constraint satisfaction'),
           ('ablation-jsd',ablations,JSD,False,'Ablation: distributional similarity'),
           ('ablation-constraints',ablations,CONSTRAINTS,True,'Ablation: constraint satisfaction'))
    heading='两城市统一结果：最终汇总' if complete else '阶段性结果：尚非最终两城市汇总'
    lines=['# '+heading,'',f'当前具有可追溯记录的独立结果：{len(records)}/{target_count}。',
        f'评估版本：{EVALUATION_VERSION}；Category 为逐小时均值，CategoryTransition 为条件转移 JSD。',
        '方法对比和消融使用同一评估实现与指标列。约束指标为百分比，JSD为原量纲；均值 ± 样本标准差来自三个采样种子。',
        '按未舍入均值逐城市、逐列标最优，不预设PCDG/Full获胜；未定义值为“—”，不填零。','']
    all_summaries={}
    preview_tables=[]
    for name,rows,metrics,percent,title in plans:
        markdown,latex,summary=table(records,rows,metrics,percent,title)
        if summary:
            latex=latex.replace(r'\begin{tabular}',r'\label{tab:two-city-'+name+'}\n'+r'\begin{tabular}',1)
            preview_tables.append(latex)
        (out/f'{name}.md').write_text('# '+title+'\n\n'+markdown,encoding='utf-8')
        (out/f'{name}.tex').write_text(latex,encoding='utf-8')
        lines+=['## '+title,'',markdown]
        all_summaries[name]=summary
    atomic_json(out/'table-values.json',all_summaries)
    preview=[r'\documentclass[10pt]{article}',r'\usepackage[a4paper,margin=1.6cm]{geometry}',
             r'\usepackage{booktabs,amsmath}',r'\begin{document}',
             r'\section*{'+('Final two-city results' if complete else 'Partial results -- not the final two-city comparison')+'}',
             'Only panels with all three sampling seeds are shown. Error bars in cells are sample standard deviations.',
             *[t+r'\clearpage' for t in preview_tables],r'\end{document}']
    (out/'table-preview.tex').write_text('\n'.join(preview)+'\n',encoding='utf-8')
    rows=[]
    for dataset in DATASETS:
        for variant,_ in (*METHODS,*ablations):
            if not all(key(dataset,s,variant) in records for s in SEEDS): continue
            if any(r['dataset']==dataset and r['variant']==variant for r in rows): continue
            m={m:aggregate(records,dataset,variant,m) for m in records[key(dataset,SEEDS[0],variant)]['metrics'] if m != 'evaluation_version'}
            rows.append(dict(dataset=dataset,variant=variant,statistics=m,
                timing_by_seed={str(s):records[key(dataset,s,variant)].get('timing',{}) for s in SEEDS}))
    atomic_json(out/'supplementary.json',rows)
    with (out/'summary-long.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['dataset','variant','metric','mean','sample_sd','defined_seeds'])
        for row in rows:
            for metric,value in row['statistics'].items():
                writer.writerow([row['dataset'],row['variant'],metric,value['mean'],value['sd'],value['defined']])
    lines+=['## 口径与限制','',
        '- JointGen与No Projection、PCDG与Full在同一城市对应完全相同的三份结果，不是分别挑选或重算的不同运行。',
        '- 当前PCDG为固定10×50、无提前停止的配对协议，不使用旧单次38.05%代替Full。',
        '- PO-CFG固定scale=1，属于有偏序条件模型极限，不宣称额外CFG外推收益。',
        '- totalJSD、空输出率、原始类别token不一致率、缺失/错序贡献及效率等见supplementary.json和summary-long.csv。',
        '- 时间缓存成本与空间采样成本分开；PostSwap的空间成本包含基础JointGen生成成本，不将仅后处理耗时与完整生成耗时比较。',
        '- Istanbul使用7035/4914原划分和4月14日兼容基础检查点。基础模型历史训练batch=512，NewYork为64，不能声称跨城市基础训练设置完全一致。',
        '- Istanbul实际语义类别9类，保留历史模型10槽位及原词表；额外槽位不加约束，不重编号。',
        '- 历史Istanbul训练缺少可回溯到当时的完整数据指纹；本轮从用户提供的固定文件开始封存，不补造历史证明。',
        '- 不汇合两城市轨迹计算一个总分，不新增下游任务，不依据测试集成绩改参数。',
        '- 标准差仅反映固定检查点下的采样波动，并非多个独立训练模型的不确定性。','']
    if manifest.get('cfg_training'):
        lines+=['## CFG 独立训练成本','',json.dumps(manifest['cfg_training'],ensure_ascii=False,indent=2),'']
    lines+=['## 文件','', '[结果和来源索引](files.md)','[全部数值与配对信息](registry.json)']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    index=['# 结果与来源索引','', '| 数据集 | 种子 | 结果 | 来源 | SHA-256 |','|---|---:|---|---|---|']
    for _,record in sorted(records.items()):
        link=record.get('delivery_path',record['path'])
        index.append(f"| {record['dataset']} | {record['seed']} | {record['variant']} | [{record['origin']}]({link}) | {record['sha256']} |")
    (out/'files.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    (out/'registry.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
