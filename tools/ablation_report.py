"""Aggregate sampling-seed variation; never reinterpret it as training-seed uncertainty."""
import csv
import json
from pathlib import Path

from experiment_io import atomic_json
from tools.ablation_common import mean_sd
from evaluations.statistical_metrics import require_evaluation_version

LABELS={'full':'Full','no_projection':'No Projection','no_existence':'No Existence',
        'no_order':'No Order','no_kl':'No KL','no_distance_kl':'No Distance KL','fixed_multipliers':'Fixed Multipliers','no_gumbel':'No Gumbel Noise'}


def make_report(directory,seeds,variants):
    directory=Path(directory)
    runs={v:{s:json.loads((directory/f'seed-{s}'/v/'metrics.json').read_text()) for s in seeds} for v in variants}
    for by_seed in runs.values():
        for record in by_seed.values():
            require_evaluation_version(record['metrics'])
    summary={}
    keys=[key for key in runs[variants[0]][seeds[0]]['metrics'] if key != 'evaluation_version']
    for variant in variants:
        summary[variant]={key:mean_sd([runs[variant][s]['metrics'][key] for s in seeds]) for key in keys}
        summary[variant]['spatial_wall_seconds']=mean_sd([runs[variant][s]['timing']['spatial_wall_seconds'] for s in seeds])
    differences={}
    if 'full' in variants:
        for variant in variants:
            differences[variant]={}
            for key in keys:
                vals=[None if runs[variant][s]['metrics'][key] is None or runs['full'][s]['metrics'][key] is None
                      else runs[variant][s]['metrics'][key]-runs['full'][s]['metrics'][key] for s in seeds]
                differences[variant][key]=dict(by_seed=dict(zip(map(str,seeds),vals)),**mean_sd(vals))
            paired=[]
            for seed in seeds:
                full=json.loads((directory/f'seed-{seed}'/'full'/'per_condition.json').read_text())
                other=json.loads((directory/f'seed-{seed}'/variant/'per_condition.json').read_text())
                for a,b in zip(full,other):
                    if a['index']!=b['index']:
                        raise RuntimeError('Paired condition indices differ')
                    paired.append(dict(seed=seed,index=a['index'],strict_ovr_difference=None
                        if a['strict_ovr'] is None else b['strict_ovr']-a['strict_ovr']))
            atomic_json(directory/f'paired-{variant}.json',paired)
    atomic_json(directory/'summary.json',summary)
    atomic_json(directory/'paired_differences.json',differences)
    with (directory/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['variant','metric','mean','sample_sd','sampling_seeds'])
        for variant,values in summary.items():
            for key,value in values.items():
                writer.writerow([variant,key,value['mean'],value['sample_sd'],value['n']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['svg.fonttype']='none'
    x=np.arange(len(variants))
    missing=[100*summary[v]['missing_contribution']['mean'] for v in variants]
    ordering=[100*summary[v]['order_contribution']['mean'] for v in variants]
    fig,ax=plt.subplots(figsize=(10,5.4))
    ax.bar(x,missing,label='Missing-category contribution',color='#c96858')
    ax.bar(x,ordering,bottom=missing,label='Ordering contribution',color='#467da8')
    ax.set_xticks(x,[LABELS[v] for v in variants],rotation=25,ha='right')
    ax.set_ylabel('Strict violation rate (%)')
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ('png','svg'):
        fig.savefig(directory/f'violation-decomposition.{suffix}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.5))
    for variant in variants:
        a,b=summary[variant]['totalJSD'],summary[variant]['strict_ovr']
        if a['mean'] is None:
            continue
        ax.errorbar(a['mean'],100*b['mean'],xerr=a['sample_sd'] or 0,
                    yerr=100*(b['sample_sd'] or 0),fmt='o',capsize=3,label=LABELS[variant])
    ax.set_xlabel('totalJSD (existing evaluation definition; lower is better)')
    ax.set_ylabel('Strict violation rate (%)')
    ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.tight_layout()
    for suffix in ('png','svg'):
        fig.savefig(directory/f'constraint-quality-tradeoff.{suffix}',dpi=180)
    plt.close(fig)
    def cell(value,percent=False):
        if value['mean'] is None:
            return '未定义'
        k=100 if percent else 1
        sd='—' if value['sample_sd'] is None else f"{k*value['sample_sd']:.3f}"
        return f"{k*value['mean']:.3f} ± {sd}"
    lines=['# PCDG 消融实验：固定检查点，三种采样种子','',
        '固定时间缓存、独立空间/投影随机流；有效投影固定10×50，无提前停止。未重新训练。',
        '以下均值与样本标准差只反映采样随机性，不是独立训练种子的不确定性。','',
        '| 变体 | 严格违反率 % | 缺失贡献 % | 错序贡献 % | 类别对覆盖率 % | totalJSD | 空间阶段秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        values=summary[v]
        cols=[cell(values[k],k!='totalJSD' and k!='spatial_wall_seconds') for k in
              ('strict_ovr','missing_contribution','order_contribution','pair_coverage','totalJSD','spatial_wall_seconds')]
        lines.append('| '+LABELS[v]+' | '+' | '.join(cols)+' |')
    lines+=['','## 解释与成本口径','',
        '- 不能将旧的38.05%成绩代入本表：随机协议和提前停止策略已改变，Full也重新生成。',
        '- Fixed Multipliers仅去掉λ/μ更新，不是删除所有ALM罚项。',
        '- KL方向仍为模型分布到投影分布，KL按原batch分母归一化，罚项保持求和。',
        '- 零梯度比例只统计每次投影首次更新前、有效且尚未满足的启用约束探针。',
        '- 时间缓存生成成本在各cache/result.json中单列；空间耗时不包含缓存生成和模型加载，不与旧未缓存端到端耗时直接比较。',
        '- 空输出保留；没有共同出现的类别对时，OVR_skip未定义而不是0；缺少事件分布时统计量也标记未定义。',
        '- 原始类别token不一致率在评估重映射marks之前计算；逐条件缺失/错序分解沿用OVR宏平均权重。',
        '- 所有统计指标与三种子配对差值见summary.csv、summary.json、paired_differences.json。',
        '- 图见violation-decomposition.png/svg和constraint-quality-tradeoff.png/svg。',
        '- 原始结果、源分片、缓存、逐条件诊断、随机流状态和审计文件均保留。','']
    (directory/'report.md').write_text('\n'.join(lines),encoding='utf-8')
