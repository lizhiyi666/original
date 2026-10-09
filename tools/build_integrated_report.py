"""Build the integrated research report from verified, immutable local results."""
from collections import Counter
from pathlib import Path
import json
import os
import statistics
from xml.sax.saxutils import escape
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluations.statistical_metrics import EVALUATION_VERSION, STATISTICAL_NAMES

from integrate_experiment_results import ROOT, RUNS, OUT, LEGACY, BASELINES, read, save, digest

SOURCE = OUT / 'sources/two-city-v1'
METHODS = [('no_projection','JointGen'),('postswap','PostSwap'),('energy','EnergyGuide'),('cfg','PO-CFG (s=1)'),('full','PCDG')]
ABLATIONS = [('no_projection','No Projection'),('no_existence','No Existence'),('no_order','No Order'),('no_kl','No KL'),('fixed_multipliers','Fixed Multipliers'),('no_gumbel','No Gumbel Noise'),('full','Full')]
JSD = list(STATISTICAL_NAMES)
CONS = ['strict_ovr','ovr_skip','pair_coverage','category_coverage','Unsat']
CON_LABELS = ['OVR_strict','OVR_skip','Pair coverage','Category coverage','Unsat']
PANELS = [('methods-jsd','方法对比：分布相似性',METHODS,JSD,JSD,False),
          ('methods-constraints','方法对比：约束满足',METHODS,CONS,CON_LABELS,True),
          ('ablation-jsd','消融实验：分布相似性',ABLATIONS,JSD,JSD,False),
          ('ablation-constraints','消融实验：约束满足',ABLATIONS,CONS,CON_LABELS,True)]
DATASETS = ['Istanbul_PO1_OOD','NewYork_PO1_OOD']


def relative(path):
    return Path(os.path.relpath(path, OUT)).as_posix()


def link(label, path):
    return f'[{label}](<{relative(path)}>)'


def fmt(value, percent=False):
    if value['mean'] is None: return '-'
    factor,digits=(100,2) if percent else (1,4)
    return f"{value['mean']*factor:.{digits}f} ± {value['sd']*factor:.{digits}f}"


def require_v2_source():
    schema = SOURCE / 'evaluation-schema.json'
    if not schema.exists() or read(schema).get('evaluation_version') != EVALUATION_VERSION:
        raise ValueError('Integrated report requires v2 metrics; preserve historical reports and recompute into a new source')


def collect():
    require_v2_source()
    assert read(OUT/'local-audit.json')['state']=='passed'
    # Calibration remains separate: it uses training data, not held-out OOD data.
    calibration=[r for r in read(ROOT/'calibration_runs/perfcal-v1/results.json') if r['kind']!='regression']
    errors=[]
    for r in calibration:
        path=ROOT/'calibration_runs/perfcal-v1'/r['label']/'generated.pkl'
        if not path.exists() or digest(path)!=r['output_sha256']: errors.append(path.as_posix())
    save(OUT/'calibration-audit.json',dict(state='passed' if not errors else 'failed',checked=len(calibration),errors=errors,
                                        split='train',independent_validation=False))
    if errors: raise RuntimeError(errors)
    series=[('two-city-v1',RUNS/'two-city-v1'),('pcdg-ablation-v1',RUNS/'pcdg-ablation-v1'),
            (BASELINES,RUNS/BASELINES),(LEGACY,RUNS/LEGACY),('perfcal-v1',ROOT/'calibration_runs/perfcal-v1')]
    inventory=[]
    for name,folder in series:
        for p in sorted(folder.rglob('*')):
            if p.is_file(): inventory.append(dict(series=name,path=p.as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    # Older Hydra directories contain logs/configs, not comparable audited result tables.
    old_logs=[dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted((ROOT/'outputs').rglob('*')) if p.is_file()]
    save(OUT/'file-inventory.json',dict(files=inventory,older_training_logs=old_logs,
         note='Raw historical logs are catalogued only. They are not counted as audited OOD results. Final two-city metadata overrides are mapped in merge-audit.json.'))
    return calibration,inventory,old_logs


def markdown(calibration,inventory,old_logs):
    require_v2_source()
    tables=read(SOURCE/'table-values.json')
    merge=read(OUT/'merge-audit.json')
    index=read(OUT/'result-index.json')
    lines=['# 实验结果总汇','', '整合日期：2026-10-05。范围：当前工作区与前序计划中的五组正式实验/工程校准。',
           '', '**结论：双城市主实验已完成 60/60；计入早期单种子 OOD 对照后，去重共有 65 份 OOD 生成结果。另有 10 组训练集工程校准，不计入 OOD 成绩。**',
           '', '## 从这里阅读','',
           '- '+link('7页综合报告与表格 PDF',OUT/'experiment-summary.pdf'),
           '- '+link('双城市最终完整报告',SOURCE/'report.md'),
           '- '+link('逐结果索引（含本地路径、哈希、种子和分片）',OUT/'result-index.json'),
           '- '+link('可筛选长表 CSV（原始最终文件，未改写）',SOURCE/'summary-long.csv'),
           '- '+link('LaTeX 四表预览源文件',SOURCE/'table-preview.tex'),
           '- '+link('本地核验',OUT/'local-audit.json')+' / '+link('服务器核验',OUT/'server-audit.json')+' / '+link('文件清单',OUT/'file-inventory.json'),
           '', '## 实验关系与去重','',
           '| 实验组 | 有效结果 | 本次汇总角色 | 来源 |','|---|---:|---|---|',
           f'| 两城市统一对比与消融 | 60 | 主结果；两城市各 30，39 新增 + 21 复用 | {link("最终报告",SOURCE/"report.md")} |',
           f'| NewYork 三种子消融 | 21 | 全部已进入上述 60 份，不再累加 | {link("消融报告",RUNS/"pcdg-ablation-v1/report.md")} |',
           f'| NewYork 单种子基线 | 3 新增 | 历史参照；表内 native/projection 复用早期 2 份 | {link("基线报告",RUNS/BASELINES/"report.md")} |',
           f'| NewYork 完整 OOD 重采样 | 2 | 历史 native/projection 对照 | {link("重采样报告",RUNS/LEGACY/"report.md")} |',
           f'| 训练集工程校准 | 10 | 128/512 条训练条件；不能作为 OOD 成绩 | {link("校准报告",ROOT/"calibration_runs/perfcal-v1/report.md")} |',
           '', 'JointGen = No Projection，PCDG = Full：每个城市的对应三份文件及哈希完全相同，只是表中别名。两城市主表共有 20 个独立“城市 × 方法/变体”组合，每个组合 3 个采样种子。',
           '', '## 主要结果','', '| 数据集 | JointGen 严格违反率 | PCDG 严格违反率 | 降低（百分点） | PCDG 类别对覆盖率 |', '|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        t=tables['methods-constraints'][ds]
        delta=100*(t['no_projection']['strict_ovr']['mean']-t['full']['strict_ovr']['mean'])
        lines.append(f"| {ds} | {fmt(t['no_projection']['strict_ovr'],True)}% | {fmt(t['full']['strict_ovr'],True)}% | {delta:.2f} | {fmt(t['full']['pair_coverage'],True)}% |")
    lines += ['', '- 五方法对比中，PCDG 在两城市的严格违反率、类别对覆盖率、类别覆盖率及 Unsat 均值最好，但并非所有分布相似性指标最好。',
              '- PO-CFG 在 NewYork 的 6 个分布指标中有 5 个均值最优；Istanbul 的 Distance、G-RANK 最优。约束满足与分布拟合存在取舍。',
              '- PostSwap 的 OVR_skip 为 0，但不能补回缺失类别；两城市严格违反率仍较高，不能把 skip=0 解读为全部约束满足。',
              '- 消融结果支持存在性项、顺序项和乘子更新在当前协议中的作用；去 KL、去 Gumbel 并未一致恶化结果，不应宣称所有模块均有显著收益。',
              '- 以上是均值层面的描述，未做显著性检验。三种子样本标准差只反映固定检查点的采样随机性。',
              '', '## 双城市主表与消融表','']
    for name,title,*_ in PANELS:
        lines += ['### '+title,'', (SOURCE/(name+'.md')).read_text(encoding='utf-8').split('\n',1)[1],
                  link('对应 LaTeX 表片段',SOURCE/(name+'.tex')),'']
    lines += ['## 历史实验：单独保留，不混算','',
              '旧单次投影严格违反率为 38.05%，三种子 Full 为 38.48 ± 1.04%。随机协议和提前停止策略不同，二者不可替换，也不能把差异解释为模型退化。',
              '早期单次基线五方法结果详见 '+link('历史对比',RUNS/BASELINES/'report.md')+'，完整数字原样保存在 comparison.json；两份早期结果被该报告复用，不重复计数。',
              '', '## 工程校准：仅训练集','',
              '| 阶段/候选 | 条件数 | GPU | 温度 | batch | 严格违反率 % | 类别对覆盖率 % | 条/秒 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in calibration:
        lines.append(f"| {r['label']} | {r['samples']} | {r['physical_gpu']} | {r['temperature']} | {r['batch_size']} | {r['metrics']['strict_ovr']*100:.2f} | {r['metrics']['pair_coverage']*100:.2f} | {r['samples_per_second']:.3f} |")
    lines += ['', '温度初筛为 128 条，batch 筛选和第二卡复核为 512 条，跨阶段不可直接比较质量。固定设置为 T=3、batch=64、10×50；纯投影同预算短测试加速 1.184×，不等同于端到端采样加速。',
              '', '## 审计与保留策略','',
              f"- 服务器核验：60 份结果、120 个源分片、39 份同步凭据、501 个受保护文件、59 个生产代码文件及模型/数据指纹通过。",
              f"- 完整服务器清单 1,148 个文件，1,413,658,451 字节。新增本地文件 {merge['counts']['added']} 个；{merge['counts']['same_hash']} 个同哈希跳过；{merge['counts']['versioned_conflict']} 个冲突文件版本化保存。整合前的 19 个本地文件全部保持原哈希。",
              '- 本地 60+5 份 OOD 结果、120 个主实验源分片及 10 组工程校准输出通过 SHA-256 校验。264 个表格统计单元独立复算通过。跨 Python 版本的最大浮点差为 6.94e-18，不影响显示结果。',
              '- 本次重新执行的是文件哈希和汇总统计核验。逐轨迹指标、分片合并内容及随机流一致性的完整审计引用已封存的服务器主审计，不声称本次重新执行了 GPU 实验。',
              '- 旧 two-city-v1 根目录的 report/status/registry 可能仍是中间版本，已刻意保留。请以本目录 sources/two-city-v1 的最终元数据为准；实际生成文件位于原运行目录，result-index.json 提供准确本地路径。',
              '- LaTeX 源文件及四个片段原样保留。内置编译器返回 “Unable to find standard directories for platform”，未确认 TeX 编译；交付 PDF 是由相同已核验数值独立排版的预览，不冒充 LaTeX 编译产物。',
              f'- 文件清单覆盖 {len(inventory)} 个实验/校准文件；另外列出 outputs 中 {len(old_logs)} 个历史日志/配置文件，它们没有本轮一致口径的审计指标，因此不并入结果统计。',
              '', '## 必须保留的实验限制','',
              '- 不合并两城市轨迹计算一个总分；不把训练集校准结果当验证集或测试集成绩。',
              '- NewYork 为 3160/2108、Istanbul 为 7035/4914 的训练/测试划分。历史基础训练 batch 分别为 64/512，不能宣称跨城市训练设置完全一致。',
              '- Istanbul 保留 9 类语义、10 个历史模型槽位；历史训练缺乏当时完整数据指纹，本轮封存不能补证历史溯源。',
              '- CFG scale 固定为 1，不宣称额外外推收益。NewYork CFG 1000 轮/50000 步；Istanbul CFG 1000 轮/110000 步；时间权重冻结，独立空间训练成本另计。',
              '- 所有现有训练/采样结果原样保留，本次没有重新训练、重新采样或修改实验参数。',
              '', '## 65 份 OOD 结果本地索引','', '| 编号 | 数据集 | 种子 | 方法/变体 | 来源 | 文件 |','|---:|---|---:|---|---|---|']
    for i,r in enumerate(index,1):
        lines.append(f"| {i} | {r['dataset']} | {r['seed']} | {r['variant']} | {r['origin']} | {link('PKL',Path(r['local_path']))} |")
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def pdf(calibration):
    require_v2_source()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNHeading','C:/Windows/Fonts/simhei.ttf'))
    width,height=landscape(A4); usable=width-80
    style=ParagraphStyle('body',fontName='CN',fontSize=11,leading=17,spaceAfter=9,wordWrap='CJK')
    small=ParagraphStyle('small',parent=style,fontSize=9,leading=14)
    title=ParagraphStyle('title',parent=style,fontName='CNHeading',fontSize=23,leading=29,spaceAfter=13,textColor=colors.HexColor('#163D52'))
    sub=ParagraphStyle('sub',parent=style,fontName='CNHeading',fontSize=14,leading=20,spaceBefore=10,spaceAfter=7,textColor=colors.HexColor('#163D52'))
    story=[]
    def p(text, sty=style): return Paragraph(escape(text),sty)
    def add(text,sty=style): story.append(p(text,sty))
    def table(rows,widths=None,size=10,rowheight=None,best=()):
        t=Table(rows,colWidths=widths,rowHeights=rowheight,hAlign='LEFT',repeatRows=1)
        cmds=[('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),size),
              ('TEXTCOLOR',(0,0),(-1,0),colors.white),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#24546A')),
              ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('ALIGN',(1,1),(-1,-1),'RIGHT'),
              ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
              ('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#B8C6CF')),
              ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EFF4F7'),colors.white])]
        for col,row in best: cmds.append(('FONTNAME',(col,row),(col,row),'Helvetica-Bold'))
        t.setStyle(TableStyle(cmds)); story.append(t);story.append(Spacer(1,8))
    values=read(SOURCE/'table-values.json')
    add('轨迹生成实验结果总汇',title)
    add('2026-10-05  |  最终结果、历史对照与工程校准',sub)
    add('双城市主实验：60/60 完成。39 份新增、21 份复用；早期单种子对照另有 5 份独立结果。10 组训练集工程校准单独报告。')
    rows=[['Dataset','JointGen OVR (%)','PCDG OVR (%)','Reduction (pp)','PCDG pair cov. (%)']]
    for ds in DATASETS:
        d=values['methods-constraints'][ds]
        rows.append([ds.replace('_PO1_OOD',''),fmt(d['no_projection']['strict_ovr'],True),fmt(d['full']['strict_ovr'],True),
                     f"{100*(d['no_projection']['strict_ovr']['mean']-d['full']['strict_ovr']['mean']):.2f}",fmt(d['full']['pair_coverage'],True)])
    table(rows,[110,170,160,140,usable-580])
    add('主要结论',sub)
    for text in ['PCDG 在五方法对比中的严格违反率、两项覆盖率及 Unsat 均值最好，但不是所有分布指标最好；PO-CFG 在多项分布指标上更优。',
                 'PostSwap 的 OVR_skip 为零不表示所有约束满足：它不能补回缺失类别。约束满足与分布拟合必须同时报告。',
                 '存在性项、顺序项与乘子更新的作用有当前结果支持；去 KL / 去 Gumbel 未一致恶化，不宣称所有模块都有显著收益。']:
        add(text)
    add('统计口径',sub)
    add('数值为固定检查点下三个采样种子（135398、135399、135400）的均值 ± 样本标准差；非独立训练不确定性。粗体标识同城市、同表的未舍入最优均值，不表示统计显著性。',small)
    for name,heading,variants,metrics,labels,percent in PANELS:
        story.append(PageBreak());add(heading,title)
        add(('所有数值以百分比表示。' if percent else 'JSD 保持原量纲，越低越好。')+'固定检查点，三个采样种子。',small)
        for ds in DATASETS:
            add(ds+'  |  '+('n=4914' if ds.startswith('Istanbul') else 'n=2108'),sub)
            stats=values[name][ds]
            rows=[['Method']+[label+(' (+)' if m in ('pair_coverage','category_coverage') else ' (-)') for m,label in zip(metrics,labels)]]
            best=[]
            for v,label in variants: rows.append([label]+[fmt(stats[v][m],percent) for m in metrics])
            for col,m in enumerate(metrics,1):
                nums=[stats[v][m]['mean'] for v,_ in variants if stats[v][m]['mean'] is not None]
                goal=(max if m in ('pair_coverage','category_coverage') else min)(nums)
                best += [(col,row) for row,(v,_) in enumerate(variants,1) if stats[v][m]['mean']==goal]
            table(rows,[135]+[(usable-135)/len(metrics)]*len(metrics),size=9.5,best=best)
        add('(+): higher is better; (-): lower is better. JointGen = No Projection; PCDG = Full. Source: sources/two-city-v1/table-values.json',small)
    story.append(PageBreak());add('历史对照与训练集工程校准',title)
    add('早期单种子 NewYork OOD：仅作历史参照，不与主实验混算',sub)
    comparison=read(RUNS/BASELINES/'comparison.json')
    rows=[['Method (historical)','Strict OVR (%)','Pair cov. (%)','Category cov. (%)','totalJSD']]
    for method,label in [('baseline1','Native'),('baseline2','PostSwap'),('baseline3','EnergyGuide'),('baseline4','PO-CFG (s=1)'),('projection','Projection')]:
        m=comparison[method]['metrics']
        rows.append([label,f"{100*m.get('strict_ovr',m.get('OVR_ref_strict')):.2f}",f"{100*m['pair_coverage']:.2f}",f"{100*m.get('category_coverage',m.get('coverage')):.2f}",f"{m['totalJSD']:.6f}"])
    table(rows,[165]+[(usable-165)/4]*4,size=9,rowheight=22)
    add('工程校准：128/512 条模型已见训练条件，不能作为独立验证或 OOD 成绩',sub)
    rows=[['Candidate','N','GPU','T','batch','Strict OVR (%)','Pair cov. (%)','samples/s']]
    for r in calibration:
        rows.append([r['label'],r['samples'],r['physical_gpu'],r['temperature'],r['batch_size'],f"{100*r['metrics']['strict_ovr']:.2f}",f"{100*r['metrics']['pair_coverage']:.2f}",f"{r['samples_per_second']:.3f}"])
    table(rows,[180,45,45,45,55,120,120,usable-610],size=8.5,rowheight=20)
    add('跨阶段样本池不同，不直接比较质量；最终固定 T=3、batch=64、10×50。旧单次 38.05% 不能替代主实验 Full 三种子均值。',small)
    story.append(PageBreak());add('审计、来源与解释边界',title)
    for heading,text in [
        ('交付与完整性','服务器 1,148 个文件（约 1.41 GB）逐文件核验；60 份主结果、120 个源分片、39 份同步凭据、501 个受保护文件通过。原本地 19 个文件未改；16 个冲突元数据保留新旧两版。'),
        ('本次重新执行的检查','65 份 OOD 结果及 10 组工程校准输出哈希通过；264 个表格统计单元复算通过。逐轨迹指标、配对随机流和分片内容一致性引用封存的服务器审计。本次未重训、未重采样。'),
        ('训练口径与历史溯源','NewYork/Istanbul 基础训练历史 batch 为 64/512。Istanbul 的历史训练数据缺少当时完整指纹；保留 9 语义类别、10 模型槽位。不能把本轮封存当作历史数据身份的证明。'),
        ('CFG 与时间成本','CFG scale=1，不宣称额外外推收益。NewYork/Istanbul CFG 各训练 1000 轮，分别 50000/110000 步；独立空间训练成本约 4013.10/8345.91 秒。时间模型冻结，空间采样和缓存成本分开。'),
        ('最终文件入口','README.md 为完整中文总报告；result-index.json 为 65 份 OOD 输出及主实验源分片索引；sources/two-city-v1 包含最终报告、CSV、四个 LaTeX 片段及统计 JSON；merge-audit.json 映射冲突文件。'),
        ('LaTeX 编译状态','原 LaTeX 文件未改。内置编译器因平台目录错误未能编译；本 PDF 由同一份已核验统计 JSON 独立排版，不是 TeX 编译产物。旧根目录阶段性报告保留，请使用 sources 中的最终版本。')]:
        add(heading,sub);add(text)
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#5B6B76'))
        canvas.drawString(40,21,'PCDG | Experiment results integration | 2026-10-05')
        canvas.drawRightString(width-40,21,str(doc.page))
    doc=SimpleDocTemplate(str(OUT/'experiment-summary.pdf'),pagesize=(width,height),leftMargin=40,rightMargin=40,topMargin=32,bottomMargin=36,
                          title='轨迹生成实验结果总汇',author='Research experiment integration')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    calibration,inventory,old_logs=collect()
    markdown(calibration,inventory,old_logs)
    pdf(calibration)
    print(json.dumps(dict(experiment_files=len(inventory),older_log_files=len(old_logs),calibration_results=len(calibration),report=str(OUT/'README.md')),ensure_ascii=False))
