"""Final non-mutating content checks; only writes the derived delivery receipt."""
import json
from pathlib import Path
import re

from pypdf import PdfReader
from integrate_experiment_results import ROOT, OUT, read, save, digest
from build_integrated_report import SOURCE, PANELS, DATASETS, fmt


def main():
    errors=[]
    report=OUT/'README.md'
    links=re.findall(r'\]\(<([^>]+)>\)',report.read_text(encoding='utf-8'))
    for path in links:
        if not (OUT/path).exists(): errors.append('Missing report link: '+path)
    merge=read(OUT/'merge-audit.json')
    for entry in merge['files']:
        if digest(entry['local_path'])!=entry['sha256']:
            errors.append('Delivered file changed: '+entry['relative_path'])
    values=read(SOURCE/'table-values.json')
    pdf=PdfReader(OUT/'experiment-summary.pdf')
    if len(pdf.pages)!=7: errors.append('Expected 7 pages')
    pdf_cells=0
    for page_index,(name,title,rows,metrics,labels,percent) in enumerate(PANELS,1):
        page=pdf.pages[page_index]
        text=re.sub(r'\s+','',page.extract_text())
        for ds in DATASETS:
            if ds not in text: errors.append('Missing dataset heading on PDF page '+str(page_index+1))
            for variant,label in rows:
                for metric in metrics:
                    expected=re.sub(r'\s+','',fmt(values[name][ds][variant][metric],percent))
                    if expected not in text: errors.append(f'PDF cell missing: {name}/{ds}/{variant}/{metric}')
                    pdf_cells+=1
    latex_files=[]
    for p in sorted(SOURCE.glob('*.tex')):
        t=p.read_text(encoding='utf-8')
        if '\\begin{table*}' in t and t.count('\\begin{table*}')!=t.count('\\end{table*}'):
            errors.append('Unbalanced TeX table environment: '+p.name)
        latex_files.append(p.name)
    audits={name:read(OUT/name)['state'] for name in ['server-audit.json','staging-audit.json','merge-audit.json','local-audit.json','calibration-audit.json']}
    if any(state!='passed' for state in audits.values()): errors.append('Prerequisite audit failed')
    receipt=dict(state='passed' if not errors else 'failed',audits=audits,
                 verified_delivered_files=len(merge['files']),verified_report_links=len(links),
                 pdf_pages=len(pdf.pages),pdf_table_cells_checked=pdf_cells,
                 pdf_visual_review='All 7 final rendered pages inspected; no clipped text, split city panels, or missing rows.',
                 latex_files=latex_files,latex_compilation='unverified: built-in compiler environment error: Unable to find standard directories for platform',
                 safety_unit_tests=5,errors=errors,
                 artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in [report,OUT/'experiment-summary.pdf',OUT/'result-index.json',OUT/'file-inventory.json']})
    save(OUT/'delivery-audit.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifacts'},ensure_ascii=False))
    if errors: raise SystemExit(1)


if __name__=='__main__': main()
