"""Read-only manuscript integrity checks; no scientific code or experiment imports.

Run with a Python environment providing pypdf. Build main.pdf first with latexmk.
An integrity pass is NOT a scientific-readiness or official CCS approval.
"""
from collections import Counter
from pathlib import Path
import re
import subprocess
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper-www2027"
BASELINE = "c828598216bd4949fd12f1e59aa1eca9ccba0e72"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def data_rows(text):
    return Counter(line.strip() for line in text.splitlines()
                   if " & " in line and re.search(r"\d+\.\d+", line)
                   and line.rstrip().endswith(r"\\"))


def main():
    original = git("show", f"{BASELINE}:paper-www2027/main.tex")
    current = (PAPER / "main.tex").read_text(encoding="utf-8")
    patterns = {
        "title": r"\\title\{[^\n]+\}",
        "abstract": r"\\begin\{abstract\}.*?\\end\{abstract\}",
        "keywords": r"\\keywords\{.*?\}",
        "author order": r"\\author\{[^}]+\}",
    }
    for label, pattern in patterns.items():
        old = re.findall(pattern, original, re.S)
        new = re.findall(pattern, current, re.S)
        assert old and old == new, f"Locked {label} changed"
        print(f"PASS locked {label}")
    for filename in ("method-comparison.tex", "ablation.tex", "stagewise.tex"):
        old = git("show", f"{BASELINE}:paper-www2027/tables/{filename}")
        new = (PAPER / "tables" / filename).read_text(encoding="utf-8")
        assert data_rows(old) and data_rows(old) == data_rows(new), filename
        print(f"PASS unchanged numeric rows / precision / bolding: {filename}")
    for path in [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\\todo\s*\{", text), path
        assert "\ufffd" not in text, f"Replacement character in {path}"
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace")
    failures = re.findall(r"^!.*|^.*(?:LaTeX Error|Package \w+ Error|Undefined control sequence).*$", log, re.M)
    assert not failures, failures
    assert not re.search(r"(?:Citation|Reference).*undefined|There were undefined|Rerun to get", log)
    assert "Overfull" not in log
    print("PASS no TODO calls, LaTeX errors, unresolved references/citations, or overfull boxes")
    aux = (PAPER / "main.aux").read_text(encoding="utf-8")
    body_page = int(re.search(r"\\newlabel\{end:main\}\{\{[^}]+\}\{(\d+)\}", aux).group(1))
    appendix_page = int(re.search(r"\\newlabel\{app:algo\}\{\{[^}]+\}\{(\d+)\}", aux).group(1))
    assert body_page <= 8 and appendix_page <= 9
    reader = PdfReader(PAPER / "main.pdf")
    assert len(reader.pages) <= 12
    texts = [page.extract_text() for page in reader.pages]
    assert "Relevance to the Web" in texts[0]
    assert not re.search(r"Zhiyi|Harbin|hit\.edu|25s151146|zhaokaiqi|\bTODO\b|lizhiyi666", "\n".join(texts), re.I)
    metadata = str(reader.metadata)
    if reader.trailer["/Root"].get("/Metadata"):
        metadata += reader.trailer["/Root"]["/Metadata"].get_object().get_data().decode("utf-8", errors="replace")
    assert not re.search(r"Zhiyi|Kaiqi|Harbin|hit\.edu|Li and Zhao", metadata, re.I)
    urls = []
    for page in reader.pages:
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A", {})
            if hasattr(action, "get_object"):
                action = action.get_object()
            uri = str(action.get("/URI", ""))
            if uri:
                urls.append(uri)
    assert not any(re.search(r"lizhiyi666|hit\.edu|25s151146|file:|[CD]:", uri, re.I) for uri in urls)
    font_rows = subprocess.check_output(["pdffonts", str(PAPER / "main.pdf")]).decode().splitlines()[2:]
    assert font_rows and all(row.split()[-5] == "yes" for row in font_rows if row.strip())
    assert not any("Type 3" in row for row in font_rows)
    changed = git("diff", "--name-only", BASELINE, "--", "experiment_runs", "constraint_projection.py", "geometry_projection.py", "distribution_balance.py", "discrete_diffusion", "evaluations")
    assert not changed, f"Scientific inputs or implementation changed: {changed}"
    print(f"PASS PDF: {len(reader.pages)} pages; main text ends p{body_page}; appendix p{appendix_page}; embedded fonts; anonymous text/metadata/links")
    print("PASS no changes to sealed experiments or audited scientific implementation")
    print("OPEN: locked abstract's 'eliminates' conflicts with nonzero decoder mismatch; official CCS export still required")


if __name__ == "__main__":
    main()
