"""Read-only manuscript integrity checks; no scientific code or experiment imports.

Run with a Python environment providing pypdf. Build main.pdf first with latexmk.
An integrity pass is NOT a scientific-readiness or official CCS approval.
"""
from collections import Counter
import argparse
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper-www2027"
# Modification checkpoint for this revision; it is already synchronized with origin.
BASELINE = "c28af0efc8bc8383d9a29109f034a2bcac46ae81"
# Official https://dl.acm.org/ccs Generate Code export, checked 2026-10-18.
OFFICIAL_CCS = {
    ("10002951.10003227.10003236.10003101", "Information systems~Location based services", "500"),
    ("10002951.10003227.10003351", "Information systems~Data mining", "300"),
    ("10010147.10010257", "Computing methodologies~Machine learning", "500"),
    ("10010147.10010257.10010293.10010294", "Computing methodologies~Neural networks", "100"),
    ("10003120.10003130.10003131.10003292", "Human-centered computing~Social networks", "300"),
}


def verify_ccs(current):
    blocks = re.findall(r"\\begin\{CCSXML\}(.*?)\\end\{CCSXML\}", current, re.S)
    assert len(blocks) == 1, "Expected exactly one CCSXML block"
    root = ET.fromstring(blocks[0].strip())
    assert root.tag == "ccs2012"
    concepts = [tuple(node.findtext(field) for field in
                      ("concept_id", "concept_desc", "concept_significance"))
                for node in root.findall("concept")]
    assert len(concepts) == len(OFFICIAL_CCS) and set(concepts) == OFFICIAL_CCS
    descriptions = re.findall(r"\\ccsdesc\[(\d+)\]\{([^}]+)\}", current)
    expected = {(weight, description) for _, description, weight in OFFICIAL_CCS}
    assert len(descriptions) == len(expected) and set(descriptions) == expected
    print("PASS CCSXML IDs / descriptions / weights and five matching ccsdesc commands")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def data_rows(text):
    rows = []
    for line in text.splitlines():
        if " & " not in line or not re.search(r"\d+\.\d+", line) or not line.rstrip().endswith(r"\\"):
            continue
        numbers = tuple(round(float(value), 3)
                        for value in re.findall(r"(?<![A-Za-z])\d+\.\d+", line))
        structure = re.sub(r"(?<![A-Za-z])\d+\.\d+", "<NUM>", line.strip())
        rows.append((structure, numbers))
    return rows


def equivalent_data_rows(original, current, tolerance=0.00101):
    """Allow display rounding while preserving row structure and values."""
    if len(original) != len(current):
        return False
    for (old_structure, old_numbers), (new_structure, new_numbers) in zip(original, current):
        if old_structure != new_structure or len(old_numbers) != len(new_numbers):
            return False
        if any(abs(old - new) > tolerance for old, new in zip(old_numbers, new_numbers)):
            return False
    return True


def verify_locked_metadata(original, current):
    """Keep title, keywords and author order fixed; abstract edits are scoped below."""
    patterns = {
        "title": r"\\title\{[^\n]+\}",
        "keywords": r"\\keywords\{.*?\}",
        "author order": r"\\author\{[^}]+\}",
    }
    for label, pattern in patterns.items():
        old = re.findall(pattern, original, re.S)
        new = re.findall(pattern, current, re.S)
        assert old and old == new, f"Locked {label} changed"
        print(f"PASS locked {label}")
    assert len(re.findall(r"\\begin\{abstract\}.*?\\end\{abstract\}", current, re.S)) == 1
    assert r"this trade-off is largely \emph{separable}" in current
    assert "missing or out of order" in current
    assert "missing categories count as" in current
    print("PASS abstract states the first constraint result in plain language")


def main_float_labels():
    """Read the main-text include tree, excluding bibliography and appendices."""
    def expand(text):
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        for name in re.findall(r"\\input\{([^}]+)\}", text):
            yield from expand((PAPER / (name + ".tex")).read_text(encoding="utf-8"))
        yield text
    main = (PAPER / "main.tex").read_text(encoding="utf-8").split(r"\bibliographystyle", 1)[0]
    return {label for text in expand(main)
            for label in re.findall(r"\\label\{((?:fig|tab|alg):[^}]+)\}", text)}


def verify_page_layout(aux, total_pages, float_labels):
    def page(label):
        match = re.search(r"\\newlabel\{" + re.escape(label)
                          + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
        assert match, f"Missing page label: {label}"
        return int(match.group(1))

    body_page, reference_page, appendix_page = map(page, ("end:main", "start:references", "app:algo"))
    assert body_page == 8, f"Main text ends on p{body_page}, expected p8"
    assert reference_page == 9, f"References start on p{reference_page}, expected p9"
    assert reference_page < appendix_page <= total_pages, "Appendix must follow references"
    assert total_pages <= 12, f"PDF has {total_pages} pages, maximum 12"
    assert float_labels, "No main-text floats found"
    for label in sorted(float_labels):
        assert 1 <= page(label) <= 8, f"Main-text float outside first eight pages: {label}"
    return body_page, reference_page, appendix_page


def main(build_dir=None):
    build_dir = Path(build_dir) if build_dir is not None else PAPER
    original = git("show", f"{BASELINE}:paper-www2027/main.tex")
    current = (PAPER / "main.tex").read_text(encoding="utf-8")
    verify_ccs(current)
    verify_locked_metadata(original, current)
    for filename in ("method-comparison.tex", "ablation.tex", "stagewise.tex"):
        old = git("show", f"{BASELINE}:paper-www2027/tables/{filename}")
        new = (PAPER / "tables" / filename).read_text(encoding="utf-8")
        assert data_rows(old) and equivalent_data_rows(data_rows(old), data_rows(new)), filename
        print(f"PASS unchanged numeric rows / precision / bolding: {filename}")
    for path in [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\\todo\s*\{", text), path
        assert "\ufffd" not in text, f"Replacement character in {path}"
    log = (build_dir / "main.log").read_text(encoding="utf-8", errors="replace")
    failures = re.findall(r"^!.*|^.*(?:LaTeX Error|Package \w+ Error|Undefined control sequence).*$", log, re.M)
    assert not failures, failures
    assert not re.search(r"(?:Citation|Reference).*undefined|There were undefined|Rerun to get", log)
    assert "Overfull" not in log
    print("PASS no TODO calls, LaTeX errors, unresolved references/citations, or overfull boxes")
    aux = (build_dir / "main.aux").read_text(encoding="utf-8")
    reader = PdfReader(build_dir / "main.pdf")
    body_page, reference_page, appendix_page = verify_page_layout(
        aux, len(reader.pages), main_float_labels())
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
    font_rows = subprocess.check_output(["pdffonts", str(build_dir / "main.pdf")]).decode().splitlines()[2:]
    assert font_rows and all(row.split()[-5] == "yes" for row in font_rows if row.strip())
    assert not any("Type 3" in row for row in font_rows)
    changed = git("diff", "--name-only", BASELINE, "--", "experiment_runs", "constraint_projection.py", "geometry_projection.py", "distribution_balance.py", "discrete_diffusion", "evaluations")
    assert not changed, f"Scientific inputs or implementation changed: {changed}"
    print(f"PASS PDF: {len(reader.pages)} pages; main text ends p{body_page}; all main floats within p1-8; references p{reference_page}; appendix p{appendix_page}; embedded fonts; anonymous text/metadata/links")
    print("PASS no changes to sealed experiments or audited scientific implementation")
    print("PASS abstract metric-first clarification and non-abstract metadata locks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=PAPER,
                        help="Directory containing main.pdf, main.log and main.aux; sources stay in paper-www2027")
    main(parser.parse_args().build_dir)
