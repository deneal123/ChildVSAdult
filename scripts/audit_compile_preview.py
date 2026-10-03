"""Check isolated current-source PDFs without claiming an upload/portal proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from latex.build_submission import _check_source_privacy, _latex_reference_errors
from scripts.audit_submission_snapshot import literal_figures, pdf_pages, stripped_source


def preview(root, submission, *, extract=pdf_pages):
    paper = root / "latex/papers/journal-1-tbiom"
    strip = paper / "submission/strip_comments.py"
    refs = root / "latex/shared/refs.bib"
    cls = root / "latex/shared/vendor/ieee-tnnls/IEEEtran.cls"
    inputs = [
        strip,
        refs,
        cls,
        root / "latex/build_submission.py",
        root / "scripts/audit_submission_snapshot.py",
        Path(__file__),
    ]
    outputs, pages = [], {}
    for stem, directory in (("main", "manuscript"), ("supplement", "supplement")):
        source = paper / f"en/{stem}.tex"
        staged = submission / directory
        if stripped_source(source, strip) != (staged / f"{stem}.tex").read_text(encoding="utf-8"):
            raise ValueError("staged source differs from current master")
        inputs.append(source)
        for canonical, local in ((refs, staged / "refs.bib"), (cls, staged / "IEEEtran.cls")):
            if canonical.read_bytes() != local.read_bytes():
                raise ValueError("staged references/class differ")
        log = staged / f"{stem}.log"
        if _latex_reference_errors(log.read_text(encoding="utf-8", errors="replace")):
            raise ValueError("undefined or duplicate references/citations")
        pages[stem] = extract(staged / f"{stem}.pdf")
        for name in literal_figures(source.read_text(encoding="utf-8")):
            canonical, local = root / "latex/shared/figures" / name, staged / "figures" / name
            if canonical.read_bytes() != local.read_bytes():
                raise ValueError("staged figure differs")
            inputs.append(canonical)
        _check_source_privacy([staged / f"{stem}.tex", staged / "refs.bib"])
        outputs.extend(path for path in staged.rglob("*") if path.is_file())
    if not 1 <= pages["main"] <= 10 or pages["supplement"] < 1:
        raise ValueError("page limit violated")
    if (submission / "upload").exists():
        raise ValueError("compile-only preview must not contain an upload package")
    return (
        {
            "pages": pages,
            "current_source_matches": True,
            "references_checked": True,
            "textually_nonblank_pages": sum(pages.values()),
            "source_privacy_scanner_passed": True,
            "upload_package_created": False,
            "compiler_execution_attested": False,
            "visual_review_completed": False,
            "portal_proof_checked": False,
            "publication_ready": False,
            "scope": "staged source/PDF/log consistency only; process execution separately observed by root",
        },
        inputs,
        outputs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh proof destination required")
    result, inputs, outputs = preview(PROJECT_ROOT, args.submission.resolve())
    inputs = list(dict.fromkeys(path.resolve() for path in inputs))
    before = [file_record(path) for path in inputs]
    before_outputs = [file_record(path) for path in outputs]
    checked, _, checked_outputs = preview(PROJECT_ROOT, args.submission.resolve())
    if (
        result != checked
        or before != [file_record(path) for path in inputs]
        or before_outputs != [file_record(path) for path in checked_outputs]
    ):
        raise ValueError("preview changed during verification")
    args.out.mkdir(parents=True)
    output = args.out / "summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="compile-only-current-source-pdf-consistency",
        parameters={"scope": "no index/upload refresh"},
        metrics=result,
        inputs=inputs,
        outputs=[output, *outputs],
    )
    native = json.loads(target.read_text(encoding="utf-8"))
    if before != native["inputs"] or before_outputs != native["outputs"][1:]:
        target.unlink()
        raise ValueError("proof inputs/outputs changed during write; marker withdrawn")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
