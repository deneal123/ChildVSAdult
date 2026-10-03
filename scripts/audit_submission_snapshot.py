"""Read-only current-source/package consistency audit, not a compilation attestation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, sha256_file, write_experiment_manifest
from latex.build_submission import _latex_reference_errors, _localize
from scripts import export_curation_evidence as curation
from scripts import export_lfw_evidence as lfw
from scripts import export_public_evidence as public
from scripts import export_roc_v2_evidence as roc

UPLOAD_MEMBERS = {"main.pdf", "supplement.pdf", "manuscript-source.zip", "artifact-manifest.json",
                  "publication-evidence.zip", "lfw-evidence.zip", "curation-evidence.zip", "roc-v2-evidence.zip"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def literal_figures(tex):
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", tex)
    if len(names) != len(re.findall(r"\\includegraphics\b", tex)):
        raise ValueError("unsupported figure syntax; audit refused")
    if any(Path(name).name != name or not name.endswith(".pdf") for name in names):
        raise ValueError("nonliteral or non-PDF figure reference")
    return set(names)


def stripped_source(source, strip_script):
    # Execute the trusted local helper definitions without writing a bytecode cache.
    namespace = {"__name__": "submission_strip"}
    exec(compile(strip_script.read_text(encoding="utf-8"), str(strip_script), "exec"), namespace)
    lines = _localize(source.read_text(encoding="utf-8")).split("\n")
    out = []
    for line in lines:
        code = namespace["strip_comment"](line)
        if code.strip():
            out.append(code.rstrip())
        elif not line.strip():
            out.append("")
    return "\n".join(out)


def pdf_pages(path):
    info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True, timeout=30)
    match = re.search(r"^Pages:\s+(\d+)\s*$", info.stdout, re.MULTILINE)
    if not match:
        raise ValueError("PDF page count unavailable")
    text = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True,
                          text=True, check=True, timeout=30).stdout
    pages = text.split("\f")
    if not pages[-1].strip():
        pages.pop()
    if len(pages) != int(match[1]) or any(not page.strip() for page in pages):
        raise ValueError("textually blank or unreadable PDF page")
    return len(pages)


def check_upload(upload, extract=pdf_pages):
    if {path.name for path in upload.iterdir()} != UPLOAD_MEMBERS:
        raise ValueError("unexpected upload membership")
    if any(not (upload / name).is_file() for name in UPLOAD_MEMBERS):
        raise ValueError("non-file upload member")
    manifest = json.loads((upload / "artifact-manifest.json").read_text(encoding="utf-8"))
    rows = manifest["artifacts"]
    names = [row["file"] for row in rows]
    if len(names) != len(set(names)) or set(names) != UPLOAD_MEMBERS - {"artifact-manifest.json"}:
        raise ValueError("artifact manifest membership mismatch")
    counts = {}
    for row in rows:
        path = upload / row["file"]
        if row["sha256"] != sha256_file(path) or row["bytes"] != path.stat().st_size:
            raise ValueError("upload checksum or byte count mismatch")
        if path.suffix == ".pdf":
            count = extract(path)
            if type(row["pages"]) is not int or row["pages"] != count:
                raise ValueError("declared PDF page count mismatch")
            counts[path.stem] = count
        elif row["pages"] is not None:
            raise ValueError("non-PDF page count")
    if not 1 <= counts["main"] <= 10 or counts["supplement"] < 1:
        raise ValueError("main page limit or supplement violated")
    return counts


def check_sources(root, submission):
    source = root / "latex/papers/journal-1-tbiom"
    strip = source / "submission/strip_comments.py"
    inputs = [strip, root / "latex/build_submission.py"]
    staged = {"main": submission / "manuscript", "supplement": submission / "supplement"}
    archive_expected = {}
    for stem, directory in staged.items():
        canonical = source / "en" / f"{stem}.tex"
        tex = directory / f"{stem}.tex"
        inputs.extend((canonical, tex, directory / f"{stem}.log", directory / f"{stem}.pdf"))
        if stripped_source(canonical, strip).encode("utf-8") != tex.read_bytes():
            raise ValueError("staged source differs from current localized/comment-stripped master")
        if _latex_reference_errors((directory / f"{stem}.log").read_text(encoding="utf-8", errors="replace")):
            raise ValueError("undefined or duplicate references in staged log")
        if (submission / "upload" / f"{stem}.pdf").read_bytes() != (directory / f"{stem}.pdf").read_bytes():
            raise ValueError("staged/upload PDF mismatch")
        for name, original in (("refs.bib", root / "latex/shared/refs.bib"),
                               ("IEEEtran.cls", root / "latex/shared/vendor/ieee-tnnls/IEEEtran.cls")):
            target = directory / name
            inputs.extend((original, target))
            if target.read_bytes() != original.read_bytes():
                raise ValueError("staged bibliography/class differs from canonical")
        figures = literal_figures(tex.read_text(encoding="utf-8"))
        existing = {p.name for p in (directory / "figures").iterdir()} if (directory / "figures").exists() else set()
        if figures != existing or any(Path(name).name != name for name in figures):
            raise ValueError("staged figure membership mismatch")
        for name in figures:
            original, target = root / "latex/shared/figures" / name, directory / "figures" / name
            inputs.extend((original, target))
            if original.read_bytes() != target.read_bytes():
                raise ValueError("staged figure differs from canonical")
            if stem == "main":
                archive_expected[f"figures/{name}"] = target.read_bytes()
    for name in ("main.tex", "main.bbl", "refs.bib", "IEEEtran.cls"):
        path = staged["main"] / name
        inputs.append(path)
        if not path.read_bytes().strip():
            raise ValueError("empty mandatory source file")
        archive_expected[name] = path.read_bytes()
    with zipfile.ZipFile(submission / "upload/manuscript-source.zip") as archive:
        index_name = "artifacts/publication_artifact_index.json"
        index_path = root / "metrics/publication_artifact_index.json"
        inputs.append(index_path)
        if archive.read(index_name) != index_path.read_bytes():
            raise ValueError("archived index differs from current local index")
        index = json.loads(archive.read(index_name))
        if index.get("entry_count") != len(index.get("entries", [])):
            raise ValueError("archived index count mismatch")
        archive_expected[index_name] = archive.read(index_name)
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(archive_expected):
            raise ValueError("source ZIP membership mismatch")
        if any(archive.read(name) != data for name, data in archive_expected.items()):
            raise ValueError("source ZIP bytes differ from checked staged source")
    return inputs, index


def aggregate_binding(root, result_path, manifest_path, original_hash, inputs):
    source, manifest = root / result_path, root / manifest_path
    inputs.extend((source, manifest))
    raw = json.loads(source.read_text(encoding="utf-8"))
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if sha256_file(source) != original_hash or file_record(source) not in metadata.get("outputs", []):
        raise ValueError("original aggregate binding mismatch")
    return raw


def check_roc_evidence(root, upload):
    """Check ROC projections against current aggregates/tables, not private-data clearance."""
    path, inputs = upload / "roc-v2-evidence.zip", [Path(roc.__file__)]
    with zipfile.ZipFile(path) as archive:
        certificate = json.loads(archive.read("CERTIFICATE.json"))
        if (certificate["policy"] != "tbiom-reviewed-roc-v2-aggregates-v1"
                or certificate["exporter_sha256"] != sha256_file(Path(roc.__file__))
                or certificate["training_identity_independence"] != "unverified"
                or set(certificate["aggregate_bindings"]) != set(roc.EXPORTS)):
            raise ValueError("unreviewed ROC-v2 policy/exporter/identity claim")
        if any(certificate[key] is not False for key in ("publication_ready", "full_reproduction", "privacy_certified",
                "ethics_clearance_attested", "disclosure_review_completed", "private_records_listed")):
            raise ValueError("unreviewed ROC-v2 clearance claim")
        expected = {**certificate["members_excluding_certificate"], "CERTIFICATE.json": digest(archive.read("CERTIFICATE.json"))}
        roc.verify_archive(path, expected)
        for label, (directory, experiment) in roc.EXPORTS.items():
            result_path = Path("metrics") / directory / "summary.json"
            mp = result_path.with_suffix(".manifest.json")
            binding = certificate["aggregate_bindings"][label]
            raw = aggregate_binding(root, result_path, mp, binding["original_aggregate_sha256"], inputs)
            native = json.loads((root / mp).read_text(encoding="utf-8"))
            if native["experiment"] != experiment or native["metrics"] != raw:
                raise ValueError("ROC-v2 native aggregate linkage mismatch")
            exported = archive.read(f"results/{label}.json")
            if digest(exported) != binding["exported_aggregate_sha256"] or json.loads(exported) != roc.project(label, raw):
                raise ValueError("ROC-v2 projection differs from current aggregate")
            if json.loads(archive.read(f"manifests/{label}.sanitized.json")) != roc.sanitized(native, sha256_file(root / mp)):
                raise ValueError("ROC-v2 sanitized manifest mismatch")
            if label in roc.PRESENTATIONS:
                pe, tables = roc.PRESENTATIONS[label]
                pp = root / "metrics" / directory / "presentation.manifest.json"
                inputs.append(pp)
                presentation = json.loads(pp.read_text(encoding="utf-8"))
                if (presentation["experiment"] != pe or any(file_record(p) not in presentation["inputs"] for p in (root / result_path, root / mp))
                        or json.loads(archive.read(f"manifests/{label}.presentation.sanitized.json")) != roc.sanitized(presentation, sha256_file(pp))):
                    raise ValueError("ROC-v2 presentation linkage mismatch")
                for filename, renderer in tables.items():
                    table = pp.parent / filename
                    inputs.append(table)
                    if (file_record(table) not in presentation["outputs"] or table.read_text(encoding="utf-8") != renderer(raw)
                            or archive.read(f"tables/{filename}").decode("utf-8") != renderer(raw)):
                        raise ValueError("ROC-v2 generated table mismatch")
    return inputs


def check_evidence(root, upload):
    inputs = []
    for module, filename in ((public, "publication-evidence.zip"), (lfw, "lfw-evidence.zip"),
                             (curation, "curation-evidence.zip")):
        inputs.append(Path(module.__file__))
        with zipfile.ZipFile(upload / filename) as archive:
            certificate = json.loads(archive.read("CERTIFICATE.json"))
            if module is public:
                if certificate["export_policy"] != public.EXPORT_POLICY or certificate["file_whitelist"] != list(public.EXPORTS):
                    raise ValueError("unreviewed public export policy")
                if certificate["exporter"]["sha256"] != sha256_file(Path(module.__file__)):
                    raise ValueError("exporter source changed")
                if any(certificate["unresolved"][key] is not False for key in (
                    "asserts_training_independence", "asserts_publication_ready", "asserts_experiment_complete")):
                    raise ValueError("unreviewed clearance claim")
                expected = {key: digest(archive.read(key)) for key in public.FIXED_MEMBERS}
                if len(certificate["entries"]) != len(public.EXPORTS):
                    raise ValueError("certificate entry count mismatch")
                for result, row in zip(public.EXPORTS, certificate["entries"], strict=True):
                    result_member = f"results/{Path(result).name}"
                    manifest_member = f"manifests/{Path(result).name}.sanitized.json"
                    manifest_path = str(Path(result).with_suffix(".manifest.json"))
                    if (row["original_relpath"], row["member"], row["manifest_member"]) != (result, result_member, manifest_member):
                        raise ValueError("certificate aggregate mapping mismatch")
                    raw = aggregate_binding(root, result, manifest_path, row["original_sha256"], inputs)
                    if row["verification"]["manifest_sha256"] != sha256_file(root / manifest_path):
                        raise ValueError("original manifest changed")
                    clean, _ = public.sanitize_value(raw, root)
                    if json.loads(archive.read(result_member)) != clean:
                        raise ValueError("public projection differs from current original")
                    expected[result_member] = row["exported_sha256"]
                    expected[manifest_member] = row["manifest_exported_sha256"]
            else:
                if certificate["exporter_sha256"] != sha256_file(Path(module.__file__)):
                    raise ValueError("exporter source changed")
                if certificate["publication_ready"] is not False or certificate["full_reproduction"] is not False:
                    raise ValueError("unreviewed clearance claim")
                expected = {**certificate["members_excluding_certificate"],
                            "CERTIFICATE.json": digest(archive.read("CERTIFICATE.json"))}
                if module is lfw:
                    if certificate["policy"] != "tbiom-reviewed-lfw-aggregates-v1":
                        raise ValueError("unreviewed LFW policy")
                    if (certificate["training_identity_independence"] != "unverified"
                            or certificate["checkpoint_training_provenance"] != "unverified"
                            or certificate["private_records_listed"] is not False):
                        raise ValueError("unreviewed LFW provenance claim")
                    directory = Path(lfw.DIRECTORY)
                    raw = aggregate_binding(root, directory / "lfw_bound_evaluation.json",
                        directory / "lfw_bound_evaluation.manifest.json", certificate["original_result_sha256"], inputs)
                    if json.loads(archive.read("results/lfw.json")) != lfw.project_result(raw):
                        raise ValueError("LFW projection differs from current original")
                    table = root / directory / "lfw_table.tex"
                    inputs.append(table)
                    if certificate["original_table_sha256"] != sha256_file(table) or archive.read("tables/lfw.tex").decode() != lfw.render(raw):
                        raise ValueError("LFW table binding mismatch")
                else:
                    if certificate["policy"] != "tbiom-reviewed-curation-aggregates-v1":
                        raise ValueError("unreviewed curation policy")
                    if any(certificate[key] is not False for key in (
                            "human_adjudication_completed", "disclosure_review_completed",
                            "historical_pipeline_provenance_recovered", "private_records_listed")):
                        raise ValueError("unreviewed human/disclosure claim")
                    projected = {}
                    for name, (directory, _) in curation.EXPORTS.items():
                        directory = Path("metrics") / directory
                        raw = aggregate_binding(root, directory / "summary.json", directory / "summary.manifest.json",
                            certificate["aggregate_bindings"][name]["original_aggregate_sha256"], inputs)
                        projected[name] = curation.project_result(name, raw)
                        if json.loads(archive.read(f"results/{name}.json")) != projected[name]:
                            raise ValueError("curation projection differs from current original")
                    curation.validate_consistency(projected)
                for name in expected:
                    if name.endswith("sanitized.json"):
                        projection = json.loads(archive.read(name))
                        if {"inputs", "outputs", "command", "parameters"} & projection.keys():
                            raise ValueError("record lists/commands in count-only projection")
                        if module is lfw:
                            source_name = {"manifests/evaluation.sanitized.json": "lfw_bound_evaluation.manifest.json",
                                           "manifests/presentation.sanitized.json": "lfw_presentation.manifest.json"}[name]
                            original_manifest = root / lfw.DIRECTORY / source_name
                        else:
                            label = Path(name).name.removesuffix(".sanitized.json")
                            original_manifest = root / "metrics" / curation.EXPORTS[label][0] / "summary.manifest.json"
                        inputs.append(original_manifest)
                        if projection["original_manifest_sha256"] != sha256_file(original_manifest):
                            raise ValueError("sanitized manifest binding mismatch")
            module.verify_archive(upload / filename, expected)
            for name in archive.namelist():
                payload = archive.read(name)
                public.scan_unsafe(json.loads(payload) if name.endswith(".json") else payload.decode("utf-8"))
    inputs.extend(check_roc_evidence(root, upload))
    return inputs


def audit(root, submission, extract=pdf_pages):
    root, submission = Path(root).resolve(), Path(submission).resolve()
    upload = submission / "upload"
    canonical = root / "latex/papers/journal-1-tbiom"
    originals = [canonical / "en/main.tex", canonical / "en/supplement.tex",
                 canonical / "submission/strip_comments.py", root / "latex/shared/refs.bib",
                 root / "latex/shared/vendor/ieee-tnnls/IEEEtran.cls", root / "latex/build_submission.py",
                 Path(__file__), Path(public.__file__), Path(lfw.__file__), Path(curation.__file__),
                 root / "scripts/render_lfw_evidence.py", root / "src/age_gap/common/manifest.py",
                 root / "metrics/publication_artifact_index.json",
                 root / lfw.DIRECTORY / "lfw_presentation.manifest.json"]
    originals.extend((Path(roc.__file__), root / "scripts/render_fgnet_metrics_v2.py", root / "scripts/render_internal_metrics_v2.py"))
    for directory, _ in roc.EXPORTS.values():
        originals.extend(root / "metrics" / directory / name for name in ("summary.json", "summary.manifest.json"))
    for label, (_, tables) in roc.PRESENTATIONS.items():
        directory = root / "metrics" / roc.EXPORTS[label][0]
        originals.extend(directory / name for name in ("presentation.manifest.json", *tables))
    results = [Path(path) for path in public.EXPORTS]
    results.extend((Path(lfw.DIRECTORY) / "lfw_bound_evaluation.json", Path(lfw.DIRECTORY) / "lfw_table.tex"))
    results.extend(Path("metrics") / directory / "summary.json" for directory, _ in curation.EXPORTS.values())
    originals.extend(root / path for path in results)
    originals.extend(root / path.with_suffix(".manifest.json") for path in results if path.suffix == ".json")
    initial = [path for path in submission.rglob("*") if path.is_file()]
    before = {path: file_record(path) for path in set(initial + originals)}
    for tex in (canonical / "en/main.tex", canonical / "en/supplement.tex"):
        for name in literal_figures(stripped_source(tex, canonical / "submission/strip_comments.py")):
            figure = root / "latex/shared/figures" / name
            if figure not in before:
                before[figure] = file_record(figure)
    counts = check_upload(upload, extract)
    inputs, index = check_sources(root, submission)
    inputs.extend(check_evidence(root, upload))
    inputs.extend(upload / name for name in UPLOAD_MEMBERS)
    if set(initial) != {path for path in submission.rglob("*") if path.is_file()}:
        raise ValueError("submission membership changed during audit")
    if before != {path: file_record(path) for path in before}:
        raise ValueError("source/submission files changed during audit")
    inputs.extend(before)
    return {
        "main_pages": counts["main"], "supplement_pages": counts["supplement"],
        "upload_artifacts": len(UPLOAD_MEMBERS) - 1, "source_zip_matches_staged_sources": True,
        "staged_sources_match_current_master": True, "staged_and_upload_pdf_bytes_match": True,
        "textual_nonblank_pages_checked": sum(counts.values()), "evidence_archive_members": [18, 6, 8, 13],
        "archived_index_entries": index["entry_count"], "archived_index_complete": index["complete"],
        "archived_index_missing_expected": len(index["missing_expected_manifests"]),
        "compiler_execution_attested": False, "all_pages_visually_reviewed": False,
        "full_private_input_reverification": False, "portal_proof_checked": False,
        "privacy_certified": False, "publication_ready": False,
        "scope": "current-source/package byte consistency and textual PDF checks; not compilation provenance or publication clearance",
    }, sorted(set(inputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh audit directory required")
    result, inputs = audit(args.root, args.submission_dir)
    args.out.mkdir(parents=True)
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(args.out / "summary.manifest.json", experiment="submission-snapshot-consistency",
        parameters={"mode": "read-only", "seed": None, "source": "isolated final-source candidate"},
        metrics=result, inputs=inputs, outputs=[summary])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
