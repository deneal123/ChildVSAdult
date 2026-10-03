import json
import subprocess
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts import reevaluate_cacd_serial as serial


def test_allowlisted_streamed_extraction_preserves_exact_member_bytes(tmp_path):
    source = tmp_path / "crops.npz"
    values = {name: (name.encode() * 40000) for name in serial.MEMBERS}
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in values.items():
            archive.writestr(name, value)
    destination = tmp_path / "mapped"
    serial.extract_mapped(source, destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == values
    with pytest.raises(FileExistsError):
        serial.extract_mapped(source, destination)


@pytest.mark.parametrize("extra", ["../a.npy", "unknown.npy", "a.npy"])
def test_archive_allowlist_and_duplicate_guard(tmp_path, extra):
    source = tmp_path / "crops.npz"
    with warnings.catch_warnings(), zipfile.ZipFile(source, "w") as archive:
        warnings.simplefilter("ignore", UserWarning)
        for name in serial.MEMBERS:
            archive.writestr(name, b"dummy")
        archive.writestr(extra, b"forbidden")
    with pytest.raises(ValueError, match="archive members"):
        serial.extract_mapped(source, tmp_path / "mapped")
    assert not (tmp_path / "a.npy").exists()


def test_stream_embedding_side_order_and_partial_batches(tmp_path):
    a, b = np.arange(5), np.arange(5) + 10
    calls = []

    def embed(batch):
        calls.append(batch.tolist())
        vectors = np.c_[np.ones(len(batch)), batch].astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    path = tmp_path / "embeddings.npy"
    serial.stream_embeddings(a, b, embed, path, batch_size=2, dimension=2)
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    assert vectors.shape == (10, 2)
    assert vectors.dtype == np.float32
    assert calls == [[0, 1], [2, 3], [4], [10, 11], [12, 13], [14]]
    np.testing.assert_allclose(vectors[:, 1] / vectors[:, 0], np.r_[a, b])
    with pytest.raises(ValueError, match="fresh"):
        serial.stream_embeddings(a, b, embed, path, dimension=2)


@pytest.mark.parametrize("kind", ["missing", "nan", "not_normalized"])
def test_embedding_batch_failures_have_no_success_marker(tmp_path, kind):
    def embed(batch):
        vectors = np.tile(np.array([1.0, 0.0], np.float32), (len(batch), 1))
        if kind == "missing":
            return vectors[:-1]
        if kind == "nan":
            vectors[0, 0] = np.nan
        else:
            vectors[0, 0] = 2
        return vectors

    with pytest.raises(ValueError, match="normalized complete"):
        serial.stream_embeddings(
            np.arange(3), np.arange(3), embed, tmp_path / "partial.npy", dimension=2
        )
    assert not list(tmp_path.glob("*.manifest.json"))


def test_cosine_final_partial_chunk_does_not_cross_side_boundary():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(8000, 2)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    actual = serial.paired_scores(vectors, dimension=2)
    expected = np.sum(vectors[:4000] * vectors[4000:], axis=1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (4000,)
    assert actual.dtype == np.float32


def test_native_input_output_corruption_and_preexecution_binding(tmp_path):
    source, output, manifest = (
        tmp_path / name for name in ("in.json", "out.json", "run.manifest.json")
    )
    source.write_text("original", encoding="utf-8")
    output.write_text("result", encoding="utf-8")
    before = [serial.file_record(source)]
    serial.write_bound(
        manifest, "test", {"execution_complete": True}, [source], [output], expected_inputs=before
    )
    serial.verify(manifest, "test")
    output.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        serial.verify(manifest, "test")
    source.write_text("changed", encoding="utf-8")
    refused = tmp_path / "refused.manifest.json"
    with pytest.raises(ValueError, match="before manifest"):
        serial.write_bound(refused, "test", {}, [source], [output], expected_inputs=before)
    assert not refused.exists()


def controller_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(serial, "PROJECT_ROOT", tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    source = tmp_path / "data/external/cacd_vs_aligned.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source input-only preflight; not real image data")
    original = tmp_path / "data/external/source.tar"
    original.write_bytes(b"original")
    serial.write_bound(
        source.with_suffix(".manifest.json"), "cacd-vs-alignment-cache", {}, [original], [source]
    )
    paths = [
        home / ".cache/torch/checkpoints/20180408-102900-casia-webface.pt",
        *(tmp_path / "models" / f"bb_facenet_seed{seed}.pt" for seed in (42, 1, 2)),
        *(tmp_path / name for name in ("uv.lock", "pyproject.toml")),
        *(
            tmp_path / "scripts" / name
            for name in (
                "reevaluate_cacd_metrics_v2.py",
                "cacd_metrics_v2.py",
                "benchmark_metrics_v2.py",
                "verification_metrics_v2.py",
                "evaluate_lfw_bound.py",
                "fgnet_retrieval_study.py",
                "export_lfw_evidence.py",
            )
        ),
    ]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode())
    return tmp_path / "run"


def test_preflight_does_not_launch_children_and_refuses_reuse(tmp_path, monkeypatch):
    out = controller_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unexpected child launch"))
    # Native manifests obtain git metadata through subprocess; replace only that helper.
    from age_gap.common import manifest

    monkeypatch.setattr(manifest, "_git_state", lambda: {})
    serial.controller(out, False)
    plan, _ = serial.read_plan(out)
    assert plan["execution_complete"] is False
    assert plan["image_inference_authorized_by_execute"] is False
    assert not (out / "private").exists()
    with pytest.raises(ValueError, match="input-only"):
        serial.execution_plan(out)
    with pytest.raises(FileExistsError):
        serial.controller(out, False)


def test_child_failure_stops_campaign_and_records_exact_phase(tmp_path, monkeypatch):
    out = controller_fixture(tmp_path, monkeypatch)
    from age_gap.common import manifest

    monkeypatch.setattr(manifest, "_git_state", lambda: {})
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["MKL_NUM_THREADS"] == "1"
        assert kwargs["cwd"] == tmp_path
        kwargs["stdout"].write("synthetic prepare failure\n")
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="prepare exited 9"):
        serial.controller(out, True)
    assert len(commands) == 1
    assert commands[0] == serial.child_command(out, "prepare")
    failed = serial.verify(out / "FAILED.manifest.json", "cacd-vs-serial-incomplete")
    assert failed["metrics"]["phase"] == "prepare"
    assert failed["metrics"]["exit_code"] == 9
    assert failed["metrics"]["execution_complete"] is False
    assert not (out / "summary.manifest.json").exists()


def test_shared_scientific_protocol_and_fixed_child_arguments():
    from scripts.fgnet_retrieval_study import PREPROCESSING
    from scripts.reevaluate_cacd_metrics_v2 import KEYS

    assert serial.PREPROCESSING == PREPROCESSING
    assert serial.KEYS == KEYS
    command = serial.child_command(Path("R:/example"), "score", "tuned_seed1")
    assert command[-4:] == ["--phase", "score", "--role", "tuned_seed1"]
    assert "--execute" in command
    assert "--device" not in command


@pytest.mark.parametrize("changed", ["data/external/source.tar", "models/bb_facenet_seed1.pt"])
def test_original_archive_and_checkpoint_changes_invalidate_plan(tmp_path, monkeypatch, changed):
    out = controller_fixture(tmp_path, monkeypatch)
    serial.controller(out, False)
    original = tmp_path / changed
    original.write_bytes(b"changed original")
    with pytest.raises(ValueError, match="checksum"):
        serial.read_plan(out)


def test_serial_controller_runs_one_child_in_fixed_role_order(tmp_path, monkeypatch):
    out = controller_fixture(tmp_path, monkeypatch)
    from age_gap.common import manifest

    monkeypatch.setattr(manifest, "_git_state", lambda: {})
    roles = []

    def run(command, **kwargs):
        phase = command[command.index("--phase") + 1]
        role = command[command.index("--role") + 1] if "--role" in command else None
        roles.append((phase, role))
        kwargs["stdout"].write(f"synthetic phase {phase}/{role}\n")
        if phase == "aggregate":
            output = out / "summary.json"
            output.write_text("synthetic result", encoding="utf-8")
            serial.write_bound(
                out / "summary.manifest.json",
                "cacd-vs-local-serial-4checkpoint-roc-v2",
                {"execution_complete": True},
                [out / "plan.json"],
                [output],
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    serial.controller(out, True)
    assert roles == [
        ("prepare", None),
        *(("score", key) for key in serial.KEYS),
        ("aggregate", None),
    ]
    assert json.loads((out / "plan.json").read_text())["model_processes_simultaneous"] == 1


def test_completed_phase_refuses_unbound_embedding_and_result(tmp_path):
    result_path, embedding, other = (
        tmp_path / name for name in ("result.json", "embedding.npy", "other.npy")
    )
    result = {"execution_complete": True}
    serial.atomic_json(result_path, result)
    embedding.write_bytes(b"embedding")
    other.write_bytes(b"other")
    manifest = tmp_path / "phase.manifest.json"
    serial.write_bound(manifest, "phase", result, [other], [result_path, embedding])
    serial.completed_phase(manifest, "phase", result_path, [embedding])
    with pytest.raises(ValueError, match="output/result"):
        serial.completed_phase(manifest, "phase", result_path, [other])
    different = {"execution_complete": True, "not_bound": True}
    serial.atomic_json(result_path, different)
    serial.write_bound(manifest, "phase", result, [other], [result_path, embedding])
    with pytest.raises(ValueError, match="output/result"):
        serial.completed_phase(manifest, "phase", result_path, [embedding])
