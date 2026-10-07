"""CPU-only tests for host inference admission and CLI ownership."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from visionbrain import cli, loader, pilot_eval
from visionbrain.frame_selector import FrameScores
from visionbrain.inference_admission import InferenceAdmission


@contextmanager
def _held_by_child(lock_path: Path):
    """Hold a real flock in another process until the context exits."""
    source_dir = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_dir), env.get("PYTHONPATH", "")]
    )
    code = (
        "import sys; "
        "from visionbrain.inference_admission import InferenceAdmission; "
        "handle = InferenceAdmission(sys.argv[1]).try_acquire('external-holder'); "
        "assert handle is not None; "
        "print('ready', flush=True); "
        "sys.stdin.read(); "
        "handle.release()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(5), "admission holder did not become ready"
        assert process.stdout.readline().strip() == "ready"
        yield
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        if process.stderr is not None:
            process.stderr.close()
        if process.stdout is not None:
            process.stdout.close()


def _set_detect_args(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["visionbrain", "detect", "--image", "unused.png", "--query", "test"],
    )


def _set_pilot_eval_args(monkeypatch, video_path: Path, ground_truth_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visionbrain",
            "pilot-eval",
            "--video",
            str(video_path),
            "--ground-truth",
            str(ground_truth_path),
        ],
    )


def _empty_scores() -> FrameScores:
    return FrameScores(
        video_path="video.mp4",
        total_frames=0,
        fps=30.0,
        duration_s=0.0,
        frames_scored=0,
        is_relevant=False,
        quick_answer="",
    )


def _can_acquire(lock_path: Path) -> bool:
    handle = InferenceAdmission(lock_path).try_acquire("test-probe")
    if handle is None:
        return False
    handle.release()
    return True


def test_cli_busy_reports_holder_and_skips_inference_callback(
    tmp_path, monkeypatch, capsys
) -> None:
    lock_path = tmp_path / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    _set_detect_args(monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "cmd_detect", lambda args: calls.append(args.command))

    with _held_by_child(lock_path):
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 1
    assert calls == []
    diagnostic = capsys.readouterr().err
    assert "inference admission busy" in diagnostic
    assert "external-holder" in diagnostic


def test_cli_success_holds_admission_for_callback_and_releases_after_return(
    tmp_path, monkeypatch
) -> None:
    lock_path = tmp_path / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    _set_detect_args(monkeypatch)
    observed = []

    def callback(args) -> None:
        observed.append((args.command, _can_acquire(lock_path)))

    monkeypatch.setattr(cli, "cmd_detect", callback)
    cli.main()

    assert observed == [("detect", False)]
    assert _can_acquire(lock_path)


def test_cli_exception_releases_admission(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    _set_detect_args(monkeypatch)

    def callback(_args) -> None:
        raise RuntimeError("callback failed")

    monkeypatch.setattr(cli, "cmd_detect", callback)
    with pytest.raises(RuntimeError, match="callback failed"):
        cli.main()

    assert _can_acquire(lock_path)


def test_pilot_eval_busy_refuses_before_default_scoring(
    tmp_path, monkeypatch
) -> None:
    lock_path = tmp_path / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    calls = []
    monkeypatch.setattr(
        pilot_eval,
        "_default_score_fn",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    ground_truth = {"query": "person", "events": []}

    with _held_by_child(lock_path):
        with pytest.raises(RuntimeError, match="inference admission busy") as exc:
            pilot_eval.run_pilot_eval("video.mp4", ground_truth)

    assert calls == []
    assert "external-holder" in str(exc.value)


def test_pilot_eval_releases_admission_after_default_scorer_failure(
    tmp_path, monkeypatch
) -> None:
    lock_path = tmp_path / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    observed = []

    def fail_scoring(*args, **kwargs):
        observed.append(_can_acquire(lock_path))
        raise RuntimeError("scorer failed")

    monkeypatch.setattr(pilot_eval, "_default_score_fn", fail_scoring)
    ground_truth = {"query": "person", "events": []}

    with pytest.raises(RuntimeError, match="scorer failed"):
        pilot_eval.run_pilot_eval("video.mp4", ground_truth)

    assert observed == [False]
    assert _can_acquire(lock_path)


def test_pilot_eval_injected_scorer_remains_lock_free_when_busy(
    tmp_path, monkeypatch
) -> None:
    lock_path = tmp_path / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    ground_truth = {"query": "person", "events": []}

    with _held_by_child(lock_path):
        report = pilot_eval.run_pilot_eval(
            "video.mp4", ground_truth, score_fn=lambda *args, **kwargs: _empty_scores()
        )

    assert report.events_total == 0


def test_cli_pilot_eval_uses_function_owned_admission_once(
    tmp_path, monkeypatch
) -> None:
    lock_path = tmp_path / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    video_path = tmp_path / "video.mp4"
    ground_truth_path = tmp_path / "ground-truth.json"
    video_path.write_bytes(b"synthetic")
    ground_truth_path.write_text(json.dumps({"query": "person", "events": []}))
    _set_pilot_eval_args(monkeypatch, video_path, ground_truth_path)
    observed = []

    def score_while_admitted(*args, **kwargs):
        observed.append(_can_acquire(lock_path))
        return _empty_scores()

    monkeypatch.setattr(pilot_eval, "_default_score_fn", score_while_admitted)
    cli.main()

    assert observed == [False]
    assert _can_acquire(lock_path)


def test_cli_pilot_eval_surfaces_busy_before_default_scoring(
    tmp_path, monkeypatch, capsys
) -> None:
    lock_path = tmp_path / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    video_path = tmp_path / "video.mp4"
    ground_truth_path = tmp_path / "ground-truth.json"
    video_path.write_bytes(b"synthetic")
    ground_truth_path.write_text(json.dumps({"query": "person", "events": []}))
    _set_pilot_eval_args(monkeypatch, video_path, ground_truth_path)
    calls = []
    monkeypatch.setattr(
        pilot_eval,
        "_default_score_fn",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with _held_by_child(lock_path):
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 1
    assert calls == []
    diagnostic = capsys.readouterr().err
    assert "inference admission busy" in diagnostic
    assert "external-holder" in diagnostic


def test_cli_admission_filesystem_error_fails_closed(tmp_path, monkeypatch, capsys) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    lock_path = blocker / "nested" / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    _set_detect_args(monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "cmd_detect", lambda args: calls.append(args.command))

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert calls == []
    assert "inference admission unavailable" in capsys.readouterr().err


def test_status_help_and_version_remain_lock_free(tmp_path, monkeypatch):
    lock_path = tmp_path / "inference.lock"
    monkeypatch.setenv("VB_INFERENCE_LOCK", str(lock_path))
    holder = InferenceAdmission(lock_path).try_acquire("test-holder")
    assert holder is not None
    try:
        statuses = []
        monkeypatch.setattr(loader, "print_status", lambda: statuses.append(True))
        monkeypatch.setattr(sys, "argv", ["visionbrain", "status"])
        cli.main()
        assert statuses == [True]
        assert holder.is_mine

        for arguments in (["--help"], ["--version"], ["detect", "--help"]):
            monkeypatch.setattr(sys, "argv", ["visionbrain", *arguments])
            with pytest.raises(SystemExit) as exc:
                cli.main()
            assert exc.value.code == 0
            assert holder.is_mine
    finally:
        holder.release()
