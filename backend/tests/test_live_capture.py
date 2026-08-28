"""Tests for the server-side RTSP live path.

Covers the three pieces that carry real risk and are cheap to exercise without
a camera: source classification, the reconnect state machine, and the
single-session registry. The pipeline itself (recognizer → behaviour → DB) is
covered by the upload-path tests; what is new here is *how frames arrive*.
"""
from __future__ import annotations

import threading
from uuid import uuid4

import pytest

from backend.pipeline.capture import (
    StreamDropped,
    VideoCapture,
    is_stream_source,
)
from backend.pipeline import live_worker as lw


# ── Source classification ────────────────────────────────────────────────────

@pytest.mark.parametrize("source", [
    "rtsp://192.168.1.50:554/Streaming/Channels/101",
    "rtsps://cam.local/stream",
    "http://cam.local/video.mjpg",
    "https://cam.local/video.mjpg",
    "udp://127.0.0.1:23000",
    "tcp://127.0.0.1:23000",
    "0", "1", 0,
])
def test_stream_sources_are_recognised(source):
    assert is_stream_source(source) is True


@pytest.mark.parametrize("source", [
    "/app/data/videos/lecture.mp4",
    "data/videos/lecture.mp4",
    "lecture.mkv",
    "file:///tmp/x.mp4",        # not one of the accepted schemes
])
def test_files_are_not_streams(source):
    assert is_stream_source(source) is False


def test_scheme_match_is_case_insensitive():
    assert is_stream_source("RTSP://CAM.LOCAL/stream") is True


# ── Reconnect state machine ──────────────────────────────────────────────────

class _FakeCapture(VideoCapture):
    """VideoCapture whose frames() drops a scripted number of times.

    Subclassed rather than mocked so frames_with_reconnect() runs verbatim —
    the bug this guards against (a returning generator reading as clean EOF and
    making the retry loop a no-op) lived in exactly that interaction.
    """

    def __init__(self, drops: int, frames_per_life: int = 2, open_fails: int = 0):
        self.source = "rtsp://fake/stream"
        self.is_stream = True
        self._drops_left = drops
        self._frames_per_life = frames_per_life
        self._open_fails = open_fails
        self.opens = 0
        self.closes = 0
        self._n = 0

    def open(self):
        self.opens += 1
        if self._open_fails > 0:
            self._open_fails -= 1
            raise IOError("cannot open")
        return self

    def close(self):
        self.closes += 1

    def frames(self):
        for _ in range(self._frames_per_life):
            self._n += 1
            yield self._n, self._n * 1000, None
        if self._drops_left > 0:
            self._drops_left -= 1
            raise StreamDropped("feed died")
        return          # genuine EOF


def test_reconnects_and_resumes_after_a_drop():
    cap = _FakeCapture(drops=2, frames_per_life=2)
    frames = list(cap.frames_with_reconnect(reconnect_delay=0.0, max_reconnects=5))
    # 3 lives x 2 frames: the drops must not truncate the stream.
    assert len(frames) == 6
    assert cap.opens == 2        # two reconnects; the initial open is the caller's


def test_frame_numbers_stay_monotonic_across_reconnects():
    cap = _FakeCapture(drops=2, frames_per_life=2)
    nums = [n for n, _, _ in cap.frames_with_reconnect(reconnect_delay=0.0, max_reconnects=5)]
    assert nums == sorted(nums)
    assert len(set(nums)) == len(nums)


def test_gives_up_after_the_retry_budget():
    cap = _FakeCapture(drops=99, frames_per_life=1)
    events = []
    frames = list(cap.frames_with_reconnect(
        reconnect_delay=0.0, max_reconnects=3,
        on_state=lambda e, **kw: events.append((e, kw)),
    ))
    assert len(frames) == 4                       # initial life + 3 retries
    assert [e for e, _ in events].count("dropped") == 3
    assert events[-1][0] == "exhausted"


def test_backoff_grows_and_is_capped():
    cap = _FakeCapture(drops=99, frames_per_life=1)
    delays = []
    list(cap.frames_with_reconnect(
        reconnect_delay=1.0, max_reconnects=6, max_backoff=8.0,
        on_state=lambda e, **kw: delays.append(kw.get("delay")) if e == "dropped" else None,
    ))
    assert delays == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_on_state_reports_reconnection():
    cap = _FakeCapture(drops=1, frames_per_life=1)
    events = []
    list(cap.frames_with_reconnect(
        reconnect_delay=0.0, max_reconnects=3,
        on_state=lambda e, **kw: events.append(e),
    ))
    assert "dropped" in events and "reconnected" in events


def test_a_broken_callback_does_not_stop_capture():
    cap = _FakeCapture(drops=1, frames_per_life=2)

    def boom(event, **kw):
        raise RuntimeError("callback is broken")

    frames = list(cap.frames_with_reconnect(
        reconnect_delay=0.0, max_reconnects=3, on_state=boom,
    ))
    assert len(frames) == 4


def test_a_failed_reopen_burns_a_retry_but_keeps_going():
    """A camera still rebooting refuses the first reopen; the loop must retry.

    The failed attempt consumes a retry (it is a real attempt) but must not end
    capture — the next one succeeds and frames resume.
    """
    cap = _FakeCapture(drops=2, frames_per_life=1, open_fails=1)
    events = []
    frames = list(cap.frames_with_reconnect(
        reconnect_delay=0.0, max_reconnects=4,
        on_state=lambda e, **kw: events.append(e),
    ))
    assert cap.opens == 2                     # one refused, one accepted
    assert len(frames) == 3                   # frames kept flowing either side
    assert events.count("dropped") == 2
    assert "reconnected" in events            # only the successful reopen reports it


# ── URL validation ───────────────────────────────────────────────────────────

def test_validate_accepts_a_stream_url():
    assert lw.validate_stream_url("  rtsp://cam/stream  ") == "rtsp://cam/stream"


@pytest.mark.parametrize("bad", ["", "   ", "/data/videos/lecture.mp4", "not a url"])
def test_validate_rejects_non_streams(bad):
    with pytest.raises(lw.LiveStartError):
        lw.validate_stream_url(bad)


# ── Single-session registry ──────────────────────────────────────────────────

class _StubWorker:
    """Stands in for LiveSessionWorker; the registry only reads is_running."""

    def __init__(self, running: bool = True):
        self.session_id = uuid4()
        self._running = running

    @property
    def is_running(self) -> bool:
        return self._running


@pytest.fixture(autouse=True)
def _clean_registry():
    with lw._registry_lock:
        lw._workers.clear()
    yield
    with lw._registry_lock:
        lw._workers.clear()


def test_registry_admits_one_worker():
    w = _StubWorker()
    lw.register(w)
    assert lw.get_worker(w.session_id) is w
    assert lw.active_worker() is w


def test_registry_refuses_a_second_running_session():
    first = _StubWorker(running=True)
    lw.register(first)
    with pytest.raises(lw.LiveStartError, match="already running"):
        lw.register(_StubWorker())
    # The refusal must not evict the incumbent.
    assert lw.get_worker(first.session_id) is first


def test_a_finished_worker_is_reaped_and_the_slot_reopens():
    done = _StubWorker(running=False)
    lw.register(done)
    fresh = _StubWorker(running=True)
    lw.register(fresh)                      # must not raise
    assert lw.get_worker(done.session_id) is None
    assert lw.active_worker() is fresh


def test_unregister_frees_the_slot():
    w = _StubWorker()
    lw.register(w)
    lw.unregister(w.session_id)
    assert lw.get_worker(w.session_id) is None
    assert lw.active_worker() is None
    lw.register(_StubWorker())              # slot is reusable


def test_concurrent_registration_admits_exactly_one():
    """The lock must hold under a real race, not just sequential calls."""
    winners, losers = [], []
    barrier = threading.Barrier(8)

    def attempt():
        w = _StubWorker()
        barrier.wait()
        try:
            lw.register(w)
            winners.append(w)
        except lw.LiveStartError:
            losers.append(w)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1
    assert len(losers) == 7
