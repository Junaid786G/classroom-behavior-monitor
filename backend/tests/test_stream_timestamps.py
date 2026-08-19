"""Regression tests for WS live-mode video-time derivation.

These lock in the fix for the 25x clock compression. The live client decimates
its source (frontend _SKIP_FRAMES = 24, i.e. every 25th frame) but the server
used to compute `timestamp_ms = msg_index * (1000 // fps)`, treating consecutive
WS messages as consecutive source frames. Video time therefore advanced 40 ms per
message instead of 1000 ms — 25x too slow — which scaled every behaviour dwell
timer (_DISTRACTED_SECONDS, _SLEEPING_SECONDS) by the same factor and made
DISTRACTED/SLEEPING effectively unreachable in live mode.

The observed session ef72cdeb-4890-40fa-b2b6-786b54abac79 is the reference case:
an 11,499-frame / 25 fps video (460.0 s) produced max start_frame 459 and max
start_ms 18,360 (= 18.4 s of a 460 s video).
"""
import pytest

from backend.pipeline import behavior as bmod
from backend.routers.stream import _source_frame_and_ms


# The real session that exposed the bug.
_SRC_FRAMES = 11_499
_SRC_FPS = 25.0
_CLIENT_STEP = 25          # frontend _SKIP_FRAMES (24) + 1
_OBSERVED_MAX_FRAME = 459
_OBSERVED_MAX_START_MS = 18_360   # what the buggy clock actually recorded


def test_contiguous_client_is_unchanged():
    """frame_step=1 (a client sending every frame) must behave exactly as before:
    message i is source frame i at i/fps seconds."""
    for i in (0, 1, 25, 1000):
        frame, ms = _source_frame_and_ms(i, frame_step=1, fps=_SRC_FPS)
        assert frame == i
        assert ms == pytest.approx(i * 1000.0 / _SRC_FPS, abs=1)


def test_decimating_client_recovers_real_video_time():
    """With the client's decimation declared, one message advances a full second
    of video time (25 frames at 25 fps), not 40 ms."""
    _, ms0 = _source_frame_and_ms(0, _CLIENT_STEP, _SRC_FPS)
    _, ms1 = _source_frame_and_ms(1, _CLIENT_STEP, _SRC_FPS)
    assert ms0 == 0
    assert ms1 == 1000


def test_frame_numbers_are_source_indices():
    """Persisted frame numbers must be source-video indices, so DB rows are
    comparable to the footage. Message i maps to source frame i*step."""
    assert _source_frame_and_ms(0, _CLIENT_STEP, _SRC_FPS)[0] == 0
    assert _source_frame_and_ms(1, _CLIENT_STEP, _SRC_FPS)[0] == 25
    assert _source_frame_and_ms(10, _CLIENT_STEP, _SRC_FPS)[0] == 250


def test_reference_session_now_spans_the_whole_video():
    """The end-to-end check against the real session.

    460 messages covered the entire 11,499-frame video. The last message must now
    land at the true end of the footage (~460 s), not at the 18.4 s the buggy
    clock recorded.
    """
    n_messages = _SRC_FRAMES // _CLIENT_STEP          # 459 -> indices 0..459
    last_frame, last_ms = _source_frame_and_ms(n_messages, _CLIENT_STEP, _SRC_FPS)

    assert last_frame == n_messages * _CLIENT_STEP == 11_475
    assert last_frame <= _SRC_FRAMES          # never points past the footage
    assert last_ms == pytest.approx(_SRC_FRAMES / _SRC_FPS * 1000, rel=0.01)
    assert last_ms / 1000.0 == pytest.approx(460.0, abs=5.0)

    # The specific regression: the old clock compressed this to 18.4 s.
    buggy_ms = n_messages * (1000 // int(_SRC_FPS))
    assert buggy_ms == _OBSERVED_MAX_START_MS
    assert last_ms / buggy_ms == pytest.approx(_CLIENT_STEP, rel=0.01)


def test_dwell_timers_are_reachable_after_the_fix():
    """The functional consequence: DISTRACTED must be reachable within a
    plausible number of live frames.

    _DISTRACTED_SECONDS is 5.0. Under the old clock that needed 125 messages
    (125 s of uninterrupted off-axis gaze); under the corrected clock it needs 5.
    """
    def messages_to_dwell(seconds, frame_step):
        i = 0
        while True:
            i += 1
            _, ms = _source_frame_and_ms(i, frame_step, _SRC_FPS)
            if ms / 1000.0 >= seconds:
                return i

    fixed = messages_to_dwell(bmod._DISTRACTED_SECONDS, _CLIENT_STEP)
    buggy = messages_to_dwell(bmod._DISTRACTED_SECONDS, 1)

    assert fixed == 5
    assert buggy == 125
    assert buggy == fixed * _CLIENT_STEP


def test_sleeping_dwell_is_reachable():
    """Same check for the shorter SLEEPING dwell."""
    _, ms = _source_frame_and_ms(2, _CLIENT_STEP, _SRC_FPS)
    assert ms / 1000.0 >= bmod._SLEEPING_SECONDS   # 1.5 s reached by message 2


def test_non_25fps_source_is_honoured():
    """fps comes from the client's probed source, not a hardcoded constant, so
    30 fps footage is timed correctly too."""
    _, ms = _source_frame_and_ms(1, frame_step=30, fps=30.0)
    assert ms == 1000
    _, ms = _source_frame_and_ms(3, frame_step=15, fps=30.0)
    assert ms == 1500


def test_timestamps_are_monotonic_and_integral():
    """Timestamps must be non-decreasing ints — they feed dwell-timer subtraction."""
    prev = -1
    for i in range(200):
        _, ms = _source_frame_and_ms(i, _CLIENT_STEP, 29.97)   # non-integer fps
        assert isinstance(ms, int)
        assert ms > prev
        prev = ms
