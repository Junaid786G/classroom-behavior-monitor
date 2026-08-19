"""Why EYE_STATE_METHOD defaults to "ear", and the limitation that remains.

=============================================================================
THE DECISION (2026-08-19)
=============================================================================
Both eye gates were replayed over identical frames of test_clip_8min.mp4
(11,499 frames @ 25fps, 14 students present) at the live-monitor sampling rate.
Every SLEEPING event was then scored against that student's own median EAR,
computed from all their measured frames — a baseline independent of which gate
produced the event, since EAR is derived from landmarks before either gate runs.

                                     cnn      ear
    total SLEEPING events             66       73
    carry-forward (no measurement)     4        3
    measured, with baseline           60       66
    measured & eyes NOT closed        35        0
    ------------------------------------------------
    false-positive rate              58%       0%

"Eyes NOT closed" means EAR >= 90% of that student's own median — i.e. the eye
was as open as in a typical frame for them. The two gates agreed on only 2 of
137 events. The CNN placed 49 of its 66 calls on one student (Aneeq) whose eyes
are visibly open in the crops, at EAR up to 0.345 against his 0.178 median; and
it produced 1 event for the student (saad) who was visibly asleep in runs of
9, 11 and 14 consecutive samples, where the EAR gate produced 32.

This supersedes the earlier informal impression that the CNN gate was better.
That impression was never frame-validated; this is.

NOT established by this work: results come from one 8-minute clip in one room
at 30-70px face sizes. The CNN gate may well be the better choice at higher
resolution. Re-validate before generalising.

=============================================================================
KNOWN LIMITATION — short closures are undercounted (ali's case)
=============================================================================
The EAR gate's *detection* is sound; its *dwell* interacts badly with the
live-monitor sampling rate.

Verified for ali, the student with the narrowest EAR margin in the cohort:
  - His 20 lowest-EAR non-triggering frames were dumped and inspected.
    ALL 20 genuinely show closed eyes. Zero false positives.
  - His rolling-75 baseline is 0.119 — NOT "stuck near zero". An older
    diagnosis reporting ~0.02-0.09 for him was reading his raw EAR floor
    (p10 = 0.057), not the baseline the gate actually uses.
  - He had 33 eyes-closed frames across 412 but produced 1 SLEEPING event.

The cause is arithmetic, not the gate. The live monitor samples every 25th
frame (frontend _SKIP_FRAMES = 24), so messages are 1s of video apart, and
_SLEEPING_SECONDS = 1.5 is measured from the first closed frame. Elapsed
therefore runs 0s, 1.0s, 2.0s — SLEEPING can only fire on the THIRD consecutive
closed sample. Runs of 1 or 2 can never fire, whatever their true duration.

    ali    : 26 closed-eye runs — 21 of length 1, 3 of length 2, 2 of length 3
    Samaan : 57 runs — 44x1, 9x2, 3x3, 1x4
    saad   : 62 runs — 40x1, 9x2, 6x3, 4x4, and runs of 9, 11 and 14

saad's long runs clear the bar easily. 24 of ali's 26 runs cannot.

WHAT THIS MEANS: ali's true sleeping rate may be undercounted. It is NOT
established that it is — at 1 fps a 2-second doze and two blinks caught a
second apart are indistinguishable, and 33 closed frames in 412 (8%) is also
consistent with ordinary blinking. The crops look like droop rather than
mid-blink, but that is a judgement about 40-pixel eyes, not evidence.

WHY IT IS NOT FIXED HERE: resolving it requires raising the live-monitor
client's sampling rate, which changes throughput and cost for every session.
That is a deliberate scope decision taken on 2026-08-19, not an oversight.
The tests below pin the arithmetic so the trade-off stays visible.
"""
import pytest

from backend.config import get_settings
from backend.pipeline import behavior as bmod
from backend.routers.stream import _source_frame_and_ms

_LIVE_FRAME_STEP = 25      # frontend _SKIP_FRAMES (24) + 1
_SRC_FPS = 25.0


def _consecutive_samples_to_sleep(frame_step, fps=_SRC_FPS, seconds=None):
    """How many consecutive eyes-closed samples SLEEPING needs at this rate."""
    secs = bmod._SLEEPING_SECONDS if seconds is None else seconds
    _, t0 = _source_frame_and_ms(0, frame_step, fps)
    i = 0
    while True:
        i += 1
        _, t = _source_frame_and_ms(i, frame_step, fps)
        if (t - t0) / 1000.0 >= secs:
            return i + 1


def test_ear_gate_is_the_configured_default():
    """Pins the 2026-08-19 decision. If this fails someone changed the gate —
    read this module's docstring before assuming that is correct."""
    assert get_settings().eye_state_method == "ear"


def test_live_sampling_requires_three_consecutive_closed_samples():
    """The mechanism behind ali's undercount."""
    assert _consecutive_samples_to_sleep(_LIVE_FRAME_STEP) == 3


def test_two_sample_runs_can_never_fire_at_live_rate():
    """A 2-run spans 1.0s of video time — below _SLEEPING_SECONDS (1.5s)."""
    _, t0 = _source_frame_and_ms(0, _LIVE_FRAME_STEP, _SRC_FPS)
    _, t1 = _source_frame_and_ms(1, _LIVE_FRAME_STEP, _SRC_FPS)
    assert (t1 - t0) / 1000.0 == 1.0
    assert 1.0 < bmod._SLEEPING_SECONDS


def test_denser_sampling_lowers_the_detectable_closure_floor():
    """Quantifies what raising the client's rate would buy, for whoever
    revisits the scope decision. Shorter floor = shorter real closures
    become detectable, and blink-vs-doze becomes separable."""
    floor_1fps = (_consecutive_samples_to_sleep(25) - 1) * (25 / _SRC_FPS)
    floor_5fps = (_consecutive_samples_to_sleep(5) - 1) * (5 / _SRC_FPS)
    assert floor_1fps == pytest.approx(2.0)
    assert floor_5fps == pytest.approx(1.6)
    assert floor_5fps < floor_1fps
