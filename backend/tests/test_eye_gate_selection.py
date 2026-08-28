"""Why EYE_STATE_METHOD defaults to "ear", and the limitation that remains.

=============================================================================
THE ABSOLUTE EAR NUMBERS BELOW ARE HISTORICAL (noted 2026-08-28)
=============================================================================
Every EAR figure in this module -- ali's 0.0976 median, saad's 0.023-0.056,
the 0.107/0.121 open readings -- was measured when _crop_face resized a
RECTANGULAR box to 256x256 and so scaled EAR by the crop's aspect ratio. That
was fixed on 2026-08-28 (see _square_face_box); on this footage the same eyes
now measure x1.227 higher (512 paired samples). Re-measuring would change every
number here and none of the conclusions: the cnn-vs-ear comparison and the
aliasing analysis both rest on RATIOS against each student's own baseline, and
a uniform scale factor cancels in a ratio. The figures are left as recorded
rather than silently rescaled, so the evidence stays the evidence that was
actually taken. Do not compare them directly against EAR values measured today.

The sustained closure at f6775-6975 documented below is now also the source of
the classroom fixtures in backend/tests/test_behavior_eye_geometry.py.

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
SAMPLING-RATE ALIASING — short apparent closures can be FABRICATED
=============================================================================
This section previously recorded a suspected UNDERCOUNT for ali (short
eyes-closed bursts that never reached the dwell). A 5fps re-sampling experiment
on 2026-08-19 DISCONFIRMED that, and found the opposite risk. The original
concern is kept below only so the reversal is legible.

WHAT WAS SUSPECTED: ali had 33 eyes-closed frames across 412 at 1fps but
produced only 1 SLEEPING event, because his closures came in runs of 1-2
consecutive samples and the dwell needs 3. At 1fps a 2s doze and two blinks a
second apart are indistinguishable, so it was left open.

WHAT THE 5fps RE-SAMPLING SHOWED: the same clip was replayed at frame_step=5
(200ms resolution, 1,996 usable samples for ali) and scored two ways — against
the live rolling-75 baseline, and against a fixed rate-independent baseline
(median EAR 0.0976) so run lengths compare across rates:

    5fps, live baseline  : 89 runs — 64x1, 17x2, 5x3, 2x4, 1x5   longest 0.8s
    5fps, fixed baseline : 43 runs — 33x1,  6x2, 3x3, 1x4        longest 0.6s
    runs clearing the 1.5s dwell, either criterion: 0

Ali's longest eye closure in the entire 8-minute clip is 0.8s. Nothing comes
near 1.5s. The distribution — most runs confined to a single 200ms sample,
tailing off sharply — is a blink profile (blinks run 100-400ms). He was not
asleep, and 1fps was NOT undercounting him.

THE REAL RISK IS THE REVERSE. Both 1fps "runs of 3" dissolve under 5fps:

    frames 1750/1775/1800 read 0.049/0.049/0.058 at 1fps — three consecutive
    closed samples. The 5fps frames BETWEEN them read 0.107, 0.121, 0.118,
    0.130, 0.129, 0.124 — eyes wide open. Three separate blinks that happened
    to land one second apart, aliased into an apparent 2.0s closure.

    frames 5175..5225 — the run that produced ali's ONLY SLEEPING event — is
    the same aliasing plus detection gaps (ali is absent at 5175, 5180, 5210,
    5215; the frames between read 0.102-0.120).

So ali's single SLEEPING event is a FALSE POSITIVE manufactured by the sampling
rate. Under the fixed rate-independent baseline, 1fps yields ZERO runs clearing
the dwell — both apparent runs depend on the drifting live baseline.

The comparison is trustworthy: on the 384 source frames both runs sampled, mean
|delta EAR| = 0.00008 (max 0.021), i.e. the pipeline measures identically at
both rates, so the run-length difference is signal and not run-to-run variance.

CONFIRMED FOR saad AND Samaan TOO. Both were re-sampled at 5fps and each of
their 1fps runs of >=3 samples was expanded: if a closure is genuine the 5fps
frames INSIDE its span must also read closed; if they read open, the run was
independent blinks aliased together.

    run class                        ALIASED  PARTIAL  SUSTAINED  total
    short (3-4 samples, 2-3s)             11        3          0     14
    long  (9-14 samples, 8-13s)            0        2          1      3

ZERO of the 14 short runs are sustained; 11 are outright aliased. Examples:

    Samaan f10950..11000 — 1fps saw 3 consecutive closed samples (2.0s). At
      5fps ALL 8 intervening frames read open: 0.163 0.145 0.152 0.079 0.077
      0.076. Entirely fabricated.
    Samaan f3125..3200   — 11 of 12 intervening frames open.
    saad   f3900..3950   — 7 of 8 open (0.140 0.229 0.211 0.102 0.118).
    saad   f4125..4175   — 7 of 8 open, peaking at 0.266.

The long runs are real:

    saad f6775..6975 (9 samples, 8s)  — 0 of 32 intervening frames open,
      EAR 0.023-0.056 throughout. Unambiguous sleep.
    saad f8050..8300 (11 samples, 10s) — only 2 of 40 open, both at the very
      start; from f8060 every frame reads 0.025-0.043. Real sleep (the PARTIAL
      label is an artifact of the >=50%-open cutoff used to classify).
    saad f325..650  (14 samples, 13s) — 10 of 52 open; starts genuinely closed
      then opens. Real closure whose DURATION 1fps overstated.

RULE OF THUMB: at 1fps, a 3-4 sample run is usually WRONG, not merely weak. A
run of 9+ samples is trustworthy, though its duration may be overstated.

EVENT-COUNT IMPACT: a run of length L yields L-2 SLEEPING events, so saad's six
3-sample and four 4-sample runs contribute ~14 events against ~28 from his three
long runs — roughly a third of his events rest on aliasing. Samaan is worse: none
of his 4 runs are sustained, so most or all of his 5 events are aliased.

This does NOT undermine the cnn-vs-ear decision above. That comparison turned on
frames where the CNN called eyes closed while EAR showed them objectively OPEN —
a per-frame error, not a duration artifact — and a 58%-vs-0% gap is far too large
for aliasing to explain. What is affected is the EAR gate's EVENT COUNTS, which
are inflated for short closures.

WHY NOTHING WAS CHANGED: _SLEEPING_SECONDS, the EAR gate, behavior.py and the
client sampling rate are all untouched. Raising the client's rate would remove
most of this aliasing; it also changes throughput and cost for every session.
That remains a deliberate scope decision, now taken with the measurement in hand
rather than without it.

Still open: only ali, saad and Samaan were re-sampled — the remaining 11
students' short runs have not been checked, and the same aliasing should be
assumed present for them until measured.
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
    """Pins the aliasing threshold: at 1fps a SLEEPING event can rest on as
    little as 3 sampled frames spanning 2.0s, which 5fps showed can be three
    independent blinks a second apart rather than one closure."""
    assert _consecutive_samples_to_sleep(_LIVE_FRAME_STEP) == 3


def test_two_sample_runs_can_never_fire_at_live_rate():
    """A 2-run spans 1.0s of video time — below _SLEEPING_SECONDS (1.5s)."""
    _, t0 = _source_frame_and_ms(0, _LIVE_FRAME_STEP, _SRC_FPS)
    _, t1 = _source_frame_and_ms(1, _LIVE_FRAME_STEP, _SRC_FPS)
    assert (t1 - t0) / 1000.0 == 1.0
    assert 1.0 < bmod._SLEEPING_SECONDS


def test_denser_sampling_lowers_the_detectable_closure_floor():
    """Quantifies what raising the client's rate would buy, for whoever
    revisits the scope decision. The 5fps experiment used exactly this: at
    200ms resolution ali's closures resolved to a 0.8s maximum, exposing the
    1fps events as aliased blinks."""
    floor_1fps = (_consecutive_samples_to_sleep(25) - 1) * (25 / _SRC_FPS)
    floor_5fps = (_consecutive_samples_to_sleep(5) - 1) * (5 / _SRC_FPS)
    assert floor_1fps == pytest.approx(2.0)
    assert floor_5fps == pytest.approx(1.6)
    assert floor_5fps < floor_1fps
