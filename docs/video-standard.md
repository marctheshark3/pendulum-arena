# Recorded benchmark video standard

The published fifteen-link video is the presentation reference. Future counts
use its **1280 × 1080, 50 fps** composition: dark background, cream headings,
teal accents, fixed half-metre grid, complete chain and rail, capsule palette,
cart wheels, phase and clock, scalar Cart/Force telemetry, upright streak and
hold status, largest wrapped absolute angle, and the Rage Industries /
neuralyogi.com footer. Applied force is shown as a scalar numeric readout,
matching the published fifteen-link presentation.

The original fifteen renderer is `render_canonical_video.py`, SHA256
`550066d65d04d7b4cb7139680142883c004620cb4127163ecff4ce3b6e48101d`.
Historical renderers, recordings, source freezes, registry records and videos
stay unchanged. `scripts/render_telemetry_video.py` is a NEW presentation
adapter; it does not confer benchmark qualification or rewrite prior evidence.

## What each frame means

Frame zero displays **post-command state 1**, the command that produced it
(`actions[0]`) and simulation time 0.02 seconds. Frame 1499 displays state 1500,
action 1499 and 30.00 seconds. Encoded frame PTS is `(post_command - 1) / 50`;
the displayed simulation time is `post_command / 50`. The initial state is not
a frame. There are exactly 1,500 post-command frames, thirty seconds, no added
freeze, interpolation, dropped samples, state inference or physics execution.

The recording contains native relative qpos followed by qvel, of dimension
`2 × (links + 1)`. Hinge positions accumulate before wrapping to absolute
angles. Representable angles already in `[-pi, pi)` are retained unchanged.
Upright means **every absolute angle is strictly below 0.15 rad** at a
post-command sample; only those samples count toward the 250-tick hold. The
status latches after the best streak earns five seconds, exactly like fifteen.

Force is the actual delivered normalized action, promoted to binary64 **before
multiplying by 80 N**. It is not the unclipped raw controller request, a queued
command or a force inferred from motion. Native float32 delivery is required;
fifteen's original binary64 storage of those same float32 values is supported.
When saved native timestamps exist, all 1,501 clocks are checked for the full
30 seconds and four-times-5-ms cadence. Fifteen's older tape lacks native clock
samples; its labels use the explicitly declared 50 Hz simulated cadence, and
provenance records that distinction.

Geometry uses the count-specific compiled-style segment length
`float(format(3 / links, '.9g'))`, radius 0.025 m and pivot z = 0.07 m. Sixteen's
segments are exactly 0.1875 m. The camera is fixed from complete-recording node
bounds with the same minimum ranges and margins as fifteen; it never follows
the cart or conceals a rail excursion. Parking is an explicit, source-checked
policy parameter: fifteen has 800 commands, sixteen has 850. Post-command 850
still says HANGING SETTLE; 851 says SWING UP AND BALANCE for sixteen.

## Render a fresh presentation

Use a separate rendering environment with existing NumPy, Pillow, DejaVu Sans,
FFmpeg and ffprobe. No MuJoCo or controller package is imported or needed. Do
not install rendering dependencies into a frozen scientific runtime.

```sh
python3 scripts/render_telemetry_video.py \
  --source-receipt /path/to/original/episode-receipt.json \
  --expected-source-sha256 ACTUAL_RECEIPT_SHA256 \
  --recording /path/to/original/tape.npz \
  --expected-recording-sha256 ACTUAL_TAPE_SHA256 \
  --links 16 --seed 216500 --park-ticks 850 \
  --out /fresh/canonical16-telemetry.mp4 \
  --proof /fresh/render-proof
```

For a quick visual check, replace `--out` with
`--preview-frame 1250 --preview-out /fresh/post-command-1250.png` and use a fresh
proof directory. Preview still validates the complete recording and writes all
telemetry rows; it renders only the requested frame and starts no encoder.

The source adapter accepts the original current episode receipt or fifteen's
original reserved aggregate. It verifies the pinned receipt and recording,
count, seed, task constants, phase boundary, complete saved score, finite state,
rail, command shape, float32 delivery and optional raw-action/time alignment.
All outputs are fresh; partial failures remain available for inspection.

Full provenance uses `pendulum_arena.telemetry_video.v1`; preview uses
`pendulum_arena.telemetry_preview.v1`. It binds the recording, receipt, renderer,
fonts, interpreter and encoding tools; records camera, geometry, all telemetry
rows and inspection PNGs; and verifies the encoded dimensions, count, cadence
and duration with ffprobe. The plain output is silent. A soundtrack is a
separate share derivative with its own provenance binding this presentation.
Neither presentation replaces the historical scientific video or its hashes.

Focused mapping/refusal tests need NumPy only:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test -p test_telemetry_video.py
```

Fifty Hz describes simulated action cadence, not a wall-clock realtime claim.
Rendering is presentation of saved simulation, not a new controller run,
independent reproduction, hardware result or global-priority claim.
