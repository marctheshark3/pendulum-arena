"""Render saved post-command states in the published fifteen-link video style.

No simulator, model, controller, reset generator or historical renderer is
imported. This is a new presentation adapter, outside every historical freeze.
"""
from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import numpy as np

WIDTH, HEIGHT, FPS, COMMANDS = 1280, 1080, 50, 1500
FORCE_N, RAIL_M, ANGLE_RAD, HOLD_TICKS = 80., 3., .15, 250
REFERENCE_RENDERER_SHA256 = "550066d65d04d7b4cb7139680142883c004620cb4127163ecff4ce3b6e48101d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_row(path, *, name=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular non-symlink source file required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return {"file": name or path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def checked_row(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), "explicit SHA256 required")
    row = file_row(path)
    require(row["sha256"] == expected, "source bytes differ from the supplied SHA256")
    return row


def absolute_angles(states, links):
    theta = np.cumsum(states[:, 1:links + 1], axis=1)
    wrapped = (theta + np.pi) % (2 * np.pi) - np.pi
    # Preserve representable near-zero errors, exactly as the benchmark scorer.
    return np.where((theta >= -np.pi) & (theta < np.pi), theta, wrapped)


@dataclass(frozen=True)
class Recording:
    states: np.ndarray
    actions: np.ndarray
    times: np.ndarray | None
    links: int


def load_recording(path, links):
    require(type(links) is int and 1 <= links <= 100, "explicit positive supported link count required")
    with np.load(path, allow_pickle=False) as saved:
        require({"states", "actions"} <= set(saved.files), "saved states/actions required")
        states, actions = saved["states"].copy(), saved["actions"].copy()
        times = saved["times"].copy() if "times" in saved else None
        raw = saved["raw_actions"].copy() if "raw_actions" in saved else None
    require(states.shape == (COMMANDS + 1, 2 * (links + 1)) and states.dtype == np.float64,
            "1501 binary64 relative q/qdot states with the declared count required")
    require(actions.shape == (COMMANDS,) and actions.dtype in (np.dtype("float32"), np.dtype("float64")),
            "1500 delivered floating-point actions required")
    require(np.isfinite(states).all() and np.isfinite(actions).all(), "finite complete recording required")
    require(np.all(np.abs(states[:, 0]) <= RAIL_M) and np.all(np.abs(actions) <= 1.),
            "in-rail states and bounded delivered actions required")
    require(np.array_equal(actions, actions.astype(np.float32).astype(actions.dtype)),
            "actions must be actual float32 deliveries (legacy binary64 storage is accepted)")
    if raw is not None:
        require(raw.shape == (COMMANDS,) and raw.dtype == np.float64 and np.isfinite(raw).all(),
                "finite aligned binary64 raw actions required")
        require(np.array_equal(np.clip(raw, -1., 1.).astype(np.float32), actions.astype(np.float32)),
                "raw-to-delivered action alignment differs")
    if times is not None:
        require(times.shape == (COMMANDS + 1,) and times.dtype == np.float64 and np.isfinite(times).all(),
                "finite aligned native clock required")
        require(abs(float(times[0])) <= 1e-10 and abs(float(times[-1]) - 30.) <= 1e-8
                and float(np.max(np.abs(np.diff(times) - 1 / FPS))) <= 1e-10,
                "native clock must cover all 1500 commands at 50 Hz from zero")
    for array in (states, actions, times):
        if array is not None:
            array.setflags(write=False)
    return Recording(states, actions, times, links)


def source_binding(receipt_path, recording_row, recording, *, seed, park_ticks):
    """Verify either a current original episode or the original fifteen aggregate."""
    receipt = json.loads(Path(receipt_path).read_text())
    require(type(seed) is int and seed == 200000 + 1000 * recording.links + 500,
            "the declared count's actual reserved video seed is required")
    require(type(park_ticks) is int and 0 <= park_ticks < COMMANDS, "literal parking command count required")
    require(receipt.get("stage") == "reserved", "original reserved episode evidence required")
    if "recording" in receipt and "identity" in receipt:
        row = receipt["recording"]
        require(row["sha256"] == recording_row["sha256"] and row["bytes"] == recording_row["bytes"],
                "recording is not bound by the original episode receipt")
        require(receipt["seed"] == seed and receipt["exact"] is False
                and receipt["source_pins_unchanged"] is True
                and receipt["reset_count_after_initialization"] == 0,
                "wrong, changed or reset reserved video episode")
        identity = receipt["identity"]
        p = identity["protocol"]
        expected = {"n_links": recording.links, "total_length_m": 3., "total_mass_kg": 1.,
                    "cart_mass_kg": 1., "physics_dt_s": .005, "frame_skip": 4,
                    "episode_seconds": 30., "force_limit_N": FORCE_N, "rail_limit_m": RAIL_M,
                    "upright_limit_rad": ANGLE_RAD, "required_hold_seconds": 5.,
                    "cart_damping": .02, "total_hinge_damping": .015,
                    "joint_armature": .0005, "capsule_radius_m": .025, "reset_noise_std": .05}
        require(all(p.get(key) == value for key, value in expected.items()), "different recorded task/count")
        require(identity["policy_parameters"]["park_ticks"] == park_ticks, "recorded phase boundary differs")
        score = receipt["score"]
        layout = "original episode receipt"
    else:
        require(receipt.get("passed") is True and receipt.get("complete") is True,
                "complete passing original reserved aggregate required")
        require(recording.links == 15 and park_ticks == 800
                and receipt.get("identities", {}).get("candidate", {}).get("model.xml")
                == "73876177ae2950fe42752878f79f72d5334a4344a177d0b619b21af668b349d7",
                "legacy adapter is specifically the original fifteen-link model and phase")
        cases = [case for case in receipt["cases"] if case["gate"] == "video" and case["seed"] == seed]
        require(len(cases) == 1, "one original held-out video case required")
        score = cases[0]
        require(score.get("admitted") is True and score["recording_sha256"] == recording_row["sha256"],
                "legacy video recording/admission binding differs")
        layout = "original fifteen reserved aggregate"
    rows = telemetry(recording, park_ticks)
    require(score["success"] is True and score["complete"] is True and score["ticks"] == COMMANDS
            and score["longest_upright_ticks"] == rows[-1]["best_upright_ticks"]
            and score["longest_upright_seconds"] in (rows[-1]["best_upright_ticks"] / FPS,
                rows[-1]["best_upright_ticks"] * (1 / FPS))
            and score["maximum_cart_m"] == float(np.max(np.abs(recording.states[:, 0]))),
            "recomputed complete saved-state score differs from original receipt")
    require(rows[-1]["best_upright_ticks"] >= HOLD_TICKS, "the saved episode does not earn its upright hold")
    return rows, layout


def telemetry(recording, park_ticks):
    angles = np.max(np.abs(absolute_angles(recording.states, recording.links)), axis=1)
    # Promote F32 before multiplying: NumPy F32*80 would round telemetry again.
    forces = recording.actions.astype(np.float64) * FORCE_N
    streak = best = 0
    rows = []
    for index in range(1, COMMANDS + 1):
        streak = streak + 1 if angles[index] < ANGLE_RAD else 0
        best = max(best, streak)
        rows.append({"post_command": index, "encoded_frame_index": index - 1,
                     "encoded_pts_seconds": (index - 1) / FPS, "display_time_seconds": index / FPS,
                     "native_time_seconds": None if recording.times is None else float(recording.times[index]),
                     "cart_m": float(recording.states[index, 0]), "delivered_force_N": float(forces[index - 1]),
                     "max_wrapped_absolute_angle_rad": float(angles[index]),
                     "upright_streak_ticks": streak, "best_upright_ticks": best, "hold_met": best >= HOLD_TICKS,
                     "phase": "HANGING SETTLE" if index <= park_ticks else "SWING UP AND BALANCE"})
    return rows


def geometry(recording):
    length = float(format(3 / recording.links, ".9g"))
    theta = np.cumsum(recording.states[:, 1:recording.links + 1], axis=1)
    xs = np.column_stack([recording.states[:, 0], recording.states[:, 0, None]
                          + np.cumsum(length * np.sin(theta), axis=1)])
    zs = np.column_stack([np.full(len(theta), .07), .07 + np.cumsum(length * np.cos(theta), axis=1)])
    return xs, zs, length


class FrameRenderer:
    def __init__(self, recording, rows, seed):
        from PIL import Image, ImageDraw, ImageFont
        self.Image, self.ImageDraw = Image, ImageDraw
        self.recording, self.rows = recording, rows
        self.xs, self.zs, self.length = geometry(recording)
        xlo, xhi = min(-3.35, float(self.xs.min()) - .15), max(3.35, float(self.xs.max()) + .15)
        zlo, zhi = min(-3.15, float(self.zs.min()) - .15), max(3.25, float(self.zs.max()) + .15)
        self.scale = min((WIDTH - 120) / (xhi - xlo), 750 / (zhi - zlo))
        self.center_x, self.center_z = (xlo + xhi) / 2, (zlo + zhi) / 2
        self.camera = dict(xlo=xlo, xhi=xhi, zlo=zlo, zhi=zhi, pixels_per_meter=self.scale,
                           center_x=self.center_x, center_z=self.center_z, fixed_for_complete_recording=True)
        self.fonts = {size: ImageFont.truetype("DejaVuSans.ttf", size) for size in (19, 23, 27, 34)}
        self.bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        self.font_rows = [file_row(self.fonts[19].path), file_row(self.bold.path)]
        self.colors = [tuple(round(v * 255) for v in colorsys.hls_to_rgb(i * .618034 % 1, .68, .72))
                       for i in range(recording.links)]
        self.base = Image.new("RGB", (WIDTH, HEIGHT), "#0b1115")
        g = ImageDraw.Draw(self.base)
        for x in np.arange(np.floor(xlo), xhi + .5, .5):
            g.line([self.point(x, zlo), self.point(x, zhi)], fill="#142229", width=1)
        for z in np.arange(np.floor(zlo), zhi + .5, .5):
            g.line([self.point(xlo, z), self.point(xhi, z)], fill="#142229", width=1)
        rail_z = -.11
        g.line([self.point(-3, rail_z), self.point(3, rail_z)], fill="#6d8791", width=3)
        for x in (-3, 3):
            px, py = self.point(x, rail_z)
            g.line([(px, py - 10), (px, py + 10)], fill="#f6b86b", width=3)
        g.rectangle((0, 0, WIDTH, 118), fill="#0b1115")
        g.text((46, 24), f"{recording.links} LINKS  /  PUBLISHED BENCHMARK", font=self.bold, fill="#f5e8cf")
        g.text((47, 73), "3 m / 1 kg chain     |     80 N cart     |     ±3 m rail     |     MuJoCo 3.3.5",
               font=self.fonts[23], fill="#90a6ae")
        g.rectangle((0, 903, WIDTH, HEIGHT), fill="#0b1115")
        g.line([(0, 903), (WIDTH, 903)], fill="#55e6c3", width=2)
        g.text((46, 1001), "RAGE INDUSTRIES", font=self.bold, fill="#f5e8cf")
        g.text((WIDTH - 46, 1011), "neuralyogi.com", font=self.fonts[27], fill="#55e6c3", anchor="rt")
        g.text((46, 1052), f"Recorded simulation  ·  Actual held-out episode {seed}  ·  50 Hz simulated cadence",
               font=self.fonts[19], fill="#90a6ae")

    def point(self, x, z):
        return WIDTH / 2 + (x - self.center_x) * self.scale, 515 - (z - self.center_z) * self.scale

    def frame(self, index):
        require(type(index) is int and 1 <= index <= COMMANDS, "post-command preview index must be 1..1500")
        im = self.base.copy()
        draw = self.ImageDraw.Draw(im)
        radius = max(2, round(.025 * self.scale))
        for link in range(self.recording.links):
            a = self.point(self.xs[index, link], self.zs[index, link])
            b = self.point(self.xs[index, link + 1], self.zs[index, link + 1])
            draw.line([a, b], fill=self.colors[link], width=2 * radius)
            for x, y in (a, b):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=self.colors[link])
            draw.ellipse((a[0] - 2, a[1] - 2, a[0] + 2, a[1] + 2), fill="#0b1115")
        state = self.recording.states[index]
        a, b = self.point(state[0] - .18, .07), self.point(state[0] + .18, -.07)
        draw.rounded_rectangle((a[0], a[1], b[0], b[1]), radius=4,
                               fill="#26494d", outline="#55e6c3", width=2)
        for dx in (-.115, .115):
            x, y = self.point(state[0] + dx, -.085)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#0b1115", outline="#90a6ae", width=2)
        row = self.rows[index - 1]
        draw.text((WIDTH - 46, 136), f"{row['display_time_seconds']:05.2f} / 30.00 s",
                  font=self.fonts[27], fill="#f5e8cf", anchor="rt")
        draw.text((46, 136), row["phase"], font=self.fonts[23], fill="#55e6c3")
        draw.text((46, 925), f"Cart  {row['cart_m']:+.3f} m       Force  {row['delivered_force_N']:+.2f} N",
                  font=self.fonts[23], fill="#f5e8cf")
        label = "5 SECOND HOLD MET" if row["hold_met"] else f"Upright streak  {row['upright_streak_ticks'] / FPS:.2f} / 5.00 s"
        draw.text((WIDTH - 46, 925), label, font=self.fonts[23],
                  fill="#55e6c3" if row["hold_met"] else "#f5e8cf", anchor="rt")
        draw.text((46, 964), f"Largest wrapped absolute angle  {row['max_wrapped_absolute_angle_rad']:.5f} rad    |    Required < 0.15 rad",
                  font=self.fonts[19], fill="#90a6ae")
        return im


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--expected-recording-sha256", required=True)
    parser.add_argument("--links", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--park-ticks", type=int, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--preview-frame", type=int)
    parser.add_argument("--preview-out", type=Path)
    parser.add_argument("--encoder-threads", type=int, default=4)
    args = parser.parse_args()
    preview = args.preview_frame is not None
    require((preview and args.preview_out is not None and args.out is None)
            or (not preview and args.out is not None and args.preview_out is None),
            "choose preview-frame/preview-out OR a complete --out video")
    require(1 <= args.encoder_threads <= 16, "bounded encoder thread count required")
    target = args.preview_out if preview else args.out
    require(not target.exists() and not target.is_symlink() and not args.proof.exists()
            and not args.proof.is_symlink(), "refusing to overwrite an output or proof directory")
    source_row = checked_row(args.source_receipt, args.expected_source_sha256)
    recording_row = checked_row(args.recording, args.expected_recording_sha256)
    recording = load_recording(args.recording, args.links)
    rows, layout = source_binding(args.source_receipt, recording_row, recording,
                                  seed=args.seed, park_ticks=args.park_ticks)
    renderer = FrameRenderer(recording, rows, args.seed)
    if preview:
        require(1 <= args.preview_frame <= COMMANDS, "post-command preview index must be 1..1500")
    args.proof.mkdir(parents=True, exist_ok=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.proof / "telemetry.json", rows)
    frames = []
    probe = None
    tools = {}
    if preview:
        with target.open("xb") as image_file:
            renderer.frame(args.preview_frame).save(image_file, format="PNG")
        frames.append({"post_command": args.preview_frame, **file_row(target)})
    else:
        for tool in ("ffmpeg", "ffprobe"):
            path = shutil.which(tool)
            require(path is not None, f"{tool} must already be installed in the render environment")
            tools[tool] = file_row(Path(path).resolve())
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
                   "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
                   "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                   "-threads", str(args.encoder_threads), "-pix_fmt", "yuv420p", "-map_metadata", "-1",
                   "-movflags", "+faststart", str(target)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for index in range(1, COMMANDS + 1):
                frame = renderer.frame(index)
                encoder.stdin.write(frame.tobytes())
                if index in {1, args.park_ticks, args.park_ticks + 1, 1000, 1250, 1500}:
                    path = args.proof / f"post-command-{index:04d}.png"
                    frame.save(path)
                    frames.append({"post_command": index, **file_row(path)})
                if index % 250 == 0:
                    print(json.dumps({"rendered_post_commands": index, "total": COMMANDS}), flush=True)
        finally:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
            status = encoder.wait()
        require(status == 0, "video encoder failed; partial output is retained")
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames,duration", "-of", "json", str(target)], text=True))
        stream, = probe["streams"]
        require(stream["width"] == WIDTH and stream["height"] == HEIGHT
                and stream["r_frame_rate"] == "50/1" and int(stream["nb_read_frames"]) == COMMANDS
                and float(stream["duration"]) == 30., "encoded video does not match 1500 frames / 30 s / 50 fps")
    # Recheck source bytes after rendering; no source artifact is rewritten.
    checked_row(args.source_receipt, args.expected_source_sha256)
    checked_row(args.recording, args.expected_recording_sha256)
    import PIL
    result = {"schema": "pendulum_arena.telemetry_preview.v1" if preview else "pendulum_arena.telemetry_video.v1",
              "passed": True, "n_links": args.links, "seed": args.seed, "park_ticks": args.park_ticks,
              "source_receipt": source_row, "source_recording": recording_row, "source_layout": layout,
              "source_renderer": file_row(__file__, name="scripts/render_telemetry_video.py"),
              "reference_style": {"renderer": "original fifteen scripts/render_canonical_video.py",
                                  "sha256": REFERENCE_RENDERER_SHA256, "historical_sources_modified": False},
              "output": file_row(target), "width": WIDTH, "height": HEIGHT, "fps": FPS,
              "rendered_frames": 1 if preview else COMMANDS, "simulated_seconds": 30.,
              "post_command_indices": [args.preview_frame] if preview else [1, COMMANDS],
              "initial_frame_included": False, "extra_held_frames": 0, "audio": False,
              "stream_probe": probe, "camera": renderer.camera,
              "geometry": {"link_count": args.links, "serialized_link_length_m": renderer.length,
                           "capsule_radius_m": .025, "cart_pivot_z_m": .07,
                           "angle_coordinates": "cumulative native relative hinge positions"},
              "telemetry": file_row(args.proof / "telemetry.json"), "inspection_frames": frames,
              "fonts": renderer.font_rows, "encoder_tools": tools,
              "render_runtime": {"python": platform.python_version(), "numpy": np.__version__, "Pillow": PIL.__version__,
                                 "executable": file_row(Path(sys.executable).resolve()),
                                 "encoder_threads": 0 if preview else args.encoder_threads},
              "telemetry_alignment": "post-command state i; delivered action i-1 promoted to binary64 before x80; encoded PTS=(i-1)/50",
              "native_clock_saved": recording.times is not None,
              "physics_steps_during_render": 0, "controller_calls_during_render": 0,
              "qualification_awarded_by_renderer": False,
              "claim_scope": "new presentation of a hash-bound actual held-out recording; no hardware or real-time claim"}
    write_json(args.proof / "receipt.json", result)
    print(json.dumps({"proof": str(args.proof / "receipt.json"), "output": str(target), "preview": preview}), flush=True)


if __name__ == "__main__":
    main()
