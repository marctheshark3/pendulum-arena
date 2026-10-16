"""Render recorded relative-coordinate states; no plant or policy execution."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform
import subprocess

import numpy as np

from .canonical_benchmark import CanonicalProtocol, absolute_angles
from .canonical_evaluation import (
    audit_episode, pinned_file, require, value_sha256, write_json_exclusive,
)


def chain_positions(state, protocol):
    """Forward geometry only, with the same nine-digit link lengths as XML."""
    state = np.asarray(state)
    require(state.shape == (2 * (protocol.n_links + 1),) and np.isfinite(state).all(),
            "finite canonical state required")
    length = float(format(protocol.total_length_m / protocol.n_links, ".9g"))
    theta = np.cumsum(state[1:protocol.n_links + 1])
    offsets = length * np.column_stack((np.sin(theta), np.cos(theta)))
    return np.vstack(([state[0], .07], np.cumsum(offsets, axis=0) + [state[0], .07]))


def frame_image(state, action, tick, protocol, *, width=1280, height=720):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), "#101822")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, round(20 * width / 1280)))
    small = ImageFont.load_default(size=max(10, round(16 * width / 1280)))
    scale = min(width / 13., (height - 120) / 6.8)

    def point(x, z):
        return width / 2 + x * scale, height / 2 + 20 - z * scale

    for x in range(-6, 7):
        draw.line((point(x, -3.2), point(x, 3.2)), fill="#1b2b39")
    for z in range(-3, 4):
        draw.line((point(-6.2, z), point(6.2, z)), fill="#1b2b39")
    draw.line((point(-3., -.08), point(3., -.08)), fill="#798c9a", width=4)
    for x in (-3., 3.):
        a, b = point(x, -.24), point(x, .1)
        draw.line((a, b), fill="#ff8585", width=3)
    cartx, carty = point(float(state[0]), 0)
    draw.rounded_rectangle((cartx - .18 * scale, carty - .07 * scale,
                            cartx + .18 * scale, carty + .07 * scale),
                           radius=3, fill="#7bb7ff")
    nodes = chain_positions(state, protocol)
    palette = ("#55e6c3", "#ffbd66", "#fb91b9", "#7bb7ff", "#bf9aff", "#d6ee81")
    for index, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        draw.line((point(*a), point(*b)), fill=palette[index % len(palette)],
                  width=max(3, round(.05 * scale)))
        x, y = point(*a)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#f5f7fa")
    maximum = float(np.max(np.abs(absolute_angles(state, protocol.n_links))))
    draw.text((28, 20), f"{protocol.n_links} passive links | published canonical benchmark",
              fill="#f5f7fa", font=font)
    draw.text((28, 51), "Recorded states | 50 Hz | 3 m / 1 kg chain | 80 N force | 3 m rail",
              fill="#9fb4c4", font=small)
    draw.text((28, height - 38),
              f"t = {(tick + 1) * protocol.control_dt:05.2f} s     force = {action * 80:+06.2f} N"
              f"     max |angle| = {maximum:.4f} rad", fill="#f5f7fa", font=small)
    return image


def render_saved_episode(episode_receipt, out):
    checked = audit_episode(episode_receipt)
    require(checked["score"] is not None and checked["score"]["success"] is True,
            "complete successful recorded episode required")
    protocol = CanonicalProtocol(**checked["document"]["identity"]["protocol"])
    require(len(checked["actions"]) == 1500, "full thirty-second source tape required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    target = out / "episode.mp4"
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "50", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(target)]
    with (out / "encoder.log").open("xb") as log:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=log)
        try:
            for tick, (state, action) in enumerate(zip(checked["states"][1:], checked["actions"])):
                process.stdin.write(frame_image(state, float(action), tick, protocol).tobytes())
            process.stdin.close()
            require(process.wait() == 0, "video encoder failed; retained partial output and log")
        except BaseException:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.wait()
            raise
    # Re-read the original evidence after rendering; no stale pass cache.
    audit_episode(episode_receipt, identity=checked["document"]["identity"])
    receipt = {
        "schema": "rage_mill.canonical_saved_state_video.v1", "episode_receipt": episode_receipt,
        "recording": checked["document"]["recording"],
        "identity_sha256": value_sha256(checked["document"]["identity"]),
        "video": pinned_file(target), "encoder_log": pinned_file(out / "encoder.log"),
        "renderer_source": pinned_file(__file__), "encoder_argv": argv,
        "render_runtime": {"python": platform.python_version(), "numpy": np.__version__,
                           "pillow": importlib.metadata.version("pillow")},
        "encoder_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        "frames": 1500, "fps": 50, "simulated_seconds": 30., "reset_count": 0,
        "dynamics_reexecuted": False, "interpolated_states": False,
        "state_samples": "every actual post-command state, in order, without initial sample",
        "controller_qualified": False, "scope": "saved-state visualization; release verified separately",
    }
    return write_json_exclusive(out / "receipt.json", receipt)
