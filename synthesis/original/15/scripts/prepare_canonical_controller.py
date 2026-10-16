"""Package a tested diagnostic controller without rebuilding any feedback array."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rage_mill.canonical_benchmark import CanonicalEnv
from rage_mill.canonical_campaign import file_sha256, write_json
from rage_mill.canonical_control import CanonicalPolicy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prototype", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bounded-planning", action="store_true")
    args = parser.parse_args()
    receipt = json.loads((args.prototype/"receipt.json").read_text())
    if not (receipt["native_bias"] and receipt["native_value"] and receipt["lookahead"] == 1
            and receipt["optimizer"] == "bounded" and receipt["fixed_prefix_ticks"] == 0
            and receipt["balance_after"] == 1500):
        raise ValueError("prototype has unsupported runtime policy options")
    with np.load(args.prototype/"controller.npz", allow_pickle=False) as saved:
        keys = ("states", "actions", "gains", "offsets", "roots", "residuals", "balance_gain", "park_gain")
        arrays = {key: saved[key].copy() for key in keys}
        park_ticks = int(saved["park_ticks"])
    with np.load(args.design, allow_pickle=False) as saved:
        arrays.update(terminal_root=saved["terminal_root"].copy(), Q=saved["Q"].copy())
        if not np.array_equal(arrays["balance_gain"], saved["gain"]):
            raise ValueError("prototype uses a different terminal design")
    parameters = dict(park_ticks=park_ticks, cart_target=receipt["cart_target"],
                      native_value_after=180, control_cost=receipt["tracking_cost"],
                      joint_horizon=receipt["joint_lookahead"], joint_after=390, optimizer_xtol=2e-8)
    CanonicalPolicy(arrays, **parameters)
    if args.bounded_planning:
        parameters = dict(controller=parameters, forecast_profiles=[
            dict(joint_horizon=horizon, optimizer_xtol=tolerance)
            for tolerance in (2e-8, 1e-9, 1e-7, 1e-10, 5e-8, 5e-9, 5e-7, 1e-6)
            for horizon in (1, 2, 3, 4)])
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out/"controller.npz", **arrays)
    write_json(args.out/"policy.json", parameters)
    (args.out/"model.xml").write_text(CanonicalEnv().xml)
    write_json(args.out/"provenance.json", dict(
        schema="rage_mill.canonical15_controller.v1", controller_qualified=False,
        method=("bounded native route planning from the observed hanging-settle state"
                if args.bounded_planning else "native value control and joint native capture forecasts"),
        prototype=str(args.prototype.resolve()),
        prototype_receipt_sha256=file_sha256(args.prototype/"receipt.json"),
        prototype_controller_sha256=file_sha256(args.prototype/"controller.npz"),
        design=str(args.design.resolve()), design_sha256=file_sha256(args.design),
        reference_sha256=receipt["reference_sha256"],
        model_receipt_sha256=receipt["model_receipt_sha256"],
        offline_riccati_precision=receipt["riccati_precision"],
        parking_decay=receipt["parking_decay"], reserved_episodes=0))
    identity = {name: file_sha256(args.out/name) for name in
                ("controller.npz", "policy.json", "model.xml", "provenance.json")}
    write_json(args.out/"identity.json", identity)
    print(json.dumps(identity, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
