#!/usr/bin/env python3
"""Validate a Marvin demo before starting the final training run."""

import argparse
import json
import os
import pickle

import numpy as np


EXPECTED_CONTROL = {
    "impedance_mode": "cartesian",
    "action_scale": [10.0, 0.03, 1.0],
    "position_action_limit": 1.0,
    "policy_hz": 10,
    "stream_control_hz": 200,
    "stream_linear_velocity_mm_s": 60.0,
    "stream_linear_acceleration_mm_s2": 300.0,
    "stream_yaw_velocity_deg_s": 20.0,
    "stream_yaw_acceleration_deg_s2": 120.0,
}
EXPECTED_IMAGES = {
    "side_classifier",
    "side_policy",
    "wrist_1",
    "wrist_1buchong",
    "wrist_2",
}


def _equal(actual, expected):
    if isinstance(expected, (int, float, list)):
        return np.allclose(actual, expected, rtol=0.0, atol=1e-8)
    return actual == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_path")
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="Validate transition shapes but allow a legacy pickle without sidecar metadata.",
    )
    args = parser.parse_args()

    metadata_path = args.demo_path + ".metadata.json"
    errors = []
    if not os.path.isfile(metadata_path):
        if not args.allow_missing_metadata:
            errors.append(f"missing final-version metadata: {metadata_path}")
        metadata = None
    else:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        control = metadata.get("control", {})
        for key, expected in EXPECTED_CONTROL.items():
            actual = control.get(key)
            if actual is None or not _equal(actual, expected):
                errors.append(f"metadata control.{key}: expected {expected}, got {actual}")

    with open(args.demo_path, "rb") as f:
        transitions = pickle.load(f)
    if not isinstance(transitions, list) or not transitions:
        errors.append("pickle must contain a non-empty transition list")
    else:
        required = {
            "observations", "actions", "next_observations",
            "rewards", "masks", "dones",
        }
        if not required.issubset(transitions[0]):
            errors.append(f"transition keys missing: {sorted(required - transitions[0].keys())}")

        actions = np.asarray([transition["actions"] for transition in transitions])
        if actions.ndim != 2 or actions.shape[1] != 5:
            errors.append(f"expected actions shape (N, 5), got {actions.shape}")
        elif not np.isfinite(actions).all():
            errors.append("actions contain NaN or Inf")

        obs = transitions[0]["observations"]
        next_obs = transitions[0]["next_observations"]
        for label, value in (("state", obs.get("state")), ("next state", next_obs.get("state"))):
            shape = np.asarray(value).shape
            if shape != (1, 24):
                errors.append(f"expected {label} shape (1, 24), got {shape}")
        image_keys = set(obs) - {"state"}
        if image_keys != EXPECTED_IMAGES:
            errors.append(f"expected image keys {sorted(EXPECTED_IMAGES)}, got {sorted(image_keys)}")
        for key in image_keys & EXPECTED_IMAGES:
            shape = np.asarray(obs[key]).shape
            if shape != (1, 128, 128, 3):
                errors.append(f"expected {key} shape (1, 128, 128, 3), got {shape}")

        episode_count = sum(bool(transition["dones"]) for transition in transitions)
        if episode_count == 0 or not bool(transitions[-1]["dones"]):
            errors.append("demo must contain complete, terminal episodes")

    if errors:
        print("Demo validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(
        f"Demo validation OK: {args.demo_path} | "
        f"transitions={len(transitions)}, episodes={episode_count}, metadata={'yes' if metadata else 'no'}"
    )


if __name__ == "__main__":
    main()
