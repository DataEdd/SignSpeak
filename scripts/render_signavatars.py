#!/usr/bin/env python3
"""Render SignAvatars SMPL-X motions to video.

Usage:
    python scripts/render_signavatars.py HELLO
    python scripts/render_signavatars.py "HELLO AGAIN" --output output/
    python scripts/render_signavatars.py --list
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.avatar.renderer_smplx import AvatarSMPLXRenderer, SMPLXRenderSettings, SMPLXSequence


def load_mapping():
    """Load WLASL video ID to gloss mapping."""
    mapping_path = PROJECT_ROOT / "data" / "signavatars" / "wlasl_mapping.json"
    if not mapping_path.exists():
        print(f"Error: Mapping file not found: {mapping_path}")
        print("Run the integration setup first.")
        sys.exit(1)

    with open(mapping_path) as f:
        return json.load(f)


def get_pkl_path(gloss: str, mapping: dict) -> Path:
    """Get SignAvatars pkl file path for a gloss."""
    gloss = gloss.upper()
    gloss_to_ids = mapping["gloss_to_ids"]

    if gloss not in gloss_to_ids:
        return None

    pkl_dir = PROJECT_ROOT / "data" / "signavatars" / "word2motion" / "wlasl_pkls_cropFalse_defult_shape"

    for vid_id in gloss_to_ids[gloss]:
        candidate = pkl_dir / f"{vid_id}.pkl"
        if candidate.exists():
            return candidate

    return None


def render_gloss(gloss: str, renderer: AvatarSMPLXRenderer, mapping: dict, output_dir: Path) -> Path:
    """Render a single gloss to video."""
    pkl_path = get_pkl_path(gloss, mapping)
    if not pkl_path:
        print(f"  [SKIP] {gloss}: No SignAvatars data")
        return None

    print(f"  Loading {gloss}...")
    seq = SMPLXSequence.load_signavatars(pkl_path, gloss=gloss.upper())

    print(f"  Rendering {len(seq.frames)} frames...")
    frames = renderer.render_smplx_sequence(seq)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{gloss.upper()}_{timestamp}.mp4"

    result = renderer._export_frames(np.array(frames), output_path, 30.0)
    print(f"  Saved: {result} ({result.stat().st_size/1024:.1f} KB)")

    return result


def render_sequence(glosses: list, output_dir: Path) -> Path:
    """Render multiple glosses concatenated."""
    mapping = load_mapping()
    settings = SMPLXRenderSettings(model_path=str(PROJECT_ROOT / "data" / "models"))
    renderer = AvatarSMPLXRenderer(settings)

    all_frames = []
    rendered_glosses = []

    for gloss in glosses:
        pkl_path = get_pkl_path(gloss, mapping)
        if not pkl_path:
            print(f"  [SKIP] {gloss}: No SignAvatars data")
            continue

        print(f"  Rendering {gloss}...")
        seq = SMPLXSequence.load_signavatars(pkl_path, gloss=gloss.upper())
        frames = renderer.render_smplx_sequence(seq)
        all_frames.extend(frames)
        rendered_glosses.append(gloss.upper())

        # Add transition frames
        if len(glosses) > 1:
            bg = np.full((settings.height, settings.width, 3),
                        (26, 26, 46), dtype=np.uint8)  # background color
            for _ in range(5):
                all_frames.append(bg)

    if not all_frames:
        print("No frames to render")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_".join(rendered_glosses[:3])
    if len(rendered_glosses) > 3:
        name += f"_plus{len(rendered_glosses)-3}"
    output_path = output_dir / f"{name}_{timestamp}.mp4"

    result = renderer._export_frames(np.array(all_frames), output_path, 30.0)
    print(f"\nSaved: {result} ({result.stat().st_size/1024:.1f} KB)")

    return result


def list_available():
    """List available SignAvatars glosses."""
    mapping = load_mapping()
    pkl_dir = PROJECT_ROOT / "data" / "signavatars" / "word2motion" / "wlasl_pkls_cropFalse_defult_shape"

    available = []
    for gloss, vid_ids in mapping["gloss_to_ids"].items():
        for vid_id in vid_ids:
            if (pkl_dir / f"{vid_id}.pkl").exists():
                available.append(gloss)
                break

    available.sort()
    print(f"Available SignAvatars glosses ({len(available)}):\n")

    # Print in columns
    cols = 5
    for i in range(0, len(available), cols):
        row = available[i:i+cols]
        print("  ".join(f"{g:<15}" for g in row))


def main():
    parser = argparse.ArgumentParser(description="Render SignAvatars SMPL-X motions")
    parser.add_argument("glosses", nargs="*", help="Gloss(es) to render (space-separated)")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--list", "-l", action="store_true", help="List available glosses")

    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if not args.glosses:
        parser.print_help()
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # Parse glosses (handle "HELLO WORLD" as two glosses)
    glosses = []
    for g in args.glosses:
        glosses.extend(g.upper().split())

    print(f"Rendering: {' '.join(glosses)}")
    print(f"Output: {args.output}/\n")

    if len(glosses) == 1:
        mapping = load_mapping()
        settings = SMPLXRenderSettings(model_path=str(PROJECT_ROOT / "data" / "models"))
        renderer = AvatarSMPLXRenderer(settings)
        render_gloss(glosses[0], renderer, mapping, args.output)
    else:
        render_sequence(glosses, args.output)


if __name__ == "__main__":
    main()
