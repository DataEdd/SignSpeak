#!/usr/bin/env python3
"""Download required data for SignSpeak.

Sets up the WLASL sign video dataset and SMPL body model files
needed to run the translation and rendering pipeline.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --wlasl-only
    python scripts/download_data.py --smpl-only
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def setup_directories():
    """Create the expected data directory structure."""
    dirs = [
        "data/signs/verified",
        "data/signs/pending",
        "data/signs/imported",
        "data/cache",
        "data/exports",
        "data/models/smplx",
        "output",
    ]
    for d in dirs:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created {d}/")


def download_wlasl():
    """Download WLASL (Word-Level American Sign Language) dataset.

    WLASL is a large-scale video dataset for Word-Level American Sign Language
    Recognition, containing 2,000 glosses performed by over 100 signers.

    Reference: Li et al., "Word-level Deep Sign Language Recognition from
    Video: A New Large-scale Dataset and Methods Comparison", WACV 2020
    """
    print("\n=== WLASL Dataset ===")
    print("The WLASL dataset provides word-level ASL video clips.")
    print()
    print("To download WLASL data:")
    print("  1. Visit: https://dxli94.github.io/WLASL/")
    print("  2. Download the WLASL2000 video files")
    print("  3. Place videos in: data/signs/verified/<GLOSS>/video.mp4")
    print()
    print("Each sign should have this structure:")
    print("  data/signs/verified/HELLO/")
    print("    sign.json   (metadata - included in repo)")
    print("    video.mp4   (sign video clip)")
    print()

    # Verify mapping file exists
    mapping_path = PROJECT_ROOT / "data" / "wlasl_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
        n_glosses = len(mapping.get("gloss_to_ids", {}))
        print(f"  Gloss mapping file found: {n_glosses} glosses available")
    else:
        print("  Warning: wlasl_mapping.json not found in data/")


def download_smpl():
    """Download SMPL-X body model for avatar rendering.

    SMPL-X is an expressive body model that jointly models the body,
    face, and hands. Required for the avatar rendering pipeline.

    Reference: Pavlakos et al., "Expressive Body Capture: 3D Hands,
    Face, and Body from a Single Image", CVPR 2019
    """
    print("\n=== SMPL-X Body Model ===")
    print("SMPL-X models are needed for 3D avatar rendering.")
    print()
    print("To download SMPL-X:")
    print("  1. Register at: https://smpl-x.is.tue.mpg.de/")
    print("  2. Download the SMPL-X model files (MALE, FEMALE, NEUTRAL)")
    print("  3. Place .npz files in: data/models/smplx/")
    print()
    print("Expected files:")
    print("  data/models/smplx/SMPLX_MALE.npz")
    print("  data/models/smplx/SMPLX_FEMALE.npz")
    print("  data/models/smplx/SMPLX_NEUTRAL.npz")
    print()

    # Check if models already exist
    model_dir = PROJECT_ROOT / "data" / "models" / "smplx"
    for gender in ["MALE", "FEMALE", "NEUTRAL"]:
        path = model_dir / f"SMPLX_{gender}.npz"
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  Found: SMPLX_{gender}.npz ({size_mb:.1f} MB)")
        else:
            print(f"  Missing: SMPLX_{gender}.npz")


def main():
    parser = argparse.ArgumentParser(description="Download SignSpeak data dependencies")
    parser.add_argument("--wlasl-only", action="store_true", help="Only show WLASL instructions")
    parser.add_argument("--smpl-only", action="store_true", help="Only show SMPL instructions")
    args = parser.parse_args()

    print("SignSpeak Data Setup")
    print("=" * 50)
    print()
    print("Setting up directory structure...")
    setup_directories()

    if args.wlasl_only:
        download_wlasl()
    elif args.smpl_only:
        download_smpl()
    else:
        download_wlasl()
        download_smpl()

    print()
    print("=" * 50)
    print("Setup complete. Follow the instructions above to download")
    print("the required datasets for full functionality.")
    print()
    print("Note: The translation API works without video data by returning")
    print("gloss sequences. Video generation requires the sign video clips.")


if __name__ == "__main__":
    main()
