#!/usr/bin/env python3
"""SignAvatars Dataset Integration Guide.

SignAvatars is a large-scale 3D sign language motion dataset with SMPL-X annotations.
This script helps set up and use SignAvatars data with SignBridge.

Dataset Info:
- 70K motion sequences from 153 signers
- 8.34M frames with SMPL-X body model annotations
- Subsets: word2motion (WLASL), language2motion, hamnosys2motion

How to Get the Data:
1. Fill out the form: https://docs.google.com/forms/d/e/1FAIpQLSc6xQJJMf_R4xJ1sIwDL6FBIYw4HbVVv_HUgCqeiguWX5XGPg/viewform
2. Download will include:
   - word2motion/annotations/SMPL-X/*.pkl (WLASL signs)
   - language2motion/annotations/SMPL-X/*.pkl (continuous signing)
3. Place data in data/signavatars/

Usage:
    python scripts/signavatars_setup.py --check     # Check setup
    python scripts/signavatars_setup.py --import    # Import to SignBridge

Citation:
    @inproceedings{yu2024signavatars,
      title={SignAvatars: A large-scale 3D sign language holistic motion dataset and benchmark},
      author={Yu, Zhengdi and Huang, Shaoli and Cheng, Yongkang and Birdal, Tolga},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2024}
    }
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_signavatars_setup():
    """Check if SignAvatars data is available."""
    data_dir = PROJECT_ROOT / "data" / "signavatars"
    word2motion = data_dir / "word2motion" / "annotations" / "SMPL-X"

    print("SignAvatars Setup Check")
    print("=" * 50)

    if not data_dir.exists():
        print(f"[!] Data directory not found: {data_dir}")
        print("\nTo get SignAvatars data:")
        print("1. Fill out: https://docs.google.com/forms/d/e/1FAIpQLSc6xQJJMf_R4xJ1sIwDL6FBIYw4HbVVv_HUgCqeiguWX5XGPg/viewform")
        print("2. Extract to: data/signavatars/")
        return False

    print(f"[+] Data directory found: {data_dir}")

    if word2motion.exists():
        pkl_files = list(word2motion.glob("*.pkl"))
        print(f"[+] word2motion: {len(pkl_files)} SMPL-X files")
    else:
        print(f"[-] word2motion not found")

    # Check SMPL-X model
    model_path = PROJECT_ROOT / "data" / "models" / "smplx"
    if model_path.exists():
        print(f"[+] SMPL-X model available: {model_path}")
    else:
        print(f"[-] SMPL-X model not found: {model_path}")

    return True


def import_signavatars_to_signbridge():
    """Import SignAvatars word2motion data to SignBridge format."""
    from packages.avatar.renderer_smplx import SMPLXSequence

    data_dir = PROJECT_ROOT / "data" / "signavatars"
    word2motion = data_dir / "word2motion" / "annotations" / "SMPL-X"
    output_dir = PROJECT_ROOT / "data" / "signs" / "signavatars"

    if not word2motion.exists():
        print("Error: SignAvatars word2motion data not found")
        print(f"Expected at: {word2motion}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load WLASL mapping
    wlasl_json = data_dir / "word2motion" / "text" / "WLASL_v0.3.json"
    if wlasl_json.exists():
        import json
        with open(wlasl_json) as f:
            wlasl_data = json.load(f)
        print(f"Loaded WLASL mapping with {len(wlasl_data)} entries")

    # Process each .pkl file
    pkl_files = list(word2motion.glob("*.pkl"))
    print(f"Found {len(pkl_files)} SMPL-X motion files")

    for i, pkl_file in enumerate(pkl_files[:10]):  # Process first 10 as demo
        try:
            seq = SMPLXSequence.load_signavatars(pkl_file)
            print(f"  [{i+1}] {seq.gloss}: {len(seq.frames)} frames")

            # Could save in SignBridge format here
            sign_dir = output_dir / seq.gloss.upper()
            sign_dir.mkdir(exist_ok=True)
            # Save converted data...

        except Exception as e:
            print(f"  Error loading {pkl_file.name}: {e}")

    print(f"\nImported to: {output_dir}")


def demo_render_signavatars():
    """Demo rendering a SignAvatars motion sequence."""
    from packages.avatar.renderer_smplx import (
        AvatarSMPLXRenderer,
        SMPLXRenderSettings,
        SMPLXSequence,
    )

    data_dir = PROJECT_ROOT / "data" / "signavatars"
    word2motion = data_dir / "word2motion" / "annotations" / "SMPL-X"

    if not word2motion.exists():
        print("SignAvatars data not found. Using MediaPipe-converted poses instead.")
        return

    # Find first .pkl file
    pkl_files = list(word2motion.glob("*.pkl"))
    if not pkl_files:
        print("No .pkl files found")
        return

    pkl_file = pkl_files[0]
    print(f"Loading: {pkl_file}")

    # Load sequence
    seq = SMPLXSequence.load_signavatars(pkl_file)
    print(f"Loaded {len(seq.frames)} frames for '{seq.gloss}'")

    # Render
    settings = SMPLXRenderSettings(model_path=str(PROJECT_ROOT / "data" / "models"))
    renderer = AvatarSMPLXRenderer(settings)

    output_path = PROJECT_ROOT / "output" / f"{seq.gloss}_signavatars.mp4"
    output_path.parent.mkdir(exist_ok=True)

    # Render first 60 frames as demo
    seq.frames = seq.frames[:60]
    frames = renderer.render_smplx_sequence(seq)

    print(f"Rendered {len(frames)} frames")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignAvatars setup and integration")
    parser.add_argument("--check", action="store_true", help="Check setup status")
    parser.add_argument("--import-data", action="store_true", help="Import SignAvatars data")
    parser.add_argument("--demo", action="store_true", help="Demo render")

    args = parser.parse_args()

    if args.check or (not args.import_data and not args.demo):
        check_signavatars_setup()
    elif args.import_data:
        import_signavatars_to_signbridge()
    elif args.demo:
        demo_render_signavatars()
