#!/usr/bin/env python3
"""Extract MediaPipe poses from all verified signs."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.avatar.pose_extractor import PoseExtractor, extract_all_signs


def main():
    signs_dir = PROJECT_ROOT / "data" / "signs"

    print("Extracting poses from verified signs...")
    print(f"Signs directory: {signs_dir}")
    print()

    results = extract_all_signs(
        signs_dir=signs_dir,
        subdirs=["verified"],
        overwrite=False,  # Skip existing
    )

    print(f"\nDone! Processed {len(results)} signs")
    print(f"Pose files saved alongside videos (poses.json)")


if __name__ == "__main__":
    main()
