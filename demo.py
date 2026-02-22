#!/usr/bin/env python3
"""SignBridge Demo - Enter text, see avatar sign it.

Usage:
    python demo.py                    # Interactive mode
    python demo.py "hello world"      # Direct input
    python demo.py --list             # Show available signs
    python demo.py --no-captions      # Disable captions
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.avatar.renderer_smplx import PyRenderSMPLXRenderer, SMPLXRenderSettings, SMPLXSequence
from packages.avatar.motion_blender import MotionBlender, BlendSettings, EasingType
from packages.video.caption_stacker import add_captions_to_video, CaptionConfig


class SignDemo:
    def __init__(self, captions: bool = True):
        self.project_root = PROJECT_ROOT
        self.mapping = self._load_mapping()
        self.pkl_dir = self.project_root / "data" / "signavatars" / "word2motion" / "wlasl_pkls_cropFalse_defult_shape"
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(exist_ok=True)

        # Build available glosses list
        self.available_glosses = self._get_available_glosses()

        # Lazy load renderer
        self._renderer = None

        # Caption settings
        self.captions_enabled = captions

    def _load_mapping(self):
        mapping_path = self.project_root / "data" / "signavatars" / "wlasl_mapping.json"
        if not mapping_path.exists():
            print(f"Error: Mapping file not found: {mapping_path}")
            sys.exit(1)
        with open(mapping_path) as f:
            return json.load(f)

    def _get_available_glosses(self):
        available = []
        for gloss, vid_ids in self.mapping["gloss_to_ids"].items():
            for vid_id in vid_ids:
                if (self.pkl_dir / f"{vid_id}.pkl").exists():
                    available.append(gloss)
                    break
        return sorted(available)

    @property
    def renderer(self):
        if self._renderer is None:
            print("\nInitializing avatar renderer...")
            settings = SMPLXRenderSettings(model_path=str(self.project_root / "data" / "models"))
            self._renderer = PyRenderSMPLXRenderer(settings, use_texture=False)
            print("Ready!\n")
        return self._renderer

    def get_pkl_path(self, gloss: str) -> Path:
        """Get SignAvatars pkl file path for a gloss."""
        gloss = gloss.upper()
        if gloss not in self.mapping["gloss_to_ids"]:
            return None

        for vid_id in self.mapping["gloss_to_ids"][gloss]:
            candidate = self.pkl_dir / f"{vid_id}.pkl"
            if candidate.exists():
                return candidate
        return None

    def text_to_glosses(self, text: str) -> list:
        """Convert input text to available glosses.

        Handles compound phrases like 'A LOT', 'ALL DAY' etc.
        Uses greedy matching - tries longest phrase first.
        Strips punctuation from words.
        """
        import re

        # Clean and normalize text
        text = text.upper()
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()

        glosses = []
        i = 0

        while i < len(words):
            matched = False

            # Try matching longest phrases first (up to 4 words)
            for length in range(min(4, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i+length])

                if phrase in self.available_glosses:
                    glosses.append(phrase)
                    i += length
                    matched = True
                    break

            if not matched:
                # No match found for this word
                word = words[i]
                print(f"  [Skip] '{word}' - no sign available")
                i += 1

        return glosses

    def render_glosses(self, glosses: list, original_text: str = None) -> Path:
        """Render a sequence of glosses to video with natural transitions."""
        if not glosses:
            print("No glosses to render!")
            return None

        # Load all sign sequences
        sequences = []
        rendered = []

        for gloss in glosses:
            pkl_path = self.get_pkl_path(gloss)
            if not pkl_path:
                continue

            print(f"  Loading: {gloss}")
            seq = SMPLXSequence.load_signavatars(pkl_path, gloss=gloss)
            sequences.append(seq)
            rendered.append(gloss)

        if not sequences:
            print("No sequences loaded!")
            return None

        # Blend sequences with natural transitions
        if len(sequences) > 1:
            print(f"\n  Blending {len(sequences)} signs with natural transitions...")
            blender = MotionBlender(BlendSettings(
                transition_frames=12,
                easing_type=EasingType.SINE,
                use_slerp=True,
                use_overlapping=True,
            ))
            blended_seq = blender.blend_sequences(sequences)
        else:
            blended_seq = sequences[0]

        # Render the blended sequence
        print(f"  Rendering {len(blended_seq.frames)} frames...")
        all_frames = self.renderer.render_smplx_sequence(blended_seq)

        if not all_frames:
            print("No frames rendered!")
            return None

        # Generate output filename
        timestamp = datetime.now().strftime("%H%M%S")
        name = "_".join(rendered[:3])
        if len(rendered) > 3:
            name += f"_+{len(rendered)-3}"

        # Export avatar video
        avatar_path = self.output_dir / f"_avatar_{name}_{timestamp}.mp4"
        self.renderer._export_frames(np.array(all_frames), avatar_path, 30.0)

        # Add captions if enabled
        if self.captions_enabled and original_text:
            output_path = self.output_dir / f"demo_{name}_{timestamp}.mp4"
            caption_text = original_text if original_text else " ".join(rendered)

            print(f"\n  Adding captions...")
            add_captions_to_video(
                avatar_video=str(avatar_path),
                caption_text=caption_text,
                output_path=str(output_path),
                config=CaptionConfig(width=720, height=100)
            )

            # Clean up avatar-only video
            avatar_path.unlink(missing_ok=True)
            return output_path
        else:
            # Rename to final output
            output_path = self.output_dir / f"demo_{name}_{timestamp}.mp4"
            avatar_path.rename(output_path)
            return output_path

    def run_interactive(self):
        """Run interactive demo mode."""
        print("=" * 50)
        print("  SignBridge Avatar Demo")
        print("=" * 50)
        print(f"\nAvailable signs: {len(self.available_glosses)}")
        print("NOTE: Current vocabulary is limited to A-words only")
        print("      (e.g., ACT, AGAIN, AFRICA, ABOUT, AGREE...)")
        print()
        print("Try: 'act again' or 'about africa'")
        print("Type 'list' to see all available signs")
        print("Type 'quit' to exit\n")

        while True:
            try:
                text = input("Enter text > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if not text:
                continue

            if text.lower() == 'quit':
                print("Bye!")
                break

            if text.lower() == 'list':
                self.print_available()
                continue

            # Convert text to glosses
            print(f"\nInput: '{text}'")
            glosses = self.text_to_glosses(text)

            if not glosses:
                print("No matching signs found. Try 'list' to see available signs.\n")
                continue

            print(f"Glosses: {' '.join(glosses)}")

            # Render with original text for captions
            output = self.render_glosses(glosses, original_text=text)

            if output:
                print(f"\nOutput: {output}")
                print(f"Size: {output.stat().st_size / 1024:.1f} KB")

                # Auto-open on macOS
                import subprocess
                subprocess.run(["open", str(output)], capture_output=True)

            print()

    def print_available(self):
        """Print available glosses in columns."""
        print(f"\nAvailable signs ({len(self.available_glosses)}):\n")
        cols = 5
        for i in range(0, len(self.available_glosses), cols):
            row = self.available_glosses[i:i+cols]
            print("  " + "  ".join(f"{g:<12}" for g in row))
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SignBridge Avatar Demo")
    parser.add_argument("text", nargs="*", help="Text to sign (or interactive if omitted)")
    parser.add_argument("--list", "-l", action="store_true", help="List available signs")
    parser.add_argument("--no-captions", action="store_true", help="Disable captions")

    args = parser.parse_args()

    demo = SignDemo(captions=not args.no_captions)

    if args.list:
        demo.print_available()
        return

    if args.text:
        # Direct mode
        text = " ".join(args.text)
        print(f"\nInput: '{text}'")
        glosses = demo.text_to_glosses(text)

        if glosses:
            print(f"Glosses: {' '.join(glosses)}")
            output = demo.render_glosses(glosses, original_text=text)
            if output:
                print(f"\nOutput: {output}")
                import subprocess
                subprocess.run(["open", str(output)], capture_output=True)
    else:
        # Interactive mode
        demo.run_interactive()


if __name__ == "__main__":
    main()
