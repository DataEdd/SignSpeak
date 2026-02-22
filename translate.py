#!/usr/bin/env python3
"""SignBridge Unified Translation - English to ASL video.

Combines:
1. How2Sign sentence database (30,596 full sentences)
2. WLASL word database (124 individual signs)
3. ASL Grammar Rules (time-first, article removal, WH-questions, etc.)

Strategy:
- First tries to find matching sentence in How2Sign
- Falls back to grammar-aware word-by-word signing with WLASL

Usage:
    python translate.py "Hello, how are you?"
    python translate.py "Thank you very much"
    python translate.py --interactive
"""

import sys
import csv
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
import json
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.avatar.renderer_smplx import PyRenderSMPLXRenderer, SMPLXRenderSettings, SMPLXSequence
from packages.avatar.motion_blender import MotionBlender, BlendSettings, EasingType
from packages.video.caption_stacker import add_captions_to_video, CaptionConfig

# Import the grammar/translation system
from packages.translation.grammar_rules import apply_all_rules, detect_non_manual_markers, NonManualMarker
from packages.translation.gloss_converter import GlossConverter, GlossSequence, TranslationQuality
from packages.translation.validator import GlossValidator, ValidationMode


class WLASLSignLookup:
    """Adapter to make WLASL word dictionary work with GlossValidator."""

    def __init__(self, words: dict):
        """
        Args:
            words: Dictionary mapping gloss -> pkl_path
        """
        self.words = words
        self._glosses = set(words.keys())

    def get_sign(self, gloss: str) -> Optional[Path]:
        """Get a sign by gloss."""
        return self.words.get(gloss.upper())

    def get_verified_sign(self, gloss: str) -> Optional[Path]:
        """Get a verified sign (same as get_sign for WLASL)."""
        return self.get_sign(gloss)

    def has_gloss(self, gloss: str) -> bool:
        """Check if gloss exists."""
        return gloss.upper() in self._glosses


class ASLTranslator:
    """Unified English to ASL translator with grammar rules."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(exist_ok=True)

        # Load databases
        print("Loading ASL databases...")
        self.sentences = self._load_sentences()
        self.words = self._load_words()
        print(f"  {len(self.sentences)} sentences (How2Sign)")
        print(f"  {len(self.words)} words (WLASL)")

        # Create sign lookup and grammar converter
        self._sign_lookup = WLASLSignLookup(self.words)
        self._validator = GlossValidator(self._sign_lookup)
        self._converter = GlossConverter(
            validator=self._validator,
            validation_mode=ValidationMode.PERMISSIVE,
            allow_fingerspelling=False,  # Skip unknown words for now
        )

        # Lazy load renderer
        self._renderer = None

    def _load_sentences(self) -> list:
        """Load How2Sign sentence database."""
        csv_path = self.project_root / "data" / "signavatars" / "language2motion" / "how2sign_train.csv"
        pkl_dir = self.project_root / "data" / "signavatars" / "language2motion" / "how2sign_pkls_cropTrue_shapeTrue"

        sentences = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    pkl_name = row['SENTENCE_NAME'] + '.pkl'
                    pkl_path = pkl_dir / pkl_name
                    if pkl_path.exists():
                        sentences.append({
                            'text': row['SENTENCE'],
                            'text_lower': row['SENTENCE'].lower(),
                            'pkl': pkl_path,
                        })
        except FileNotFoundError:
            pass
        return sentences

    def _load_words(self) -> dict:
        """Load WLASL word database."""
        mapping_path = self.project_root / "data" / "signavatars" / "wlasl_mapping.json"
        pkl_dir = self.project_root / "data" / "signavatars" / "word2motion" / "wlasl_pkls_cropFalse_defult_shape"

        words = {}
        try:
            with open(mapping_path) as f:
                mapping = json.load(f)

            for gloss, vid_ids in mapping["gloss_to_ids"].items():
                for vid_id in vid_ids:
                    pkl_path = pkl_dir / f"{vid_id}.pkl"
                    if pkl_path.exists():
                        words[gloss] = pkl_path
                        break
        except FileNotFoundError:
            pass
        return words

    @property
    def renderer(self):
        if self._renderer is None:
            print("\nInitializing avatar renderer...")
            settings = SMPLXRenderSettings(model_path=str(self.project_root / "data" / "models"))
            self._renderer = PyRenderSMPLXRenderer(settings, use_texture=False)
            print("Ready!\n")
        return self._renderer

    def find_sentence_match(self, text: str, threshold: float = 0.85) -> dict:
        """Find matching sentence in How2Sign database.

        Args:
            text: Input English text
            threshold: Minimum similarity ratio (0-1) for a match

        Returns:
            Best matching sentence dict or None
        """
        text_lower = text.lower().strip()
        text_clean = re.sub(r'[^\w\s]', '', text_lower)

        best_match = None
        best_ratio = 0

        for sent in self.sentences:
            # Exact match
            if sent['text_lower'].strip() == text_lower:
                return sent

            # Clean match (ignoring punctuation)
            sent_clean = re.sub(r'[^\w\s]', '', sent['text_lower'])
            if sent_clean == text_clean:
                return sent

            # Fuzzy match
            ratio = SequenceMatcher(None, text_clean, sent_clean).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = sent

        if best_ratio >= threshold:
            return best_match
        return None

    def translate_with_grammar(self, text: str) -> GlossSequence:
        """Apply ASL grammar rules to translate English to gloss sequence.

        This uses the full translation pipeline:
        1. Tokenize and classify words
        2. Remove articles (a, an, the)
        3. Remove be-verbs (is, are, was, were)
        4. Remove auxiliaries (do, does, did, will, etc.)
        5. Simplify verbs to base form (went -> GO, running -> RUN)
        6. Apply Time-Topic-Comment ordering (time markers first)
        7. Move WH-questions to end (WHAT, WHERE, etc.)
        8. Move negation to end (NOT)
        9. Detect non-manual markers (questions, negation)

        Args:
            text: English sentence

        Returns:
            GlossSequence with glosses, non-manual markers, and metadata
        """
        return self._converter.translate(text)

    def text_to_words_simple(self, text: str) -> list:
        """Simple word extraction (fallback, no grammar).

        DEPRECATED: Use translate_with_grammar() instead.
        """
        text = text.upper()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()

        available = []
        for word in words:
            if word in self.words:
                available.append(word)

        return available

    def translate(self, text: str, max_frames: int = 300, slow: bool = True) -> Path:
        """Translate English text to ASL video.

        Strategy:
        1. Try to find matching sentence in How2Sign
        2. Fall back to grammar-aware translation with WLASL

        Args:
            text: English text to translate
            max_frames: Maximum frames (for demo speed)
            slow: Use slower playback for clarity

        Returns:
            Path to output video
        """
        print(f"\n{'='*60}")
        print(f"Translating: '{text}'")
        print(f"{'='*60}")

        # Try sentence match first
        sentence = self.find_sentence_match(text)
        non_manual_markers = []

        if sentence:
            print(f"\n[SENTENCE MATCH] Found in How2Sign:")
            print(f"  '{sentence['text']}'")
            seq = SMPLXSequence.load_signavatars(
                sentence['pkl'],
                gloss=text[:30],
                neutral_shape=True
            )
            method = "sentence"
            # Still detect NM markers for display
            non_manual_markers = detect_non_manual_markers(text, [])
        else:
            # Use grammar-aware translation
            print(f"\n[GRAMMAR TRANSLATION] Applying ASL grammar rules...")

            # Get the raw glosses first (before validation) to show transformation
            raw_glosses = apply_all_rules(text)
            print(f"  English: '{text}'")
            print(f"  Grammar rules applied:")
            print(f"    → Glosses: {' '.join(raw_glosses)}")

            # Full translation with validation
            translation = self.translate_with_grammar(text)
            non_manual_markers = translation.non_manual_markers

            # Show non-manual markers
            if non_manual_markers:
                print(f"  Non-manual markers detected:")
                for nm in non_manual_markers:
                    print(f"    → {nm.marker_type}: {nm.description}")

            # Show quality info
            print(f"  Translation quality: {translation.quality.value}")
            if translation.validation:
                print(f"  Coverage: {translation.validation.coverage:.0%}")
                if translation.validation.missing_glosses:
                    print(f"  Missing signs: {', '.join(translation.validation.missing_glosses)}")

            # Filter to only glosses we have signs for
            available_glosses = [g for g in translation.glosses if self._sign_lookup.has_gloss(g)]

            if not available_glosses:
                # Fallback to simple word matching
                print(f"\n  No grammar-translated signs available, trying simple match...")
                available_glosses = self.text_to_words_simple(text)

            if not available_glosses:
                print("  No matching signs found!")
                print(f"  Available words: {', '.join(sorted(self.words.keys())[:10])}...")
                return None

            print(f"  Final glosses: {' '.join(available_glosses)}")

            # Load word sequences
            sequences = []
            for gloss in available_glosses:
                pkl_path = self.words[gloss]
                word_seq = SMPLXSequence.load_signavatars(pkl_path, gloss=gloss, neutral_shape=True)
                sequences.append(word_seq)

            # Blend with natural transitions
            if len(sequences) > 1:
                blender = MotionBlender(BlendSettings(
                    transition_frames=12,
                    easing_type=EasingType.SINE,
                    use_slerp=True,
                    use_overlapping=True,
                ))
                seq = blender.blend_sequences(sequences)
            else:
                seq = sequences[0]

            method = "grammar"

        # Limit frames for demo
        if len(seq.frames) > max_frames:
            print(f"  Trimming {len(seq.frames)} frames to {max_frames}")
            seq.frames = seq.frames[:max_frames]

        print(f"\n  Rendering {len(seq.frames)} frames...")
        frames = self.renderer.render_smplx_sequence(seq)

        # Generate output
        timestamp = datetime.now().strftime("%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in text[:25])
        avatar_path = self.output_dir / f"_translate_{timestamp}.mp4"

        # Slow down for clarity (20 fps instead of 30)
        fps = 20.0 if slow else 30.0
        self.renderer._export_frames(np.array(frames), avatar_path, fps)

        # Add captions
        output_path = self.output_dir / f"translate_{safe_name}_{timestamp}.mp4"
        print(f"  Adding captions...")
        add_captions_to_video(
            avatar_video=str(avatar_path),
            caption_text=text,
            output_path=str(output_path),
            config=CaptionConfig(width=720, height=100)
        )
        avatar_path.unlink(missing_ok=True)

        print(f"\n{'='*60}")
        print(f"Output: {output_path}")
        print(f"Method: {method}")
        if non_manual_markers:
            nm_types = [nm.marker_type for nm in non_manual_markers]
            print(f"Non-manual: {', '.join(nm_types)}")
            print(f"  (Facial expressions not yet rendered - V3 feature)")
        print(f"{'='*60}")

        return output_path

    def show_grammar_demo(self, text: str):
        """Demo the grammar transformation without rendering.

        Useful for understanding how English is converted to ASL gloss order.
        """
        print(f"\n{'='*60}")
        print(f"Grammar Demo: '{text}'")
        print(f"{'='*60}")

        # Show raw transformation
        raw_glosses = apply_all_rules(text)
        print(f"\nStep-by-step transformation:")
        print(f"  Input:    '{text}'")
        print(f"  Glosses:  {' '.join(raw_glosses)}")

        # Full translation
        translation = self.translate_with_grammar(text)

        print(f"\nNon-manual markers:")
        if translation.non_manual_markers:
            for nm in translation.non_manual_markers:
                print(f"  [{nm.marker_type}] {nm.description}")
                print(f"     Applies to glosses {nm.start_gloss_index} to {nm.end_gloss_index}")
        else:
            print(f"  (none detected)")

        print(f"\nValidation:")
        print(f"  Quality:  {translation.quality.value}")
        print(f"  Coverage: {translation.validation.coverage:.0%}")
        print(f"  Valid:    {', '.join(translation.validation.valid_glosses) or '(none)'}")
        print(f"  Missing:  {', '.join(translation.validation.missing_glosses) or '(none)'}")

        # Show example grammar rules
        print(f"\nGrammar rules applied:")
        print(f"  ✓ Articles removed (a, an, the)")
        print(f"  ✓ Be-verbs removed (is, are, was, were)")
        print(f"  ✓ Auxiliaries removed (do, does, did, will)")
        print(f"  ✓ Verbs simplified to base form")
        print(f"  ✓ Time markers moved to front")
        if any(nm.marker_type == 'wh_question' for nm in translation.non_manual_markers):
            print(f"  ✓ WH-word moved to end")
        if any(nm.marker_type == 'negation' for nm in translation.non_manual_markers):
            print(f"  ✓ Negation moved to end")

        print()

    def list_sentences(self, query: str = None, limit: int = 20):
        """List available sentences."""
        if query:
            results = [s for s in self.sentences if query.lower() in s['text_lower']][:limit]
            print(f"\nSentences containing '{query}' ({len(results)}):")
        else:
            results = self.sentences[:limit]
            print(f"\nSample sentences ({limit} of {len(self.sentences)}):")

        for i, s in enumerate(results, 1):
            text = s['text'][:70] + "..." if len(s['text']) > 70 else s['text']
            print(f"  {i}. {text}")

    def list_words(self):
        """List available words."""
        words = sorted(self.words.keys())
        print(f"\nAvailable words ({len(words)}):")
        cols = 6
        for i in range(0, len(words), cols):
            row = words[i:i+cols]
            print("  " + "  ".join(f"{w:<12}" for w in row))

    def run_interactive(self):
        """Interactive translation mode."""
        print("\n" + "=" * 60)
        print("  SignBridge ASL Translator (V2 with Grammar)")
        print("=" * 60)
        print(f"\nDatabases loaded:")
        print(f"  - {len(self.sentences)} sentences (How2Sign)")
        print(f"  - {len(self.words)} words (WLASL)")
        print(f"\nGrammar rules enabled:")
        print(f"  - Time-first ordering")
        print(f"  - Article/be-verb removal")
        print(f"  - WH-question reordering")
        print(f"  - Negation placement")
        print(f"\nCommands:")
        print("  <text>              - Translate to ASL")
        print("  grammar <text>      - Show grammar transformation only")
        print("  sentences <query>   - List matching sentences")
        print("  words               - List available words")
        print("  quit                - Exit")
        print()

        while True:
            try:
                cmd = input("Translate > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if not cmd:
                continue

            if cmd.lower() == 'quit':
                print("Bye!")
                break

            if cmd.lower() == 'words':
                self.list_words()
                continue

            if cmd.lower().startswith('sentences'):
                query = cmd[9:].strip() if len(cmd) > 9 else None
                self.list_sentences(query)
                continue

            if cmd.lower().startswith('grammar '):
                text = cmd[8:].strip()
                if text:
                    self.show_grammar_demo(text)
                continue

            # Translate
            output = self.translate(cmd)
            if output:
                import subprocess
                subprocess.run(["open", str(output)], capture_output=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SignBridge ASL Translator")
    parser.add_argument("text", nargs="*", help="Text to translate")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--sentences", "-s", type=str, help="List sentences containing query")
    parser.add_argument("--words", "-w", action="store_true", help="List available words")
    parser.add_argument("--grammar", "-g", action="store_true", help="Show grammar transformation only")

    args = parser.parse_args()

    translator = ASLTranslator()

    if args.sentences:
        translator.list_sentences(args.sentences)
        return

    if args.words:
        translator.list_words()
        return

    if args.grammar and args.text:
        text = " ".join(args.text)
        translator.show_grammar_demo(text)
        return

    if args.interactive or not args.text:
        translator.run_interactive()
        return

    # Direct translation
    text = " ".join(args.text)
    output = translator.translate(text)
    if output:
        import subprocess
        subprocess.run(["open", str(output)], capture_output=True)


if __name__ == "__main__":
    main()
