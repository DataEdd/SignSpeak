#!/usr/bin/env python3
"""Import MVP signs from WLASL dataset directly to verified/."""

import json
import shutil
from pathlib import Path
from datetime import date

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
WLASL_DIR = PROJECT_ROOT / "data" / "downloads"
VERIFIED_DIR = PROJECT_ROOT / "data" / "signs" / "verified"

# MVP words to import
MVP_WORDS = [
    "book", "drink", "computer", "go", "help", "yes", "no",
    "please", "want", "like", "good", "bad", "happy", "sad",
    "you", "we", "they", "who", "what", "where", "when", "why", "how",
    "see", "know", "think", "understand", "learn", "sorry", "friend",
    "family", "mother", "father", "work", "school", "home", "eat", "water",
    "fine", "deaf", "name", "nice", "meet", "again", "love", "need",
    "hello", "before", "chair", "clothes", "candy", "cousin", "thin",
    "walk", "year", "all", "black", "cool", "white", "woman", "man",
    # Additional essential signs
    "i", "your", "my", "sign", "language", "me", "here", "there",
    "time", "today", "now", "day", "night", "morning", "afternoon",
    "can", "have", "give", "get", "come", "stay", "live", "feel",
    "ask", "tell", "say", "call", "wait", "stop", "start", "finish",
    "different", "same", "new", "old", "big", "small", "more", "many"
]


def load_wlasl_metadata():
    """Load WLASL metadata."""
    meta_path = WLASL_DIR / "WLASL_v0.3.json"
    with open(meta_path) as f:
        return json.load(f)


def find_video(video_id: str) -> Path | None:
    """Find video file by ID."""
    videos_dir = WLASL_DIR / "videos"
    for ext in [".mp4", ".webm", ".avi"]:
        path = videos_dir / f"{video_id}{ext}"
        if path.exists():
            return path
    return None


def get_video_duration_ms(video_path: Path) -> int:
    """Get video duration using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True
        )
        duration_sec = float(result.stdout.strip())
        return int(duration_sec * 1000)
    except:
        return 2000  # Default 2 seconds


def import_sign(entry: dict) -> bool:
    """Import a single sign to verified/."""
    gloss = entry["gloss"].upper()
    instances = entry.get("instances", [])

    if not instances:
        print(f"  {gloss}: No instances")
        return False

    # Find first available video
    video_path = None
    instance = None
    for inst in instances:
        video_id = inst.get("video_id", "")
        path = find_video(video_id)
        if path:
            video_path = path
            instance = inst
            break

    if not video_path:
        print(f"  {gloss}: No video found")
        return False

    # Create destination directory
    dest_dir = VERIFIED_DIR / gloss
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy video
    video_dest = dest_dir / "video.mp4"
    shutil.copy2(video_path, video_dest)

    # Get video duration
    duration_ms = get_video_duration_ms(video_dest)

    # Determine category
    categories = {
        "hello": "greeting", "goodbye": "greeting", "please": "greeting",
        "sorry": "greeting", "nice": "greeting", "meet": "greeting",
        "yes": "response", "no": "response", "fine": "response",
        "good": "adjective", "bad": "adjective", "happy": "emotion",
        "sad": "emotion", "love": "emotion",
        "i": "pronoun", "you": "pronoun", "we": "pronoun", "they": "pronoun",
        "me": "pronoun", "my": "pronoun", "your": "pronoun",
        "who": "question", "what": "question", "where": "question",
        "when": "question", "why": "question", "how": "question",
        "mother": "family", "father": "family", "family": "family",
        "friend": "people", "woman": "people", "man": "people",
        "go": "verb", "want": "verb", "like": "verb", "see": "verb",
        "know": "verb", "think": "verb", "understand": "verb", "learn": "verb",
        "work": "verb", "eat": "verb", "drink": "verb", "need": "verb",
        "have": "verb", "give": "verb", "get": "verb", "come": "verb",
        "stay": "verb", "live": "verb", "feel": "verb", "ask": "verb",
        "tell": "verb", "say": "verb", "call": "verb", "wait": "verb",
        "stop": "verb", "start": "verb", "finish": "verb", "can": "verb",
        "school": "place", "home": "place", "here": "place", "there": "place",
        "book": "object", "computer": "object", "water": "object",
        "chair": "object", "clothes": "object", "candy": "object",
        "sign": "object", "language": "object",
        "time": "time", "today": "time", "now": "time", "day": "time",
        "night": "time", "morning": "time", "afternoon": "time",
        "different": "adjective", "same": "adjective", "new": "adjective",
        "old": "adjective", "big": "adjective", "small": "adjective",
        "more": "quantity", "many": "quantity",
    }
    category = categories.get(gloss.lower(), "general")

    # Create metadata
    metadata = {
        "gloss": gloss,
        "english": [gloss.lower().replace("_", " ")],
        "category": category,
        "source": "wlasl",
        "status": "verified",
        "quality_score": 4,
        "verified_by": "auto-import",
        "verified_date": date.today().isoformat(),
        "video": {
            "file": "video.mp4",
            "fps": instance.get("fps", 25),
            "duration_ms": duration_ms
        },
        "timing": {
            "sign_start_ms": 0,
            "sign_end_ms": duration_ms
        }
    }

    # Save metadata
    meta_path = dest_dir / "sign.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {gloss}: OK ({duration_ms}ms)")
    return True


def main():
    print("Loading WLASL metadata...")
    data = load_wlasl_metadata()

    # Create lookup by gloss
    by_gloss = {entry["gloss"].lower(): entry for entry in data}

    print(f"\nImporting {len(MVP_WORDS)} MVP signs to {VERIFIED_DIR}...")

    success = 0
    for word in MVP_WORDS:
        if word.lower() in by_gloss:
            if import_sign(by_gloss[word.lower()]):
                success += 1
        else:
            print(f"  {word.upper()}: Not in dataset")

    print(f"\nDone! Imported {success}/{len(MVP_WORDS)} signs")
    print(f"Signs are in: {VERIFIED_DIR}")


if __name__ == "__main__":
    main()
