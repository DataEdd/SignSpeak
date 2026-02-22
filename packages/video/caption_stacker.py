"""
Caption Stacker for SignBridge V2
Generates TikTok-style animated captions and stacks them below avatar video.

Based on 13hacks/SignBridge/caption_stacker.py
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("Pillow required. Run: pip install pillow")


@dataclass
class CaptionConfig:
    """Caption styling configuration."""
    width: int = 720
    height: int = 120
    bg_color: tuple = (26, 26, 46)  # Match avatar background
    text_color: tuple = (255, 255, 255)
    highlight_color: tuple = (255, 215, 0)  # Gold
    past_color: tuple = (80, 80, 80)
    future_color: tuple = (160, 160, 160)
    font_size: int = 42
    fps: int = 30


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    return float(data.get('format', {}).get('duration', 0))


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a nice font, with fallback."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf"
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()


def generate_caption_frames(
    text: str,
    duration: float,
    config: CaptionConfig = None
) -> list:
    """
    Generate caption frames with TikTok-style word highlighting.

    Returns list of numpy arrays (frames).
    """
    config = config or CaptionConfig()
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return []

    time_per_word = duration / word_count
    total_frames = int(duration * config.fps)

    font_large = load_font(config.font_size)
    font_small = load_font(int(config.font_size * 0.75))

    frames = []

    for frame_idx in range(total_frames):
        t = frame_idx / config.fps

        # Create frame
        img = Image.new('RGB', (config.width, config.height), config.bg_color)
        draw = ImageDraw.Draw(img)

        # Current word index
        current_idx = min(int(t / time_per_word), word_count - 1)

        # Show window: 1 before, current, 2 after
        start_idx = max(0, current_idx - 1)
        end_idx = min(word_count, current_idx + 3)
        visible_words = words[start_idx:end_idx]

        # Calculate widths
        spacing = 15
        total_width = 0
        word_widths = []

        for i, word in enumerate(visible_words):
            actual_idx = start_idx + i
            font = font_large if actual_idx == current_idx else font_small
            bbox = draw.textbbox((0, 0), word, font=font)
            w = bbox[2] - bbox[0]
            word_widths.append(w)
            total_width += w + spacing

        total_width -= spacing

        # Center text
        x = (config.width - total_width) // 2
        y = config.height // 2

        # Draw words
        for i, word in enumerate(visible_words):
            actual_idx = start_idx + i

            if actual_idx < current_idx:
                color = config.past_color
                font = font_small
                y_offset = 4
            elif actual_idx == current_idx:
                color = config.highlight_color
                font = font_large
                y_offset = 0
            else:
                color = config.future_color
                font = font_small
                y_offset = 4

            draw.text((x, y + y_offset), word, fill=color, font=font, anchor="lm")
            x += word_widths[i] + spacing

        frames.append(np.array(img))

    return frames


def stack_videos_ffmpeg(
    avatar_path: str,
    captions_path: str,
    output_path: str,
    target_width: int = 720
) -> str:
    """Stack avatar (top) and captions (bottom) using FFmpeg."""
    filter_complex = (
        f"[0:v]scale={target_width}:-2[top];"
        f"[1:v]scale={target_width}:-2[bottom];"
        f"[top][bottom]vstack=inputs=2[out]"
    )

    cmd = [
        'ffmpeg', '-y',
        '-i', str(avatar_path),
        '-i', str(captions_path),
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:v', 'libx264',
        '-crf', '20',
        '-preset', 'fast',
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg stacking failed: {result.stderr}")

    return output_path


def export_frames_to_video(frames: list, output_path: str, fps: float = 30) -> str:
    """Export frames to video using ffmpeg."""
    if not frames:
        raise ValueError("No frames to export")

    height, width = frames[0].shape[:2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass
    finally:
        if process.stdin:
            process.stdin.close()

    process.stderr.read()
    process.wait()
    return output_path


def add_captions_to_video(
    avatar_video: str,
    caption_text: str,
    output_path: str,
    config: CaptionConfig = None
) -> str:
    """
    Main function: Add captions below avatar video.

    Args:
        avatar_video: Path to avatar video
        caption_text: Text to display as captions
        output_path: Where to save final stacked video
        config: Caption styling config

    Returns:
        Path to final video
    """
    config = config or CaptionConfig()
    output_path = Path(output_path)

    # Get avatar duration
    duration = get_video_duration(avatar_video)
    print(f"  Avatar duration: {duration:.2f}s")

    # Generate caption frames
    print(f"  Generating captions: '{caption_text}'")
    caption_frames = generate_caption_frames(caption_text, duration, config)

    if not caption_frames:
        print("  No caption frames generated, returning original video")
        return avatar_video

    # Export captions to temp video
    captions_video = output_path.parent / f"_captions_temp.mp4"
    export_frames_to_video(caption_frames, str(captions_video), config.fps)

    # Stack videos
    print(f"  Stacking videos...")
    stack_videos_ffmpeg(
        avatar_video,
        str(captions_video),
        str(output_path),
        target_width=config.width
    )

    # Cleanup temp file
    captions_video.unlink(missing_ok=True)

    print(f"  Final video: {output_path}")
    return str(output_path)
