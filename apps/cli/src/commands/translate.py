"""Translate English text to ASL."""

from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from packages.translation import translate as translate_text, GlossConverter
from packages.video import Compositor, compose_sequence

from ..utils.config import get_signs_dir, get_store
from ..utils.display import console


def translate(
    text: str = typer.Argument(..., help="English text to translate"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output video file path"
    ),
    avatar: bool = typer.Option(
        False, "--avatar", "-a", help="Use 3D skeleton avatar instead of video clips"
    ),
    avatar_type: str = typer.Option(
        "skeleton", "--avatar-type", "-t",
        help="Avatar type: skeleton (fast), smplx (realistic)"
    ),
    speed: str = typer.Option(
        "normal", "--speed", "-s", help="Playback speed: slow, normal, fast"
    ),
    format: str = typer.Option(
        "mp4", "--format", "-f", help="Output format: mp4, webm, gif"
    ),
    show_glosses: bool = typer.Option(
        True, "--glosses/--no-glosses", help="Show glosses in output"
    ),
) -> None:
    """Translate English text to ASL video.

    Converts the text to ASL glosses using grammar rules, then generates
    a video from sign clips.

    Example:
        signbridge translate "Hello, how are you?" -o hello.mp4
        signbridge translate "See you tomorrow" --speed slow
    """
    store = get_store()
    signs_dir = get_signs_dir()

    # Translate to glosses
    result = translate_text(text, store=store)
    gloss_list = result.glosses

    if show_glosses:
        console.print(f"[bold]Glosses:[/] {' '.join(gloss_list)}")

        # Show translation quality info
        if result.validation:
            coverage = result.validation.coverage
            color = "green" if coverage >= 0.8 else "yellow" if coverage >= 0.5 else "red"
            console.print(f"[dim]Coverage:[/] [{color}]{coverage:.0%}[/]")

            if result.validation.missing_glosses:
                console.print(
                    f"[yellow]Missing signs:[/] {', '.join(result.validation.missing_glosses)}"
                )

    # Generate video if output specified
    if output:
        # Map speed to transition duration
        speed_map = {"slow": 250, "normal": 150, "fast": 100}
        transition_ms = speed_map.get(speed, 150)

        # Check we have enough signs
        if result.validation and result.validation.coverage < 0.5:
            console.print(
                "[yellow]Warning:[/] Less than 50% of signs available in database"
            )
            if not typer.confirm("Continue anyway?"):
                raise typer.Exit(0)

        # Filter to available glosses only
        available_glosses = [
            g for g in gloss_list
            if store.get_verified_sign(g) is not None
        ]

        if not available_glosses:
            console.print("[red]Error:[/] No available signs for this translation")
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if avatar:
                avatar_label = "SMPL-X" if avatar_type == "smplx" else "skeleton"
                progress.add_task(f"Rendering {avatar_label} avatar...", total=None)
                try:
                    from packages.avatar import PoseSequence

                    # Load pose sequences
                    sequences = []
                    for gloss in available_glosses:
                        pose_path = signs_dir / "verified" / gloss / "poses.json"
                        if pose_path.exists():
                            sequences.append(PoseSequence.load(pose_path))
                        else:
                            console.print(f"[yellow]No poses for {gloss}, skipping[/]")

                    if not sequences:
                        console.print("[red]Error:[/] No pose data available")
                        raise typer.Exit(1)

                    # Choose renderer based on avatar type
                    if avatar_type == "smplx":
                        from packages.avatar import AvatarSMPLXRenderer, SMPLXRenderSettings
                        settings = SMPLXRenderSettings(model_path="data/models")
                        renderer = AvatarSMPLXRenderer(settings)
                    else:
                        from packages.avatar import AvatarMatplotlibRenderer
                        renderer = AvatarMatplotlibRenderer()

                    output_path = renderer.render_multiple(sequences, output)
                    console.print(f"[green]Saved {avatar_label} avatar video to {output_path}[/]")
                except Exception as e:
                    console.print(f"[red]Error rendering avatar:[/] {e}")
                    raise typer.Exit(1)
            else:
                progress.add_task("Composing video...", total=None)
                try:
                    output_path = compose_sequence(
                        glosses=available_glosses,
                        signs_dir=signs_dir / "verified",
                        output_path=str(output),
                        transition_duration_ms=transition_ms,
                    )
                    console.print(f"[green]Saved to {output_path}[/]")
                except Exception as e:
                    console.print(f"[red]Error composing video:[/] {e}")
                    raise typer.Exit(1)


def glosses(
    text: str = typer.Argument(..., help="English text to translate"),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate against sign database"
    ),
) -> None:
    """Translate English to ASL glosses (no video).

    Shows the gloss sequence that would be signed.

    Example:
        signbridge glosses "I am happy"
        # Output: I HAPPY
    """
    store = get_store() if validate else None

    result = translate_text(text, store=store)

    # Output just the glosses for scripting
    console.print(" ".join(result.glosses))

    # Show validation info if available
    if validate and result.validation:
        if result.validation.missing_glosses:
            console.print(
                f"\n[dim]Missing:[/] [yellow]{', '.join(result.validation.missing_glosses)}[/]",
                highlight=False,
            )
        if result.fingerspelled:
            console.print(
                f"[dim]Fingerspelled:[/] {', '.join(result.fingerspelled)}",
                highlight=False,
            )
