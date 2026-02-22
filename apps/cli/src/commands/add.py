"""Add new signs to the database."""

from pathlib import Path
from typing import Optional

import typer
from rich.prompt import Confirm

from ..utils.config import get_store
from ..utils.display import console, print_sign


def add(
    gloss: str = typer.Argument(..., help="Sign gloss (e.g., HELLO)"),
    video: Path = typer.Option(..., "--video", "-v", help="Path to video file"),
    english: str = typer.Option(
        ..., "--english", "-e", help="English translations (comma-separated)"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Sign category (e.g., greeting)"
    ),
    source: str = typer.Option(
        "recorded", "--source", "-s", help="Source of the sign"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite if exists"
    ),
) -> None:
    """Add a new sign to the database.

    The sign will be added to the pending directory for verification.

    Example:
        signbridge add HELLO --video hello.mp4 --english "hello,hi,hey"
    """
    store = get_store()
    gloss = gloss.upper()

    # Check if video exists
    if not video.exists():
        console.print(f"[red]Error:[/] Video file not found: {video}")
        raise typer.Exit(1)

    # Check for existing sign
    existing = store.get_sign(gloss)
    if existing:
        if not force:
            console.print(
                f"[yellow]Warning:[/] Sign '{gloss}' already exists in {existing.status.value}"
            )
            if not Confirm.ask("Delete and re-add?"):
                raise typer.Exit(0)
            store.delete_sign(gloss)
        else:
            store.delete_sign(gloss)

    # Parse English translations
    english_list = [e.strip() for e in english.split(",") if e.strip()]

    try:
        sign = store.add_sign(
            gloss=gloss,
            video_path=video,
            english=english_list,
            category=category or "",
            source=source,
        )
        console.print(f"[green]Added {gloss} to pending[/]")
        print_sign(sign)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
