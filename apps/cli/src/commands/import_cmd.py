"""Import signs from external sources."""

from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn

from packages.database import create_importer

from ..utils.config import get_store
from ..utils.display import console


def import_signs(
    source: str = typer.Argument(..., help="Source type: wlasl, how2sign"),
    path: Path = typer.Option(..., "--source", "-s", help="Path to source data"),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Maximum signs to import"
    ),
    gloss: Optional[str] = typer.Option(
        None, "--gloss", "-g", help="Import a specific sign by gloss"
    ),
    verify: bool = typer.Option(
        False, "--verify", "-v", help="Auto-verify imported signs"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be imported"
    ),
) -> None:
    """Import signs from external sources.

    Supported sources:
    - wlasl: Word-Level ASL dataset
    - how2sign: How2Sign dataset

    Example:
        signbridge import wlasl --source /path/to/wlasl/
        signbridge import wlasl --source /data/wlasl --limit 100
        signbridge import wlasl --source /data/wlasl --gloss HELLO
    """
    store = get_store()

    # Validate source path
    if not path.exists():
        console.print(f"[red]Error:[/] Source path not found: {path}")
        raise typer.Exit(1)

    try:
        importer = create_importer(source, path, store)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)

    # Import a single sign
    if gloss:
        console.print(f"Importing {gloss.upper()} from {source}...")

        if dry_run:
            console.print(f"[dim]Would import {gloss.upper()}[/]")
            return

        sign = importer.import_sign(gloss)
        if sign:
            console.print(f"[green]Imported {sign.gloss}[/]")
        else:
            console.print(f"[yellow]Sign {gloss.upper()} not found in {source}[/]")
        return

    # List available signs in dry-run mode
    if dry_run:
        if hasattr(importer, "list_available"):
            available = importer.list_available()
            if limit:
                available = available[:limit]
            console.print(f"[bold]{len(available)} signs available[/]")
            console.print(", ".join(available[:50]))
            if len(available) > 50:
                console.print(f"[dim]... and {len(available) - 50} more[/]")
        else:
            console.print("[dim]Dry run: would import signs from source[/]")
        return

    # Full import
    console.print(f"[bold]Importing from {source}...[/]\n")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # We don't know total ahead of time, so use indeterminate
        task = progress.add_task("Importing...", total=None)

        success, errors = importer.import_all(limit=limit)

        progress.update(task, completed=success)

    # Report results
    console.print(f"\n[green]Successfully imported {success} signs[/]")

    if errors:
        console.print(f"[yellow]{len(errors)} errors occurred:[/]")
        for error in errors[:10]:
            console.print(f"  - {error}")
        if len(errors) > 10:
            console.print(f"  [dim]... and {len(errors) - 10} more[/]")

    # Show next steps
    console.print(f"\n[dim]Imported signs are in the 'imported/{source}' directory.[/]")
    console.print("[dim]Use 'signbridge verify --interactive' to review them.[/]")
