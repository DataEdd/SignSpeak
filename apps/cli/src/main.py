"""SignBridge CLI - Command-line tools for sign management and translation.

Usage:
    signbridge <command> [options]

Commands:
    add         Add a new sign to the database
    verify      Verify pending signs
    list        List signs in the database
    search      Search signs by gloss or English
    translate   Translate English to ASL video
    glosses     Convert English to ASL glosses
    import      Import signs from external sources
    show        Show details for a specific sign
    stats       Show database statistics
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .commands.add import add
from .commands.verify import verify
from .commands.list import list_signs, search
from .commands.translate import translate, glosses
from .commands.import_cmd import import_signs
from .utils.config import get_store, get_search
from .utils.display import console, print_sign, print_stats

# Create the main app
app = typer.Typer(
    name="signbridge",
    help="SignBridge CLI - Sign language translation tools",
    add_completion=False,
)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"SignBridge CLI v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=version_callback, is_eager=True,
        help="Show version and exit"
    ),
) -> None:
    """SignBridge CLI - Command-line tools for sign management and translation."""
    pass


# Register commands
app.command("add")(add)
app.command("verify")(verify)
app.command("list")(list_signs)
app.command("search")(search)
app.command("translate")(translate)
app.command("glosses")(glosses)

# Import is a reserved word, so we use a different function name
app.command("import")(import_signs)


@app.command("show")
def show(
    gloss: str = typer.Argument(..., help="Sign gloss to show"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all details"),
) -> None:
    """Show details for a specific sign.

    Example:
        signbridge show HELLO
        signbridge show THANK-YOU --verbose
    """
    store = get_store()
    sign = store.get_sign(gloss.upper())

    if not sign:
        console.print(f"[red]Sign '{gloss.upper()}' not found[/]")
        raise typer.Exit(1)

    print_sign(sign, verbose=verbose)


@app.command("stats")
def stats() -> None:
    """Show database statistics.

    Example:
        signbridge stats
    """
    search_obj = get_search()
    stats_data = search_obj.get_statistics()
    print_stats(stats_data)


@app.command("count")
def count() -> None:
    """Show sign counts by status.

    Example:
        signbridge count
    """
    store = get_store()
    counts = store.count_signs()

    console.print("[bold]Sign Counts[/]")
    console.print(f"  [green]Verified:[/]  {counts['verified']}")
    console.print(f"  [yellow]Pending:[/]   {counts['pending']}")
    console.print(f"  [blue]Imported:[/]  {counts['imported']}")
    console.print(f"  [red]Rejected:[/]  {counts['rejected']}")
    console.print(f"  [bold]Total:[/]     {sum(counts.values())}")


@app.command("delete")
def delete(
    gloss: str = typer.Argument(..., help="Sign gloss to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a sign from the database.

    Example:
        signbridge delete HELLO
        signbridge delete HELLO --force
    """
    store = get_store()
    sign = store.get_sign(gloss.upper())

    if not sign:
        console.print(f"[red]Sign '{gloss.upper()}' not found[/]")
        raise typer.Exit(1)

    if not force:
        print_sign(sign)
        if not typer.confirm(f"Delete {gloss.upper()}?"):
            raise typer.Exit(0)

    if store.delete_sign(gloss.upper()):
        console.print(f"[green]Deleted {gloss.upper()}[/]")
    else:
        console.print(f"[red]Failed to delete {gloss.upper()}[/]")
        raise typer.Exit(1)


@app.command("export")
def export_cmd(
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json, csv"),
    verified_only: bool = typer.Option(
        True, "--verified-only/--all", help="Only export verified signs"
    ),
) -> None:
    """Export signs from the database.

    Example:
        signbridge export -o signs.json
        signbridge export -o all_signs.json --all
    """
    import json
    import csv

    store = get_store()

    if verified_only:
        signs = store.list_verified()
    else:
        signs = store.list_signs()

    if not signs:
        console.print("[yellow]No signs to export[/]")
        return

    if format.lower() == "json":
        data = [sign.to_dict() for sign in signs]
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
    elif format.lower() == "csv":
        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gloss", "english", "category", "source", "quality_score"])
            for sign in signs:
                writer.writerow([
                    sign.gloss,
                    ";".join(sign.english),
                    sign.category,
                    sign.source,
                    sign.quality_score or "",
                ])
    else:
        console.print(f"[red]Unknown format: {format}[/]")
        raise typer.Exit(1)

    console.print(f"[green]Exported {len(signs)} signs to {output}[/]")


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
