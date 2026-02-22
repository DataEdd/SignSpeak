"""List and search signs."""

from typing import Optional

import typer

from packages.database import SignStatus

from ..utils.config import get_store, get_search
from ..utils.display import console, print_sign, print_sign_table, print_stats


def list_signs(
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status: verified, pending, imported, rejected"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", help="Filter by source"
    ),
    quality: Optional[int] = typer.Option(
        None, "--quality", "-q", min=1, max=5, help="Minimum quality score"
    ),
    limit: int = typer.Option(
        50, "--limit", "-l", help="Maximum signs to show"
    ),
    stats: bool = typer.Option(
        False, "--stats", help="Show statistics only"
    ),
) -> None:
    """List signs in the database.

    By default, lists all verified signs. Use --status to filter by status.

    Example:
        signbridge list                    # All verified signs
        signbridge list --status pending   # Pending signs
        signbridge list --category greeting
        signbridge list --stats            # Show statistics
    """
    search_obj = get_search()

    # Stats only
    if stats:
        stats_data = search_obj.get_statistics()
        print_stats(stats_data)
        return

    # Parse status
    status_enum = None
    if status:
        try:
            status_enum = SignStatus(status.lower())
        except ValueError:
            console.print(f"[red]Error:[/] Invalid status '{status}'")
            console.print("Valid statuses: verified, pending, imported, rejected")
            raise typer.Exit(1)

    # Default to verified if no filters specified
    verified_only = status is None and category is None and source is None

    # Search with filters
    signs = search_obj.search(
        category=category,
        min_quality=quality,
        status=status_enum,
        source=source,
        verified_only=verified_only,
    )

    # Apply limit
    if len(signs) > limit:
        signs = signs[:limit]
        truncated = True
    else:
        truncated = False

    # Display
    title = "Signs"
    if status:
        title = f"{status.capitalize()} Signs"
    elif verified_only:
        title = "Verified Signs"

    print_sign_table(signs, title=title)

    if truncated:
        console.print(f"[dim](showing {limit} of {len(signs)}+, use --limit to see more)[/]")


def search(
    query: str = typer.Argument(..., help="Search query"),
    verified_only: bool = typer.Option(
        False, "--verified", "-v", help="Only search verified signs"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show detailed sign info"
    ),
) -> None:
    """Search signs by gloss or English translation.

    Example:
        signbridge search "hello"
        signbridge search "thank" --verified
    """
    search_obj = get_search()

    signs = search_obj.search(query=query, verified_only=verified_only)

    if not signs:
        console.print(f"[dim]No signs matching '{query}'[/]")
        return

    console.print(f"[bold]Found {len(signs)} sign(s) matching '{query}'[/]\n")

    if verbose:
        for sign in signs:
            print_sign(sign, verbose=True)
    else:
        print_sign_table(signs, title=f"Search: {query}")
