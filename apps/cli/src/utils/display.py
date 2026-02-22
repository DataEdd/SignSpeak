"""Rich display utilities for CLI output."""

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from packages.database import Sign, SignStatus

console = Console()


def status_color(status: SignStatus) -> str:
    """Get color for a status."""
    colors = {
        SignStatus.VERIFIED: "green",
        SignStatus.PENDING: "yellow",
        SignStatus.IMPORTED: "blue",
        SignStatus.REJECTED: "red",
    }
    return colors.get(status, "white")


def quality_indicator(score: Optional[int]) -> str:
    """Get quality indicator string."""
    if score is None:
        return "-"
    return "*" * score + " " * (5 - score)


def print_sign(sign: Sign, verbose: bool = False) -> None:
    """Print a single sign with details."""
    status_str = f"[{status_color(sign.status)}]{sign.status.value}[/]"

    title = f"[bold]{sign.gloss}[/bold] ({status_str})"

    lines = [
        f"[dim]English:[/] {', '.join(sign.english) or '-'}",
        f"[dim]Category:[/] {sign.category or '-'}",
        f"[dim]Source:[/] {sign.source}",
    ]

    if sign.quality_score:
        lines.append(f"[dim]Quality:[/] {quality_indicator(sign.quality_score)} ({sign.quality_score}/5)")

    if sign.verified_by:
        lines.append(f"[dim]Verified by:[/] {sign.verified_by} on {sign.verified_date or '-'}")

    if verbose and sign.video:
        lines.append(f"[dim]Video:[/] {sign.video.file}")
        if sign.video.duration_ms:
            lines.append(f"[dim]Duration:[/] {sign.video.duration_ms}ms")
        if sign.video.resolution:
            lines.append(f"[dim]Resolution:[/] {sign.video.resolution}")

    if verbose and sign.path:
        lines.append(f"[dim]Path:[/] {sign.path}")

    content = "\n".join(lines)
    console.print(Panel(content, title=title, expand=False))


def print_sign_table(signs: list[Sign], title: str = "Signs") -> None:
    """Print a table of signs."""
    if not signs:
        console.print(f"[dim]No {title.lower()} found[/]")
        return

    table = Table(title=title)
    table.add_column("Gloss", style="bold")
    table.add_column("English")
    table.add_column("Status")
    table.add_column("Quality")
    table.add_column("Category")

    for sign in signs:
        status_str = Text(sign.status.value, style=status_color(sign.status))
        quality_str = quality_indicator(sign.quality_score) if sign.quality_score else "-"

        table.add_row(
            sign.gloss,
            ", ".join(sign.english[:2]) + ("..." if len(sign.english) > 2 else ""),
            status_str,
            quality_str,
            sign.category or "-",
        )

    console.print(table)


def print_stats(stats: dict) -> None:
    """Print database statistics."""
    table = Table(title="Database Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Verified Signs", f"[green]{stats.get('total_verified', 0)}[/]")
    table.add_row("Pending Review", f"[yellow]{stats.get('total_pending', 0)}[/]")

    if "categories" in stats:
        table.add_row("Categories", str(len(stats["categories"])))

    if "sources" in stats:
        for source, count in stats["sources"].items():
            table.add_row(f"  {source}", str(count))

    console.print(table)
