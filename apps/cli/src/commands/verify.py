"""Verify pending signs."""

from typing import Optional

import typer
from rich.prompt import Prompt, IntPrompt, Confirm

from packages.database import SignStatus

from ..utils.config import get_store, get_verifier
from ..utils.display import console, print_sign


def verify(
    gloss: Optional[str] = typer.Argument(None, help="Sign gloss to verify (or interactive mode)"),
    score: Optional[int] = typer.Option(
        None, "--score", "-s", min=1, max=5, help="Quality score (1-5)"
    ),
    by: Optional[str] = typer.Option(
        None, "--by", "-b", help="Verifier name"
    ),
    reject: bool = typer.Option(
        False, "--reject", "-r", help="Reject the sign instead of verifying"
    ),
    reason: Optional[str] = typer.Option(
        None, "--reason", help="Rejection reason (with --reject)"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive verification mode"
    ),
) -> None:
    """Verify a pending sign and move to production.

    If no gloss is provided, enters interactive mode to process the
    verification queue.

    Example:
        signbridge verify HELLO --score 5 --by "john"
        signbridge verify --interactive
    """
    verifier = get_verifier()
    store = get_store()

    # Interactive mode
    if interactive or gloss is None:
        _interactive_verify(verifier, store)
        return

    gloss = gloss.upper()
    sign = store.get_sign(gloss)

    if not sign:
        console.print(f"[red]Error:[/] Sign '{gloss}' not found")
        raise typer.Exit(1)

    if sign.status == SignStatus.VERIFIED:
        console.print(f"[yellow]Sign '{gloss}' is already verified[/]")
        raise typer.Exit(0)

    # Show current sign
    print_sign(sign, verbose=True)

    # Rejection flow
    if reject:
        rejection_reason = reason or Prompt.ask("Reason for rejection")
        rejected_by = by or Prompt.ask("Your name")
        sign = verifier.reject(gloss, rejection_reason, rejected_by)
        console.print(f"[red]Rejected {gloss}[/]")
        return

    # Verification flow
    if score is None:
        # Run automated quality check first
        result = verifier.check_sign_quality(sign)
        if result.issues:
            console.print("[yellow]Issues found:[/]")
            for issue in result.issues:
                console.print(f"  - {issue}")
        if result.suggestions:
            console.print("[dim]Suggestions:[/]")
            for suggestion in result.suggestions:
                console.print(f"  - {suggestion}")

        score = IntPrompt.ask("Quality score (1-5)", default=result.score)

    if by is None:
        by = Prompt.ask("Your name")

    try:
        sign = verifier.verify(gloss, score, by)
        console.print(f"[green]Verified {gloss} with score {score}/5[/]")
        print_sign(sign)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


def _interactive_verify(verifier, store) -> None:
    """Interactive verification workflow."""
    queue = verifier.get_verification_queue()

    if not queue:
        console.print("[dim]No signs awaiting verification[/]")
        return

    console.print(f"[bold]{len(queue)} signs in verification queue[/]\n")

    for i, sign in enumerate(queue, 1):
        console.print(f"[dim]--- {i}/{len(queue)} ---[/]")
        print_sign(sign, verbose=True)

        # Show video path for manual review
        if sign.path:
            video_path = sign.path / (sign.video.file if sign.video else "video.mp4")
            console.print(f"\n[dim]Video:[/] {video_path}")

        # Run quality check
        result = verifier.check_sign_quality(sign)
        if result.issues:
            console.print("\n[yellow]Issues:[/]")
            for issue in result.issues:
                console.print(f"  - {issue}")

        console.print()

        # Get action
        action = Prompt.ask(
            "Action",
            choices=["verify", "reject", "skip", "quit"],
            default="verify" if result.passed else "skip",
        )

        if action == "quit":
            break
        elif action == "skip":
            continue
        elif action == "reject":
            reason = Prompt.ask("Rejection reason")
            by = Prompt.ask("Your name")
            verifier.reject(sign.gloss, reason, by)
            console.print(f"[red]Rejected {sign.gloss}[/]\n")
        elif action == "verify":
            score = IntPrompt.ask("Quality score (1-5)", default=result.score)
            by = Prompt.ask("Your name")
            try:
                verifier.verify(sign.gloss, score, by)
                console.print(f"[green]Verified {sign.gloss}[/]\n")
            except ValueError as e:
                console.print(f"[red]Error:[/] {e}")
                if Confirm.ask("Reject instead?"):
                    reason = Prompt.ask("Rejection reason")
                    verifier.reject(sign.gloss, reason, by)
                    console.print(f"[red]Rejected {sign.gloss}[/]\n")

    # Show final stats
    stats = verifier.get_verification_stats()
    console.print("\n[bold]Verification Summary[/]")
    console.print(f"  Pending: {stats['pending_review']}")
    console.print(f"  Verified: {stats['verified']}")
    console.print(f"  Rejected: {stats['rejected']}")
    console.print(f"  Approval rate: {stats['approval_rate']}%")
