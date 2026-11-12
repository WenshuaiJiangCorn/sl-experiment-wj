"""This module arranges a session's directories into the layout the re-processing pipeline reads and writes.

The runtime writes its parse to the directory the analysis code reads, and the batch driver seeds from the parse that
predates sanitization and writes that read directory back. This module renames the runtime's parse out of the way, so
that the driver has a seed to read and a name to write, and it is the step to run before re-processing a project.

Run it against a project whose sessions still carry only the runtime's parse, and it archives that parse. Run it
against a project whose re-processed output was written beside the runtime's parse under its own name, and it
archives the runtime's parse and promotes the output into its place.

Notes:
    Each session is promoted in two renames, and a session is left in a recoverable state between them. The module
    resolves what each session still needs from the directories it holds, so a pass interrupted at any point is
    finished by running it again rather than by repairing anything by hand.

    A rename never overwrites. A session whose target name is already taken is reported and skipped, as renaming onto
    an existing directory would destroy whatever that directory holds.

    The source directories are renamed, never copied, so the promotion moves no data and costs no additional space.
"""

import sys
import time
import argparse
from typing import Any
from pathlib import Path

# The name the previous parse is renamed to.
_ARCHIVED_NAME = "unsanitized_processed"

# The name the analysis code reads, which the re-processed output is renamed to.
_PROMOTED_NAME = "processed"

# The name the batch driver wrote the re-processed output under.
_OUTPUT_NAME = "processed_v2"

# The name of the directory holding the raw .npz log archives, used to resolve the session directories.
_LOG_DIRECTORY_NAME = "linear_track_data_log"

# The number of attempts made to rename a directory held open by another process, and the base delay, in seconds,
# between them. The delay grows with each attempt.
_RENAME_ATTEMPTS = 5
_RENAME_RETRY_DELAY = 1.0


def discover_sessions(project_directory: Path) -> list[Path]:
    """Finds every session directory holding a raw microcontroller log archive.

    Args:
        project_directory: The path to the project directory to walk.

    Returns:
        The path of each session directory, sorted.
    """
    return sorted(path.parent for path in project_directory.rglob(_LOG_DIRECTORY_NAME) if path.is_dir())


def _rename_directory(source: Path, destination: Path, attempts: int = _RENAME_ATTEMPTS) -> None:
    """Renames the source directory to the destination, retrying while it is held open by another process.

    Notes:
        A synchronization client indexing a session holds its directories open in short bursts, which makes a rename
        fail with a permission error that succeeds moments later.

    Args:
        source: The path to rename.
        destination: The path to rename it to.
        attempts: The number of rename attempts to make before giving up.

    Raises:
        FileExistsError: If the destination already exists, as a rename must never overwrite it.
        OSError: If the directory is still held open after the final attempt.
    """
    if destination.exists():
        message = f"Unable to rename {source.name} to {destination.name}: the destination already exists."
        raise FileExistsError(message)

    for attempt in range(1, attempts + 1):
        try:
            source.rename(destination)
            return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(_RENAME_RETRY_DELAY * attempt)


def resolve_actions(session: Path) -> tuple[str, list[tuple[Path, Path]]]:
    """Determines what the target session still needs to reach the promoted layout.

    Notes:
        Resolves the state from the directories the session holds rather than from a record of previous runs, which
        is what lets an interrupted pass be finished by running the module again.

    Args:
        session: The path to the session directory.

    Returns:
        The state of the session, and the renames that bring it to the promoted layout, in the order to apply them.
    """
    archived = session.joinpath(_ARCHIVED_NAME)
    promoted = session.joinpath(_PROMOTED_NAME)
    output = session.joinpath(_OUTPUT_NAME)

    # The promoted layout: the earlier parse is archived and the re-processed output carries the read name.
    if archived.is_dir() and promoted.is_dir() and not output.exists():
        return "done", []

    # The runtime layout: the session holds only the parse the runtime wrote, which has to be archived before the
    # batch driver can re-process the session, as the driver reads that directory as its seed.
    if not archived.exists() and promoted.is_dir() and not output.exists():
        return "runtime", [(promoted, archived)]

    # Archived and awaiting re-processing: the batch driver writes the read directory back.
    if archived.is_dir() and not promoted.exists() and not output.exists():
        return "ready", []

    # A re-processed output written beside the earlier parse under its own name, which both still carry.
    if not archived.exists() and promoted.is_dir() and output.is_dir():
        return "pending", [(promoted, archived), (output, promoted)]

    # Interrupted between the two renames: the earlier parse is archived and the output still carries its own name.
    if archived.is_dir() and not promoted.exists() and output.is_dir():
        return "half", [(output, promoted)]

    return "unexpected", []


def promote_session(session: Path, dry_run: bool) -> dict[str, Any]:
    """Promotes the re-processed output of a single session.

    Args:
        session: The path to the session directory to promote.
        dry_run: Determines whether to report the renames without applying them.

    Returns:
        A record holding the session path, its resolved state, the outcome status, and the error text of a failure.
    """
    state, actions = resolve_actions(session)
    record: dict[str, Any] = {"session": str(session), "state": state, "status": "ok", "error": ""}

    if state == "unexpected":
        held = sorted(entry.name for entry in session.iterdir() if entry.is_dir())
        record["status"] = "skipped"
        record["error"] = f"unexpected layout, holds: {held}"
        return record

    if state == "done":
        record["status"] = "already promoted"
        return record

    if state == "ready":
        record["status"] = "ready for re-processing"
        return record

    if dry_run:
        record["status"] = "would promote"
        return record

    try:
        for source, destination in actions:
            _rename_directory(source, destination)
    except OSError as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"

    return record


def main() -> int:
    """Promotes the re-processed output of every session under the project directory.

    Returns:
        The exit code, which is 1 when any session failed or was skipped and 0 otherwise.
    """
    parser = argparse.ArgumentParser(description="Promotes the re-processed output of every session of a project.")
    parser.add_argument("project_directory", type=Path, help="The path to the project directory to walk.")
    parser.add_argument("--dry-run", action="store_true", help="Report the renames without applying them.")
    arguments = parser.parse_args()

    sessions = discover_sessions(arguments.project_directory)
    print(f"Discovered {len(sessions)} session(s).", flush=True)

    records = [promote_session(session, dry_run=arguments.dry_run) for session in sessions]

    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1

    for record in records:
        if record["status"] in ("failed", "skipped"):
            session_path = Path(record["session"])
            print(
                f"   {record['status'].upper()} {session_path.parent.name}/{session_path.name}: {record['error']}",
                flush=True,
            )

    print("\n" + ", ".join(f"{status}: {count}" for status, count in sorted(counts.items())), flush=True)
    if arguments.dry_run:
        print("Dry run: nothing was renamed.", flush=True)

    return 1 if counts.get("failed") or counts.get("skipped") else 0


if __name__ == "__main__":
    sys.exit(main())
