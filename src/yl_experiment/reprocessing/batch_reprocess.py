"""This module provides a batch driver that re-processes every experiment session under a project directory.

The driver walks the project tree, resolves each session that holds a raw microcontroller log archive, and runs the
re-processing pipeline of the reprocess_legacy_logs module against it. Each session is written to the directory the
analysis code reads, seeded from the directory holding the parse that predates sanitization, so the result carries
the files this pipeline does not regenerate alongside the ones it does.

Notes:
    The output directory is the one the analysis code reads, so a session is re-processed in place. Seeding copies
    the earlier parse over that directory before the pipeline regenerates anything, which means a forced re-run that
    fails partway leaves the read directory holding the earlier parse until the run is repeated. A session that is
    already complete is skipped outright, so an ordinary resume never reaches that state.

    Valve calibration is never supplied, so the valve modules are skipped and the valve data of the previous parse is
    carried into the output unchanged. Only the lick and analog modules are re-parsed.

    A session that fails is recorded and skipped rather than ending the run, which keeps one unreadable archive from
    stranding the sessions that follow it. The failure and its error are written to the run log.

    The run is resumable. A session whose output directory already holds every file this driver regenerates is
    skipped unless the caller forces it, so an interrupted run continues where it stopped.

    The extraction pipeline parallelizes each session internally. Running several sessions at once therefore
    oversubscribes the machine unless the per-session worker count is lowered to match, which the 'workers' argument
    of this driver does.
"""

import csv
import sys
import time
import shutil
import argparse
import functools
import traceback
import contextlib
import concurrent.futures
from typing import Any
from pathlib import Path

# The directory holding the modules of the parent package. Several of those modules import their siblings as
# top-level modules rather than as package members, which only resolves while that directory is on the import path,
# so the driver puts it there before importing the pipeline.
_PARENT_PACKAGE_DIRECTORY = Path(__file__).resolve().parent.parent

# The name of the directory holding the raw .npz log archives inside a session directory.
_LOG_DIRECTORY_NAME = "linear_track_data_log"

# The name of the directory holding the parse that predates sanitization, used to seed the output directory. This
# directory holds the files the re-processing pipeline does not regenerate, which is what makes it the seed.
_SEED_DIRECTORY_NAME = "unsanitized_processed"

# The name of the output directory written for each session, which is the directory the analysis code reads.
_OUTPUT_DIRECTORY_NAME = "processed"

# The name of the per-session file capturing the console output of the re-processing pipeline.
_SESSION_LOG_NAME = "reprocessing.log"

# The number of attempts made to remove a directory held open by another process, and the base delay, in seconds,
# between them. The delay grows with each attempt.
_REMOVAL_ATTEMPTS = 5
_REMOVAL_RETRY_DELAY = 1.0

# The files this driver regenerates. These alone cannot mark a session as processed, as seeding copies files under
# the same names from the previous parse before any of them is regenerated.
_REGENERATED_FILES = ("analog_signal.feather", "left_lick_sensor.feather", "right_lick_sensor.feather")

# The name of the directory the extraction pipeline writes its per-module tables and its processing tracker to.
_EXTRACTION_DIRECTORY_NAME = "microcontroller_data"

# The per-module table each re-parsed module leaves in the extraction directory, matched by glob so that the pattern
# holds for any controller ID. A session is only complete once the extraction actually produced these, which
# distinguishes a finished session from one that failed after its output directory was seeded.
_EXTRACTED_TABLE_PATTERNS = (
    "controller_*_module_2_1.feather",
    "controller_*_module_2_2.feather",
    "controller_*_module_3_1.feather",
)

# The files the output directory carries over from the previous parse, which this driver never regenerates. The valve
# files are listed here because the driver supplies no valve calibration, so the valves are always skipped.
_SEEDED_FILES = (
    "left_camera_timestamps.feather",
    "right_camera_timestamps.feather",
    "top_camera_timestamps.feather",
    "left_valve_data.feather",
    "right_valve_data.feather",
)

# The entries the extraction pipeline writes into the output directory beside the parsed files.
_PIPELINE_ENTRIES = ("extraction_configuration.yaml", "microcontroller_data", "sanitized_logs")


def discover_sessions(project_directory: Path) -> list[Path]:
    """Finds every session directory holding a raw microcontroller log archive.

    Args:
        project_directory: The path to the project directory to walk.

    Returns:
        The path of each session directory, sorted.
    """
    return sorted(path.parent for path in project_directory.rglob(_LOG_DIRECTORY_NAME) if path.is_dir())


def _remove_directory(directory: Path, attempts: int = _REMOVAL_ATTEMPTS) -> None:
    """Removes the target directory, retrying while it is held open by another process.

    Notes:
        A synchronization client indexing the output directory holds its files open in short bursts, which makes a
        removal fail with a permission error that succeeds moments later. Retrying rather than failing keeps a
        transient lock from stranding a session, as a partial removal leaves the session incomplete.

    Args:
        directory: The path to the directory to remove.
        attempts: The number of removal attempts to make before giving up.

    Raises:
        OSError: If the directory is still held open after the final attempt.
    """
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(directory)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(_REMOVAL_RETRY_DELAY * attempt)


def missing_outputs(session: Path) -> list[str]:
    """Determines which output entries the target session still lacks.

    Notes:
        Requires the extracted per-module tables as well as the parsed files. Seeding copies the previous parse into
        the output directory under the very names this driver regenerates, so those names alone cannot distinguish a
        finished session from one that failed between seeding and extraction.

    Args:
        session: The path to the session directory.

    Returns:
        The names of the missing entries, which is an empty list for a complete session.
    """
    output = session.joinpath(_OUTPUT_DIRECTORY_NAME)
    if not output.is_dir():
        return ["(no output directory)"]

    missing = [
        name
        for name in (*_REGENERATED_FILES, *_SEEDED_FILES, *_PIPELINE_ENTRIES)
        if not output.joinpath(name).exists()
    ]

    extraction_directory = output.joinpath(_EXTRACTION_DIRECTORY_NAME)
    if not extraction_directory.is_dir():
        missing.extend(_EXTRACTED_TABLE_PATTERNS)
    else:
        missing.extend(
            pattern for pattern in _EXTRACTED_TABLE_PATTERNS if not any(extraction_directory.glob(pattern))
        )

    return missing


def is_processed(session: Path) -> bool:
    """Determines whether the target session already holds a complete output directory.

    Args:
        session: The path to the session directory.

    Returns:
        True when the output directory lacks nothing.
    """
    return not missing_outputs(session)


def verify_sessions(sessions: list[Path]) -> tuple[int, list[tuple[Path, list[str]]]]:
    """Checks that every session holds a complete output directory.

    Notes:
        Checks the carried-over files as well as the regenerated ones, so a session whose seeding silently produced
        nothing is reported rather than counted as complete.

    Args:
        sessions: The session directories to check.

    Returns:
        The number of complete sessions, and the incomplete sessions paired with the entries each one is missing.
    """
    complete = 0
    incomplete: list[tuple[Path, list[str]]] = []

    for session in sessions:
        missing = missing_outputs(session)
        if missing:
            incomplete.append((session, missing))
        else:
            complete += 1

    return complete, incomplete


def process_session(session: Path, workers: int = 0) -> dict[str, Any]:
    """Re-processes a single session and reports the outcome.

    Notes:
        Imports the pipeline inside the function so that each worker process of a pool loads it once in its own
        interpreter, rather than inheriting a partially initialized module through the process start-up. The import
        path is extended first, as several modules the pipeline pulls in import their siblings as top-level modules.

        The pipeline writes progress to the console, which interleaves unreadably across concurrent sessions, so the
        output of each session is captured into a log file inside that session's output directory.

    Args:
        session: The path to the session directory to process.
        workers: The worker count to give the extraction pipeline. A non-positive value leaves the pipeline's own
            default in place, which resolves the width from the archive.

    Returns:
        A record holding the session path, the outcome status, the elapsed time, and the error text of a failure.
    """
    if str(_PARENT_PACKAGE_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(_PARENT_PACKAGE_DIRECTORY))

    import yl_experiment.reprocessing.reprocess_legacy_logs as pipeline_module
    from ataraxis_communication_interface import run_log_processing_pipeline

    # Pins the extraction width so that several sessions running at once do not oversubscribe the machine. The
    # pipeline module resolved the name at import, so rebinding it here is what the re-processing call observes.
    keyword_arguments: dict[str, Any] = {"display_progress": False}
    if workers > 0:
        keyword_arguments["workers"] = workers
    pipeline_module.run_log_processing_pipeline = functools.partial(
        run_log_processing_pipeline, **keyword_arguments
    )

    started = time.monotonic()
    record: dict[str, Any] = {"session": str(session), "status": "ok", "seconds": 0.0, "error": ""}

    output_directory = session.joinpath(_OUTPUT_DIRECTORY_NAME)
    try:
        # An incomplete output directory left by an earlier attempt still holds that attempt's processing tracker,
        # which would suppress the extraction jobs of this attempt and leave the session incomplete again. Removing
        # the extraction directory resets the tracker along with the partial tables it governs.
        extraction_directory = output_directory.joinpath(_EXTRACTION_DIRECTORY_NAME)
        if extraction_directory.is_dir():
            _remove_directory(extraction_directory)

        output_directory.mkdir(parents=True, exist_ok=True)
        with output_directory.joinpath(_SESSION_LOG_NAME).open("w", encoding="utf-8") as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                pipeline_module.reprocess_log_archive(
                    log_directory=session.joinpath(_LOG_DIRECTORY_NAME),
                    output_directory=output_directory,
                    seed_from=session.joinpath(_SEED_DIRECTORY_NAME),
                )
    except Exception as error:  # noqa: BLE001 - one failed session must not end the batch.
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()

    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def _drain(results: Any, writer: Any, log_file: Any, total: int, started: float) -> Any:
    """Writes each result to the run log as it arrives and reports progress.

    Notes:
        Renders a progress bar when the output is a terminal, and falls back to one line per session otherwise, as a
        bar redrawing itself through carriage returns is unreadable once the output is redirected to a file.

    Args:
        results: The iterable of session records to drain.
        writer: The csv writer to write each record through.
        log_file: The open run log file, flushed after every record so an interrupted run leaves a usable log.
        total: The number of sessions the run processes.
        started: The monotonic timestamp the run started at.

    Yields:
        Each record, unchanged.
    """
    interactive = sys.stdout.isatty()
    progress_bar = None
    if interactive:
        from tqdm import tqdm

        progress_bar = tqdm(total=total, unit="session", dynamic_ncols=True, smoothing=0.1)

    failures = 0
    try:
        for index, record in enumerate(results, start=1):
            writer.writerow(record)
            log_file.flush()

            if record["status"] == "failed":
                failures += 1
            session_path = Path(record["session"])

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"{failures} failed | {session_path.parent.name}/{session_path.name}")
                # A failure is written above the bar so it stays visible once the bar moves on.
                if record["status"] == "failed":
                    progress_bar.write(f"FAIL {session_path.parent.name}/{session_path.name}: {record['error']}")
            else:
                elapsed = time.monotonic() - started
                rate = elapsed / index
                remaining = (total - index) * rate
                marker = "OK  " if record["status"] == "ok" else "FAIL"
                print(
                    f"[{index:>4}/{total}] {marker} {session_path.parent.name}/{session_path.name} "
                    f"{record['seconds']:>6.1f}s | avg {rate:.1f}s | eta {remaining / 3600:.2f} h",
                    flush=True,
                )
                if record["status"] == "failed":
                    print(f"        {record['error']}", flush=True)

            yield record
    finally:
        if progress_bar is not None:
            progress_bar.close()


def main() -> int:
    """Runs the batch re-processing driver.

    Returns:
        The exit code, which is 1 when any session failed and 0 otherwise.
    """
    parser = argparse.ArgumentParser(description="Batch re-processes every experiment session of a project.")
    parser.add_argument("project_directory", type=Path, help="The path to the project directory to walk.")
    parser.add_argument("--log", type=Path, required=True, help="The path to the .csv run log to write.")
    parser.add_argument(
        "--sessions", type=int, default=0, help="Process at most this many sessions. 0 processes every session."
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="The number of sessions to process at the same time."
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="The extraction worker count per session. 0 keeps the default."
    )
    parser.add_argument("--force", action="store_true", help="Re-process sessions that already hold an output.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be processed and exit.")
    parser.add_argument(
        "--verify", action="store_true", help="Check that every session holds a complete output and exit."
    )
    arguments = parser.parse_args()

    sessions = discover_sessions(arguments.project_directory)

    if arguments.verify:
        complete, incomplete = verify_sessions(sessions)
        print(f"Complete: {complete}/{len(sessions)} session(s).", flush=True)
        for session, missing in incomplete:
            print(f"   INCOMPLETE {session.parent.name}/{session.name}: missing {', '.join(missing)}", flush=True)
        return 1 if incomplete else 0

    pending = sessions if arguments.force else [session for session in sessions if not is_processed(session)]
    if arguments.sessions:
        pending = pending[: arguments.sessions]

    print(f"Discovered {len(sessions)} session(s); {len(pending)} pending.", flush=True)
    if arguments.dry_run:
        for session in pending[:20]:
            print(f"   would process: {session}", flush=True)
        return 0
    if not pending:
        return 0

    print(
        f"Concurrency: {arguments.concurrency} session(s), workers per session: "
        f"{arguments.workers or 'default'}.",
        flush=True,
    )

    arguments.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with arguments.log.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["session", "status", "seconds", "error"], extrasaction="ignore")
        writer.writeheader()
        log_file.flush()

        worker = functools.partial(process_session, workers=arguments.workers)

        if arguments.concurrency > 1:
            # A multiprocessing pool runs its workers as daemons, which the extraction pipeline cannot work under, as
            # it starts child processes of its own. The executor's workers are not daemons, so they may have children.
            with concurrent.futures.ProcessPoolExecutor(max_workers=arguments.concurrency) as executor:
                futures = [executor.submit(worker, session) for session in pending]
                results = (future.result() for future in concurrent.futures.as_completed(futures))
                records = list(_drain(results, writer, log_file, len(pending), started))
        else:
            records = list(
                _drain((worker(session) for session in pending), writer, log_file, len(pending), started)
            )

    succeeded = sum(1 for record in records if record["status"] == "ok")
    failed = sum(1 for record in records if record["status"] == "failed")
    elapsed = time.monotonic() - started
    print(f"\nComplete. {succeeded} succeeded, {failed} failed, {elapsed / 3600:.2f} h elapsed.", flush=True)
    print(f"Run log: {arguments.log}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
