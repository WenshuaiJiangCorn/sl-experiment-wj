"""This module renders a live progress bar for a batch re-processing run that is already in flight.

The batch driver flushes its run log after every session, so the log doubles as the run's progress state. This module
tails that log and renders the progress bar in a separate terminal, which lets a run started in the background, or
started without a terminal attached, still be watched.

Notes:
    Reads the run log only, so watching a run costs it nothing and several watchers can run at once.
"""

import csv
import sys
import time
import argparse
from typing import Any
from pathlib import Path

# The interval, in seconds, between the run log polls.
_POLL_INTERVAL = 2.0


def read_records(log_path: Path) -> list[dict[str, Any]]:
    """Reads the records the run log holds.

    Notes:
        Tolerates a partially written trailing row, as the log is read while the driver is writing to it.

    Args:
        log_path: The path to the .csv run log the driver writes.

    Returns:
        The records read, which is an empty list when the log does not exist yet.
    """
    if not log_path.exists():
        return []

    try:
        with log_path.open("r", newline="", encoding="utf-8") as log_file:
            return [row for row in csv.DictReader(log_file) if row.get("status")]
    except (OSError, csv.Error):
        return []


def main() -> int:
    """Renders the progress bar until the run reaches the expected session count.

    Returns:
        The exit code, which is 0 once the run reaches the expected total.
    """
    parser = argparse.ArgumentParser(description="Renders a live progress bar for a batch re-processing run.")
    parser.add_argument("log", type=Path, help="The path to the .csv run log the batch driver writes.")
    parser.add_argument("--total", type=int, required=True, help="The number of sessions the run processes.")
    arguments = parser.parse_args()

    from tqdm import tqdm

    started = time.monotonic()
    seen = 0
    with tqdm(total=arguments.total, unit="session", dynamic_ncols=True, smoothing=0.1) as progress_bar:
        while seen < arguments.total:
            records = read_records(arguments.log)
            if len(records) > seen:
                for record in records[seen:]:
                    if record["status"] == "failed":
                        session_path = Path(record["session"])
                        progress_bar.write(
                            f"FAIL {session_path.parent.name}/{session_path.name}: {record.get('error', '')}"
                        )
                progress_bar.update(len(records) - seen)
                seen = len(records)

                failures = sum(1 for record in records if record["status"] == "failed")
                mean_seconds = sum(float(record["seconds"]) for record in records) / max(len(records), 1)
                progress_bar.set_postfix_str(f"{failures} failed | {mean_seconds:.0f}s/session")

            if seen >= arguments.total:
                break
            time.sleep(_POLL_INTERVAL)

    print(f"\nRun reached {seen}/{arguments.total} session(s) in {(time.monotonic() - started) / 3600:.2f} h.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
