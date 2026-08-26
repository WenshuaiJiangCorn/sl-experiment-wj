"""This module provides tooling for correcting the data payloads of .npz log archives recorded by firmware that
serialized the sensor's zero readout at the wrong width.

Firmware predating the 2026-08-11 fix sent the zero (pull-down) readout of the lick and analog modules through the
explicit-prototype SendData overload, passing a bare integer literal as the data object:

    SendData(static_cast<uint8_t>(kCustomStatusCodes::kChanged), kPrototypes::kOneUint16, 0);

The message header therefore declares prototype code 7, whose data object occupies 2 bytes, while the literal is
deduced as a 32-bit int and serialized as 4 bytes. Non-zero readouts pass a uint16_t variable and are unaffected, so
only the zero readouts carry an oversized payload.

The updated extraction pipeline validates every payload against its declared prototype and refuses an archive that
holds such a message, which makes every archive recorded before the firmware fix unreadable. This module rewrites
the affected messages at their declared width, which restores the archives without altering any value.

Notes:
    An oversized payload is narrowed only when every one of its bytes is zero, which makes the narrowed payload
    decode to the same value the original did. A payload carrying any non-zero byte is data this module cannot
    interpret, so the archive is refused instead of guessed at.

    Validation runs over the whole archive before any output is written, so a refused archive never leaves a
    partially written or partially corrected file behind.

    The source archive is never modified. Every correction is written to a new archive.
"""

import shutil
from pathlib import Path
import zipfile
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_communication_interface import SerialProtocols, SerialPrototypes

# The number of leading bytes every logged message reserves for the source ID and the message timestamp.
_LOG_MESSAGE_PREFIX_SIZE = 9

# The number of leading payload bytes a MODULE_DATA message reserves for the protocol, the module type and ID codes,
# the command code, the event code, and the prototype code.
_MODULE_DATA_HEADER_SIZE = 6

# The filename suffix of the entries a .npz archive stores its arrays under.
_ARCHIVE_ENTRY_SUFFIX = ".npy"

# The number of command line arguments the module expects when run as a script.
_CLI_ARGUMENT_COUNT = 3


@dataclass()
class SanitizationReport:
    """Stores the outcome of sanitizing a single log archive."""

    archive_path: Path
    """The path to the source archive."""
    output_path: Path
    """The path to the written archive."""
    total_entries: int = 0
    """The total number of entries the source archive holds."""
    data_messages: int = 0
    """The number of MODULE_DATA messages the archive holds."""
    corrected_messages: int = 0
    """The number of messages whose payload was narrowed to its declared width."""
    corrections_by_module: dict[tuple[int, int], int] = field(default_factory=dict)
    """The number of corrected messages for each (module_type, module_id) pair."""

    def __repr__(self) -> str:
        """Returns a string representation of the SanitizationReport instance."""
        return (
            f"SanitizationReport(archive={self.archive_path.name}, entries={self.total_entries}, "
            f"data_messages={self.data_messages}, corrected={self.corrected_messages})"
        )


def _expected_payload_size(prototype_code: int) -> int | None:
    """Returns the data-payload size, in bytes, that the input prototype code declares.

    Args:
        prototype_code: The prototype code read from a MODULE_DATA message header.

    Returns:
        The declared payload size in bytes, or None if the code is not a recognized prototype.
    """
    dtype_string = SerialPrototypes.get_dtype_for_code(code=prototype_code)
    if dtype_string is None:
        return None

    prototype = SerialPrototypes.get_prototype_for_code(code=np.uint8(prototype_code))
    element_count = 1 if np.ndim(prototype) == 0 else int(np.size(prototype))

    return int(np.dtype(dtype_string).itemsize) * element_count


def _resolve_correction(message: np.ndarray) -> tuple[int, int, int] | None:
    """Determines whether the input logged message carries an oversized data payload.

    Args:
        message: The logged message, stored as the [source_id][timestamp][payload] byte layout.

    Returns:
        The module type code, the module ID code, and the declared payload size of a MODULE_DATA message whose
        payload is wider than its prototype declares, or None when the message needs no correction.

    Raises:
        ValueError: If the message declares a payload wider than its prototype and carries a non-zero byte inside
            it, or if the message carries a payload narrower than its prototype declares. Neither case can be
            corrected without inventing or discarding data.
    """
    payload = message[_LOG_MESSAGE_PREFIX_SIZE:]
    if payload.size == 0 or int(payload[0]) != int(SerialProtocols.MODULE_DATA):
        return None
    if payload.size < _MODULE_DATA_HEADER_SIZE:
        return None

    module_type = int(payload[1])
    module_id = int(payload[2])
    prototype_code = int(payload[5])

    expected_size = _expected_payload_size(prototype_code=prototype_code)
    if expected_size is None:
        # An unrecognized prototype code carries no width to validate against, so the message is left as it is.
        return None

    data_payload = payload[_MODULE_DATA_HEADER_SIZE:]
    actual_size = int(data_payload.size)
    if actual_size == expected_size:
        return None

    if actual_size < expected_size:
        message_text = (
            f"Unable to sanitize the message logged by the module {module_type} {module_id}. The message declares "
            f"the prototype code {prototype_code}, whose data object occupies {expected_size} bytes, but it carries "
            f"a {actual_size}-byte data payload. A payload narrower than its prototype cannot be restored, as the "
            f"missing bytes were never logged."
        )
        console.error(message=message_text, error=ValueError)

    if bool(np.any(data_payload)):
        message_text = (
            f"Unable to sanitize the message logged by the module {module_type} {module_id}. The message declares "
            f"the prototype code {prototype_code}, whose data object occupies {expected_size} bytes, but it carries "
            f"a {actual_size}-byte data payload holding at least one non-zero byte "
            f"({data_payload.tolist()}). Narrowing this payload would discard logged data, so the archive is "
            f"refused. This module only narrows payloads whose every byte is zero, which the firmware defect this "
            f"module corrects always produced."
        )
        console.error(message=message_text, error=ValueError)

    return module_type, module_id, expected_size


def sanitize_log_archive(archive_path: Path, output_path: Path, *, overwrite: bool = False) -> SanitizationReport:
    """Rewrites the target log archive with every oversized data payload narrowed to its declared width.

    Notes:
        Validates the whole archive before writing anything, so a refused archive leaves no output behind. Copies
        every entry that needs no correction verbatim, which preserves the archive's entry names, entry order, and
        per-entry compression.

    Args:
        archive_path: The path to the source .npz log archive.
        output_path: The path to the .npz archive to write.
        overwrite: Determines whether to overwrite the output archive when it already exists.

    Returns:
        The report describing the entries inspected and the messages corrected.

    Raises:
        FileNotFoundError: If the source archive does not exist.
        FileExistsError: If the output archive exists and the 'overwrite' flag is False.
        ValueError: If the archive holds a payload this module cannot narrow without inventing or discarding data.
    """
    if not archive_path.exists() or not archive_path.is_file():
        message = f"Unable to sanitize the log archive. The file does not exist at the expected path: {archive_path}."
        console.error(message=message, error=FileNotFoundError)

    if output_path.exists() and not overwrite:
        message = (
            f"Unable to sanitize the log archive {archive_path}. The output archive already exists at "
            f"{output_path}. Enable the 'overwrite' flag to replace it."
        )
        console.error(message=message, error=FileExistsError)

    ensure_directory_exists(output_path.parent)
    report = SanitizationReport(archive_path=archive_path, output_path=output_path)

    with zipfile.ZipFile(archive_path) as source_zip:
        entries = source_zip.infolist()
    report.total_entries = len(entries)

    # The first pass validates every message and records what has to change. Deferring the write until the whole
    # archive validates keeps a refused archive from leaving a partially corrected output file behind.
    corrections: dict[str, int] = {}
    with np.load(archive_path, allow_pickle=False, mmap_mode="r") as archive:
        for entry in entries:
            key = entry.filename.removesuffix(_ARCHIVE_ENTRY_SUFFIX)
            message = archive[key]
            payload = message[_LOG_MESSAGE_PREFIX_SIZE:]
            if payload.size > 0 and int(payload[0]) == int(SerialProtocols.MODULE_DATA):
                report.data_messages += 1

            resolution = _resolve_correction(message=message)
            if resolution is None:
                continue

            module_type, module_id, expected_size = resolution
            corrections[key] = expected_size
            report.corrected_messages += 1
            module = (module_type, module_id)
            report.corrections_by_module[module] = report.corrections_by_module.get(module, 0) + 1

    # The second pass writes the output archive, narrowing the messages the first pass recorded.
    with (
        np.load(archive_path, allow_pickle=False, mmap_mode="r") as archive,
        zipfile.ZipFile(output_path, mode="w", allowZip64=True) as destination_zip,
    ):
        for entry in entries:
            key = entry.filename.removesuffix(_ARCHIVE_ENTRY_SUFFIX)
            message = np.asarray(archive[key])

            expected_size = corrections.get(key)
            if expected_size is not None:
                keep = _LOG_MESSAGE_PREFIX_SIZE + _MODULE_DATA_HEADER_SIZE + expected_size
                message = message[:keep]

            # Preserving the source entry's name and compression keeps the output a drop-in replacement.
            entry_info = zipfile.ZipInfo(filename=entry.filename, date_time=entry.date_time)
            entry_info.compress_type = entry.compress_type
            with destination_zip.open(entry_info, mode="w") as stream:
                np.lib.format.write_array(stream, message, allow_pickle=False)

    return report


def sanitize_log_directory(
    log_directory: Path, output_directory: Path, *, overwrite: bool = False
) -> list[SanitizationReport]:
    """Sanitizes every log archive under the target directory and assembles a drop-in replacement directory.

    Notes:
        Archives that need no correction are copied unchanged, and so is every other file the directory holds. The
        resulting directory therefore stands in for the source directory wherever the source was used, including as
        the input of the extraction pipeline.

    Args:
        log_directory: The path to the directory holding the source .npz log archives. The whole tree is walked, so
            archives nested at any depth are processed.
        output_directory: The path to the directory to assemble.
        overwrite: Determines whether to overwrite output files that already exist.

    Returns:
        The report of each archive processed, ordered by path.

    Raises:
        FileNotFoundError: If the log directory does not exist.
        ValueError: If any archive holds a payload this module cannot narrow.
    """
    if not log_directory.exists() or not log_directory.is_dir():
        message = (
            f"Unable to sanitize the log archives. The directory does not exist at the expected path: "
            f"{log_directory}."
        )
        console.error(message=message, error=FileNotFoundError)

    if not console.enabled:
        console.enable()

    ensure_directory_exists(output_directory)
    reports: list[SanitizationReport] = []

    for source_path in sorted(log_directory.rglob("*")):
        if source_path.is_dir():
            continue

        destination_path = output_directory.joinpath(source_path.relative_to(log_directory))
        ensure_directory_exists(destination_path.parent)

        if source_path.suffix == ".npz":
            report = sanitize_log_archive(
                archive_path=source_path, output_path=destination_path, overwrite=overwrite
            )
            reports.append(report)
            level = LogLevel.SUCCESS if report.corrected_messages else LogLevel.INFO
            modules_note = (
                f" across modules {sorted(report.corrections_by_module)}" if report.corrections_by_module else ""
            )
            console.echo(
                message=(
                    f"{source_path.name}: corrected {report.corrected_messages} of "
                    f"{report.data_messages} data messages{modules_note}."
                ),
                level=level,
            )
        else:
            # Every non-archive file is carried over so the output directory stands in for the source directory.
            if destination_path.exists() and not overwrite:
                continue
            shutil.copy2(source_path, destination_path)

    total_corrected = sum(report.corrected_messages for report in reports)
    console.echo(
        message=(
            f"Log archive sanitization: complete. Corrected {total_corrected} message(s) across "
            f"{len(reports)} archive(s). Sanitized directory: {output_directory}."
        ),
        level=LogLevel.SUCCESS,
    )

    return reports


if __name__ == "__main__":
    import sys

    if not console.enabled:
        console.enable()

    if len(sys.argv) != _CLI_ARGUMENT_COUNT:
        console.echo(
            message="usage: python archive_sanitization.py <source_log_directory> <output_directory>",
            level=LogLevel.ERROR,
        )
        raise SystemExit(2)

    sanitize_log_directory(log_directory=Path(sys.argv[1]), output_directory=Path(sys.argv[2]), overwrite=True)
