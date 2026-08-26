"""This module provides an end-to-end pipeline for turning raw .npz log archives into the per-module .feather files
the analysis code consumes.

The pipeline chains the three stages a legacy archive has to pass through. It sanitizes the archive's data payloads,
registers the microcontroller manifest that archives recorded before the manifest was introduced do not carry, runs
the extraction pipeline, and parses each module's extracted messages into its own .feather file.

Notes:
    Every module is resolved by its type and identifier codes, so a module that reported no messages is skipped and
    reported instead of displacing the modules that follow it.

    Only the target controller's archive is staged and sanitized, so the camera archives sharing the log directory
    are neither copied nor rewritten.

    The raw log directory is never modified. The staged archive, the manifest, the extraction configuration, and the
    extracted message tables are all written under the output directory.

    Reuses the parsing functions of the data_processing module, which keeps the .feather files this pipeline writes
    identical in layout to the ones a live runtime produces.
"""

import shutil
from pathlib import Path

import numpy as np
from .data_processing import _parse_lick_data, _parse_valve_data, _parse_analog_data, _read_module_partition
from .microcontroller import (
    _CONTROLLED_ID,
    _CONTROLLER_NAME,
    _LICK_DETECTION_THRESHOLD,
    ModuleTypeCodes,
    fit_valve_calibration,
)
from .archive_sanitization import sanitize_log_archive
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import find_log_archive
from ataraxis_communication_interface import (
    EXTRACTION_CONFIGURATION_FILENAME,
    MICROCONTROLLER_MANIFEST_FILENAME,
    OutputLayout,
    ExtractionConfig,
    ModuleSourceData,
    ModuleExtractionConfig,
    ControllerExtractionConfig,
    resolve_module_path,
    run_log_processing_pipeline,
    write_microcontroller_manifest,
)

# Describes every hardware module the linear track runtime registers: its type and identifier codes, its
# human-readable name, the event codes to extract for it, and the name of the .feather file its parsed data is
# written to.
_MODULE_ROSTER: tuple[tuple[ModuleTypeCodes, int, str, tuple[int, ...], str], ...] = (
    (ModuleTypeCodes.VALVE_MODULE, 1, "Left Valve", (51, 52, 53), "left_valve_data.feather"),
    (ModuleTypeCodes.VALVE_MODULE, 2, "Right Valve", (51, 52, 53), "right_valve_data.feather"),
    (ModuleTypeCodes.LICK_MODULE, 1, "Left Lick Sensor", (51,), "left_lick_sensor.feather"),
    (ModuleTypeCodes.LICK_MODULE, 2, "Right Lick Sensor", (51,), "right_lick_sensor.feather"),
    (ModuleTypeCodes.ANALOG_MODULE, 1, "Analog Input", (51,), "analog_signal.feather"),
)

# The name of the subdirectory that stores the sanitized archive and the manifest the extraction pipeline reads.
_STAGING_DIRECTORY_NAME = "sanitized_logs"

# The number of command line arguments the module expects when run as a script.
_CLI_ARGUMENT_COUNT = 3


def _seed_output_directory(source_directory: Path, output_directory: Path) -> list[str]:
    """Copies the files of a previous parse into the output directory.

    Pre-populating the output directory carries over every file the previous parse produced, including the ones this
    pipeline does not regenerate, such as the camera frame timestamps and the valve data of a valve whose calibration
    was never recorded. The pipeline then overwrites only the files it produces, which leaves the rest in place.

    Notes:
        Copies the files sitting directly in the source directory and skips its subdirectories. A subdirectory of a
        previous run holds the extraction pipeline's own staging and tracker files, and seeding those would let a
        stale processing tracker suppress the extraction jobs of this runtime.

    Args:
        source_directory: The path to the directory holding the previous parse.
        output_directory: The path to the directory to pre-populate.

    Returns:
        The names of the copied files, sorted, and an empty list when the source directory does not exist.
    """
    if not source_directory.is_dir() or source_directory.resolve() == output_directory.resolve():
        return []

    copied: list[str] = []
    for source_file in sorted(source_directory.iterdir()):
        if not source_file.is_file():
            continue
        # copy2 preserves the source timestamps, which keeps a carried-over file identifiable as the older parse.
        shutil.copy2(source_file, output_directory.joinpath(source_file.name))
        copied.append(source_file.name)

    return copied


def reprocess_log_archive(
    log_directory: Path,
    output_directory: Path,
    controller_id: int = int(_CONTROLLED_ID),
    *,
    left_valve_calibration_data: tuple[tuple[int | float, int | float], ...] | None = None,
    right_valve_calibration_data: tuple[tuple[int | float, int | float], ...] | None = None,
    lick_threshold: np.uint16 = _LICK_DETECTION_THRESHOLD,
    seed_from: Path | None = None,
    sanitize: bool = True,
) -> dict[str, Path]:
    """Extracts and parses the data of every hardware module from a raw microcontroller log archive.

    Notes:
        Valve data is only parsed for a valve whose calibration data the caller supplies. Dispensed volumes are
        reconstructed from the valve's pulse durations through the calibration that was in use during the runtime, so
        a valve parsed under unrelated calibration constants yields wrong volumes. A valve left without calibration is
        skipped and reported, and any previously written data for that valve is left untouched. This matters for
        archives whose runtime calibration was not recorded, where the previously parsed volumes are the only correct
        copy that exists.

        Does not extract camera frame timestamps. The camera archives sharing the log directory are left untouched.

    Args:
        log_directory: The path to the directory holding the raw .npz log archives. The whole tree beneath this path
            is searched, so archives nested at any depth are found.
        output_directory: The path to the directory where to write the parsed .feather files. The staged archive, the
            extraction configuration, and the intermediate message tables are written here as well.
        controller_id: The identifier code of the microcontroller whose archive is processed.
        left_valve_calibration_data: The calibration data that the left valve used during the runtime that produced
            the archive, used to translate valve pulse durations into dispensed fluid volumes. When omitted, the left
            valve is skipped and its previously written data is left untouched.
        right_valve_calibration_data: The calibration data that the right valve used during the runtime that produced
            the archive. When omitted, the right valve is skipped.
        lick_threshold: The voltage threshold, in raw 12-bit ADC units, for classifying a sensor readout as a lick.
        seed_from: The path to a directory holding a previous parse of this session. Its files are copied into the
            output directory before processing starts, and processing then overwrites only the files it produces. This
            keeps the output a complete session, carrying over both the files this pipeline does not regenerate, such
            as the camera frame timestamps, and the valve data of a valve whose calibration is not supplied.
        sanitize: Determines whether to correct the archive's data payloads before extracting it. Archives recorded
            by firmware predating the 2026-08-11 fix require this, and archives recorded after it are unaffected by
            it, so disabling this only makes sense to confirm an archive needs no correction.

    Returns:
        The path of the written .feather file for each module this runtime parsed, keyed by the module's name. Modules
        that reported no data, and valves whose calibration was not supplied, are absent from the mapping even when a
        seeded file for them is present in the output directory.

    Raises:
        FileNotFoundError: If the log directory does not exist or holds no archive for the requested controller.
        ValueError: If the archive holds a data payload that cannot be corrected without inventing or discarding
            data.
    """
    if not console.enabled:
        console.enable()

    # Resolves the archive up front so a missing or ambiguous archive is reported before any output is written.
    archive_path = find_log_archive(log_directory=log_directory, source_id=str(controller_id))
    console.echo(message=f"Re-processing the log archive: {archive_path}.", level=LogLevel.INFO)

    ensure_directory_exists(output_directory)

    # Pre-populates the output directory so the files this pipeline does not regenerate survive into it.
    if seed_from is not None:
        seeded = _seed_output_directory(source_directory=seed_from, output_directory=output_directory)
        console.echo(
            message=(
                f"Seeded the output directory with {len(seeded)} file(s) from {seed_from}."
                if seeded
                else f"No files to seed from {seed_from}."
            ),
            level=LogLevel.INFO,
        )

    # Stages only the target controller's archive, which keeps the camera archives sharing the log directory from
    # being copied, and keeps the raw log directory free of the manifest the extraction pipeline requires.
    staging_directory = output_directory.joinpath(_STAGING_DIRECTORY_NAME)
    ensure_directory_exists(staging_directory)
    staged_archive = staging_directory.joinpath(archive_path.name)

    if sanitize:
        report = sanitize_log_archive(archive_path=archive_path, output_path=staged_archive, overwrite=True)
        level = LogLevel.SUCCESS if report.corrected_messages else LogLevel.INFO
        console.echo(
            message=(
                f"Payload sanitization: corrected {report.corrected_messages} of {report.data_messages} data "
                f"messages."
            ),
            level=level,
        )
    else:
        # Staging the archive unchanged keeps the rest of the pipeline reading from one place either way.
        staged_archive.write_bytes(archive_path.read_bytes())

    # Archives recorded before the manifest was introduced carry none, and the extraction pipeline reads the manifest
    # to confirm the archive was produced by this library.
    if not staging_directory.joinpath(MICROCONTROLLER_MANIFEST_FILENAME).exists():
        write_microcontroller_manifest(
            log_directory=staging_directory,
            controller_id=controller_id,
            controller_name=_CONTROLLER_NAME,
            modules=tuple(
                ModuleSourceData(module_type=int(module_type), module_id=module_id, name=name)
                for module_type, module_id, name, _, _ in _MODULE_ROSTER
            ),
        )

    # Declares every module in the roster as an extraction target.
    extraction_config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=controller_id,
                modules=tuple(
                    ModuleExtractionConfig(module_type=int(module_type), module_id=module_id, event_codes=event_codes)
                    for module_type, module_id, _, event_codes, _ in _MODULE_ROSTER
                ),
                kernel=None,
            )
        ]
    )
    config_path = output_directory.joinpath(EXTRACTION_CONFIGURATION_FILENAME)
    extraction_config.to_yaml(file_path=config_path)

    run_log_processing_pipeline(
        log_directory=staging_directory,
        output_directory=output_directory,
        config=config_path,
    )

    data_directory = output_directory.joinpath(OutputLayout.DIRECTORY_NAME)
    source_id = str(controller_id)

    # Fits the calibration of each valve whose calibration the caller supplied. A valve left without calibration is
    # skipped entirely, which leaves any previously written data for that valve untouched.
    left_valve_fit = (
        fit_valve_calibration(valve_calibration_data=left_valve_calibration_data)
        if left_valve_calibration_data is not None
        else None
    )
    right_valve_fit = (
        fit_valve_calibration(valve_calibration_data=right_valve_calibration_data)
        if right_valve_calibration_data is not None
        else None
    )

    written_files: dict[str, Path] = {}
    skipped_modules: list[str] = []

    for module_type, module_id, name, _, output_name in _MODULE_ROSTER:
        module_path = resolve_module_path(
            output_directory=data_directory,
            source_id=source_id,
            module_type=int(module_type),
            module_id=module_id,
        )

        # A module that reported no messages yields no extracted table, or an empty one. Recording it as skipped
        # keeps every other module addressed by its own identity rather than by its position in the result.
        if not module_path.exists():
            skipped_modules.append(f"{name} (no extracted data file)")
            continue

        partition = _read_module_partition(
            data_directory=data_directory,
            source_id=source_id,
            module_type=int(module_type),
            module_id=module_id,
        )
        if not partition:
            skipped_modules.append(f"{name} (reported no messages)")
            continue

        if module_type == ModuleTypeCodes.VALVE_MODULE:
            valve_fit = left_valve_fit if module_id == 1 else right_valve_fit

            # Volumes are only meaningful under the calibration the runtime used. Without it, the valve is left alone
            # rather than overwritten with volumes derived from unrelated constants.
            if valve_fit is None:
                skipped_modules.append(f"{name} (no calibration supplied, existing data left untouched)")
                continue

        output_file = output_directory.joinpath(output_name)
        if module_type == ModuleTypeCodes.VALVE_MODULE:
            scale_coefficient, nonlinearity_exponent = valve_fit
            _parse_valve_data(
                partition=partition,
                output_file=output_file,
                scale_coefficient=scale_coefficient,
                nonlinearity_exponent=nonlinearity_exponent,
            )
        elif module_type == ModuleTypeCodes.LICK_MODULE:
            _parse_lick_data(partition=partition, output_file=output_file, lick_threshold=lick_threshold)
        else:
            _parse_analog_data(partition=partition, output_file=output_file)

        written_files[name] = output_file
        console.echo(message=f"{name}: parsed into {output_file.name}.", level=LogLevel.SUCCESS)

    if skipped_modules:
        console.echo(
            message=f"The following modules reported no data and were skipped: {', '.join(skipped_modules)}.",
            level=LogLevel.WARNING,
        )

    console.echo(message=f"Log archive re-processing: complete. Output: {output_directory}.", level=LogLevel.SUCCESS)

    return written_files


if __name__ == "__main__":
    import sys

    if not console.enabled:
        console.enable()

    if len(sys.argv) != _CLI_ARGUMENT_COUNT:
        console.echo(
            message="usage: python reprocess_legacy_logs.py <raw_log_directory> <output_directory>",
            level=LogLevel.ERROR,
        )
        raise SystemExit(2)

    reprocess_log_archive(log_directory=Path(sys.argv[1]), output_directory=Path(sys.argv[2]))
