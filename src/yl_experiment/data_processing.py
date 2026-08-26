"""This module provides methods for processing the data acquired by the microcontroller at runtime."""

from typing import Any
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from .microcontroller import AMCInterface, ModuleTypeCodes, _SineStateCodes
from ataraxis_data_structures import DataLogger
from ataraxis_communication_interface import (
    EXTRACTION_CONFIGURATION_FILENAME,
    OutputLayout,
    ExtractionConfig,
    ModuleExtractionConfig,
    ControllerExtractionConfig,
    get_event_data,
    partition_events,
    resolve_module_path,
    get_event_timestamps,
    run_log_processing_pipeline,
)


def _read_module_partition(
    data_directory: Path, source_id: str, module_type: int, module_id: int
) -> dict[int, pl.DataFrame]:
    """Reads the extracted message file of the target hardware module and partitions it by event code.

    Args:
        data_directory: The path to the directory that stores the files written by the extraction pipeline.
        source_id: The identifier of the microcontroller that manages the module.
        module_type: The type (family) code of the hardware module.
        module_id: The unique identifier code of the hardware module.

    Returns:
        The sub-table of each event code the module message file holds, keyed by that event code.
    """
    module_path = resolve_module_path(
        output_directory=data_directory,
        source_id=source_id,
        module_type=module_type,
        module_id=module_id,
    )
    return partition_events(module_dataframe=pl.read_ipc(source=module_path, memory_map=True))


def _interpolate_data(
    timestamps: NDArray[np.uint64],
    data: NDArray[Any],
    seed_timestamps: NDArray[np.uint64],
    is_discrete: bool,
) -> NDArray[Any]:
    """Interpolates data values for the provided seed timestamps.

    Primarily, this service function is used to time-align different datastreams from the same source.

    Notes:
        This function expects seed_timestamps and timestamps arrays to be monotonically increasing.

        Discrete interpolated data is returned as an array with the same datatype as the input data. Continuous
        interpolated data is always returned as float_64 datatype.

        This function is specifically designed to work with Sun lab time data, which uses the unsigned integer format.

    Args:
        timestamps: The one-dimensional numpy array that stores the timestamps for the source data.
        data: The one-dimensional numpy array that stores the source datapoints.
        seed_timestamps: The one-dimensional numpy array that stores the timestamps for which to interpolate the data
            values.
        is_discrete: A boolean flag that determines whether the data is discrete or continuous.

    Returns:
        A numpy array with the same dimension as the seed_timestamps array that stores the interpolated data values.
    """
    # Discrete data
    if is_discrete:
        # Preallocates the output array
        interpolated_data = np.empty(seed_timestamps.shape, dtype=data.dtype)

        # Handles boundary conditions in bulk using boolean masks. All seed timestamps below the minimum source
        # timestamp are statically set to data[0], and all seed timestamps above the maximum source timestamp are set
        # to data[-1].
        below_min = seed_timestamps < timestamps[0]
        above_max = seed_timestamps > timestamps[-1]
        within_bounds = ~(below_min | above_max)  # The portion of the seed that is within the source timestamp boundary

        # Assigns out-of-bounds values in-bulk
        interpolated_data[below_min] = data[0]
        interpolated_data[above_max] = data[-1]

        # Processes within-boundary timestamps by finding the last known certain value to the left of each seed
        # timestamp and setting each seed timestamp to that value.
        if np.any(within_bounds):
            indices = np.searchsorted(timestamps, seed_timestamps[within_bounds], side="right") - 1
            interpolated_data[within_bounds] = data[indices]

        return interpolated_data

    # Continuous data. Note, due to interpolation, continuous data is always returned using float_64 datatype.
    return np.interp(seed_timestamps, timestamps, data)  # type: ignore[no-any-return]


def _parse_valve_data(
    partition: dict[int, pl.DataFrame],
    output_file: Path,
    scale_coefficient: np.float64,
    nonlinearity_exponent: np.float64,
) -> None:
    """Extracts and saves the data acquired by the ValveModule during runtime as a .feather file.

    Args:
        partition: The event-code-keyed partition of the messages logged by the module during runtime.
        output_file: The path to the output .feather file where to save the extracted data.
        scale_coefficient: Stores the scale coefficient used in the fitted power law equation that translates valve
            pulses into dispensed water volumes.
        nonlinearity_exponent: Stores the nonlinearity exponent used in the fitted power law equation that
            translates valve pulses into dispensed water volumes.

    Notes:
        Stores the duration of each valve pulse alongside the volume derived from it. The duration is the quantity the
        module actually logged, so it does not depend on the calibration used at parsing time. Keeping it makes the
        dispensed volumes recomputable under a different calibration without re-extracting the archive.
    """
    # This function looks for event-codes 51 (Valve Open) and event-codes 52 (Valve Closed). Both codes report state
    # transitions and carry no data payload, so only their timestamps are read.

    # The way this module is implemented guarantees there is at least one code 52 message, but there may be no code
    # 51 messages.
    open_timestamps = get_event_timestamps(partition=partition, event_code=51)
    closed_timestamps = get_event_timestamps(partition=partition, event_code=52)

    # If there were no valve open events, no water was dispensed. In this case, uses the first code 52 timestamp
    # to report a zero-volume reward and ends the runtime early.
    if open_timestamps.size == 0:
        module_dataframe = pl.DataFrame(
            {
                "time_us": closed_timestamps[:1],
                "pulse_duration_us": np.zeros(shape=closed_timestamps[:1].size, dtype=np.uint64),
                "dispensed_water_volume_uL": np.zeros(shape=closed_timestamps[:1].size, dtype=np.float64),
            }
        )
        module_dataframe.write_ipc(file=output_file, compression="uncompressed")
        return

    # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype. Although valve
    # trigger values are boolean, they are translated into the total volume of water, in microliters, dispensed to the
    # animal at each time-point and store that value as a float64.
    n_on = open_timestamps.size
    n_off = closed_timestamps.size
    total_length = n_on + n_off
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    volume: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

    # The water is dispensed gradually while the valve stays open. Therefore, the full reward volume is dispensed
    # when the valve goes from open to closed. Based on calibration data, uses a conversion factor to translate
    # the time the valve remains open into the fluid volume dispensed to the animal, which is then used to convert each
    # Open/Close cycle duration into the dispensed volume.

    # Extracts Open (Code 51) trigger codes. Statically assigns the value '1' to denote Open signals.
    timestamps[:n_on] = open_timestamps
    volume[:n_on] = 1  # Open state

    # Extracts Closed (Code 52) trigger codes.
    timestamps[n_on:] = closed_timestamps
    volume[n_on:] = 0  # Closed state

    # Sorts both arrays based on timestamps.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    volume = volume[sort_indices]

    # Find falling and rising edges. Falling edges are valve-closing events, rising edges are valve-opening events.
    edges = np.diff(volume, prepend=volume[0])
    rising_edges = np.where(edges == 1)[0]
    falling_edges = np.where(edges == -1)[0]

    # Samples the timestamp array to only include timestamps for the falling edges. That is, when the valve has
    # finished delivering water
    reward_timestamps = timestamps[falling_edges]

    # Calculates pulse durations in microseconds for each open-close cycle. Since the original timestamp array
    # contains alternating HIGH / LOW edges, each falling edge has to match to a rising edge.
    pulse_durations_us: NDArray[np.uint64] = timestamps[falling_edges] - timestamps[rising_edges]
    pulse_durations: NDArray[np.float64] = pulse_durations_us.astype(np.float64)

    # Converts the time the Valve stayed open into the dispensed water volume, in microliters.
    # noinspection PyTypeChecker
    volumes = np.cumsum(scale_coefficient * np.power(pulse_durations, nonlinearity_exponent))
    volumes = np.round(volumes, decimals=8)

    # The processing logic above removes the initial water volume of 0. This re-adds the initial volume using the
    # first timestamp of the module data. That timestamp communicates the initial valve state, which should be 0.
    reward_timestamps = np.insert(reward_timestamps, 0, timestamps[0])
    volumes = np.insert(volumes, 0, 0.0)
    pulse_durations_us = np.insert(pulse_durations_us, 0, np.uint64(0))

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": reward_timestamps,
            "pulse_duration_us": pulse_durations_us,
            "dispensed_water_volume_uL": volumes,
        }
    )

    # Saves extracted data using Feather format and no compression to support memory-mapping the file during processing.
    module_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_lick_data(partition: dict[int, pl.DataFrame], output_file: Path, lick_threshold: np.uint16) -> None:
    """Extracts and saves the data acquired by the LickModule during runtime as a .feather file.

    Args:
        partition: The event-code-keyed partition of the messages logged by the module during runtime.
        output_file: The path to the output .feather file where to save the extracted data.
        lick_threshold: The voltage threshold for detecting the interaction with the sensor as a lick.

    Notes:
        The extraction classifies lick events based on the lick threshold used during runtime. The
        time-difference between consecutive ON and OFF event edges corresponds to the time, in microseconds, the
        tongue maintained contact with the lick tube. This may include both the time the tongue physically
        touched the tube and the time there was a conductive fluid bridge between the tongue and the lick tube.

        In addition to classifying the licks and providing binary lick state data, the extraction preserves the raw
        12-bit ADC voltages associated with each lick. This way, it is possible to spot issues with the lick detection
        system by applying a different lick threshold from the one used at runtime, potentially augmenting data
        analysis.
    """
    # LickModule only sends messages with code 51 (Voltage level changed). Therefore, this extraction pipeline has
    # to apply the threshold filter, similar to how the real-time processing method.

    # Unlike the other parsing methods, this one will always work as expected since it only deals with one code and
    # that code is guaranteed to be received for each runtime.

    # Extract timestamps and voltage levels. Timestamps use uint64 datatype. Lick sensor
    # voltage levels come in as uint16, but they are later used to generate a binary uint8 lick classification mask.
    timestamps, voltages = get_event_data(partition=partition, event_code=51, values_dtype=np.uint16)

    # Sorts all arrays by timestamp. This is technically not needed as the extracted values are already sorted by
    # timestamp, but this is still done for additional safety.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    voltages = voltages[sort_indices]

    # Creates a lick binary classification column based on the class threshold. Note, the threshold is inclusive.
    licks = (voltages >= lick_threshold).astype(np.uint8)

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": timestamps,
            "voltage_12_bit_adc": voltages,
            "lick_state": licks,
        }
    )

    # Saves extracted data using Feather format and no compression to support memory-mapping the file during processing.
    module_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_analog_data(partition: dict[int, pl.DataFrame], output_file: Path) -> None:
    """Extracts and saves the data acquired by the AnalogModule during runtime as a .feather file. Essentially the same
       as the lick data extraction, but without applying any thresholding.

    Args:
        partition: The event-code-keyed partition of the messages logged by the module during runtime.
        output_file: The path to the output .feather file where to save the extracted data.

    Notes:
        This module is used to record the timestamps of continuous analog signals, so the photometry data can be
        time-aligned and counter the internal drift of doric console timestamps. The extraction preserves the raw
        12-bit ADC voltages associated with each analog signal sample.

        The AnalogModule is no longer part of the runtime, having been replaced by the SineModule, which generates the
        synchronization signal instead of sampling it. This parser is retained for reprocessing the archives recorded
        before that change and is called by reprocess_legacy_logs, not by process_microcontroller_log.
    """
    timestamps, voltages = get_event_data(partition=partition, event_code=51, values_dtype=np.uint16)

    # Sorts all arrays by timestamp. This is technically not needed as the extracted values are already sorted by
    # timestamp, but this is still done for additional safety.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    voltages = voltages[sort_indices]

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": timestamps,
            "voltage_12_bit_adc": voltages,
        }
    )

    module_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_sine_data(partition: dict[int, pl.DataFrame], output_file: Path) -> None:
    """Extracts and saves the data generated by the SineModule during runtime as a .feather file.

    Args:
        partition: The event-code-keyed partition of the messages logged by the module during runtime.
        output_file: The path to the output .feather file where to save the extracted data.

    Notes:
        This module generates the analog synchronization signal, rather than sampling one. Each extracted row is
        therefore the DAC value the module commanded at that time, which makes the output the ground truth for
        time-aligning the photometry recording of the same waveform.

        Stop events are recorded as samples of value 0, since stopping the wave drives the DAC to 0. The wave_active
        column separates them from the commanded samples of an active wave.
    """
    # Sample Set (51) carries the 12-bit DAC value written for that sample. Stopped (52) reports a state transition
    # and carries no data payload, so only its timestamp is available.
    sample_code = int(_SineStateCodes.SAMPLE_SET)
    stop_code = int(_SineStateCodes.STOPPED)

    # A runtime that was terminated before the wave was ever started reports no samples at all, so the partition is
    # checked rather than read blindly.
    if sample_code in partition:
        sample_timestamps, dac_values = get_event_data(
            partition=partition, event_code=sample_code, values_dtype=np.uint16
        )
    else:
        sample_timestamps = np.empty(shape=0, dtype=np.uint64)
        dac_values = np.empty(shape=0, dtype=np.uint16)
    stop_timestamps = get_event_timestamps(partition=partition, event_code=stop_code)

    timestamps: NDArray[np.uint64] = np.concatenate((sample_timestamps, stop_timestamps)).astype(np.uint64)
    values: NDArray[np.uint16] = np.concatenate(
        (dac_values, np.zeros(shape=stop_timestamps.size, dtype=np.uint16))
    ).astype(np.uint16)
    wave_active: NDArray[np.bool_] = np.concatenate(
        (
            np.ones(shape=sample_timestamps.size, dtype=np.bool_),
            np.zeros(shape=stop_timestamps.size, dtype=np.bool_),
        )
    )

    # Sorts by timestamp, as the two event codes are read separately. Uses a stable sort so that a stop event sharing
    # a timestamp with a sample stays after that sample, which is the order in which the module emitted them.
    sort_indices = np.argsort(timestamps, kind="stable")

    module_dataframe = pl.DataFrame(
        {
            "time_us": timestamps[sort_indices],
            "dac_12_bit_value": values[sort_indices],
            "wave_active": wave_active[sort_indices],
        }
    )

    module_dataframe.write_ipc(file=output_file, compression="uncompressed")


def process_microcontroller_log(data_logger: DataLogger, microcontroller: AMCInterface, output_directory: Path) -> None:
    """Extracts the data recorded by all hardware modules from the .npz log archives generated by the DataLogger
    instance for the target microcontroller and saves it as .feather files.

    Notes:
        Writes the extraction configuration and the intermediate message tables produced by the extraction pipeline
        into the output directory alongside the parsed module files.

    Notes:
        This function should be called at the end of each runtime to process the logged data.

    Args:
        data_logger: The DataLogger instance used at runtime to log the microcontroller data.
        microcontroller: The AMCInterface instance used at runtime to communicate with the microcontroller.
        output_directory: The path to the directory where to save the extracted .feather files.

    """
    # Declares the hardware modules used at runtime and the event codes to extract for each of them. Valve modules
    # report Open (51), Closed (52) and Calibrated (53) states, lick modules report a single voltage-readout code (51),
    # and the sine module reports Sample Set (51) and Stopped (52).
    module_configurations = (
        ModuleExtractionConfig(
            module_type=int(ModuleTypeCodes.VALVE_MODULE), module_id=1, event_codes=(51, 52, 53)
        ),  # Left valve
        ModuleExtractionConfig(
            module_type=int(ModuleTypeCodes.VALVE_MODULE), module_id=2, event_codes=(51, 52, 53)
        ),  # Right valve
        ModuleExtractionConfig(module_type=int(ModuleTypeCodes.LICK_MODULE), module_id=1, event_codes=(51,)),
        ModuleExtractionConfig(module_type=int(ModuleTypeCodes.LICK_MODULE), module_id=2, event_codes=(51,)),
        ModuleExtractionConfig(module_type=int(ModuleTypeCodes.SINE_MODULE), module_id=1, event_codes=(51, 52)),
    )

    # The extraction pipeline is configuration-driven, so the targets above are written out as a .yaml file that the
    # pipeline then consumes. The controller and its modules are matched against the microcontroller manifest the
    # MicroControllerInterface writes into the log directory during initialization.
    extraction_config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=microcontroller.controller_id,
                modules=module_configurations,
                kernel=None,
            )
        ]
    )
    config_path = output_directory.joinpath(EXTRACTION_CONFIGURATION_FILENAME)
    extraction_config.to_yaml(file_path=config_path)

    # Extracts the messages of every configured module into feather files stored under the pipeline's output
    # subdirectory.
    run_log_processing_pipeline(
        log_directory=data_logger.output_directory,
        output_directory=output_directory,
        config=config_path,
    )

    # Reads back the extracted message tables. Each table is partitioned by event code before parsing.
    data_directory = output_directory.joinpath(OutputLayout.DIRECTORY_NAME)
    source_id = str(microcontroller.controller_id)

    # Parses the extracted data for each module and saves the output as .feather files in the requested directory:

    # Left Valve
    _parse_valve_data(
        partition=_read_module_partition(
            data_directory=data_directory,
            source_id=source_id,
            module_type=int(ModuleTypeCodes.VALVE_MODULE),
            module_id=1,
        ),
        output_file=output_directory / "left_valve_data.feather",
        scale_coefficient=microcontroller.left_valve.scale_coefficient,
        nonlinearity_exponent=microcontroller.left_valve.nonlinearity_exponent,
    )

    # Right Valve
    _parse_valve_data(
        partition=_read_module_partition(
            data_directory=data_directory,
            source_id=source_id,
            module_type=int(ModuleTypeCodes.VALVE_MODULE),
            module_id=2,
        ),
        output_file=output_directory / "right_valve_data.feather",
        scale_coefficient=microcontroller.right_valve.scale_coefficient,
        nonlinearity_exponent=microcontroller.right_valve.nonlinearity_exponent,
    )

    # Left Lick Sensor
    _parse_lick_data(
        partition=_read_module_partition(
            data_directory=data_directory,
            source_id=source_id,
            module_type=int(ModuleTypeCodes.LICK_MODULE),
            module_id=1,
        ),
        output_file=output_directory / "left_lick_sensor.feather",
        lick_threshold=microcontroller.left_lick_sensor.lick_threshold,
    )

    # Right Lick Sensor
    _parse_lick_data(
        partition=_read_module_partition(
            data_directory=data_directory,
            source_id=source_id,
            module_type=int(ModuleTypeCodes.LICK_MODULE),
            module_id=2,
        ),
        output_file=output_directory / "right_lick_sensor.feather",
        lick_threshold=microcontroller.right_lick_sensor.lick_threshold,
    )

    # Sine Wave Generator
    _parse_sine_data(
        partition=_read_module_partition(
            data_directory=data_directory,
            source_id=source_id,
            module_type=int(ModuleTypeCodes.SINE_MODULE),
            module_id=1,
        ),
        output_file=output_directory / "sine_wave.feather",
    )
