"""This module contains the functions used by the general data processing pipeline which runs on the BioHPC server.
Since some processing methods rely on custom code exposed by each AMC hardware module interface, it makes more sense to
implement these methods as part of this library. These methods are not used during data acquisition or preprocessing!
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import polars as pl
from numpy.typing import NDArray
from numpy.lib.npyio import NpzFile
from ataraxis_video_system import extract_logged_video_system_data
from ataraxis_base_utilities import ensure_directory_exists

from .module_interfaces import (
    TTLInterface,
    LickInterface,
    BreakInterface,
    ValveInterface,
    ScreenInterface,
    TorqueInterface,
    EncoderInterface,
)


def _process_camera_timestamps(log_path: Path, output_path: Path) -> None:
    """Reads the log .npz archive specified by the log_path and extracts the camera frame timestamps
    as a Polars Series saved to the output_path as a Feather file.

    Args:
        log_path: Path to the .npz log archive to be parsed.
        output_path: Path to save the output Polars Series as a Feather file.
    """
    # Extracts timestamp data from log archive
    timestamp_data = extract_logged_video_system_data(log_path)

    # Converts extracted data to Polars series.
    timestamps_series = pl.Series(name="frame_time_us", values=timestamp_data)

    # Saves extracted data using Feather format and 'lz4' compression. Lz4 allows optimizing processing time and
    # file size. These extracted files are temporary and will be removed during later processing steps.
    timestamps_series.to_frame().write_ipc(file=output_path, compression="lz4")


def _process_experiment_data(log_path: Path, output_directory: Path, cue_map: dict[int, float]) -> None:
    """Extracts the VR states, Experiment states, and the Virtual Reality cue sequence from the log generated
    by the MesoscopeExperiment instance during runtime and saves the extracted data as Polars DataFrame .feather files.

    This extraction method functions similar to camera log extraction and hardware module log extraction methods. The
    key difference is the wall cue sequence extraction, which does not have a timestamp column. Instead, it has a
    distance colum which stores the distance the animal has to run, in centimeters, to reach each cue in the sequence.
    It is expected that during data processing, the distance data will be used to align the cues to the distance ran by
    the animal during the experiment.
    """
    # Loads the archive into RAM
    archive: NpzFile = np.load(file=log_path)

    # Precreates the variables used to store extracted data
    vr_states = []
    vr_timestamps = []
    experiment_states = []
    experiment_timestamps = []
    cue_sequence: NDArray[np.uint8] = np.zeros(shape=0, dtype=np.uint8)

    # Locates the logging onset timestamp. The onset is used to convert the timestamps for logged data into absolute
    # UTC timestamps. Originally, all timestamps other than onset are stored as elapsed time in microseconds
    # relative to the onset timestamp.
    timestamp_offset = 0
    onset_us = np.uint64(0)
    timestamp: np.uint64
    for number, item in enumerate(archive.files):
        message: NDArray[np.uint8] = archive[item]  # Extracts message payload from the compressed .npy file

        # Recovers the uint64 timestamp value from each message. The timestamp occupies 8 bytes of each logged
        # message starting at index 1. If timestamp value is 0, the message contains the onset timestamp value
        # stored as 8-byte payload. Index 0 stores the source ID (uint8 value)
        if np.uint64(message[1:9].view(np.uint64)[0]) == 0:
            # Extracts the byte-serialized UTC timestamp stored as microseconds since epoch onset.
            onset_us = np.uint64(message[9:].view("<i8")[0].copy())

            # Breaks the loop onc the onset is found. Generally, the onset is expected to be found very early into
            # the loop
            timestamp_offset = number  # Records the item number at which the onset value was found.
            break

    # Once the onset has been discovered, loops over all remaining messages and extracts data stored in these
    # messages.
    for item in archive.files[timestamp_offset + 1 :]:
        message = archive[item]

        # Extracts the elapsed microseconds since timestamp and uses it to calculate the global timestamp for the
        # message, in microseconds since epoch onset.
        elapsed_microseconds = np.uint64(message[1:9].view(np.uint64)[0].copy())
        timestamp = onset_us + elapsed_microseconds

        payload = message[9:]  # Extracts the payload from the message

        # If the message is longer than 500 bytes, it is a sequence of wall cues. It is very unlikely that we
        # will log any other data with this length, so it is a safe heuristic to use.
        if len(payload) > 500:
            cue_sequence = payload.view(np.uint8).copy()  # Keeps the original numpy uint8 format

        # If the message has a length of 2 bytes and the first element is 1, the message communicates the VR state
        # code.
        elif len(payload) == 2 and payload[0] == 1:
            vr_state = np.uint8(payload[1])  # Extracts the VR state code from the second byte of the message.
            vr_states.append(vr_state)
            vr_timestamps.append(timestamp)

        # Otherwise, if the starting code is 2, the message communicates the experiment state code.
        elif len(payload) == 2 and payload[0] == 2:
            # Extracts the experiment state code from the second byte of the message.
            experiment_state = np.uint8(payload[1])
            experiment_states.append(experiment_state)
            experiment_timestamps.append(timestamp)

    # Closes the archive to free up memory
    archive.close()

    # Uses the cue_map dictionary to compute the length of each cue in the sequence. Then computes the cumulative
    # distance the animal needs to travel to reach each cue in the sequence. The first cue is associated with distance
    # of 0 (the animal starts at this cue), the distance to each following cue is the sum of all previous cue lengths.
    distance_sequence = np.zeros(len(cue_sequence), dtype=np.float64)
    distance_sequence[1:] = np.cumsum([cue_map[int(code)] for code in cue_sequence[:-1]], dtype=np.float64)

    # Converts extracted data into Polar Feather files:
    vr_dataframe = pl.DataFrame(
        {
            "time_us": vr_timestamps,
            "vr_state": vr_states,
        }
    )
    exp_dataframe = pl.DataFrame(
        {
            "time_us": experiment_timestamps,
            "experiment_state": experiment_states,
        }
    )
    cue_dataframe = pl.DataFrame(
        {
            "vr_cue": cue_sequence,
            "traveled_distance_cm": distance_sequence,
        }
    )

    # Saves the DataFrames to Feather file with lz4 compression
    vr_dataframe.write_ipc(output_directory.joinpath("vr_data.feather"), compression="lz4")
    exp_dataframe.write_ipc(output_directory.joinpath("experiment_data.feather"), compression="lz4")
    cue_dataframe.write_ipc(output_directory.joinpath("cue_data.feather"), compression="lz4")


def process_log_directory(data_directory: Path, verbose: bool = False) -> None:
    """Reads the compressed .npz log files stored in the input directory and extracts all camera frame timestamps and
    relevant behavior data stored in log files.

    This function is intended to run on the BioHPC server as part of the data processing pipeline. It is optimized to
    process all log files in parallel and extract the data stored inside the files into behavior_data directory and
    camera_frames directory.

    Notes:
        This function makes certain assumptions about the layout of the raw-data directory to work as expected.

    Args:
        data_directory: The Path to the target session raw_data directory to be processed.
        verbose: Determines whether this function should run in the verbose mode.
    """
    # Resolves the paths to the specific directories used during processing
    log_directory = data_directory.joinpath("behavior_data_log")  # Should exist inside the raw data directory
    camera_frame_directory = data_directory.joinpath("camera_frames")  # Should exist inside the raw data directory
    behavior_data_directory = data_directory.joinpath("behavior_data")
    ensure_directory_exists(behavior_data_directory)  # Generates the directory

    # Should exist inside the raw data directory
    hardware_configuration_path = data_directory.joinpath("hardware_configuration.yaml")

    # Finds all .npz log files inside the input log file directory. Assumes there are no uncompressed log files.
    compressed_files: list[Path] = [file for file in log_directory.glob("*.npz")]

    # Loads the input HardwareConfiguration file to read the hardware parameters necessary to parse the data
    hardware_configuration: RuntimeHardwareConfiguration = RuntimeHardwareConfiguration.from_yaml(  # type: ignore
        file_path=hardware_configuration_path,
    )

    # Otherwise, iterates over all compressed log files and processes them in-parallel
    with ProcessPoolExecutor() as executor:
        futures = set()
        for file in compressed_files:
            # MesoscopeExperiment log file
            if file.stem == "1_log" and hardware_configuration.cue_map is not None:
                futures.add(
                    executor.submit(
                        _process_experiment_data,
                        file,
                        behavior_data_directory,
                        hardware_configuration.cue_map,
                    )
                )

            # Face Camera timestamps
            if file.stem == "51_log":
                futures.add(
                    executor.submit(
                        _process_camera_timestamps,
                        file,
                        camera_frame_directory.joinpath("face_camera_timestamps.feather"),
                    )
                )

            # Left Camera timestamps
            if file.stem == "62_log":
                futures.add(
                    executor.submit(
                        _process_camera_timestamps,
                        file,
                        camera_frame_directory.joinpath("left_camera_timestamps.feather"),
                    )
                )

            # Right Camera timestamps
            if file.stem == "73_log":
                futures.add(
                    executor.submit(
                        _process_camera_timestamps,
                        file,
                        camera_frame_directory.joinpath("right_camera_timestamps.feather"),
                    )
                )

            # Actor AMC module data
            if file.stem == "101_log":
                # Break
                if (
                    hardware_configuration.minimum_break_strength is not None
                    and hardware_configuration.maximum_break_strength is not None
                ):
                    futures.add(
                        executor.submit(
                            BreakInterface.parse_logged_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.minimum_break_strength,
                            hardware_configuration.maximum_break_strength,
                        )
                    )

                # Valve
                if (
                    hardware_configuration.nonlinearity_exponent is not None
                    and hardware_configuration.scale_coefficient is not None
                ):
                    futures.add(
                        executor.submit(
                            ValveInterface.parse_logged_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.scale_coefficient,
                            hardware_configuration.nonlinearity_exponent,
                        )
                    )

                # Screens
                if hardware_configuration.initially_on is not None:
                    futures.add(
                        executor.submit(
                            ScreenInterface.parse_logged_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.initially_on,
                        )
                    )

            # Sensor AMC module data
            if file.stem == "152_log":
                # Lick Sensor
                if hardware_configuration.lick_threshold is not None:
                    futures.add(
                        executor.submit(
                            LickInterface.parse_logged_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.lick_threshold,
                        )
                    )

                # Torque Sensor
                if hardware_configuration.torque_per_adc_unit is not None:
                    futures.add(
                        executor.submit(
                            TorqueInterface.parse_logged_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.torque_per_adc_unit,
                        )
                    )

                # Mesoscope Frame TTL module
                if hardware_configuration.has_ttl:
                    futures.add(executor.submit(TTLInterface.parse_logged_data, file, behavior_data_directory))

            # Encoder AMC module data
            if file.stem == "203_log":
                # Encoder
                if hardware_configuration.cm_per_pulse is not None:
                    futures.add(
                        executor.submit(
                            EncoderInterface.parse_logged_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.cm_per_pulse,
                        )
                    )

        # Displays a progress bar to track the parsing status if the function is called in the verbose mode.
        if verbose:
            with tqdm(
                total=len(futures),
                desc=f"Parsing log sources",
                unit="source",
            ) as pbar:
                for future in as_completed(futures):
                    # Propagates any exceptions from the transfers
                    future.result()
                    pbar.update(1)
        else:
            for future in as_completed(futures):
                # Propagates any exceptions from the transfers
                future.result()
