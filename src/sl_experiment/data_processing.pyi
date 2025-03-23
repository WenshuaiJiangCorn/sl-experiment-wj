from pathlib import Path

from numpy.typing import NDArray as NDArray

from .module_interfaces import (
    TTLInterface as TTLInterface,
    LickInterface as LickInterface,
    BreakInterface as BreakInterface,
    ValveInterface as ValveInterface,
    ScreenInterface as ScreenInterface,
    TorqueInterface as TorqueInterface,
    EncoderInterface as EncoderInterface,
)

def _process_camera_timestamps(log_path: Path, output_path: Path) -> None:
    """Reads the log .npz archive specified by the log_path and extracts the camera frame timestamps
    as a Polars Series saved to the output_path as a Feather file.

    Args:
        log_path: Path to the .npz log archive to be parsed.
        output_path: Path to save the output Polars Series as a Feather file.
    """

def _process_experiment_data(log_path: Path, output_directory: Path, cue_map: dict[int, float]) -> None:
    """Extracts the VR states, Experiment states, and the Virtual Reality cue sequence from the log generated
    by the MesoscopeExperiment instance during runtime and saves the extracted data as Polars DataFrame .feather files.

    This extraction method functions similar to camera log extraction and hardware module log extraction methods. The
    key difference is the wall cue sequence extraction, which does not have a timestamp column. Instead, it has a
    distance colum which stores the distance the animal has to run, in centimeters, to reach each cue in the sequence.
    It is expected that during data processing, the distance data will be used to align the cues to the distance ran by
    the animal during the experiment.
    """

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
