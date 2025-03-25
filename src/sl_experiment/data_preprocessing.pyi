from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray as NDArray
from ataraxis_data_structures import YamlConfig

from .transfer_tools import transfer_directory as transfer_directory
from .packaging_tools import calculate_directory_checksum as calculate_directory_checksum
from .google_sheet_tools import (
    SurgeryData as SurgeryData,
    SurgerySheet as SurgerySheet,
    WaterSheetData as WaterSheetData,
)

@dataclass()
class RuntimeHardwareConfiguration(YamlConfig):
    """This class is used to save the runtime hardware configuration information as a .yaml file.

    This information is used to read the data saved to the .npz log files during runtime during data processing.

    Notes:
        All fields in this dataclass initialize to None. During log processing, any log associated with a hardware
        module that provides the data stored in a field will be processed, unless that field is None. Therefore, setting
        any field in this dataclass to None also functions as a flag for whether to parse the log associated with the
        module that provides this field's information.

        This class is automatically configured by MesoscopeExperiment and BehaviorTraining classes to facilitate log
        parsing.
    """

    cue_map: dict[int, float] | None = ...
    cm_per_pulse: float | None = ...
    maximum_break_strength: float | None = ...
    minimum_break_strength: float | None = ...
    lick_threshold: int | None = ...
    scale_coefficient: float | None = ...
    nonlinearity_exponent: float | None = ...
    torque_per_adc_unit: float | None = ...
    initially_on: bool | None = ...
    has_ttl: bool | None = ...

@dataclass
class SessionData(YamlConfig):
    """Provides methods for managing the data acquired during one experiment or training session.

    This class functions as the central hub for collecting the data from all local PCs involved in the data acquisition
    process and pushing it to the NAS and the BioHPC server. Its primary purpose is to maintain the session data
    structure across all supported destinations and to efficiently and safely move the data to these destinations with
    minimal redundancy and footprint. Additionally, this class generates the paths used by all other classes from
    this library to determine where to load and saved various data during runtime. Finally, it also carries out basic
    data preprocessing to optimize raw data for network transmission and long-term storage.

    As part of its initialization, the class generates the session directory for the input animal and project
    combination. Session directories use the current UTC timestamp, down to microseconds, as the directory name. This
    ensures that each session name is unique and preserves the overall session order.

    Notes:
        Do not call methods from this class directly. This class is intended to be used primarily through the runtime
        logic functions from the experiment.py module and general command-line-interfaces installed with the library.
        The only reason the class is defined as public is to support reconfiguring data destinations and session details
        when implementing custom CLI functions for projects that use this library.

        It is expected that the server, NAS, and mesoscope data directories are mounted on the host-machine via the
        SMB or equivalent protocol. All manipulations with these destinations are carried out with the assumption that
        the OS has full access to these directories and filesystems.

        This class is specifically designed for working with raw data from a single animal participating in a single
        experimental project session. Processed data is managed by the processing library methods and classes.

        This class generates an xxHash-128 checksum stored inside the ax_checksum.txt file at the root of each
        experimental session 'raw_data' directory. The checksum verifies the data of each file and the paths to each
        file relative to the 'raw_data' root directory.
    """

    project_name: str
    animal_id: str
    surgery_sheet_id: str
    water_log_sheet_id: str
    session_type: str
    credentials_path: str = ...
    local_root_directory: str = ...
    server_root_directory: str = ...
    nas_root_directory: str = ...
    mesoscope_root_directory: str = ...
    session_name: str = ...
    def __post_init__(self) -> None:
        """Generates the session name and creates the session directory structure on all involved PCs."""
    @classmethod
    def from_path(cls, path: Path) -> SessionData:
        """Initializes a SessionData instance to represent the data of an already existing session.

        Typically, this initialization mode is used to preprocess an interrupted session. This method uses the cached
        data stored in the 'session_data.yaml' file in the 'raw_data' subdirectory of the provided session directory.

        Args:
            path: The path to the session directory on the local (VRPC) machine.

        Returns:
            An initialized SessionData instance for the session whose data is stored at the provided path.

        Raises:
            FileNotFoundError: If the 'session_data.yaml' file is not found after resolving the provided path.
        """
    def to_path(self) -> None:
        """Saves the data of the instance to the 'raw_data' directory of the managed session as a 'session_data.yaml'
        file.

        This is used to save the data stored in the instance to disk, so that it can be reused during preprocessing or
        data processing. This also serves as the repository for the identification information about the project,
        animal, and session that generated the data.
        """
    @property
    def raw_data_path(self) -> Path:
        """Returns the path to the 'raw_data' directory of the managed session on the VRPC.

        This directory functions as the root directory that stores all raw data acquired during training or experiment
        runtime for a given session.
        """
    @property
    def camera_frames_path(self) -> Path:
        """Returns the path to the 'camera_frames' directory of the managed session.

        This subdirectory is stored under the 'raw_data' directory and aggregates all video camera data.
        """
    @property
    def zaber_positions_path(self) -> Path:
        """Returns the path to the 'zaber_positions.yaml' file of the managed session.

        This path is used to save the positions for all Zaber motors of the HeadBar and LickPort controllers at the
        end of the experimental session.
        """
    @property
    def session_descriptor_path(self) -> Path:
        """Returns the path to the 'session_descriptor.yaml' file of the managed session.

        This path is used to save important session information to be viewed by experimenters post-runtime and to use
        for further processing.
        """
    @property
    def hardware_configuration_path(self) -> Path:
        """Returns the path to the 'hardware_configuration.yaml' file of the managed session.

        This file stores hardware module parameters used to read and parse .npz log files during data processing.
        """
    @property
    def previous_zaber_positions_path(self) -> Path:
        """Returns the path to the 'zaber_positions.yaml' file of the previous session.

        The file is stored inside the 'persistent_data' directory of the managed animal.
        """
    @property
    def mesoscope_root_path(self) -> Path:
        """Returns the path to the root directory of the Mesoscope pc (ScanImagePC) used to store all
        mesoscope-acquired data.
        """
    @property
    def nas_root_path(self) -> Path:
        """Returns the path to the root directory of the Synology NAS (Network Attached Storage) used to store all
        training and experiment data after preprocessing (backup cold long-term storage)."""
    @property
    def server_root_path(self) -> Path:
        """Returns the path to the root directory of the BioHPC server used to process and store all training and e
        experiment data (main long-term storage)."""
    @property
    def mesoscope_persistent_path(self) -> Path:
        """Returns the path to the 'persistent_data' directory of the Mesoscope pc (ScanImagePC).

        This directory is primarily used to store the reference MotionEstimator.me files for each animal.
        """
    @property
    def local_metadata_path(self) -> Path:
        """Returns the path to the 'metadata' directory of the managed animal on the VRPC."""
    @property
    def server_metadata_path(self) -> Path:
        """Returns the path to the 'metadata' directory of the managed animal on the BioHPC server."""
    @property
    def nas_metadata_path(self) -> Path:
        """Returns the path to the 'metadata' directory of the managed animal on the Synology NAS."""
    def preprocess_session_data(self) -> None:
        """Carries out all data preprocessing tasks to prepare the data for NAS / BioHPC server transfer and future
        processing.

        This method should be called at the end of each training and experiment runtime to compress and safely transfer
        the data to its long-term storage destinations.

        Notes:
            The method will NOT delete the data from the VRPC or ScanImagePC. To safely remove the data, use the
            purge-redundant-data CLI command. The data will only be removed if it has been marked for removal by our
            data management algorithms, which ensure we have enough spare copies of the data elsewhere.
        """

@dataclass()
class LickTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to lick training sessions as a .yaml file."""

    experimenter: str
    mouse_weight_g: float
    dispensed_water_volume_ml: float = ...
    average_reward_delay_s: int = ...
    maximum_deviation_from_average_s: int = ...
    maximum_water_volume_ml: float = ...
    maximum_training_time_m: int = ...
    experimenter_notes: str = ...
    experimenter_given_water_volume_ml: float = ...

@dataclass()
class RunTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to run training sessions as a .yaml file."""

    experimenter: str
    mouse_weight_g: float
    dispensed_water_volume_ml: float = ...
    final_running_speed_cm_s: float = ...
    final_speed_duration_s: float = ...
    initial_running_speed_cm_s: float = ...
    initial_speed_duration_s: float = ...
    increase_threshold_ml: float = ...
    increase_running_speed_cm_s: float = ...
    increase_speed_duration_s: float = ...
    maximum_running_speed_cm_s: float = ...
    maximum_speed_duration_s: float = ...
    maximum_water_volume_ml: float = ...
    maximum_training_time_m: int = ...
    experimenter_notes: str = ...
    experimenter_given_water_volume_ml: float = ...

@dataclass()
class MesoscopeExperimentDescriptor(YamlConfig):
    """This class is used to save the description information specific to experiment sessions as a .yaml file."""

    experimenter: str
    mouse_weight_g: float
    dispensed_water_volume_ml: float = ...
    experimenter_notes: str = ...
    experimenter_given_water_volume_ml: float = ...

def _delete_directory(directory_path: Path) -> None:
    """Removes the input directory and all its subdirectories using parallel processing.

    This function outperforms default approaches like subprocess call with rm -rf and shutil rmtree for directories with
    a comparably small number of large files. For example, this is the case for the mesoscope frame directories, which
    are deleted ~6 times faster with this method over sh.rmtree. Potentially, it may also outperform these approaches
    for all comparatively shallow directories.

    Args:
        directory_path: The path to the directory to delete.
    """

def _get_stack_number(tiff_path: Path) -> int | None:
    """A helper function that determines the number of mesoscope-acquired tiff stacks using its file name.

    This is used to sort all TIFF stacks in a directory before recompressing them with LERC scheme. Like
    other helpers, this helper is also used to identify and remove non-mesoscope TIFFs from the dataset.
    """

def _check_stack_size(file: Path) -> int:
    """Reads the header of the input TIFF file, and if the file is a stack, extracts its size.

    This function is used to both determine the stack size of the processed TIFF files and to exclude non-mesoscope
    TIFFs from processing.

    Notes:
        This function only works with monochrome TIFF stacks generated by the mesoscope. It expects each TIFF file to
        be a stack of 2D frames.

    Args:
        file: The path to the TIFF file to evaluate.

    Returns:
        If the file is a stack, returns the number of frames (pages) in the stack. Otherwise, returns 0 to indicate that
        the file is not a stack.
    """

def _process_stack(
    tiff_path: Path, first_frame_number: int, output_dir: Path, verify_integrity: bool, batch_size: int = 250
) -> dict[str, Any]:
    """Reads a TIFF stack, extracts its frame-variant ScanImage data, and saves it as a LERC-compressed stacked TIFF
    file.

    This is a worker function called by the process_mesoscope_directory in-parallel for each stack inside each
    processed directory. It re-compresses the input TIFF stack using LERC-compression and extracts the frame-variant
    ScanImage metadata for each frame inside the stack. Optionally, the function can be configured to verify data
    integrity after compression.

    Notes:
        This function can reserve up to double the processed stack size of RAM bytes to hold the data in memory. If the
        host-computer does not have enough RAM, reduce the number of concurrent processes, reduce the batch size, or
        disable verification.

    Raises:
        RuntimeError: If any extracted frame does not match the original frame stored inside the TIFF stack.
        NotImplementedError: If extracted frame-variant metadata contains unexpected keys or expected keys for which
            we do not have a custom extraction implementation.

    Args:
        tiff_path: The path to the TIFF stack to process.
        first_frame_number: The position (number) of the first frame stored in the stack, relative to the overall
            sequence of frames acquired during the experiment. This is used to configure the output file name to include
            the range of frames stored in the stack.
        output_dir: The path to the directory where to save the processed stacks.
        verify_integrity: Determines whether to verify the integrity of compressed data against the source data.
            The conversion does not alter the source data, so it is usually safe to disable this option, as the chance
            of compromising the data is negligible. Note, enabling this function doubles the RAM usage for each worker
            process.
        batch_size: The number of frames to process at the same time. This directly determines the RAM footprint of
            this function, as frames are kept in RAM during compression. Note, verification doubles the RAM footprint,
            as it requires both compressed and uncompressed data to be kept in RAM for comparison.
    """

def _generate_ops(metadata: dict[str, Any], frame_data: NDArray[np.int16], ops_path: Path) -> None:
    """Uses frame-invariant ScanImage metadata and static default values to create an ops.json file in the directory
    specified by data_path.

    This function is an implementation of the mesoscope data extraction helper from the suite2p library. The helper
    function has been reworked to use the metadata parsed by tifffile and reimplemented in Python. Primarily, this
    function generates the 'fs', 'dx', 'dy', 'lines', 'nroi', 'nplanes' and 'mesoscan' fields of the 'ops' configuration
    file.

    Notes:
        The generated ops.json file will be saved at the location and filename specified by the ops_path.

    Args:
        metadata: The dictionary containing ScanImage metadata extracted from a mesoscope tiff stack file.
        frame_data: A numpy array containing the extracted pixel data for the first frame of the stack.
        ops_path: The path to the output ops.json file. This is generated by the ProjectData class and passed down to
            this method via the main directory processing function.
    """

def _process_invariant_metadata(file: Path, ops_path: Path, metadata_path: Path) -> None:
    """Extracts frame-invariant ScanImage metadata from the target tiff file and uses it to generate metadata.json and
    ops.json files.

    This function only needs to be called for one raw ScanImage TIFF stack acquired as part of the same experimental
    session. It extracts the ScanImage metadata that is common for all frames across all stacks and outputs it as a
    metadata.json file. This function also calls the _generate_ops() function that generates a suite2p ops.json file
    from the parsed metadata.

    Notes:
        This function is primarily designed to preserve the metadata before compressing raw TIFF stacks with the
        Limited Error Raster Compression (LERC) scheme.

    Args:
        file: The path to the mesoscope TIFF stack file. This can be any file in the directory as the
            frame-invariant metadata is the same for all stacks.
        ops_path: The path to the ops.json file that should be created by this function. This is resolved by the
            ProjectData class to match the processed project, animal, and session combination.
        metadata_path: The path to the metadata.json file that should be created by this function. This is resolved
            by the ProjectData class to match the processed project, animal, and session combination.
    """

def _preprocess_video_names(session_data: SessionData) -> None:
    """Renames the video files generated during runtime to use human-friendly camera names, rather than ID-codes.

    This is a minor preprocessing function primarily designed to make further data processing steps more human-readable.

    Notes:
        This function assumes that the runtime uses 3 cameras with IDs 51 (face camera), 62 (left camera), and 73
        (right camera).

    Args:
        session_data: The SessionData instance that manages the data for the processed session.
    """

def _pull_mesoscope_data(
    session_data: SessionData,
    num_threads: int = 30,
    remove_sources: bool = True,
    verify_transfer_integrity: bool = True,
) -> None:
    """Pulls the frames acquired by the mesoscope from the ScanImagePC to the VRPC.

    This function should be called after the data acquisition runtime to aggregate all recorded data on the VRPC
    before running the preprocessing pipeline. The function expects that the mesoscope frames source directory
    contains only the frames acquired during the current session runtime, the MotionEstimator.me and
    zstack.mat used for motion registration.

    Notes:
        It is safe to call this function for sessions that did not acquire mesoscope frames. It is designed to
        abort early if it cannot discover the cached mesoscope frames data for the target session on the ScanImagePC.

        This function expects that the data acquisition runtime has renamed the mesoscope_frames source directory for
        the session to include the session name. Manual intervention may be necessary if the runtime fails before the
        mesoscope_frames source directory is renamed.

        This function is configured to parallelize data transfer and verification to optimize runtime speeds where
        possible.

        When the function is called for the first time for a particular project and animal combination, it also
        'persists' the MotionEstimator.me file before moving all mesoscope data to the VRPC. This creates the
        reference for all further motion estimation procedures carried out during future sessions.

    Args:
        session_data: The SessionData instance that manages the data for the processed session.
        remove_sources: Determines whether to remove the transferred mesoscope frame data from the ScanImagePC.
            Generally, it is recommended to remove source data to keep ScanImagePC disk usage low. Note, setting
            this to True will only mark the data for removal. The data will not be removed until 'purge-data' command
            is used from the terminal.
        verify_transfer_integrity: Determines whether to verify the integrity of the transferred data. This is
            performed before source folder is marked for removal from the ScanImagePC if remove_sources is True.
    """

def _preprocess_mesoscope_directory(
    session_data: SessionData,
    num_processes: int,
    remove_sources: bool = True,
    batch: bool = False,
    verify_integrity: bool = False,
    batch_size: int = 250,
) -> None:
    """Loops over all multi-frame Mesoscope TIFF stacks in the mesoscope_frames, recompresses them using Limited Error
    Raster Compression (LERC) scheme, and extracts ScanImage metadata.

    This function is used as a preprocessing step for mesoscope-acquired data that optimizes the size of raw images for
    long-term storage and streaming over the network. To do so, all stacks are re-encoded using LERC scheme, which
    achieves ~70% compression ratio, compared to the original frame stacks obtained from the mesoscope. Additionally,
    this function also extracts frame-variant and frame-invariant ScanImage metadata from raw stacks and saves it as
    efficiently encoded JSON (.json) and compressed numpy archive (.npz) files to minimize disk space usage.

    Notes:
        This function is specifically calibrated to work with TIFF stacks produced by the ScanImage matlab software.
        Critically, these stacks are named using '_' to separate acquisition and stack number from the rest of the
        file name, and the stack number is always found last, e.g.: 'Tyche-A7_2022_01_25_1__00001_00067.tif'. If the
        input TIFF files do not follow this naming convention, the function will not process them. Similarly, if the
        stacks do not contain ScanImage metadata, they will be excluded from processing.

        To optimize runtime efficiency, this function employs multiple processes to work with multiple TIFFs at the
        same time. Given the overall size of each image dataset, this function can run out of RAM if it is allowed to
        operate on the entire folder at the same time. To prevent this, disable verification, use fewer processes, or
        change the batch_size to load fewer frames in memory at the same time.

        In addition to frame compression and data extraction, this function also generates the ops.json configuration
        file. This file is used during suite2p cell registration, performed as part of our standard data processing
        pipeline.

    Args:
        session_data: The SessionData instance that manages the data for the processed session.
        num_processes: The maximum number of processes to use while processing the directory. Each process is used to
            compress a stack of TIFF files in parallel.
        remove_sources: Determines whether to remove the original TIFF files after they have been processed.
        batch: Determines whether the function is called as part of batch-processing multiple directories. This is used
            to optimize progress reporting to avoid cluttering the terminal window.
        verify_integrity: Determines whether to verify the integrity of compressed data against the source data.
            The conversion does not alter the source data, so it is usually safe to disable this option, as the chance
            of compromising the data is negligible. Note, enabling this function doubles the RAM used by each parallel
            worker spawned by this function.
        batch_size: Determines how many frames are loaded into memory at the same time during processing. Note, the same
            number of frames will be loaded from each stack processed in parallel.
    """

def _preprocess_log_directory(
    session_data: SessionData, num_processes: int, remove_sources: bool = True, verify_integrity: bool = False
) -> None:
    """Compresses all .npy (uncompressed) log entries stored in the behavior log directory into one or more .npz
    archives.

    This service function is used during data preprocessing to optimize the size and format used to store all log
    entries. Primarily, this is necessary to facilitate data transfer over the network and log processing on the
    BioHPC server.

    Args:
        session_data: The SessionData instance that manages the data for the processed session.
        num_processes: The maximum number of processes to use while processing the directory.
        remove_sources: Determines whether to remove the original .npy files after they are compressed into .npz
            archives. It is recommended to have this option enabled.
        verify_integrity: Determines whether to verify the integrity of compressed data against the source data.
            It is advised to have this disabled for most runtimes, as data corruption is highly unlikely, but enabling
            this option adds a significant overhead to the processing time.

    Raises:
        RuntimeError: If the target log directory contains both compressed and uncompressed log entries.
    """

def _push_data(session_data: SessionData, parallel: bool = True, num_threads: int = 15) -> None:
    """Copies the raw_data directory from the VRPC to the NAS and the BioHPC server.

    This internal method is called as part of preprocessing to move the preprocessed data to the NAS and the server.
    This method generates the xxHash3-128 checksum for the source folder that the server processing pipeline uses to
    verify the integrity of the transferred data.

    Notes:
        The method also replaces the persisted zaber_positions.yaml file with the file generated during the managed
        session runtime. This ensures that the persisted file is always up to date with the current zaber motor
        positions.

    Args:
        session_data: The SessionData instance that manages the data for the processed session.
        parallel: Determines whether to parallelize the data transfer. When enabled, the method will transfer the
            data to all destinations at the same time (in-parallel). Note, this argument does not affect the number
            of parallel threads used by each transfer process or the number of threads used to compute the
            xxHash3-128 checksum. This is determined by the 'num_threads' argument (see below).
        num_threads: Determines the number of threads used by each transfer process to copy the files and calculate
            the xxHash3-128 checksums. Since each process uses the same number of threads, it is highly
            advised to set this value so that num_threads * 2 (number of destinations) does not exceed the total
            number of CPU cores - 4.
    """

def _preprocess_google_sheet_data(session_data: SessionData) -> None:
    """Updates the water restriction log and the surgery_data.yaml file.

    This internal method is called as part of preprocessing. Primarily, it is used to ensure that the surgery data
    extracted and stored in the 'metadata' folder of each processed animal is actual. It also updates the water
    restriction log for the managed animal to reflect the water received before and after runtime. This step improves
    user experience by ensuring all relevant data is always kept together on the NAS and BioHPC server while preventing
    the experimenter from manually updating the log after data preprocessing.

    Raises:
        ValueError: If the session_type attribute of the input SessionData instance is not one of the supported options.
    """

def _resolve_telomere_markers(server_root_path: Path, local_root_path: Path) -> None:
    """Checks the data stored on Sun lab BioHPC server for the presence of telomere.bin markers and removes all matching
    directories on the VRPC.

    Specifically, this function iterates through all raw_data directories on the VRPC, checks if the corresponding
    directory on the BioHPC server contains a telomere.bin marker, and removes the local raw_data directory if a marker
    is found.

    Args:
        server_root_path: The path to the root directory used to store all experiment and training data on the Sun lab
            BioHPC server.
        local_root_path: The path to the root directory used to store all experiment and training data on the VRPC.
    """

def _resolve_ubiquitin_markers(mesoscope_root_path: Path) -> None:
    """Checks the data stored on the ScanImage PC for the presence of ubiquitin.bin markers and removes all directories
    that contain the marker.

    This function is used to clear out cached mesoscope frame directories on the ScanImage PC once they have been safely
    copied and processed on the VRPC.

    Args:
        mesoscope_root_path: The path to the root directory used to store all mesoscope-acquired data on the ScanImage
            (Mesoscope) PC.
    """

def purge_redundant_data(
    remove_ubiquitin: bool,
    remove_telomere: bool,
    local_root_path: Path = ...,
    server_root_path: Path = ...,
    mesoscope_root_path: Path = ...,
) -> None:
    """Loops over ScanImagePC and VRPC directories that store training and experiment data and removes no longer
    necessary data caches.

    This function searches the ScanImagePC and VRPC for no longer necessary directories and removes them from the
    respective systems. ScanImagePC directories are marked for deletion once they are safely copied to the VRPC (and the
    integrity of the copied data is verified using xxHash-128 checksum). VRPC directories are marked for deletion once
    the data is safely copied to the BioHPC server and the server verifies the integrity of the copied data using
    xxHash-128 checksum.

    Notes:
        This is a service function intended to maintain the ScanImagePC and VRPC disk space. To ensure data integrity
        and redundancy at all processing stages, we do not remove the raw data from these PCs even if it has been
        preprocessed and moved to long-term storage destinations. However, once the data is moved to the BioHPC server
        and the NAS, it is generally safe to remove the copies stored on the ScanImagePC and VRPC.

        While the NAS is currently not verified for transferred data integrity, it is highly unlikely that the transfer
        process leads to data corruption. Overall, the way this process is structured ensures that at all stages of
        data processing there are at least two copies of the data stored on two different machines.

        Currently, this function does not discriminate between projects or animals. It will remove all data marked for
        deletion via the ubiquitin.bin marker or the telomere.bin marker.

    Args:
        remove_ubiquitin: Determines whether to remove ScanImagePC mesoscope_frames directories marked for deletion
            with ubiquitin.bin markers. Specifically, this allows removing directories that have been safely moved to
            the VRPC.
        remove_telomere: Determines whether to remove VRPC directories whose corresponding BioHPC-server directories
            are marked with telomere.bin markers. Specifically, this allows removing directories that have been safely
            moved to and processed by the BioHPC server.
        local_root_path: The path to the root directory of the VRPC used to store all experiment and training data.
        server_root_path: The path to the root directory of the BioHPC server used to store all experiment and
            training data.
        mesoscope_root_path: The path to the root directory of the ScanImagePC used to store all
            mesoscope-acquired frame data.
    """
