"""This module provides the main MesoscopeExperiment and BehavioralTraining classes that abstract working with Sun
lab's Mesoscope-VR system and SessionData class that abstracts working with acquired experimental data."""

import os
import warnings
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray
from numpy.lib.npyio import NpzFile
import copy
import json
from ataraxis_base_utilities import console, ensure_directory_exists, LogLevel
from ataraxis_data_structures import DataLogger, LogPackage, YamlConfig, SharedMemoryArray
from ataraxis_time.time_helpers import get_timestamp
from ataraxis_communication_interface import MicroControllerInterface, MQTTCommunication
from ataraxis_time import PrecisionTimer

from .transfer_tools import transfer_directory
from .packaging_tools import calculate_directory_checksum
from .data_preprocessing import (
    process_mesoscope_directory,
    process_camera_timestamps,
    process_module_data,
)
from .visualizers import BehaviorVisualizer
from .module_interfaces import ValveInterface
from .binding_classes import _MicroControllerInterfaces, _HeadBar, _LickPort, _VideoSystems, _ZaberPositions
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from dataclasses import dataclass
import polars as pl
from tqdm import tqdm
from pynput import keyboard


@dataclass()
class _LickTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to lick training sessions as a .yaml file."""

    session_type: str = "lick_training"
    """
    The type of the session. Currently, the following options are supported: "lick_training", "run_training", and 
    "mesoscope_experiment". This field is hardcoded and should not be modified.
    """
    dispensed_water_volume_ul: float = 0.0
    """Stores the total water volume, in microliters, dispensed during runtime."""
    average_reward_delay_s: int = 12
    """Stores the center-point for the reward delay distribution, in seconds."""
    maximum_deviation_from_average_s: int = 6
    """Stores the deviation value, in seconds, used to determine the upper and lower bounds for the reward delay 
    distribution."""
    maximum_water_volume_ml: float = 1.0
    """Stores the maximum volume of water the system is allowed to dispense during training."""
    maximum_training_time_m: int = 40
    """Stores the maximum time, in minutes, the system is allowed to run the training for."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""


@dataclass()
class _RunTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to run training sessions as a .yaml file."""

    session_type: str = "run_training"
    """
    The type of the session. Currently, the following options are supported: "lick_training", "run_training", and 
    "mesoscope_experiment". This field is hardcoded and should not be modified.
    """
    dispensed_water_volume_ul: float = 0.0
    """Stores the total water volume, in microliters, dispensed during runtime."""
    final_running_speed_cm_s: float = 0.0
    """Stores the final running speed threshold that was active at the end of training."""
    final_speed_duration_s: float = 0.0
    """Stores the final running duration threshold that was active at the end of training."""
    initial_running_speed_cm_s: float = 0.0
    """Stores the initial running speed threshold, in centimeters per second, used during training."""
    initial_speed_duration_s: float = 0.0
    """Stores the initial above-threshold running duration, in seconds, used during training."""
    increase_threshold_ml: float = 0.0
    """Stores the volume of water delivered to the animal, in milliliters, that triggers the increase in the running 
    speed and duration thresholds."""
    increase_running_speed_cm_s: float = 0.0
    """Stores the value, in centimeters per second, used by the system to increment the running speed threshold each 
    time the animal receives 'increase_threshold' volume of water."""
    increase_speed_duration_s: float = 0.0
    """Stores the value, in seconds, used by the system to increment the duration threshold each time the animal 
    receives 'increase_threshold' volume of water."""
    maximum_running_speed_cm_s: float = 0.0
    """Stores the maximum running speed threshold, in centimeters per second, the system is allowed to use during 
    training."""
    maximum_speed_duration_s: float = 0.0
    """Stores the maximum above-threshold running duration, in seconds, the system is allowed to use during training."""
    maximum_water_volume_ml: float = 1.0
    """Stores the maximum volume of water the system is allowed to dispensed during training."""
    maximum_training_time_m: int = 40
    """Stores the maximum time, in minutes, the system is allowed to run the training for."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""


@dataclass()
class _MesoscopeExperimentDescriptor(YamlConfig):
    """This class is used to save the description information specific to experiment sessions as a .yaml file."""

    session_type: str = "mesoscope_experiment"
    """
    The type of the session. Currently, the following options are supported: "lick_training", "run_training", and 
    "mesoscope_experiment". This field is hardcoded and should not be modified.
    """
    dispensed_water_volume_ul: float = 0.0
    """Stores the total water volume, in microliters, dispensed during runtime."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""


@dataclass()
class _ProcessingTracker(YamlConfig):
    """This class is used to track which data acquisition and processing pipelines have been applied to the session
    data.

    This is primarily used together with scheduled data processing tasks to ensure the same data is not processed
    multiple times. Additionally, it can be used to determine the current processing stage of the data.
    """

    data_acquisition: bool = True
    data_preprocessing: bool = False
    integrity_verification: bool = False
    data_compression: bool = False
    mesoscope_registration: bool = False
    video_tracking: bool = False


def _process_experiment_data(log_path: Path, output_directory: Path, cue_map: dict[int, float]) -> None:
    """Extracts the VR states, Experiment states, and the Virtual Reality cue sequence from the log generated
    by the instance during runtime and saves the extracted data as Polars DataFrame .feather files.

    This extraction method functions similar to how our camera log extraction and hardware module log extraction methods
    work. The key difference is the wall cue sequence extraction, which does not have a timestamp column. Instead, it
    has a distance colum which stores the distance the animal has to run, in cm, to reach each cue in the sequence. It
    is expected that during data processing, the distance data will be used to align the cues to the distance ran by
    the animal during the experiment.

    Notes:
        This function is used as the target for a multiprocessing executor used by the MesoscopeExperiment class during
        stop() method.
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
            "timestamps_us": pl.Series(name="timestamps_us", values=vr_timestamps),
            "vr_state": pl.Series(name="vr_state", values=vr_states),
        }
    )
    exp_dataframe = pl.DataFrame(
        {
            "timestamps_us": pl.Series(name="timestamps_us", values=experiment_timestamps),
            "experiment_state": pl.Series(name="experiment_state", values=experiment_states),
        }
    )
    cue_dataframe = pl.DataFrame(
        {
            "vr_cue": pl.Series(name="vr_cue", values=cue_sequence),
            "distance_cm": pl.Series(name="distance_cm", values=distance_sequence),
        }
    )

    # Saves the DataFrames to Feather file with lz4 compression
    vr_dataframe.write_ipc(output_directory.joinpath("vr_data.feather"), compression="lz4")
    exp_dataframe.write_ipc(output_directory.joinpath("experiment_data.feather"), compression="lz4")
    cue_dataframe.write_ipc(output_directory.joinpath("cue_data.feather"), compression="lz4")


class KeyboardListener:
    """Monitors the keyboard input for various runtime control signals and changes internal flags to communicate
    detected signals.

    This class is used during all training runtimes to allow the user to manually control some aspects of the
    Mesoscope-VR system and runtime. For example, it is used to abort the training runtime early and manually deliver
    rewards via the lick-tube.

    This class looks for the following key combinations to set the following flags:
        - ESC + 'q': Immediately aborts the training runtime.
        - ESC + 'r': Delivers 5 uL of water via the LickTube.

    Notes:
        While our training logic functions automatically make use of this class, it is NOT explicitly part of the
        MesoscopeExperiment class runtime. We highly encourage incorporating this class into all experiment runtimes to
        provide similar APi as done by our training runtimes.

        This monitor may pick up keyboard strokes directed at other applications during runtime. While our unique key
        combination is likely to not be used elsewhere, exercise caution when using other applications alongside the
        runtime code.

    Attributes:
        _exit_flag: Tracks whether the instance has detected the runtime abort key sequence press.
        _reward_flag: Tracks whether the instance has detected the reward delivery key sequence press.
        _speed_flag: Tracks the current user-defined modifier applied to the running speed threshold.
        _duration_flag: Tracks the current user-defined modifier applied to the running duration threshold.
        _currently_pressed: Stores the keys that are currently being pressed.
        _listener: The Listener instance used to monitor keyboard strokes.

    """

    def __init__(self):
        self._exit_flag = False
        self._reward_flag = False
        self._speed_flag = 0
        self._duration_flag = 0
        self._currently_pressed = set()

        # Set up listeners for both press and release
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        """Adds newly pressed keys to the storage set and determines whether the pressed key combination matches the
        shutdown combination.

        This method is used as the 'on_press' callback for the Listener instance.
        """
        # Updates the set with current data
        self._currently_pressed.add(str(key))

        # Checks if ESC is pressed (required for all combinations)
        if "Key.esc" in self._currently_pressed:
            # Exit combination: ESC + q
            if "'q'" in self._currently_pressed:
                self._exit_flag = True

            # Reward combination: ESC + r
            if "'r'" in self._currently_pressed:
                self._reward_flag = True

            # Speed control: ESC + Up/Down arrows
            if "Key.up" in self._currently_pressed:
                self._speed_flag += 1

            if "Key.down" in self._currently_pressed:
                self._speed_flag -= 1

            # Duration control: ESC + Left/Right arrows
            if "Key.right" in self._currently_pressed:
                self._duration_flag -= 1

            if "Key.left" in self._currently_pressed:
                self._duration_flag += 1

    def _on_release(self, key):
        """Removes no longer pressed keys from the storage set.

        This method is used as the 'on_release' callback for the Listener instance.
        """
        # Removes no longer pressed keys from the set
        key_str = str(key)
        if key_str in self._currently_pressed:
            self._currently_pressed.remove(key_str)

    @property
    def exit_signal(self) -> bool:
        """Returns True if the listener has detected the runtime abort keys combination (ESC + q) being pressed.

        This indicates that the user has requested the runtime to gracefully abort.
        """
        return self._exit_flag

    @property
    def reward_signal(self) -> bool:
        """Returns True if the listener has detected the water reward delivery keys combination (ESC + r) being
        pressed.

        This indicates that the user has requested the system to deliver 5uL water reward.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
        signal = copy.copy(self._reward_flag)
        self._reward_flag = False  # FLips the flag to False
        return signal

    @property
    def speed_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running speed threshold.

        This is used during run training to manually update the running speed threshold.
        """
        return self._speed_flag

    @property
    def duration_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running epoch duration threshold.

        This is used during run training to manually update the running epoch duration threshold.
        """
        return self._duration_flag


class SessionData:
    """Provides methods for managing the data acquired during one experiment or training session.

    This class functions as the central hub for collecting the data from all local PCs involved in the data acquisition
    process and pushing it to the NAS and the BioHPC server. Its primary purpose is to maintain the session data
    structure across all supported destinations and to efficiently and safely move the data to these destinations with
    minimal redundancy and footprint. Additionally, this class generates the paths used by all other classes from
    this library to determine where to load and saved various data during runtime.

    As part of its initialization, the class generates the session directory for the input animal and project
    combination. Session directories use the current UTC timestamp, down to microseconds, as the directory name. This
    ensures that each session name is unique and preserves the overall session order.

    Notes:
        Do not call methods from this class directly. This class is intended to be used through the MesoscopeExperiment
        and BehavioralTraining classes. The only reason the class is defined as public is to support reconfiguring major
        source / destination paths (NAS, BioHPC, etc.).

        It is expected that the server, nas, and mesoscope data directories are mounted on the host-machine via the
        SMB or equivalent protocol. All manipulations with these destinations are carried out with the assumption that
        the OS has full access to these directories and filesystems.

        This class is specifically designed for working with raw data from a single animal participating in a single
        experimental project session. Processed data is managed by the processing library methods and classes.

        This class generates an xxHash-128 checksum stored inside the ax_checksum.txt file at the root of each
        experimental session 'raw_data' directory. The checksum verifies the data of each file and the paths to each
        file relative to the 'raw_data' root directory.

    Args:
        project_name: The name of the project managed by the class.
        animal_name: The name of the animal managed by the class.
        generate_mesoscope_paths: Determines whether the managed session uses ScanImage (mesoscope) PC. Training
            sessions that do not use the Mesoscope do not need to resolve paths to mesoscope data folders and storage
            directories.
        local_root_directory: The path to the root directory where all projects are stored on the host-machine. Usually,
            this is the 'Experiments' folder on the 16TB volume of the VRPC machine.
        server_root_directory: The path to the root directory where all projects are stored on the BioHPC server
            machine. Usually, this is the 'storage/SunExperiments' (RAID 6) volume of the BioHPC server.
        nas_root_directory: The path to the root directory where all projects are stored on the Synology NAS. Usually,
            this is the non-compressed 'raw_data' directory on one of the slow (RAID 6) volumes, such as Volume 1.
        mesoscope_data_directory: The path to the directory where the mesoscope saves acquired frame data. Usually, this
            directory is on the fast Data volume (NVME) and is cleared of data after each acquisition runtime, such as
            the /mesodata/mesoscope_frames directory. Note, this class will delete and recreate the directory during its
            runtime, so it is highly advised to make sure it does not contain any important non-session-related data.

    Attributes:
        _local: The path to the host-machine directory for the managed project, animal and session combination.
            This path points to the 'raw_data' subdirectory that stores all acquired and preprocessed session data.
        _server: The path to the BioHPC server directory for the managed project, animal, and session
            combination.
        _nas: The path to the Synology NAS directory for the managed project, animal, and session combination.
        _mesoscope: The path to the root ScanImage PC (mesoscope) data directory. This directory is shared by
            all projects, animals, and sessions.
        _persistent: The path to the host-machine directory used to retain persistent data from previous session(s) of
            the managed project and animal combination. For example, this directory is used to persist Zaber positions
            and mesoscope motion estimator files when the original session directory is moved to nas and server.
        _project_name: Stores the name of the project whose data is managed by the class.
        _animal_name: Stores the name of the animal whose data is managed by the class.
        _session_name: Stores the name of the session directory whose data is managed by the class.
    """

    def __init__(
        self,
        project_name: str,
        animal_name: str,
        generate_mesoscope_paths: bool = True,
        local_root_directory: Path = Path("/media/Data/Experiments"),
        server_root_directory: Path = Path("/media/cbsuwsun/storage/sun_data"),
        nas_root_directory: Path = Path("/home/cybermouse/nas/rawdata"),
        mesoscope_data_directory: Path = Path("/home/cybermouse/scanimage/mesodata"),
    ) -> None:
        # Computes the project + animal directory paths for the local machine (VRPC)
        self._local: Path = local_root_directory.joinpath(project_name, animal_name)

        # Mesoscope is configured to use the same directories for all projects and animals
        self._mesoscope: Path = mesoscope_data_directory

        # Generates a separate directory to store persistent data. This has to be done early as _local is later
        # overwritten with the path to the raw_data directory of the created session.
        self._persistent: Path = self._local.joinpath("persistent_data")

        # Records animal and project names to attributes. Session name is resolved below
        self._project_name: str = project_name
        self._animal_name: str = animal_name

        # Acquires the UTC timestamp to use as the session name
        session_name = get_timestamp(time_separator="-")

        # Constructs the session directory path and generates the directory
        raw_session_path = self._local.joinpath(session_name)

        # Handles potential session name conflicts. While this is extremely unlikely, it is not impossible for
        # such conflicts to occur.
        counter = 0
        while raw_session_path.exists():
            counter += 1
            new_session_name = f"{session_name}_{counter}"
            raw_session_path = self._local.joinpath(new_session_name)

        # If a conflict is detected and resolved, warns the user about the resolved conflict.
        if counter > 0:
            message = (
                f"Session name conflict occurred for animal '{self._animal_name}' of project '{self._project_name}' "
                f"when adding the new session with timestamp {session_name}. The session with identical name "
                f"already exists. The newly created session directory uses a '_{counter}' postfix to distinguish "
                f"itself from the already existing session directory."
            )
            warnings.warn(message=message)

        # Saves the session name to class attribute
        self._session_name: str = raw_session_path.stem

        # Modifies the local path to point to the raw_data directory under the session directory. The 'raw_data'
        # directory acts as the root for all raw and preprocessed data generated during session runtime.
        self._local = self._local.joinpath(self._session_name, "raw_data")

        # Modifies nas and server paths to point to the session directory. The "raw_data" folder will be created during
        # data movement method runtime, so the folder is not precreated for destinations.
        self._server: Path = server_root_directory.joinpath(self._project_name, self._animal_name, self._session_name)
        self._nas: Path = nas_root_directory.joinpath(self._project_name, self._animal_name, self._session_name)

        # Ensures that root paths exist for all destinations and sources. Note
        ensure_directory_exists(self._local)
        ensure_directory_exists(self._nas)
        ensure_directory_exists(self._server)
        if not generate_mesoscope_paths:
            ensure_directory_exists(self._mesoscope)
        ensure_directory_exists(self._persistent)

    @property
    def raw_data_path(self) -> Path:
        """Returns the path to the 'raw_data' directory of the managed session.

        The raw_data is the root directory for aggregating all acquired and preprocessed data. This path is primarily
        used by MesoscopeExperiment class to determine where to save captured videos, data logs, and other acquired data
        formats. After runtime, SessionData pulls all mesoscope frames into this directory.
        """
        return self._local

    @property
    def mesoscope_frames_path(self) -> Path:
        """Returns the path to the mesoscope_frames directory of the managed session.

        This path is used during mesoscope data preprocessing to store compressed (preprocessed) mesoscope frames.
        """
        directory_path = self._local.joinpath("mesoscope_frames")

        # Since this is a directory, we need to ensure it exists before this path is returned to caller
        ensure_directory_exists(directory_path)
        return directory_path

    @property
    def ops_path(self) -> Path:
        """Returns the path to the ops.json file of the managed session.

        This path is used to save the ops.json generated from the mesoscope TIFF metadata during preprocessing. This is
        a configuration file used by the suite2p during mesoscope data registration.
        """
        return self.mesoscope_frames_path.joinpath("ops.json")

    @property
    def frame_invariant_metadata_path(self) -> Path:
        """Returns the path to the frame_invariant_metadata.json file of the managed session.

        This path is used to save the metadata shared by all frames in all TIFF stacks acquired by the mesoscope.
        Currently, this data is not used during processing.
        """
        return self.mesoscope_frames_path.joinpath("frame_invariant_metadata.json")

    @property
    def frame_variant_metadata_path(self) -> Path:
        """Returns the path to the frame_variant_metadata.npz file of the managed session.

        This path is used to save the metadata unique for each frame in all TIFF stacks acquired by the mesoscope.
        Currently, this data is not used during processing.

        Notes:
            Unlike frame-invariant metadata, this file is stored as a compressed NumPy archive (NPZ) file to optimize
            storage space usage.
        """
        return self.mesoscope_frames_path.joinpath("frame_variant_metadata.npz")

    @property
    def camera_frames_path(self) -> Path:
        """Returns the path to the camera_frames directory of the managed session.

        This path is used during camera data preprocessing to store the videos and extracted frame timestamps for all
        video cameras used to record behavior.
        """
        directory_path = self._local.joinpath("camera_frames")

        # Since this is a directory, we need to ensure it exists before this path is returned to caller
        ensure_directory_exists(directory_path)
        return directory_path

    @property
    def zaber_positions_path(self) -> Path:
        """Returns the path to the zaber_positions.yaml file of the managed session.

        This path is used to save the positions for all Zaber motors of the HeadBar and LickPort controllers at the
        end of the experimental session. This allows restoring the motors to those positions during the following
        experimental session(s).
        """
        return self._local.joinpath("zaber_positions.yaml")

    @property
    def session_descriptor_path(self) -> Path:
        """Returns the path to the session_descriptor.yaml file of the managed session.

        This path is used to save important session information to be viewed by experimenters post-runtime and to use
        for further processing. This includes the type of the session (e.g. 'lick_training') and the total volume of
        water delivered during runtime (important for water restriction).
        """
        return self._local.joinpath("session_descriptor.yaml")

    @property
    def behavioral_data_path(self) -> Path:
        """Returns the path to the behavioral_data directory of the managed session.

        This path is used during module data processing to extract the data acquired by various hardware modules from
        .npz log archives and save it as feather files.
        """
        directory_path = self._local.joinpath("behavioral_data")

        # Since this is a directory, we need to ensure it exists before this path is returned to caller
        ensure_directory_exists(directory_path)
        return directory_path

    @property
    def previous_zaber_positions_path(self) -> Path:
        """Returns the path to the zaber_positions.yaml file of the previous session.

        The file is stored inside the 'persistent' directory of the project and animal combination. The file is saved to
        the persistent directory when the original session is moved to long-term storage. Loading the file allows
        reusing LickPort and HeadBar motor positions across sessions. The contents of this file are updated after each
        experimental or training session.
        """
        file_path = self._persistent.joinpath("zaber_positions.yaml")
        return file_path

    @property
    def persistent_motion_estimator_path(self) -> Path:
        """Returns the path to the MotionEstimator.me file for the managed animal and project combination stored on the
        ScanImagePC.

        This path is used during the first training session to save the 'reference' MotionEstimator.me file established
        during the initial mesoscope ROI selection to the ScanImagePC. The same reference file is used for all following
        sessions to correct for the natural motion of the brain relative to the cranial window.
        """
        return self._mesoscope.joinpath("persistent_data", self._project_name, self._animal_name, "MotionEstimator.me")

    @property
    def data_processing_tracker_path(self) -> Path:
        """Returns the path to the processing_checklist.yaml file of the managed session.

        This path is used to generate the .yaml file that tracks the data processing stages that have been applied to
        the data inside the session directory. Primarily, this is used by the pipelines that run on the BioHPC server
        to prepare the session for multi-day processing and project dataset generation.
        """
        return self._local.joinpath("processing_checklist.yaml")

    def pull_mesoscope_data(
        self, num_threads: int = 28, remove_sources: bool = False, verify_transfer_integrity: bool = False
    ) -> None:
        """Pulls the frames acquired by the mesoscope from the ScanImage PC to the VRPC.

        This method should be called after the data acquisition runtime to aggregate all recorded data on the VRPC
        before running the preprocessing pipeline. The method expects that the mesoscope frames source directory
        contains only the frames acquired during the current session runtime, and the MotionEstimator.me and
        zstack.mat used for motion registration.

        Notes:
            This method is configured to parallelize data transfer and verification to optimize runtime speeds where
            possible.

            When the method is called for the first time for a particular project and animal combination, it also
            'persists' the MotionEstimator.me file before moving all mesoscope data to the VRPC. This creates the
            reference for all further motion estimation procedures carried out during future sessions.

        Args:
            num_threads: The number of parallel threads used for transferring the data from ScanImage (mesoscope) PC to
                the local machine. Depending on the connection speed between the PCs, it may be useful to set this
                number to the number of available CPU cores - 4.
            remove_sources: Determines whether to remove the transferred mesoscope frame data from the ScanImagePC.
                Generally, it is recommended to remove source data to keep ScanImagePC disk usage low.
            verify_transfer_integrity: Determines whether to verify the integrity of the transferred data. This is
                performed before source folder is removed from the ScanImagePC, if remove_sources is True.
        Raises:
            RuntimeError: If the mesoscope source directory does not contain motion estimator files or mesoscope frames.
        """
        # Resolves source and destination paths
        source = self._mesoscope.joinpath("mesoscope_frames")
        destination = self.raw_data_path  # The path to the raw_data subdirectory of the current session

        # Extracts the names of files stored in the source folder
        files = tuple([path for path in source.glob("*")])

        # Ensures the folder contains motion estimator data files
        if "MotionEstimator.me" not in files:
            message = (
                f"Unable to pull the mesoscope-acquired data from the ScanImage PC to the VRPC. The 'mesoscope_frames' "
                f"ScanImage PC directory does not contain the MotionEstimator.me file, which is required for further "
                f"frame data processing."
            )
            console.error(message=message, error=RuntimeError)
        if "zstack.mat" not in files:
            message = (
                f"Unable to pull the mesoscope-acquired data from the ScanImage PC to the VRPC. The 'mesoscope_frames' "
                f"ScanImage PC directory does not contain the zstack.mat file, which is required for further "
                f"frame data processing."
            )
            console.error(message=message, error=RuntimeError)

        # Prevents 'pulling' an empty folder. At a minimum, we expect 2 motion estimation files and one TIFF stack file
        if len(files) < 3:
            message = (
                f"Unable to pull the mesoscope-acquired data from the ScanImage PC to the VRPC. The 'mesoscope_frames' "
                f"ScanImage PC directory does not contain the minimum expected number of files (3). This indicates "
                f"that no frames were acquired during runtime or that the frames were saved at a different location."
            )
            console.error(message=message, error=RuntimeError)

        # If the processed project and animal combination does not have a reference MotionEstimator.me saved in the
        # persistent ScanImagePC directory, copies the MotionEstimator.me to the persistent directory. This ensures that
        # the first ever created MotionEstimator.me is saved as the reference MotionEstimator.me for further sessions.
        if not self.persistent_motion_estimator_path.exists():
            ensure_directory_exists(self.persistent_motion_estimator_path)
            shutil.copy2(src=source.joinpath("MotionEstimator.me"), dst=self.persistent_motion_estimator_path)

        # Generates the checksum for the source folder if transfer integrity verification is enabled.
        if verify_transfer_integrity:
            calculate_directory_checksum(directory=source, num_processes=None, save_checksum=True)

        # Transfers the mesoscope frames data from the ScanImage PC to the local machine.
        transfer_directory(
            source=source, destination=destination, num_threads=num_threads, verify_integrity=verify_transfer_integrity
        )

        # Removes the checksum file after the transfer is complete. The checksum will be recalculated for the whole
        # session directory during preprocessing, so there is no point in keeping the original mesoscope checksum file.
        if verify_transfer_integrity:
            destination.joinpath("ax_checksum.txt").unlink(missing_ok=True)

        # After the transfer completes successfully (including integrity verification), recreates the mesoscope_frames
        # folder to remove the transferred data from the ScanImage PC.
        if remove_sources:
            shutil.rmtree(source)
            ensure_directory_exists(source)

    def process_mesoscope_data(self) -> None:
        """Preprocesses the (pulled) mesoscope data.

        This is a wrapper around the process_mesoscope_directory() function. It compressed all mesoscope frames using
        LERC, extracts and save sframe-invariant and frame-variant metadata, and generates an ops.json file for future
        suite2p registration. This method also ensures that after processing all mesoscope data, including motion
        estimation files, are found under the mesoscope_frames directory.

        Notes:
            Additional processing for camera data is carried out by the stop() method of each experimental class. This
            is intentional, as not all classes use all cameras.
        """
        # Preprocesses the pulled mesoscope frames.
        process_mesoscope_directory(
            image_directory=self.raw_data_path,
            output_directory=self.mesoscope_frames_path,
            ops_path=self.ops_path,
            frame_invariant_metadata_path=self.frame_invariant_metadata_path,
            frame_variant_metadata_path=self.frame_variant_metadata_path,
            num_processes=28,
            remove_sources=True,
            verify_integrity=True,
        )

        # Cleans up some data inconsistencies. Moves motion estimator files to the mesoscope_frames directory generated
        # during mesoscope data preprocessing. This way, ALL mesoscope-related data is stored under mesoscope_frames.
        shutil.move(
            src=self.raw_data_path.joinpath("MotionEstimator.me"),
            dst=self.mesoscope_frames_path.joinpath("MotionEstimator.me"),
        )
        shutil.move(
            src=self.raw_data_path.joinpath("zstack.mat"),
            dst=self.mesoscope_frames_path.joinpath("zstack.mat"),
        )

    def push_data(
        self,
        parallel: bool = True,
        num_threads: int = 10,
        remove_sources: bool = False,
        verify_transfer_integrity: bool = False,
    ) -> None:
        """Copies the raw_data directory from the VRPC to the NAS and the BioHPC server.

        This method should be called after data acquisition and preprocessing to move the prepared data to the NAS and
        the server. This method generates the xxHash3-128 checksum for the source folder and, if configured, verifies
        that the transferred data produces the same checksum to ensure data integrity.

        Notes:
            This method is configured to run data transfer and checksum calculation in parallel where possible. It is
            advised to minimize the use of the host-machine while it is running this method, as most CPU resources will
            be consumed by the data transfer process.

            The method also replaces the persisted zaber_positions.yaml file with the file generated during the managed
            session runtime. This ensures that the persisted file is always up to date with the current zaber motor
            positions.

        Args:
            parallel: Determines whether to parallelize the data transfer. When enabled, the method will transfer the
                data to all destinations at the same time (in-parallel). Note, this argument does not affect the number
                of parallel threads used by each transfer process or the number of threads used to compute the
                xxHash3-128 checksum. This is determined by the 'num_threads' argument (see below).
            num_threads: Determines the number of threads used by each transfer process to copy the files and calculate
                the xxHash3-128 checksums. Since each process uses the same number of threads, it is highly
                advised to set this value so that num_threads * 2 (number of destinations) does not exceed the total
                number of CPU cores - 4.
            remove_sources: Determines whether to remove the raw_data directory from the VRPC once it has been copied
                to the NAS and Server. Depending on the overall load of the VRPC, we recommend keeping source data on
                the VRPC at least until the integrity of the transferred data is verified on the server.
            verify_transfer_integrity: Determines whether to verify the integrity of the transferred data. This is
                performed before source folder is removed from the VRPC, if remove_sources is True.
        """
        # Resolves source and destination paths
        source = self.raw_data_path

        # Destinations include short destination names used for progress reporting
        destinations = (
            (self._nas.joinpath("raw_data"), "NAS"),
            (self._server.joinpath("raw_data"), "Server"),
        )

        # Updates the zaber_positions.yaml file stored inside the persistent directory for the project+animal
        # combination with the zaber_positions.yaml file from the current session. This ensures that the zaber_positions
        # file is always set to the latest snapshot of zaber motor positions.
        if self.zaber_positions_path.exists():
            self.previous_zaber_positions_path.unlink(missing_ok=True)  # Removes the previous persisted file
            shutil.copy2(self.zaber_positions_path, self.previous_zaber_positions_path)  # Persists the current file

        # Resolves the destination paths based on the provided short destination names
        destinations = [(dest[0], dest[1]) for dest in destinations]

        # Computes the xxHash3-128 checksum for the source folder
        calculate_directory_checksum(directory=source, num_processes=None, save_checksum=True)

        # If the method is configured to transfer files in parallel, submits tasks to a ProcessPoolExecutor
        if parallel:
            with ProcessPoolExecutor(max_workers=len(destinations)) as executor:
                futures = {
                    executor.submit(
                        transfer_directory,
                        source=source,
                        destination=dest[0],
                        num_threads=num_threads,
                        verify_integrity=verify_transfer_integrity,
                    ): dest
                    for dest in destinations
                }
                for future in as_completed(futures):
                    # Propagates any exceptions from the transfers
                    future.result()

        # Otherwise, runs the transfers sequentially. Note, transferring individual files is still done in parallel, but
        # the transfer is performed for each destination sequentially.
        else:
            for destination in destinations:
                transfer_directory(
                    source=source,
                    destination=destination[0],
                    num_threads=num_threads,
                    verify_integrity=verify_transfer_integrity,
                )

        # After all transfers complete successfully, removes the source directory, if requested
        if remove_sources:
            shutil.rmtree(source)


class MesoscopeExperiment:
    """The base class for all mesoscope experiment runtimes.

    This class provides methods for conducting experiments using the Mesoscope-VR system. It abstracts most
    low-level interactions with the VR system and the mesoscope via a simple high-level state API.

    This class also provides methods for limited preprocessing of the collected data. The preprocessing methods are
    designed to be executed after each experiment runtime to prepare the data for long-term storage and transmission
    over the network. Preprocessing methods use multiprocessing to optimize runtime performance and assume that the
    VRPC is kept mostly idle during data preprocessing.

    Notes:
        Calling this initializer does not start the Mesoscope-VR components. Use the start() method before issuing other
        commands to properly initialize all remote processes. This class reserves up to 11 CPU cores during runtime.

        Use the 'axtl-ports' cli command to discover the USB ports used by Ataraxis Micro Controller (AMC) devices.

        Use the 'axvs-ids' cli command to discover the camera indices used by the Harvesters-managed and
        OpenCV-managed cameras.

        Use the 'sl-devices' cli command to discover the serial ports used by the Zaber motion controllers.

        This class statically reserves the id code '1' to label its log entries. Make sure no other Ataraxis class, such
        as MicroControllerInterface or VideoSystem, uses this id code.

    Args:
        session_data: An initialized SessionData instance. This instance is used to transfer the data between VRPC,
            ScanImagePC, BioHPC server, and the NAS during runtime. Each instance is initialized for the specific
            project, animal, and session combination for which the data is acquired.
        descriptor: A partially configured _LickTrainingDescriptor or _RunTrainingDescriptor instance. This instance is
            used to store session-specific information in a human-readable format.
        cue_length_map: A dictionary that maps each integer-code associated with a wall cue used in the Virtual Reality
            experiment environment to its length in real-world centimeters. Ity is used to map each VR cue to the
            distance the mouse needs to travel to fully traverse the wall cue region from start to end.
        screens_on: Communicates whether the VR screens are currently ON at the time of class initialization.
        experiment_state: The integer code that represents the initial state of the experiment. Experiment state codes
            are used to mark different stages of each experiment (such as rest_1, run_1, rest_2, etc...). During
            analysis, these codes can be used to segment experimental data into sections.
        actor_port: The USB port used by the actor Microcontroller.
        sensor_port: The USB port used by the sensor Microcontroller.
        encoder_port: The USB port used by the encoder Microcontroller.
        headbar_port: The USB port used by the headbar Zaber motor controllers (devices).
        lickport_port: The USB port used by the lickport Zaber motor controllers (devices).
        unity_ip: The IP address of the MQTT broker used to communicate with the Unity game engine.
        unity_port: The port number of the MQTT broker used to communicate with the Unity game engine.
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters.
        face_camera_index: The index of the face camera in the list of all available Harvester-managed cameras.
        left_camera_index: The index of the left camera in the list of all available OpenCV-managed cameras.
        right_camera_index: The index of the right camera in the list of all available OpenCV-managed cameras.
        harvesters_cti_path: The path to the GeniCam CTI file used to connect to Harvesters-managed cameras.

    Attributes:
        _started: Tracks whether the VR system and experiment runtime are currently running.
        _descriptor: Stores the session descriptor instance.
        _logger: A DataLogger instance that collects behavior log data from all sources: microcontrollers, video
            cameras, and the MesoscopeExperiment instance.
        _microcontrollers: Stores the _MicroControllerInterfaces instance that interfaces with all MicroController
            devices used during runtime.
        _cameras: Stores the _VideoSystems instance that interfaces with video systems (cameras) used during
            runtime.
        _headbar: Stores the _HeadBar class instance that interfaces with all HeadBar manipulator motors.
        _lickport: Stores the _LickPort class instance that interfaces with all LickPort manipulator motors.
        _vr_state: Stores the current state of the VR system. The MesoscopeExperiment updates this value whenever it is
            instructed to change the VR system state.
        _state_map: Maps the integer state-codes used to represent VR system states to human-readable string-names.
        _experiment_state: Stores the user-defined experiment state. Experiment states are defined by the user and
            are expected to be unique for each project and, potentially, experiment. Different experiment states can
            reuse the same VR state.
        _timestamp_timer: A PrecisionTimer instance used to timestamp log entries generated by the class instance.
        _source_id: Stores the unique identifier code for this class instance. The identifier is used to mark log
            entries made by this class instance and has to be unique across all sources that log data at the same time,
            such as MicroControllerInterfaces and VideoSystems.
        _cue_map: Stores the dictionary that maps integer-codes associated with each VR wall cue with
            its length in centimeters.
        _session_data: Stores the SessionData instance used to manage the acquired data.

    Raises:
        TypeError: If session_data or cue_length_map arguments have invalid types.
    """

    # Maps integer VR state codes to human-readable string-names.
    _state_map: dict[int, str] = {0: "Idle", 1: "Rest", 2: "Run"}

    def __init__(
        self,
        session_data: SessionData,
        descriptor: _MesoscopeExperimentDescriptor,
        cue_length_map: dict[int, float],
        screens_on: bool = False,
        experiment_state: int = 0,
        actor_port: str = "/dev/ttyACM0",
        sensor_port: str = "/dev/ttyACM1",
        encoder_port: str = "/dev/ttyACM2",
        headbar_port: str = "/dev/ttyUSB0",
        lickport_port: str = "/dev/ttyUSB1",
        unity_ip: str = "127.0.0.1",
        unity_port: int = 1883,
        valve_calibration_data: tuple[tuple[int | float, int | float], ...] = (
            (15000, 1.8556),
            (30000, 3.4844),
            (45000, 7.1846),
            (60000, 10.0854),
        ),
        face_camera_index: int = 0,
        left_camera_index: int = 0,
        right_camera_index: int = 2,
        harvesters_cti_path: Path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"),
    ) -> None:
        # Activates the console to display messages to the user if the console is disabled when the class is
        # instantiated.
        if not console.enabled:
            console.enable()

        # Creates the _started flag first to avoid leaks if the initialization method fails.
        self._started: bool = False
        self._descriptor: _MesoscopeExperimentDescriptor = descriptor

        # Input verification:
        if not isinstance(session_data, SessionData):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a SessionData instance for "
                f"'session_data' argument, but instead encountered {session_data} of type "
                f"{type(session_data).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(cue_length_map, dict):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a dictionary for 'cue_length_map' "
                f"argument, but instead encountered {cue_length_map} of type {type(cue_length_map).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Defines other flags used during runtime:
        self._vr_state: int = 0  # Stores the current state of the VR system
        self._experiment_state: int = experiment_state  # Stores user-defined experiment state
        self._timestamp_timer: PrecisionTimer = PrecisionTimer("us")  # A timer used to timestamp log entries
        self._source_id: np.uint8 = np.uint8(1)  # Reserves source ID code 1 for this class

        # This dictionary is used to convert distance traveled by the animal into the corresponding sequence of
        # traversed cues (corridors).
        self._cue_map: dict[int, float] = cue_length_map

        # Saves the SessionData instance to class attribute so that it can be used from class methods. Since SessionData
        # resolves session directory structure at initialization, the instance is ready to resolve all paths used by
        # the experiment class instance.
        self._session_data: SessionData = session_data

        # Initializes the DataLogger instance used to log data from all microcontrollers, camera frame savers, and this
        # class instance.
        self._logger: DataLogger = DataLogger(
            output_directory=session_data.raw_data_path,
            instance_name="behavior",  # Creates behavior_log subfolder under raw_data
            sleep_timer=0,
            exist_ok=True,
            process_count=1,
            thread_count=10,
        )

        # Initializes the binding class for all MicroController Interfaces.
        self._microcontrollers: _MicroControllerInterfaces = _MicroControllerInterfaces(
            data_logger=self._logger,
            screens_on=screens_on,
            actor_port=actor_port,
            sensor_port=sensor_port,
            encoder_port=encoder_port,
            valve_calibration_data=valve_calibration_data,
            debug=False,
        )

        # Also instantiates an MQTTCommunication instance to directly communicate with Unity. Currently, this is used
        # exclusively to verify that the Unity is running and to collect the sequence of VR wall cues used by the task.
        monitored_topics = ("CueSequence/",)
        self._unity: MQTTCommunication = MQTTCommunication(
            ip=unity_ip, port=unity_port, monitored_topics=monitored_topics
        )

        # Initializes the binding class for all VideoSystems.
        self._cameras: _VideoSystems = _VideoSystems(
            data_logger=self._logger,
            output_directory=self._session_data.camera_frames_path,
            face_camera_index=face_camera_index,
            left_camera_index=left_camera_index,
            right_camera_index=right_camera_index,
            harvesters_cti_path=harvesters_cti_path,
        )

        # While we can connect to ports managed by ZaberLauncher, ZaberLauncher cannot connect to ports managed via
        # software. Therefore, we have to make sure ZaberLauncher is running before connecting to motors.
        message = (
            "Preparing to connect to HeadBar and LickPort Zaber controllers. Make sure that ZaberLauncher app is "
            "running before proceeding further. If ZaberLauncher is not running, youi WILL NOT be able to manually "
            "control the HeadBar and LickPort motor positions until you reset the runtime."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Initializes the binding classes for the HeadBar and LickPort manipulator motors.
        self._headbar: _HeadBar = _HeadBar(
            headbar_port=headbar_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )
        self._lickport: _LickPort = _LickPort(
            lickport_port=lickport_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )

    def start(self) -> None:
        """Sets up all assets used during the experiment.

        This internal method establishes the communication with the microcontrollers, data logger cores, and video
        system processes. It also requests the cue sequence from Unity game engine and starts mesoscope frame
        acquisition process.

        Notes:
            This method will not run unless the host PC has access to the necessary number of logical CPU cores
            and other required hardware resources (GPU for video encoding, etc.). This prevents using the class on
            machines that are unlikely to sustain the runtime requirements.

            As part of its runtime, this method will attempt to set Zaber motors for the HeadBar and LickPort to the
            positions optimal for mesoscope frame acquisition. Exercise caution and always monitor the system when
            it is running this method, as unexpected motor behavior can damage the mesoscope or harm the animal.

        Raises:
            RuntimeError: If the host PC does not have enough logical CPU cores available.
        """
        # Prevents (re) starting an already started VR process.
        if self._started:
            return

        # 3 cores for microcontrollers, 1 core for the data logger, 6 cores for the current video_system
        # configuration (3 producers, 3 consumer), 1 core for the central process calling this method. 11 cores
        # total.
        if not os.cpu_count() >= 11:
            message = (
                f"Unable to start the MesoscopeExperiment runtime. The host PC must have at least 11 logical CPU "
                f"cores available for this class to work as expected, but only {os.cpu_count()} cores are "
                f"available."
            )
            console.error(message=message, error=RuntimeError)

        message = "Initializing MesoscopeExperiment assets..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts the data logger
        self._logger.start()

        # Generates and logs the onset timestamp for the VR system as a whole. The MesoscopeExperiment class logs
        # changes to VR and Experiment state during runtime, so it needs to have the onset stamp, just like all other
        # classes that generate data logs.

        # Constructs the timezone-aware stamp using UTC time. This creates a reference point for all later delta time
        # readouts. The time is returned as an array of bytes.
        onset: NDArray[np.uint8] = get_timestamp(as_bytes=True)  # type: ignore
        self._timestamp_timer.reset()  # Immediately resets the timer to make it as close as possible to the onset time

        # Logs the onset timestamp. All further timestamps will be treated as integer time deltas (in microseconds)
        # relative to the onset timestamp. Note, ID of 1 is used to mark the main experiment system.
        package = LogPackage(
            source_id=self._source_id, time_stamp=np.uint8(0), serialized_data=onset
        )  # Packages the id, timestamp, and data.
        self._logger.input_queue.put(package)

        message = "DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts the face camera. This starts frame acquisition and displays acquired frames to the user. However,
        # frame saving is disabled at this time. Body cameras are also disabled. This is intentional, as at this point
        # we want to minimize the number of active processes. This is helpful if this method is called while the
        # previous session is still running its data preprocessing pipeline and needs as many free cores as possible.
        self._cameras.start_face_camera()

        # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is safe to
        # proceed with motor movements.
        message = (
            "Preparing to move HeadBar into position. Remove the mesoscope objective, swivel out the VR screens, "
            "and make sure the animal is NOT mounted on the rig. Failure to fulfill these steps may DAMAGE the "
            "mesoscope and / or HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not intersect
        # with each other, so it is safe to move both assemblies at the same time.
        self._headbar.prepare_motors(wait_until_idle=False)
        self._lickport.prepare_motors(wait_until_idle=True)
        self._headbar.wait_until_idle()

        # Sets the motors into the mounting position. The HeadBar is either restored to the previous session position or
        # is set to the default mounting position stored in non-volatile memory. The LickPort is moved to a position
        # optimized for putting the animal on the VR rig.
        self._headbar.restore_position(wait_until_idle=False)
        self._lickport.mount_position(wait_until_idle=True)
        self._headbar.wait_until_idle()

        # Gives user time to mount the animal and requires confirmation before proceeding further.
        message = (
            "Preparing to move the LickPort into position. Mount the animal onto the VR rig and install the mesoscope "
            "objetive. If necessary, adjust the HeadBar position to make sure the animal can comfortably run the task."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Restores the lickPort to the previous session's position or to the default parking position. This positions
        # the LickPort in a way that is easily accessible by the animal.
        self._lickport.restore_position()
        message = (
            "If necessary, adjust LickPort position to be easily reachable by the animal and position the mesoscope "
            "objective above the imaging field. Take extra care when moving the LickPort towards the animal! Run any "
            "mesoscope preparation procedures, such as motion correction, before proceeding further. This is the last "
            "manual checkpoint, entering 'y' after this message will begin the experiment."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Generates a snapshot of all zaber positions. This serves as an early checkpoint in case the runtime has to be
        # aborted in a non-graceful way (without running the stop() sequence). This way, next runtime will restart with
        # the calibrated zaber positions.
        head_bar_positions = self._headbar.get_positions()
        lickport_positions = self._lickport.get_positions()
        zaber_positions = _ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)
        message = "HeadBar and LickPort positions: Saved."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Enables body cameras. Starts frame saving for all cameras
        self._cameras.start_body_cameras()
        self._cameras.save_face_camera_frames()
        self._cameras.save_body_camera_frames()

        # Starts all microcontroller interfaces
        self._microcontrollers.start()

        # Establishes a direct communication with Unity over MQTT. This is in addition to some ModuleInterfaces using
        # their own communication channels.
        self._unity.connect()

        # Queries the task cue (segment) sequence from Unity. This also acts as a check for whether Unity is
        # running and is configured appropriately. The extracted sequence data is logged as a sequence of byte
        # values.
        cue_sequence = self._get_cue_sequence()
        package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=cue_sequence,
        )
        self._logger.input_queue.put(package)

        message = "Unity Game Engine: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts monitoring the sensors used regardless of the VR state. Currently, this is the lick sensor state and
        # the mesoscope frame ttl module state.
        self._microcontrollers.enable_mesoscope_frame_monitoring()
        self._microcontrollers.enable_lick_monitoring()

        # Sets the rest of the subsystems to use the REST state.
        self.vr_rest()

        # Starts mesoscope frame acquisition. This also verifies that the mesoscope responds to triggers and
        # actually starts acquiring frames using the _mesoscope_frame interface above.
        self._start_mesoscope()

        message = "Mesoscope frame acquisition: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # The setup procedure is complete.
        self._started = True

        message = "MesoscopeExperiment assets: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops and terminates the MesoscopeExperiment runtime.

        This method achieves two main purposes. First, releases the hardware resources used during the experiment
        runtime by various system components. Second, it pulls all collected data to the VRPC and runs the preprocessing
        pipeline on the data to prepare it for long-term storage and further processing.
        """

        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        message = "Terminating MesoscopeExperiment runtime..."
        console.echo(message=message, level=LogLevel.INFO)

        # Resets the _started tracker
        self._started = False

        # Switches the system into the rest state. Since REST state has most modules set to stop-friendly states,
        # this is used as a shortcut to prepare the VR system for shutdown.
        self.vr_rest()

        # Stops mesoscope frame acquisition.
        self._microcontrollers.stop_mesoscope()
        self._timestamp_timer.reset()  # Resets the timestamp timer. It is now co-opted to enforce the shutdown delay
        message = "Mesoscope stop command: Sent."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops all cameras.
        self._cameras.stop()

        # Manually stops hardware modules not stopped by the REST state. This excludes mesoscope frame monitoring, which
        # is stopped separately after the 5-second delay (see below).
        self._microcontrollers.disable_lick_monitoring()
        self._microcontrollers.disable_torque_monitoring()

        # Delays for 5 seconds to give mesoscope time to stop acquiring frames. Primarily, this ensures that all
        # mesoscope frames have recorded acquisition timestamps. This implementation times the delay relative to the
        # mesoscope shutdown command and allows running other shutdown procedures in-parallel with the mesoscope
        # shutdown processing.
        while self._timestamp_timer.elapsed < 5000000:
            continue

        # Stops mesoscope frame monitoring. At this point, the mesoscope should have stopped acquiring frames.
        self._microcontrollers.disable_mesoscope_frame_monitoring()

        # Stops all microcontroller interfaces
        self._microcontrollers.stop()

        # Stops the data logger instance
        self._logger.stop()

        message = "Data Logger: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Updates the contents of the pregenerated descriptor file and dumps it as a .yaml into the root raw_data
        # session directory. This needs to be done after the microcontrollers and loggers have been stopped to ensure
        # that the reported dispensed_water_volume_ul is accurate.
        delivered_water = self._microcontrollers.total_delivered_volume
        # Overwrites the delivered water volume with the volume recorded over the runtime.
        self._descriptor.dispensed_water_volume_ul = delivered_water
        self._descriptor.to_yaml(file_path=self._session_data.session_descriptor_path)

        # Generates the snapshot of the current HeadBar and LickPort positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are reset back to parking position.
        head_bar_positions = self._headbar.get_positions()
        lickport_positions = self._lickport.get_positions()
        zaber_positions = _ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)

        # Moves the LickPort to the mounting position to assist removing the animal from the rig.
        self._lickport.mount_position()

        # Notifies the user about the volume of water dispensed during runtime, so that they can ensure the mouse
        # get any leftover daily water limit.
        message = (
            f"During runtime, the system dispensed ~{delivered_water} uL of water to the animal. "
            f"If the animal is on water restriction, make sure it receives any additional water, if the dispensed "
            f"volume does not cover the daily water limit for that animal."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Instructs the user to remove the objective and the animal before resetting all zaber motors.
        message = (
            "Preparing to reset the HeadBar and LickPort motors. Uninstall the mesoscope objective, remove the animal "
            "from the VR rig and swivel the VR screens out. Failure to do so may DAMAGE the mesoscope objective and "
            "HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Parks both controllers and then disconnects from their Connection classes. Note, the parking is performed
        # in-parallel
        self._headbar.park_position(wait_until_idle=False)
        self._lickport.park_position(wait_until_idle=True)
        self._headbar.wait_until_idle()
        self._headbar.disconnect()
        self._lickport.disconnect()

        message = "HeadBar and LickPort motors: Reset."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Generates the data processing checklist and saves it to disk. This is used to indicate that raw data
        # acquisition was completed successfully, in case the preprocessing fails. Any session that does not have this
        # checklist is NOT guaranteed to have data integrity.
        checkist_path = self._session_data.data_processing_tracker_path
        checklist = _ProcessingTracker()
        checklist.to_yaml(file_path=checkist_path)

        # Prompts the user to add their notes to the appropriate section of the descriptor file. This has to be done
        # before processing so that the notes are properly transferred to the NAS and server. Also, this makes it more
        # obvious to the user when it is safe to start preparing for the next session and leave the current one
        # processing the data.
        message = (
            f"Data acquisition: Complete. Open the session descriptor file stored in session's raw_data folder and "
            f"update the notes session with the notes taken during runtime. This is the last manual checkpoint, "
            f"entering 'y' after this message will begin data preprocessing and, after that, transmit it to the BioHPC "
            f"server and the NAS storage. It is safe to start preparing for the next session after hitting 'y'."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        message = "Initializing data preprocessing..."
        console.echo(message=message, level=LogLevel.INFO)

        # Compresses all logs into a single .npz file. This is done both for long-term storage optimization and to
        # allow parsing the data. Note, to minimize the time taken by data preprocessing, we disable integrity
        # verification and compression. The data is just aggregated into an uncompressed .npz file for each source.
        self._logger.compress_logs(
            remove_sources=True, memory_mapping=False, verbose=True, compress=False, verify_integrity=False
        )

        # Parses behavioral data from the compressed logs and uses it to generate the behavioral_dataset.parquet file.
        # Also, extracts camera frame timestamps for each camera and saves them as a separate .parquet file to optimize
        # further camera frame processing.
        self._process_log_data()

        # Pulls the frames and motion estimation data from the ScanImagePC into the local data directory.
        self._session_data.pull_mesoscope_data()

        # Preprocesses the pulled mesoscope data.
        self._session_data.process_mesoscope_data()

        # Renames the video files generated during runtime to use human-friendly camera names, rather than ID-codes.
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("051.mp4"),
            new=self._session_data.camera_frames_path.joinpath("face_camera.mp4"),
        )
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("062.mp4"),
            new=self._session_data.camera_frames_path.joinpath("left_camera.mp4"),
        )
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("073.mp4"),
            new=self._session_data.camera_frames_path.joinpath("right_camera.mp4"),
        )

        # Pushes the processed data to the NAS and BioHPC server.
        self._session_data.push_data()

        message = "Data preprocessing: complete. MesoscopeExperiment runtime: terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def vr_rest(self) -> None:
        """Switches the VR system to the rest state.

        In the rest state, the break is engaged to prevent the mouse from moving the wheel. The encoder module is
        disabled, and instead the torque sensor is enabled. The VR screens are switched off, cutting off light emission.
        By default, the VR system starts all experimental runtimes using the REST state.
        """

        # Ensures VR screens are turned OFF
        self._microcontrollers.disable_vr_screens()

        # Engages the break to prevent the mouse from moving the wheel
        self._microcontrollers.enable_break()

        # Temporarily suspends encoder monitoring. Since the wheel is locked, the mouse should not be able to produce
        # meaningful motion data.
        self._microcontrollers.disable_encoder_monitoring()

        # Initiates torque monitoring.The torque can only be accurately measured when the wheel is locked, as it
        # requires a resistance force to trigger the sensor.
        self._microcontrollers.enable_torque_monitoring()

        # Configures the state tracker to reflect the REST state
        self._change_vr_state(1)

    def vr_run(self) -> None:
        """Switches the VR system to the run state.

        In the run state, the break is disengaged to allow the mouse to freely move the wheel. The encoder module is
        enabled to record and share live running data with Unity, and the torque sensor is disabled. The VR screens are
        switched on to render the VR environment.
        """
        # Initializes encoder monitoring.
        self._microcontrollers.enable_encoder_monitoring()

        # Disables torque monitoring. To accurately measure torque, the sensor requires a resistance force provided by
        # the break. During running, measuring torque is not very reliable and adds little value compared to the
        # encoder.
        self._microcontrollers.disable_torque_monitoring()

        # Toggles the state of the VR screens ON.
        self._microcontrollers.enable_vr_screens()

        # Disengages the break to allow the mouse to move the wheel
        self._microcontrollers.disable_break()

        # Configures the state tracker to reflect RUN state
        self._change_vr_state(2)

    def _get_cue_sequence(self) -> NDArray[np.uint8]:
        """Requests Unity game engine to transmit the sequence of virtual reality track wall cues for the current task.

        This method is used as part of the experimental runtime startup process to both get the sequence of cues and
        verify that the Unity game engine is running and configured correctly.

        Returns:
            The NumPy array that stores the sequence of virtual reality segments as byte (uint8) values.

        Raises:
            RuntimeError: If no response from Unity is received within 2 seconds or if Unity sends a message to an
                unexpected (different) topic other than "CueSequence/" while this method is running.
        """
        # Initializes a second-precise timer to ensure the request is fulfilled within a 2-second timeout
        timeout_timer = PrecisionTimer("s")

        # Sends a request for the task cue (corridor) sequence to Unity GIMBL package.
        self._unity.send_data(topic="CueSequenceTrigger/")

        # Waits at most 2 seconds to receive the response
        timeout_timer.reset()
        while timeout_timer.elapsed < 2:
            # If Unity responds with the cue sequence message, attempts to parse the message
            if self._unity.has_data:
                topic: str
                payload: bytes
                topic, payload = self._unity.get_data()
                if topic == "CueSequence/":
                    # Extracts the sequence of cues that will be used during task runtime.
                    sequence = json.loads(payload.decode("utf-8"))["cue_sequence"]
                    sequence: NDArray[np.uint8] = np.array(sequence, dtype=np.uint8)
                    return sequence

                else:
                    # If the topic is not "CueSequence/", aborts with an error
                    message = (
                        f"Received an unexpected topic {topic} while waiting for Unity to respond to the cue sequence "
                        f"request. Make sure the Unity is not configured to send data to other topics monitored by the "
                        f"MesoscopeExperiment instance until the cue sequence is resolved as part of the start() "
                        f"method runtime."
                    )
                    console.error(message=message, error=RuntimeError)

        # If the loop above is escaped, this is due to not receiving any message from Unity. Raises an error.
        message = (
            f"The MesoscopeExperiment has requested the task Cue Sequence by sending the trigger to the "
            f"'CueSequenceTrigger/' topic and received no response for 2 seconds. It is likely that the Unity game "
            f"engine is not running or is not configured to work with MesoscopeExperiment."
        )
        console.error(message=message, error=RuntimeError)

        # This backup statement should not be reached, it is here to appease mypy
        raise RuntimeError(message)  # pragma: no cover

    def _start_mesoscope(self) -> None:
        """Sends the frame acquisition start TTL pulse to the mesoscope and waits for the frame acquisition to begin.

        This method is used internally to start the mesoscope frame acquisition as part of the experiment startup
        process. It is also used to verify that the mesoscope is available and properly configured to acquire frames
        based on the input triggers.

        Raises:
            RuntimeError: If the mesoscope does not confirm frame acquisition within 2 seconds after the
                acquisition trigger is sent.
        """

        # Initializes a second-precise timer to ensure the request is fulfilled within a 2-second timeout
        timeout_timer = PrecisionTimer("s")

        # Instructs the mesoscope to begin acquiring frames
        self._microcontrollers.start_mesoscope()

        # Waits at most 2 seconds for the mesoscope to begin sending frame acquisition timestamps to the PC
        timeout_timer.reset()
        while timeout_timer.elapsed < 2:
            # If the mesoscope starts scanning a frame, the method has successfully started the mesoscope frame
            # acquisition.
            if self._microcontrollers.mesoscope_frame_count:
                return

        # If the loop above is escaped, this is due to not receiving the mesoscope frame acquisition pulses.
        message = (
            f"The MesoscopeExperiment has requested the mesoscope to start acquiring frames and received no frame "
            f"acquisition trigger for 2 seconds. It is likely that the mesoscope has not been armed for frame "
            f"acquisition or that the mesoscope trigger or frame timestamp connection is not functional."
        )
        console.error(message=message, error=RuntimeError)

        # This code is here to appease mypy. It should not be reachable
        raise RuntimeError(message)  # pragma: no cover

    def _change_vr_state(self, new_state: int) -> None:
        """Updates and logs the new VR state.

        This method is used internally to timestamp and log VR state (stage) changes, such as transitioning between
        rest and run VR states.

        Args:
            new_state: The byte-code for the newly activated VR state.
        """
        self._vr_state = new_state  # Updates the VR state

        # Notifies the user about the new VR state.
        message = f"VR State: {self._state_map[self._vr_state]}."
        console.echo(message=message, level=LogLevel.INFO)

        # Logs the VR state update. Uses header-code 1 to indicate that the logged value is the VR state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([1, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def change_experiment_state(self, new_state: int) -> None:
        """Updates and logs the new experiment state.

        Use this method to timestamp and log experiment state (stage) changes, such as transitioning between different
        task goals.

        Args:
            new_state: The integer byte-code for the new experiment state. The code will be serialized as an uint8
                value, so only values between 0 and 255 inclusive are supported.
        """
        self._experiment_state = new_state  # Updates the tracked experiment state value

        # Notifies the user about the new Experiment state.
        message = f"Experiment State: {new_state}."
        console.echo(message=message, level=LogLevel.INFO)

        # Logs the VR state update. Uses header-code 2 to indicate that the logged value is the experiment state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([2, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def _process_log_data(self) -> None:
        """Extracts the data logged during runtime from .npz archive files and uses it to generate multiple .feather
        files inside the behavioral_data and camera_frames session subfolders.

        This method prepares the data for the main processing pipelines executed on the BioHPC server.
        """
        # Submits the processing tasks to be executed in-parallel
        with ProcessPoolExecutor() as executor:
            futures = set()

            # MicroController module data
            # noinspection PyProtectedMember,PyTypeChecker
            futures.add(
                executor.submit(
                    process_module_data,
                    self._microcontrollers._lick,
                    self._session_data.behavioral_data_path.joinpath("lick_data.feather"),
                    "lick_active",
                )
            )
            # noinspection PyProtectedMember
            futures.add(
                executor.submit(
                    process_module_data,
                    self._microcontrollers._reward,
                    self._session_data.behavioral_data_path.joinpath("valve_data.feather"),
                    "dispensed_water_μL",
                )
            )
            # noinspection PyProtectedMember,PyTypeChecker
            futures.add(
                executor.submit(
                    process_module_data,
                    self._microcontrollers._torque,
                    self._session_data.behavioral_data_path.joinpath("torque_data.feather"),
                    "torque_N_cm",
                )
            )
            # noinspection PyProtectedMember,PyTypeChecker
            futures.add(
                executor.submit(
                    process_module_data,
                    self._microcontrollers._encoder,
                    self._session_data.behavioral_data_path.joinpath("encoder_data.feather"),
                    "distance_cm",
                )
            )
            # noinspection PyProtectedMember,PyTypeChecker
            futures.add(
                executor.submit(
                    process_module_data,
                    self._microcontrollers._screens,
                    self._session_data.behavioral_data_path.joinpath("screens_data.feather"),
                    "screens_active",
                )
            )
            # noinspection PyProtectedMember,PyTypeChecker
            futures.add(
                executor.submit(
                    process_module_data,
                    self._microcontrollers._break,
                    self._session_data.behavioral_data_path.joinpath("break_data.feather"),
                    "break_active",
                )
            )

            # Camera logs
            futures.add(
                executor.submit(
                    process_camera_timestamps,
                    self._cameras.face_camera_log_path,
                    self._session_data.camera_frames_path.joinpath("face_camera_timestamps.feather"),
                )
            )
            futures.add(
                executor.submit(
                    process_camera_timestamps,
                    self._cameras.left_camera_log_path,
                    self._session_data.camera_frames_path.joinpath("left_camera_timestamps.feather"),
                )
            )
            futures.add(
                executor.submit(
                    process_camera_timestamps,
                    self._cameras.right_camera_log_path,
                    self._session_data.camera_frames_path.joinpath("right_camera_timestamps.feather"),
                )
            )

            # MesoscopeExperiment log
            log_path = self._logger.output_directory.joinpath(f"{self._source_id}_log.npz")
            futures.add(
                executor.submit(
                    _process_experiment_data, log_path, self._session_data.behavioral_data_path, self._cue_map
                )
            )

            # Shows progress with tqdm when not in batch mode
            with tqdm(
                total=len(futures),
                desc=f"Processing log files",
                unit="file",
            ) as pbar:
                for _ in as_completed(futures):
                    pbar.update(1)

    @property
    def trackers(self) -> tuple[SharedMemoryArray, SharedMemoryArray, SharedMemoryArray]:
        """Returns the tracker SharedMemoryArrays for (in this order) the LickInterface, ValveInterface, and
        EncoderInterface.

        These arrays should be passed to the BehaviorVisualizer class to monitor the lick, valve and running speed data
        in real time during the experiment session.
        """
        return (
            self._microcontrollers.lick_tracker,
            self._microcontrollers.valve_tracker,
            self._microcontrollers.distance_tracker,
        )


class _BehavioralTraining:
    """The base class for all behavioral training runtimes.

    This class provides methods for running the lick and run training sessions using a subset of the Mesoscope-VR
    system. It abstracts most low-level interactions with the VR system and the mesoscope via a simple
    high-level state API.

    This class also provides methods for limited preprocessing of the collected data. The preprocessing methods are
    designed to be executed after each training runtime to prepare the data for long-term storage and transmission
    over the network. Preprocessing methods use multiprocessing to optimize runtime performance and assume that the
    VRPC is kept mostly idle during data preprocessing.

    Notes:
        Calling this initializer does not start the Mesoscope-VR components. Use the start() method before issuing other
        commands to properly initialize all remote processes. This class reserves up to 11 CPU cores during runtime.

        Use the 'axtl-ports' cli command to discover the USB ports used by Ataraxis Micro Controller (AMC) devices.

        Use the 'axvs-ids' cli command to discover the camera indices used by the Harvesters-managed and
        OpenCV-managed cameras.

        Use the 'sl-devices' cli command to discover the serial ports used by the Zaber motion controllers.

    Args:
        session_data: An initialized SessionData instance. This instance is used to transfer the data between VRPC,
            BioHPC server, and the NAS during runtime. Each instance is initialized for the specific project, animal,
            and session combination for which the data is acquired.
        descriptor: A partially configured _LickTrainingDescriptor or _RunTrainingDescriptor instance. This instance is
            used to store session-specific information in a human-readable format.
        screens_on: Communicates whether the VR screens are currently ON at the time of class initialization.
        actor_port: The USB port used by the actor Microcontroller.
        sensor_port: The USB port used by the sensor Microcontroller.
        encoder_port: The USB port used by the encoder Microcontroller.
        headbar_port: The USB port used by the headbar Zaber motor controllers (devices).
        lickport_port: The USB port used by the lickport Zaber motor controllers (devices).
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters.
        face_camera_index: The index of the face camera in the list of all available Harvester-managed cameras.
        left_camera_index: The index of the left camera in the list of all available OpenCV-managed cameras.
        right_camera_index: The index of the right camera in the list of all available OpenCV-managed cameras.
        harvesters_cti_path: The path to the GeniCam CTI file used to connect to Harvesters-managed cameras.

    Attributes:
        _started: Tracks whether the VR system and training runtime are currently running.
        _lick_training: Tracks the training state used by the instance, which is required for log parsing.
        _descriptor: Stores the session descriptor instance.
        _logger: A DataLogger instance that collects behavior log data from all sources: microcontrollers and video
            cameras.
        _microcontrollers: Stores the _MicroControllerInterfaces instance that interfaces with all MicroController
            devices used during runtime.
        _cameras: Stores the _VideoSystems instance that interfaces with video systems (cameras) used during
            runtime.
        _headbar: Stores the _HeadBar class instance that interfaces with all HeadBar manipulator motors.
        _lickport: Stores the _LickPort class instance that interfaces with all LickPort manipulator motors.
        _screen_on: Tracks whether the VR displays are currently ON.
        _session_data: Stores the SessionData instance used to manage the acquired data.

    Raises:
        TypeError: If session_data argument has an invalid type.
    """

    def __init__(
        self,
        session_data: SessionData,
        descriptor: _LickTrainingDescriptor | _RunTrainingDescriptor,
        screens_on: bool = False,
        actor_port: str = "/dev/ttyACM0",
        sensor_port: str = "/dev/ttyACM1",
        encoder_port: str = "/dev/ttyACM2",
        headbar_port: str = "/dev/ttyUSB0",
        lickport_port: str = "/dev/ttyUSB1",
        valve_calibration_data: tuple[tuple[int | float, int | float], ...] = (
            (15000, 1.8556),
            (30000, 3.4844),
            (45000, 7.1846),
            (60000, 10.0854),
        ),
        face_camera_index: int = 0,
        left_camera_index: int = 0,
        right_camera_index: int = 2,
        harvesters_cti_path: Path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"),
    ) -> None:
        # Activates the console to display messages to the user if the console is disabled when the class is
        # instantiated.
        if not console.enabled:
            console.enable()

        # Creates the _started flag first to avoid leaks if the initialization method fails.
        self._started: bool = False

        # Determines the type of training carried out by the instance. This is needed for log parsing and
        # SessionDescriptor generation.
        self._lick_training: bool = False
        self._descriptor: _LickTrainingDescriptor | _RunTrainingDescriptor = descriptor

        # Input verification:
        if not isinstance(session_data, SessionData):
            message = (
                f"Unable to initialize the BehavioralTraining class. Expected a SessionData instance for "
                f"'session_data' argument, but instead encountered {session_data} of type "
                f"{type(session_data).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Defines other flags used during runtime:
        self._screen_on: bool = screens_on  # Usually this would be false, but this is not guaranteed

        # Saves the SessionData instance to class attribute so that it can be used from class methods. Since SessionData
        # resolves session directory structure at initialization, the instance is ready to resolve all paths used by
        # the training class instance.
        self._session_data: SessionData = session_data

        # Initializes the DataLogger instance used to log data from all microcontrollers, camera frame savers, and this
        # class instance.
        self._logger: DataLogger = DataLogger(
            output_directory=session_data.raw_data_path,
            instance_name="behavior",  # Creates behavior_log subfolder under raw_data
            sleep_timer=0,
            exist_ok=True,
            process_count=1,
            thread_count=10,
        )

        # Initializes the binding class for all MicroController Interfaces.
        self._microcontrollers: _MicroControllerInterfaces = _MicroControllerInterfaces(
            data_logger=self._logger,
            actor_port=actor_port,
            sensor_port=sensor_port,
            encoder_port=encoder_port,
            valve_calibration_data=valve_calibration_data,
            debug=False,
        )

        # Initializes the binding class for all VideoSystems.
        self._cameras: _VideoSystems = _VideoSystems(
            data_logger=self._logger,
            output_directory=self._session_data.camera_frames_path,
            face_camera_index=face_camera_index,
            left_camera_index=left_camera_index,
            right_camera_index=right_camera_index,
            harvesters_cti_path=harvesters_cti_path,
        )

        # While we can connect to ports managed by ZaberLauncher, ZaberLauncher cannot connect to ports managed via
        # software. Therefore, we have to make sure ZaberLauncher is running before connecting to motors.
        message = (
            "Preparing to connect to HeadBar and LickPort Zaber controllers. Make sure that ZaberLauncher app is "
            "running before proceeding further. If ZaberLauncher is not running, youi WILL NOT be able to manually "
            "control the HeadBar and LickPort motor positions until you reset the runtime."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Initializes the binding classes for the HeadBar and LickPort manipulator motors.
        self._headbar: _HeadBar = _HeadBar(
            headbar_port=headbar_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )
        self._lickport: _LickPort = _LickPort(
            lickport_port=lickport_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )

    def start(self) -> None:
        """Sets up all assets used during the training.

        This internal method establishes the communication with the microcontrollers, data logger cores, and video
        system processes.

        Notes:
            This method will not run unless the host PC has access to the necessary number of logical CPU cores
            and other required hardware resources (GPU for video encoding, etc.). This prevents using the class on
            machines that are unlikely to sustain the runtime requirements.

            As part of its runtime, this method will attempt to set Zaber motors for the HeadBar and LickPort to the
            positions that would typically be used during the mesoscope experiment runtime. Exercise caution and always
            monitor the system when it is running this method, as unexpected motor behavior can damage the mesoscope or
            harm the animal.

            Unlike the experiment class start(), this method does not preset the hardware module states during runtime.
            Call the desired training state method to configure the hardware modules appropriately for the chosen
            runtime mode.

        Raises:
            RuntimeError: If the host PC does not have enough logical CPU cores available.
        """
        # Prevents (re) starting an already started VR process.
        if self._started:
            return

        # 3 cores for microcontrollers, 1 core for the data logger, 6 cores for the current video_system
        # configuration (3 producers, 3 consumer), 1 core for the central process calling this method. 11 cores
        # total.
        if not os.cpu_count() >= 11:
            message = (
                f"Unable to start the BehavioralTraining runtime. The host PC must have at least 11 logical CPU "
                f"cores available for this class to work as expected, but only {os.cpu_count()} cores are "
                f"available."
            )
            console.error(message=message, error=RuntimeError)

        message = "Initializing BehavioralTraining assets..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts the data logger
        self._logger.start()
        message = "DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts the face camera. This starts frame acquisition and displays acquired frames to the user. However,
        # frame saving is disabled at this time. Body cameras are also disabled. This is intentional, as at this point
        # we want to minimize the number of active processes. This is helpful if this method is called while the
        # previous session is still running its data preprocessing pipeline and needs as many free cores as possible.
        self._cameras.start_face_camera()

        # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is safe to
        # proceed with motor movements.
        message = (
            "Preparing to move HeadBar into position. Swivel out the VR screens, and make sure the animal is NOT "
            "mounted on the rig. Failure to fulfill these steps may HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not intersect
        # with each other, so it is safe to move both assemblies at the same time.
        self._headbar.prepare_motors(wait_until_idle=False)
        self._lickport.prepare_motors(wait_until_idle=True)
        self._headbar.wait_until_idle()

        # Sets the motors into the mounting position. The HeadBar is either restored to the previous session position or
        # is set to the default mounting position stored in non-volatile memory. The LickPort is moved to a position
        # optimized for putting the animal on the VR rig.
        self._headbar.restore_position(wait_until_idle=False)
        self._lickport.mount_position(wait_until_idle=True)
        self._headbar.wait_until_idle()

        # Gives user time to mount the animal and requires confirmation before proceeding further.
        message = (
            "Preparing to move the LickPort into position. Mount the animal onto the VR rig. If necessary, adjust the "
            "HeadBar position to make sure the animal can comfortably run the task."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Restores the lickPort to the previous session's position or to the default parking position. This positions
        # the LickPort in a way that is easily accessible by the animal.
        self._lickport.restore_position()

        message = (
            "If necessary, adjust LickPort position to be easily reachable by the animal. Take extra care when moving "
            "the LickPort towards the animal! This is the last manual checkpoint, entering 'y' after this message will "
            "begin the training."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Generates a snapshot of all zaber positions. This serves as an early checkpoint in case the runtime has to be
        # aborted in a non-graceful way (without running the stop() sequence). This way, next runtime will restart with
        # the calibrated zaber positions.
        head_bar_positions = self._headbar.get_positions()
        lickport_positions = self._lickport.get_positions()
        zaber_positions = _ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)
        message = "HeadBar and LickPort positions: Saved."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Enables body cameras. Starts frame saving for all cameras
        self._cameras.start_body_cameras()
        self._cameras.save_face_camera_frames()
        self._cameras.save_body_camera_frames()

        # Initializes communication with the microcontrollers
        self._microcontrollers.start()

        # The setup procedure is complete.
        self._started = True

        message = "BehavioralTraining assets: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops and terminates the BehavioralTraining runtime.

        This method achieves two main purposes. First, releases the hardware resources used during the training runtime
        by various system components. Second, it runs the preprocessing pipeline on the data to prepare it for long-term
        storage and further processing.
        """

        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        message = "Terminating BehavioralTraining runtime..."
        console.echo(message=message, level=LogLevel.INFO)

        # Resets the _started tracker
        self._started = False

        # Stops all cameras.
        self._cameras.stop()

        # Manually stops all hardware modules before shutting down the microcontrollers
        self._microcontrollers.enable_break()
        self._microcontrollers.disable_lick_monitoring()
        self._microcontrollers.disable_torque_monitoring()
        self._microcontrollers.disable_encoder_monitoring()

        # Stops all microcontroller interfaces
        self._microcontrollers.stop()

        # Stops the data logger instance
        self._logger.stop()

        message = "Data Logger: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Updates the contents of the pregenerated descriptor file and dumps it as a .yaml into the root raw_data
        # session directory. This needs to be done after the microcontrollers and loggers have been stopped to ensure
        # that the reported dispensed_water_volume_ul is accurate.
        delivered_water = self._microcontrollers.total_delivered_volume
        if self._lick_training:
            # Overwrites the delivered water volume with the volume recorded over the runtime.
            self._descriptor.dispensed_water_volume_ul = delivered_water
            self._descriptor.to_yaml(file_path=self._session_data.session_descriptor_path)

        # Generates the snapshot of the current HeadBar and LickPort positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are reset back to parking position.
        head_bar_positions = self._headbar.get_positions()
        lickport_positions = self._lickport.get_positions()
        zaber_positions = _ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)

        # Moves the LickPort to the mounting position to assist removing the animal from the rig.
        self._lickport.mount_position()

        # Notifies the user about the volume of water dispensed during runtime, so that they can ensure the mouse
        # get any leftover daily water limit.
        message = (
            f"During runtime, the system dispensed ~{delivered_water} uL of water to the animal. "
            f"If the animal is on water restriction, make sure it receives any additional water, if the dispensed "
            f"volume does not cover the daily water limit for that animal."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Instructs the user to remove the objective and the animal before resetting all zaber motors.
        message = (
            "Preparing to reset the HeadBar and LickPort motors. Remove the animal from the VR rig and swivel the VR "
            "screens out. Failure to do so may HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Parks both controllers and then disconnects from their Connection classes. Note, the parking is performed
        # in-parallel
        self._headbar.park_position(wait_until_idle=False)
        self._lickport.park_position(wait_until_idle=True)
        self._headbar.wait_until_idle()
        self._headbar.disconnect()
        self._lickport.disconnect()

        message = "HeadBar and LickPort motors: Reset."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Prompts the user to add their notes to the appropriate section of the descriptor file. This has to be done
        # before processing so that the notes are properly transferred to the NAS and server. Also, this makes it more
        # obvious to the user when it is safe to start preparing for the next session and leave the current one
        # processing the data.
        message = (
            f"Data acquisition: Complete. Open the session descriptor file stored in session's raw_data folder and "
            f"update the notes session with the notes taken during runtime. This is the last manual checkpoint, "
            f"entering 'y' after this message will begin data preprocessing and, after that, transmit it to the BioHPC "
            f"server and the NAS storage. It is safe to start preparing for the next session after hitting 'y'."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        message = "Initializing data preprocessing..."
        console.echo(message=message, level=LogLevel.INFO)

        # Compresses all logs into a single .npz file. This is done both for long-term storage optimization and to
        # allow parsing the data. Note, to minimize the time taken by data preprocessing, we disable integrity
        # verification and compression. The data is just aggregated into an uncompressed .npz file for each source.
        self._logger.compress_logs(
            remove_sources=True, memory_mapping=False, verbose=True, compress=False, verify_integrity=False
        )

        # Parses behavioral data from the compressed logs into independent Apache Arrow Feather files.
        self._process_log_data()

        # Renames the video files generated during runtime to use human-friendly camera names, rather than ID-codes.
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("051.mp4"),
            new=self._session_data.camera_frames_path.joinpath("face_camera.mp4"),
        )
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("062.mp4"),
            new=self._session_data.camera_frames_path.joinpath("left_camera.mp4"),
        )
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("073.mp4"),
            new=self._session_data.camera_frames_path.joinpath("right_camera.mp4"),
        )

        # Pushes the processed data to the NAS and BioHPC server.
        self._session_data.push_data()

        message = "Data preprocessing: complete. BehavioralTraining runtime: terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def lick_train_state(self) -> None:
        """Configures the VR system for running the lick training.

        In the rest state, the break is engaged to prevent the mouse from moving the wheel. The encoder module is
        disabled, and the torque sensor is enabled. The VR screens are switched off, cutting off light emission.
        The lick sensor monitoring is on to record animal licking data.
        """

        # Ensures VR screens are turned OFF
        self._microcontrollers.disable_vr_screens()

        # Engages the break to prevent the mouse from moving the wheel
        self._microcontrollers.enable_break()

        # Ensures that encoder monitoring is disabled
        self._microcontrollers.disable_encoder_monitoring()

        # Initiates torque monitoring
        self._microcontrollers.enable_torque_monitoring()

        # Initiates lick monitoring
        self._microcontrollers.enable_lick_monitoring()

        # Sets the tracker
        self._lick_training = True

    def run_train_state(self) -> None:
        """Configures the VR system for running the run training.

        In the rest state, the break is disengaged, allowing the mouse to run on the wheel. The encoder module is
        enabled, and the torque sensor is disabled. The VR screens are switched off, cutting off light emission.
        The lick sensor monitoring is on to record animal licking data.
        """

        # Ensures VR screens are turned OFF
        self._microcontrollers.disable_vr_screens()

        # Disengages the break, enabling the mouse to run on the wheel
        self._microcontrollers.disable_break()

        # Ensures that encoder monitoring is enabled
        self._microcontrollers.enable_encoder_monitoring()

        # Ensures torque monitoring is disabled
        self._microcontrollers.disable_torque_monitoring()

        # Initiates lick monitoring
        self._microcontrollers.enable_lick_monitoring()

        # Sets the tracker
        self._lick_training = False

    def deliver_reward(self, reward_size: float = 5.0) -> None:
        """Uses the solenoid valve to deliver the requested volume of water in microliters.

        This method is used by the training runtimes to reward the animal with water as part of the training process.
        """
        self._microcontrollers.deliver_reward(volume=reward_size)

    @property
    def trackers(self) -> tuple[SharedMemoryArray, SharedMemoryArray, SharedMemoryArray]:
        """Returns the tracker SharedMemoryArrays for (in this order) the LickInterface, ValveInterface, and
        EncoderInterface.

        These arrays should be passed to the BehaviorVisualizer class to monitor the lick, valve and running speed data
        in real time during training.
        """
        return (
            self._microcontrollers.lick_tracker,
            self._microcontrollers.valve_tracker,
            self._microcontrollers.distance_tracker,
        )

    def _process_log_data(self) -> None:
        """Extracts the data logged during runtime from .npz archive files and uses it to generate multiple .feather
        files inside the behavioral_data and camera_frames session subfolders.

        This method prepares the data for the main processing pipelines executed on the BioHPC server.
        """
        # Submits the processing tasks to be executed in-parallel
        with ProcessPoolExecutor() as executor:
            futures = set()

            # Lick sensor data and Valve data are processed for all training runtimes.
            # noinspection PyProtectedMember,PyTypeChecker
            futures.add(
                executor.submit(
                    process_module_data,
                    self._microcontrollers._lick,
                    self._session_data.behavioral_data_path.joinpath("lick_data.feather"),
                    "lick_active",
                )
            )
            # noinspection PyProtectedMember
            futures.add(
                executor.submit(
                    process_module_data,
                    self._microcontrollers._reward,
                    self._session_data.behavioral_data_path.joinpath("valve_data.feather"),
                    "dispensed_water_μL",
                )
            )
            # Depending on the training type, also processes either torque or encoder data.
            if self._lick_training:
                # noinspection PyProtectedMember,PyTypeChecker
                futures.add(
                    executor.submit(
                        process_module_data,
                        self._microcontrollers._torque,
                        self._session_data.behavioral_data_path.joinpath("torque_data.feather"),
                        "torque_N_cm",
                    )
                )
            else:
                # noinspection PyProtectedMember,PyTypeChecker
                futures.add(
                    executor.submit(
                        process_module_data,
                        self._microcontrollers._encoder,
                        self._session_data.behavioral_data_path.joinpath("encoder_data.feather"),
                        "distance_cm",
                    )
                )

            # Also processes camera logs as part of the same parallel runtime.
            futures.add(
                executor.submit(
                    process_camera_timestamps,
                    self._cameras.face_camera_log_path,
                    self._session_data.camera_frames_path.joinpath("face_camera_timestamps.feather"),
                )
            )
            futures.add(
                executor.submit(
                    process_camera_timestamps,
                    self._cameras.left_camera_log_path,
                    self._session_data.camera_frames_path.joinpath("left_camera_timestamps.feather"),
                )
            )
            futures.add(
                executor.submit(
                    process_camera_timestamps,
                    self._cameras.right_camera_log_path,
                    self._session_data.camera_frames_path.joinpath("right_camera_timestamps.feather"),
                )
            )

            # Shows progress with tqdm when not in batch mode
            with tqdm(
                total=len(futures),
                desc=f"Processing log files",
                unit="file",
            ) as pbar:
                for _ in as_completed(futures):
                    pbar.update(1)


def lick_training_logic(
    runtime: _BehavioralTraining,
    average_reward_delay: int = 12,
    maximum_deviation_from_mean: int = 6,
    maximum_water_volume: float = 1.0,
    maximum_training_time: int = 20,
) -> None:
    """Encapsulates the logic used to train animals how to operate the lick port.

    The lick training consists of delivering randomly spaced 5 uL water rewards via the Valve module to teach the
    animal that water comes out of the lick port. Each reward is delivered at a pseudorandom delay after the previous
    reward or training onset. Reward delay sequence is generated before training runtime by sampling a uniform
    distribution centered at 'average_reward_delay' with lower and upper bounds defined by
    'maximum_deviation_from_mean'. The training continues either until the valve delivers the 'maximum_water_volume' in
    milliliters or until the 'maximum_training_time' in minutes is reached, whichever comes first.

    Args:
        runtime: The initialized _BehavioralTraining instance that manages all Mesoscope-VR components used by this
            training runtime.
        average_reward_delay: The average time, in seconds, that separates two reward deliveries. This is used to
            generate the reward delay sequence as the center of the uniform distribution from which delays are sampled.
        maximum_deviation_from_mean: The maximum deviation from the average reward delay, in seconds. This determines
            the upper and lower boundaries for the data sampled from the uniform distribution centered at the
            average_reward_delay.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered during this runtime.
        maximum_training_time: The maximum time, in minutes, to run the training.

    Notes:
        All delays fall in the range of average_reward_delay +- maximum_deviation_from_mean.

        This function acts on top of the BehavioralTraining class and provides the overriding logic for the lick
        training process. During experiments, runtime logic is handled by Unity game engine, so specialized control
        functions are only required when training the animals without Unity.
    """
    # Initializes the timer used to enforce reward delays
    delay_timer = PrecisionTimer("us")

    # Uses runtime tracker extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers
    visualizer = BehaviorVisualizer(
        lick_tracker=lick_tracker, valve_tracker=valve_tracker, distance_tracker=speed_tracker
    )

    message = f"Generating the pseudorandom reward delay sequence..."
    console.echo(message=message, level=LogLevel.INFO)

    # Computes lower and upper boundaries for the reward delay
    lower_bound = average_reward_delay - maximum_deviation_from_mean
    upper_bound = average_reward_delay + maximum_deviation_from_mean

    # Converts maximum volume to uL and divides it by 5 uL (reward size) to get the number of delays to sample from
    # the delay distribution
    num_samples = np.floor((maximum_water_volume * 1000) / 5).astype(np.uint64)

    # Generates samples from a uniform distribution within delay bounds
    samples = np.random.uniform(lower_bound, upper_bound, num_samples)

    # Calculates cumulative training time for each sampled delay. This communicates the total time passed when each
    # reward is delivered to the animal
    cumulative_time = np.cumsum(samples)

    # Finds the maximum number of samples that fits within the maximum training time. This assumes that to consume 1
    # ml of water, the animal would likely need more time than the maximum allowed training time, so we need to slice
    # the sampled delay array to fit within the time boundary.
    max_samples_idx = np.searchsorted(cumulative_time, maximum_training_time * 60, side="right")

    # Slices the samples array to make the total training time be roughly the maximum requested duration. Converts each
    # delay from seconds to microseconds and rounds to the nearest integer. This is done to make delays compatible with
    # PrecisionTimer class.
    reward_delays: NDArray[np.uint64] = np.round(samples[:max_samples_idx] * 1000000, decimals=0).astype(np.uint64)

    message = (
        f"Generated a sequence of {len(reward_delays)} rewards with the total cumulative runtime of "
        f"{np.round(cumulative_time[max_samples_idx - 1] / 60, decimals=3)} minutes."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Since we preset the descriptor class before passing the runtime to this function, the maximum training time may
    # actually not be accurate. This would be the case if the training runtime is limited by the maximum allowed water
    # delivery volume and not time. In this case, updates the training time to reflect the factual training time. This
    # would be the case if the reward delays array size is the same as the cumulative time array size, indicating no
    # slicing was performed due to session time constraints.
    if len(reward_delays) == len(cumulative_time):
        # Actual session time is the accumulated delay converted from seconds to minutes at the last index.
        # noinspection PyProtectedMember
        runtime._descriptor.training_time_m = np.ceil(cumulative_time[-1] / 60)

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Starts the visualizer process
    visualizer.initialize()

    # Configures all system components to support lick training
    runtime.lick_train_state()

    # Initializes the listener instance used to detect training abort signals and manual reward trigger signals sent
    # via the keyboard.
    listener = KeyboardListener()

    message = (
        f"Initiating lick training procedure. Press 'ESC' + 'q' to immediately abort the training at any "
        f"time. Press 'ESC' + 'r' to deliver 5 uL of water to the animal."
    )
    console.echo(message=message, level=LogLevel.INFO)

    # This tracker is used to terminate the training if manual abort command is sent via the keyboard
    terminate = False

    # Loops over all delays and delivers reward via the lick tube as soon as the delay expires.
    delay_timer.reset()
    for delay in tqdm(
        reward_delays,
        desc="Running lick training",
        unit="reward",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} rewards [{elapsed}]",
    ):
        # This loop is executed while the code is waiting for the delay to pass. Anything that needs to be done during
        # the delay has to go here
        while delay_timer.elapsed < delay:
            # Updates the visualizer plot ~every 30 ms. This should be enough to reliably capture all events of
            # interest and appear visually smooth to human observers.
            visualizer.update()

            # If the listener detects the default abort sequence, terminates the runtime.
            if listener.exit_signal:
                terminate = True  # Sets the terminate flag
                break  # Breaks the while loop

            # If the listener detects a reward delivery signal, delivers the reward to the animal
            if listener.reward_signal:
                runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water

        # If the user sent the abort command, terminates the training early
        if terminate:
            message = (
                "Lick training abort signal detected. Aborting the lick training with a graceful shutdown procedure."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            break  # Breaks the for loop

        # Once the delay is up, triggers the solenoid valve to deliver water to the animal and starts timing the next
        # reward delay
        runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water
        delay_timer.reset()

    # Shutdown sequence:
    message = (
        f"Training runtime: Complete. Delaying for additional {lower_bound} seconds to ensure the animal "
        f"has time to consume the final dispensed reward."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)
    delay_timer.delay_noblock(lower_bound * 1000000)  # Converts to microseconds before delaying

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()


def calibrate_valve_logic(
    actor_port: str,
    headbar_port: str,
    lickport_port: str,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
):
    """Encapsulates the logic used to fill, empty, check, and calibrate the water valve.

    This runtime allows interfacing with the water valve outside of training and experiment runtime contexts. Usually,
    this is done at the beginning and the end of each experimental / training day to ensure the valve operates smoothly
    during runtimes.

    Args:
        actor_port: The USB port to which the Actor Ataraxis Micro Controller (AMC) is connected.
        headbar_port: The USB port used by the headbar Zaber motor controllers (devices).
        lickport_port: The USB port used by the lickport Zaber motor controllers (devices).
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters.

    Notes:
        This runtime will position the Zaber motors to facilitate working with the valve.
    """
    # Enables the console
    if not console.enabled:
        console.enable()

    # Initializes a timer used to optimize console printouts for using the valve in debug mode (which also posts
    # things to console).
    delay_timer = PrecisionTimer("s")

    message = f"Initializing calibration assets..."
    console.echo(message=message, level=LogLevel.INFO)

    # Runs all calibration procedures inside a temporary directory which is deleted at the end of runtime.
    with tempfile.TemporaryDirectory(prefix="sl_valve_") as output_path:
        output_path = Path(output_path)

        # Initializes the data logger. Due to how the MicroControllerInterface class is implemented, this is required
        # even for runtimes that do not need to save data.
        logger = DataLogger(
            output_directory=output_path,
            instance_name="temp",
            exist_ok=True,
            process_count=1,
            thread_count=10,
        )
        logger.start()

        message = f"DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Initializes HeadBar and LickPort binding classes
        headbar = _HeadBar(headbar_port, output_path.joinpath("zaber_positions.yaml"))
        lickport = _LickPort(lickport_port, output_path.joinpath("zaber_positions.yaml"))

        message = f"Zaber controllers: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Initializes the Actor MicroController with the valve module. Ignores all other modules at this time.
        valve: ValveInterface = ValveInterface(valve_calibration_data=valve_calibration_data, debug=True)
        controller: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=actor_port,
            data_logger=logger,
            module_interfaces=(valve,),
        )
        controller.start()
        controller.unlock_controller()

        # Delays for 2 seconds for the valve to initialize and send the state message. This avoids the visual clash
        # with he zaber positioning dialog
        delay_timer.delay_noblock(delay=1)

        message = f"Actor MicroController: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is safe to
        # proceed with motor movements.
        message = (
            "Preparing to move HeadBar and LickPort motors. Remove the mesoscope objective, swivel out the VR screens, "
            "and make sure the animal is NOT mounted on the rig. Failure to fulfill these steps may DAMAGE the "
            "mesoscope and / or HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not intersect
        # with each other, so it is safe to move both assemblies at the same time.
        headbar.prepare_motors(wait_until_idle=False)
        lickport.prepare_motors(wait_until_idle=True)
        headbar.wait_until_idle()

        # Moves the motors in calibration position.
        headbar.calibrate_position(wait_until_idle=False)
        lickport.calibrate_position(wait_until_idle=True)
        headbar.wait_until_idle()

        message = f"HeadBar and LickPort motors: Positioned for calibration runtime."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Notifies the user about supported calibration commands
        message = (
            "Supported Calibration commands: open, close, reference, reward, "
            "calibrate_15, calibrate_30, calibrate_45, calibrate__60. Use 'q' command to terminate the runtime."
        )
        console.echo(message=message, level=LogLevel.INFO)

        while True:
            command = input()  # Silent input to avoid visual spam.

            if command == "open":
                message = f"Opening valve."
                console.echo(message=message, level=LogLevel.INFO)
                valve.toggle(state=True)

            if command == "close":
                message = f"Closing valve."
                console.echo(message=message, level=LogLevel.INFO)
                valve.toggle(state=False)

            if command == "reward":
                message = f"Delivering 5 uL water reward."
                console.echo(message=message, level=LogLevel.INFO)
                pulse_duration = valve.get_duration_from_volume(target_volume=5.0)
                valve.set_parameters(pulse_duration=pulse_duration)
                valve.send_pulse()

            if command == "reference":
                message = f"Running the reference (200 x 5 uL pulse time) valve calibration procedure."
                console.echo(message=message, level=LogLevel.INFO)
                pulse_duration = valve.get_duration_from_volume(target_volume=5.0)
                valve.set_parameters(pulse_duration=pulse_duration)
                valve.calibrate()

            if command == "calibrate_15":
                message = f"Running 15 ms pulse valve calibration."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(pulse_duration=np.uint32(15000))  # 15 ms in us
                valve.calibrate()

            if command == "calibrate_30":
                message = f"Running 30 ms pulse valve calibration."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(pulse_duration=np.uint32(30000))  # 30 ms in us
                valve.calibrate()

            if command == "calibrate_45":
                message = f"Running 45 ms pulse valve calibration."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(pulse_duration=np.uint32(45000))  # 45 ms in us
                valve.calibrate()

            if command == "calibrate_60":
                message = f"Running 60 ms pulse valve calibration."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(pulse_duration=np.uint32(60000))  # 60 ms in us
                valve.calibrate()

            if command == "q":
                message = f"Terminating valve calibration runtime."
                console.echo(message=message, level=LogLevel.INFO)
                break

        # Instructs the user to remove the objective and the animal before resetting all zaber motors.
        message = (
            "Preparing to reset the HeadBar and LickPort motors. Remove all objects sued during calibration, such as "
            "water collection flasks, from the Mesoscope-VR cage."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Shuts down zaber bindings
        headbar.park_position(wait_until_idle=False)
        lickport.park_position(wait_until_idle=True)
        headbar.wait_until_idle()
        headbar.disconnect()
        lickport.disconnect()

        message = f"HeadBar and LickPort connections: terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Shuts down microcontroller interfaces
        controller.stop()

        message = f"Actor MicroController: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops the data logger
        logger.stop()

        message = f"DataLogger: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # The logs will be cleaned up by deleting the temporary directory when this runtime exits.


def run_train_logic(
    runtime: _BehavioralTraining,
    initial_speed_threshold: float = 0.1,
    initial_duration_threshold: float = 0.1,
    speed_increase_step: float = 0.25,
    duration_increase_step: float = 0.25,
    increase_threshold: float = 0.1,
    maximum_speed_threshold: float = 10.0,
    maximum_duration_threshold: float = 10.0,
    maximum_water_volume: float = 1.0,
    maximum_training_time: int = 40,
) -> None:
    """Encapsulates the logic used to train animals how to run on the VR wheel.

    The run training consists of making the animal run on the wheel with a desired speed, in centimeters per second,
    maintained for the desired duration of seconds. This is used train the animal to exhibit good running speed and
    endurance. Each time the animal satisfies the speed and duration threshold, it receives 5 uL of water reward, and
    the speed and durations trackers reset for the next 'epoch'. If the animal performs well and receives many water
    rewards, the speed and duration thresholds increase to make the task more challenging. This is used to progressively
    train the animal to run better and to prevent the animal from completing the training too early.

    Notes:
        This function is highly configurable and can be adapted to a wide range of training scenarios. A central aim for
        designing the function was to go away from the arbitrary, experimenter-determined reward brackets in favor of
        a more principled approach to training.

        This function acts on top of the BehavioralTraining class and provides the overriding logic for the run
        training process. During experiments, runtime logic is handled by Unity game engine, so specialized control
        functions are only required when training the animals without Unity.

    Args:
        runtime: The initialized _BehavioralTraining instance that manages all Mesoscope-VR components used by this
            training runtime.
        initial_speed_threshold: The initial speed threshold, in centimeters per second, that the animal must maintain
            to receive water rewards.
        initial_duration_threshold: The initial duration threshold, in seconds, that the animal must maintain
            above-threshold running speed to receive water rewards.
        speed_increase_step: The step size, in centimeters per second, to increase the speed threshold by each time the
            animal receives 'increase_threshold' milliliters of water.
        duration_increase_step: The step size, in seconds, to increase the duration threshold by each time the animal
            receives 'increase_threshold' milliliters of water.
        increase_threshold: The volume of water, in milliliters, the animal should receive for the speed and duration
            threshold to be increased by one step. Note, the animal will at most get 'maximum_water_volume' of water,
            so this parameter effectively controls how many increases will be made during runtime, assuming the maximum
            training time is not reached.
        maximum_speed_threshold: The maximum speed threshold, in centimeters per second, that the animal must maintain
            to receive water rewards. Once this threshold is reached, it will not be increased further regardless of how
            much water is delivered to the animal.
        maximum_duration_threshold:
            The maximum duration threshold, in seconds, that the animal must maintain above-threshold running speed to
            receive water rewards. Once this threshold is reached, it will not be increased further regardless of
            how much water is delivered to the animal.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered during this runtime.
        maximum_training_time: The maximum time, in minutes, to run the training.
    """
    # Initializes the timer that keeps the training running until the time threshold is reached
    runtime_timer = PrecisionTimer("s")

    # Also initializes the timer used to track how long the animal maintains above-threshold running speed.
    speed_timer = PrecisionTimer("ms")

    # Converts all arguments used to determine the speed and duration threshold over time into numpy variables to
    # optimize main loop runtime speed:
    initial_speed = np.float64(initial_speed_threshold)  # In centimeters per second
    maximum_speed = np.float64(maximum_speed_threshold)  # In centimeters per second
    speed_step = np.float64(speed_increase_step)  # In centimeters per second

    initial_duration = np.float64(initial_duration_threshold * 1000)  # In milliseconds
    maximum_duration = np.float64(maximum_duration_threshold * 1000)  # In milliseconds
    duration_step = np.float64(duration_increase_step * 1000)  # In milliseconds

    water_threshold = np.float64(increase_threshold * 1000)  # In microliters
    maximum_water_volume = np.float64(maximum_water_volume * 1000)  # In microliters

    # Converts the training time from minutes to seconds to make it compatible with the timer precision.
    training_time = maximum_training_time * 60

    # Uses runtime trackers extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers
    visualizer = BehaviorVisualizer(
        lick_tracker=lick_tracker, valve_tracker=valve_tracker, distance_tracker=speed_tracker
    )

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Starts the visualizer process
    visualizer.initialize()

    # Updates the threshold lines to use the initial speed and duration values
    visualizer.update_speed_thresholds(speed_threshold=initial_speed, duration_threshold=initial_duration)

    # Configures all system components to support run training
    runtime.run_train_state()

    # Initializes the listener instance used to enable keyboard-driven training runtime control.
    listener = KeyboardListener()

    message = (
        f"Initiating run training procedure. Press 'ESC' + 'q' to immediately abort the training at any "
        f"time. Press 'ESC' + 'r' to deliver 5 uL of water to the animal. Use 'ESC' + Up / Down arrows to modify the "
        f"running speed threshold. Use 'ESC' + Left / Right arrows to modify the running duration threshold."
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Creates a tqdm progress bar that tracks the overall training progress and communicates the current speed and
    # duration threshold
    progress_bar = tqdm(
        total=training_time,
        desc="Run training progress",
        unit="second",
    )

    # Tracks the number of training seconds elapsed at each progress bar update. This is used to update the progress bar
    # with each passing second of training.
    previous_time = 0

    # Tracks when speed and / or duration thresholds are updated. This is necessary to redraw the threshold lines in
    # the visualizer plot
    previous_speed_threshold = np.float64(0)
    previous_duration_threshold = np.float64(0)

    # Initializes the main training loop. The loop will run either until the total training time expires, the maximum
    # volume of water is delivered or the loop is aborted by the user.
    runtime_timer.reset()
    speed_timer.reset()  # It is critical to reset BOTh timers at the same time.
    while runtime_timer.elapsed < training_time:
        # Updates the total volume of water dispensed during runtime at each loop iteration.
        dispensed_water_volume = valve_tracker.read_data(index=1, convert_output=False)

        # Determines how many times the speed and duration thresholds have been increased based on the difference
        # between the total delivered water volume and the increase threshold. This dynamically adjusts the running
        # speed and duration thresholds with delivered water volume, ensuring the animal has to try progressively
        # harder to keep receiving water.
        increase_steps: np.float64 = np.floor(dispensed_water_volume / water_threshold)

        # Note, the thresholds account for the user input by factoring in the speed and duration modifier obtained from
        # the keyboard listener.
        speed_threshold: np.float64 = np.min(
            initial_speed + ((increase_steps + listener.speed_modifier) * speed_step), maximum_speed
        )
        speed_threshold = np.max(speed_threshold, 0)  # In case the speed threshold becomes negative, sets it to 0.
        duration_threshold: np.float64 = np.min(
            initial_duration + ((increase_steps + listener.duration_modifier) * duration_step), maximum_duration
        )
        # In case the duration threshold becomes negative, sets it to 0.
        duration_threshold = np.max(duration_threshold, 0)

        # If any of the threshold changed relative to the previous loop iteration, updates the visualizer and previous
        # threshold trackers with new data.
        if duration_threshold != previous_duration_threshold or previous_speed_threshold != speed_threshold:
            visualizer.update_speed_thresholds(speed_threshold, duration_threshold)
            previous_speed_threshold = speed_threshold
            previous_duration_threshold = duration_threshold

        # Reads the animal's running speed from the visualizer. The visualizer uses the distance tracker to calculate
        # the running speed of the animal over 100 millisecond windows. This accesses the result of this computation and
        # uses it to determine whether the animal is performing above the threshold.
        current_speed = visualizer.running_speed

        # If the speed is above the speed threshold, and the animal has been maintaining the above-threshold speed for
        # the required duration, delivers 5 uL of water. If the speed is above threshold, but the animal has not yet
        # maintained the required duration, the loop will keep cycling and accumulating the timer count. This is done
        # until the animal either reaches the required duration or drops below the speed threshold.
        if current_speed >= speed_threshold and speed_timer.elapsed >= duration_threshold:
            runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water

            # Also resets the timer. While mice typically stop to consume water rewards, which would reset the timer,
            # this guards against animals that carry on running without consuming water rewards.
            speed_timer.reset()

        # If the current speed is below the speed threshold, resets the speed timer.
        elif current_speed < speed_threshold:
            speed_timer.reset()

        # Updates the progress bar with each elapsed second. Note, this is technically not safe in case multiple seconds
        # pass between reaching this conditional, but we know empirically that the loop will run at millisecond
        # intervals, so it is not a concern.
        if runtime_timer.elapsed > previous_time:
            previous_time = runtime_timer.elapsed  # Updates the previous time for the next progress bar update
            progress_bar.update(1)

        # Updates the visualizer plot
        visualizer.update()

        # If the total volume of water dispensed during runtime exceeds the maximum allowed volume, aborts the
        # training early with a success message.
        if dispensed_water_volume > maximum_water_volume:
            message = (
                f"Run training has delivered the maximum allowed volume of water ({maximum_water_volume} ml). Aborting "
                f"the training process."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)
            break

        # If the listener detects a reward delivery signal, delivers the reward to the animal.
        if listener.reward_signal:
            runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water

        # If the user sent the abort command, terminates the training early with an error message.
        if listener.exit_signal:
            message = "Run training abort signal detected. Aborting the training with a graceful shutdown procedure."
            console.echo(message=message, level=LogLevel.ERROR)
            break

    # Close the progress bar
    progress_bar.close()

    # Shutdown sequence:
    message = f"Training runtime: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()


def run_experiment_logic() -> None:
    """Provides the reference implementation of experimental runtime that uses the Mesoscope-VR system.

    This function is not intended to be used during real experiments. Instead, it demonstrates how to implement and
    experiment and is used during testing and calibration of the Mesoscope-VR system. It uses hardcoded default runtime
    parameters and should not be modified or called by end-users.
    """

    rest_duration = 0.5 * 60  # Duration of rest phase
    run_duration = 5 * 60  # Duration of run phrase
    total_duration = rest_duration + run_duration  # Total duration of the experiment
    runtime_timer = PrecisionTimer("s")  # TImer to enforce phase durations

    # Generates the runtime class and other assets
    session_data = SessionData(project_name="TestMice", animal_name="666")
    descriptor = _MesoscopeExperimentDescriptor()
    runtime = MesoscopeExperiment(
        session_data=session_data,
        descriptor=descriptor,
        cue_length_map={0: 30, 1: 30, 2: 30, 3: 30, 4: 30},  # Ivan's task, version with 4 cues and 4 gray regions
    )

    # Uses runtime trackers extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers
    visualizer = BehaviorVisualizer(
        lick_tracker=lick_tracker, valve_tracker=valve_tracker, distance_tracker=speed_tracker
    )

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Starts the visualizer process
    visualizer.initialize()

    # Initializes the keyboard listener to support aborting test runtimes.
    listener = KeyboardListener()

    # Main runtime loop
    once = True
    runtime.change_experiment_state(1)
    runtime_timer.reset()
    console.echo(message="Delaying for the resting phase.", level=LogLevel.INFO)
    while runtime_timer.elapsed < total_duration:
        visualizer.update()  # Continuously updates the visualizer

        if once and runtime_timer.elapsed > rest_duration:
            once = False
            console.echo(message="Delaying for the running phase.", level=LogLevel.INFO)
            runtime.vr_run()
            runtime.change_experiment_state(2)

        # If the user sent the abort command, terminates the runtime early with an error message.
        if listener.exit_signal:
            message = "Experiment runtime: aborted due to user request."
            console.echo(message=message, level=LogLevel.ERROR)
            break

    # Shutdown sequence:
    message = f"Experiment runtime: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()
