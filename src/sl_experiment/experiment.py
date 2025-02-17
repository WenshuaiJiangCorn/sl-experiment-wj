"""This module provides the main MesoscopeExperiment class that abstracts working with Sun lab's mesoscope-VR system
and ProjectData class that abstracts working with experimental data."""

import os
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer

from .module_interfaces import (
    TTLInterface,
    LickInterface,
    BreakInterface,
    ValveInterface,
    ScreenInterface,
    TorqueInterface,
    EncoderInterface,
)
from ataraxis_base_utilities import console, ensure_directory_exists, LogLevel
from ataraxis_data_structures import DataLogger, LogPackage, YamlConfig
from ataraxis_time.time_helpers import get_timestamp
from ataraxis_communication_interface import MicroControllerInterface, MQTTCommunication
from ataraxis_video_system import (
    VideoSystem,
    CameraBackends,
    VideoFormats,
    VideoCodecs,
    GPUEncoderPresets,
    InputPixelFormats,
    OutputPixelFormats,
)

from .zaber_bindings import ZaberConnection, ZaberAxis
from .transfer_tools import transfer_directory
from .packaging_tools import calculate_directory_checksum
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from enum import IntEnum
from dataclasses import dataclass


@dataclass()
class _ZaberPositions(YamlConfig):
    """This class is used to save and restore Zaber motor positions between sessions by saving them as .yaml file.

    This class is specifically designed to store, save, and load the positions of the LickPort and HeadBar motors
    (axes). It is used to both store Zaber motor positions for each session for future analysis and to (optionally)
    restore the same Zaber motor positions across consecutive experimental sessions for the same project and animal
    combination.

    Notes:
        This class is designed to be used by the MesoscopeExperiment class. Do not instantiate or load this class from
        .yaml files manually. Do not modify the data stored inside the .yaml file unless you know what you are doing.

        All positions are saved using native motor units. All class fields initialize to default placeholders that are
        likely NOT safe to apply to the VR system. Do not apply the positions loaded from the file unless you are
        certain they are safe to use.

        Exercise caution when working with Zaber motors. The motors can crush the manipulated objects into the
        environment.
    """

    headbar_z: int = 0
    """The absolute position, in native motor units, of the HeadBar z-axis motor."""
    headbar_pitch: int = 0
    """The absolute position, in native motor units, of the HeadBar pitch-axis motor."""
    headbar_roll: int = 0
    """The absolute position, in native motor units, of the HeadBar roll-axis motor."""
    lickport_z: int = 0
    """The absolute position, in native motor units, of the LickPort z-axis motor."""
    lickport_x: int = 0
    """The absolute position, in native motor units, of the LickPort x-axis motor."""
    lickport_y: int = 0
    """The absolute position, in native motor units, of the LickPort y-axis motor."""


class RuntimeModes(IntEnum):
    """Stores the integer codes for the runtime modes supported by the MesoscopeExperiment class.

    The MesoscopeExperiment takes the intended runtime mode code as one of its initialization arguments. The runtime
    mode broadly determines what assets need to be created, configured, and activated during runtime. It also determines
    the range of supported VR system states.

    Notes:
        Most users should never use code 5 (Calibration). This mode is intended for users with a good understanding of
        Mesoscope-VR hardware and software operation principles.
    """

    EXPERIMENT = 1
    """The main runtime mode intended to record experimental task performance. Experiment runtimes use all available 
    Mesoscope-VR system assets: cameras, microcontroller, Unity game engine, and the mesoscope. The start() method of 
    the MesoscopeExperiment, in addition to initializing all assets, verifies that Unity game engine is running and 
    instructs the mesoscope to start acquiring frames."""

    LICK_TRAINING = 2
    """The training mode used to teach naive animals to operate the lick tube (sensor) and consume water rewards 
    dispensed from the tube. The lick training runtime does not use Unity game engine or mesoscope. The runtime does 
    use microcontroller and camera assets, however, and generates the raw_data folder for the training session."""

    RUN_TRAINING = 3
    """The training mode used to teach naive animals how to run on the unlocked treadmill wheel. The run training 
    runtime does not use Unity game engine or mesoscope. The runtime does use microcontroller and camera assets, 
    however, and generates the raw_data folder for the training session."""

    VALVE_CALIBRATION = 4
    """The service mode used to test, calibrate, fill and empty the water reward valve. Typically, this mode would be 
    used twice a day: before the first experimental runtime of the day (to fill and test the valve), and after the 
    last experimental runtime of the day (to empty the valve). This mode does not use camera, Unity game engine or 
    mesoscope assets and it disables all hardware modules other than the reward valve interface. This mode does not
    generate output data directories. Note, this mode will automatically transition HeadBar and LickPort Zaber motors 
    into the position that provides easy access to the lickport tube to assist with calibration. Make sure the mesoscope
    objective is removed and there is no animal in the HeadBar holder prior to running this mode.
    """

    CALIBRATION = 5
    """The service mode used to test all hardware modules used by the current Mesoscope-VR system setup, other than 
    the Water Valve, which is calibrated via the VALVE_CALIBRATION mode (4). This is used during Mesoscope-VR hardware 
    assembly to test and calibrate all modules. Generally, this mode should not be used after the initial system 
    assembly and should only be used by users with a good grasp of the hardware and software employed in the 
    Mesoscope-VR system."""


class ProjectData:
    """Provides methods for managing the experimental data acquired for the input animal and project combination.

    This class functions as the central hub for collecting the data from all local PCs involved in the data acquisition
    process and pushing it to the NAS and the BioHPC server. Its primary purpose is to maintain the project data
    structure across all supported destinations and to efficiently and safely move the data to these destinations with
    minimal redundancy and footprint. Additionally, this supports transferring data such as Zaber motor positions
    between experimental sessions, streamlining experiment setup.

    Note:
        Do not call methods from this class directly. This class is intended to be used through the MesoscopeExperiment
        class and using it directly may lead to unexpected behavior.

        It is expected that the server, nas, and mesoscope data directories are mounted on the host-machine via the
        SMB or equivalent protocol. All manipulations with these destinations are carried out with the assumption that
        the OS has full access to these directories and filesystems.

        This class is specifically designed for working with raw data of an experimental session for a single animal
        participating in the managed project. Processed data is managed by the processing library methods and classes.

        This class generates an xxHash-128 checksum stored inside the ax_checksum.txt file at the root of each
        experimental session 'raw_data' directory. The checksum verifies the data of each file and the paths to each
        file relative to the 'raw_data' root directory.

        If this class is instantiated with the 'create' flag for an already existing project and / or animal, it will
        not recreate the project and animal directories. Instead, the instance will use the path to the already existing
        directories. Therefore, it is completely safe and often preferred t have the 'create' flag set to True at all
        times.

    Args:
        project_name: The name of the project whose data will be managed by the class.
        animal_name: The name of the animal whose data will be managed by the class.
        create: A boolean flag indicating whether to create the project and animal directories if they do not exist. If
            the directories already exist, the class will use existing directories. This process will not modify any
            already existing data.
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
        _local: The absolute path to the host-machine directory where all projects are stored.
        _nas: The absolute path to the Synology NAS animal directory for the managed project.
        _server: The absolute path to the BioHPC server animal directory for the managed project.
        _mesoscope: The absolute path to the mesoscope data directory.
        _project_name: Stores the name of the project whose data is currently managed by the class.
        _animal_name: Stores the name of the animal whose data is currently managed by the class.
        _session_name: Stores the name of the session directory that was last generated by this class instance. If the
        class instance has not generated a session directory, this will be None.
        _previous_session_name: Stores the name of the session directory that precedes the last generated session
            directory. If there were no previous sessions, this will be None.
        _is_test: Determines whether the last created session directory is a real experimental session or a test
            session. Test sessions are used during 'calibration' runtimes.
        _sessions: Stores the sorted list of Paths to all already existing sessions for the managed animal and project
            combination.

    Raises:
        FileNotFoundError: If the project and animal directory path does not exist and the 'create' flag is not set.
    """

    def __init__(
        self,
        project_name: str,
        animal_name: str,
        create: bool = True,
        local_root_directory: Path = Path("/media/Data/Experiments"),
        server_root_directory: Path = Path("/media/cybermouse/Extra Data/server/storage"),
        nas_root_directory: Path = Path("/home/cybermouse/nas/rawdata"),
        mesoscope_data_directory: Path = Path("/home/cybermouse/scanimage/mesodata/mesoscope_frames"),
    ) -> None:
        # Ensures that each root path is absolute
        self._local: Path = local_root_directory.absolute()
        self._nas: Path = nas_root_directory.absolute()
        self._server: Path = server_root_directory.absolute()
        self._mesoscope: Path = mesoscope_data_directory.absolute()

        # Computes the project + animal directory paths for all destinations
        local_project_directory = self._local / project_name / animal_name
        nas_project_directory = self._nas / project_name / animal_name
        server_project_directory = self._server / project_name / animal_name

        # If requested, creates the directories in all destinations and locally
        if create:
            ensure_directory_exists(local_project_directory)
            ensure_directory_exists(nas_project_directory)
            ensure_directory_exists(server_project_directory)

            # Overwrites the destination directory paths with the project- and animal-adjusted paths. This is not done
            # for the local path as that path may need to be modified in different ways, depending on the arguments
            # passed to create_session() method. However, anything we push to nas and server HAS to be a valid project-
            # and animal-specific directory tree structure.
            self._nas = nas_project_directory
            self._server = server_project_directory

        # If the 'create' flag is off, raises an error if the target directory does not exist. Assumes that if the
        # directory does not exist locally, it would not exist on the NAS and Server either.
        elif not local_project_directory.exists():
            message = (
                f"Unable to initialize the ProjectData class, as the directory for the animal '{animal_name}' of the "
                f"project '{project_name}' does not exist. Initialize the class with the 'create' flag if you need to "
                f"create the project and animal directories."
            )
            console.error(message=message, error=FileNotFoundError)

        # Records animal and project names to attributes
        self._project_name: str = project_name
        self._animal_name: str = animal_name
        self._session_name: str | None = None  # Placeholder
        self._previous_session_name: str | None = None  # Placeholder
        self._is_test: bool = False

        # Generates the sorted list of all already existing sessions. This is used to discover previous sessions,
        # which may contain useful information, such as Zaber Motor positions used during the previous session
        self._sessions: list[Path] = sorted([p for p in self._local.glob("????-??-??-??-??-??-*") if p.is_dir()])

    def create_session(self, is_test: bool = False) -> None:
        """Creates a new session directory within the broader project-animal data structure.

        Uses the current UTC timestamp down to microseconds as the session folder name, which ensures that each session
        name within the project-animal structure is unique and accurately preserves the order of the sessions. For test
        sessions, the method will use an independent directory hierarchy that clears (removes) any previous test data.

        Notes:
            Do not call this method manually. This method is designed to be called by the MesoscopeExperiment class.

            Most other class methods require this method to be called at least once before they can be used.

            To retrieve the name of the generated session, use the 'session_name' property. To retrieve the full path
            to the session raw_data directory, use the 'session_path' property.

            If this is not the first session for this animal, use the 'previous_session_name' and
            'previous_session_path' properties to get the path to the previous session data.

        Args:
            is_test: A boolean flag that determines whether the method is used to create the session directory for
                a real animal or to generate a test directory used to calibrate Mesoscope-VR modules. Test directories
                do not make use of project and animal class attributes and instead statically create
                TestProject-TestAnimal-test structure under the local root directory.
        """

        # If the method is used to generate a test directory, create a new test hierarchy and returns early. This
        # ignores most of the ProjectData configuration. If a previous test directory already exists, the method will
        # remove that directory and all previous test files.
        if is_test:
            session_path = self._local.joinpath("TestProject", "TestAnimal", "test")
            # If the session path exists (from a previous runtime), cleans it up
            shutil.rmtree(session_path, ignore_errors=True)
            ensure_directory_exists(session_path)
            self._session_name = session_path.stem
            self._is_test = True  # Enables the test flag
            return

        # Otherwise, ensures that the test flag is disabled
        self._is_test = False

        # Acquires the UTC timestamp to use as the session name
        session_name = get_timestamp(time_separator="-")

        # Constructs the session directory path and generates the directory
        raw_session_path = self._local.joinpath(self._project_name, self._animal_name, session_name)

        # Handles potential session name conflicts. While this is extremely unlikely, it is not impossible for
        # such conflicts to occur.
        counter = 0
        while raw_session_path.exists():
            counter += 1
            new_session_name = f"{session_name}_{counter}"
            raw_session_path = self._local.joinpath(self._project_name, self._animal_name, new_session_name)

        # If a conflict is detected and resolved, warns the user about the resolved conflict.
        if counter > 0:
            message = (
                f"Session name conflict occurred for animal '{self._animal_name}' of project '{self._project_name}' "
                f"when adding the new session with timestamp {self._session_name}. The session with identical name "
                f"already exists. The newly created session directory uses a '_{counter}' postfix to distinguish "
                f"itself from the already existing session directory."
            )
            warnings.warn(message=message)

        # Creates the session directory and the raw_data subdirectory
        ensure_directory_exists(raw_session_path.joinpath("raw_data"))
        self._session_name = raw_session_path.stem

        # Retrieves the name of the previous session before appending the newly generated path to the sessions' list. If
        # the session list is empty, keeps the previous name set to None.
        if len(self._sessions) != 0:
            self._previous_session_name = self._sessions[-1].stem
        self._sessions.append(raw_session_path)

    @property
    def session_path(self) -> Path:
        """Returns the full path to the last generated session's raw_data directory.

        This path is used by the MesoscopeExperiment to set up the output directories for the mesoscope and
        behavioral data acquired during experimental runtime.

        Raises:
            ValueError: If create_session() has not been called to generate the experimental session directory.
        """
        if self._session_name is None:
            message = (
                f"Unable to retrieve the 'raw_data' folder path for the last generated session of the animal "
                f"'{self._animal_name}' and project '{self._project_name}'. Call create_session() method before using "
                f"this property."
            )
            console.error(message=message, error=ValueError)
        return self._local.joinpath(self._project_name, self._animal_name, self._session_name, "raw_data")

    @property
    def ops_path(self) -> Path:
        """Returns the full path to the ops.json file of the last generated session.

        This path is used to save the ops.json generated from the mesoscope TIFF metadata. Ops.json is a configuration
        file used by the suite2p and other processing pipelines to process acquired mesoscope data.

        Raises:
            ValueError: If create_session() has not been called to generate the experimental session directory.
        """
        if self._session_name is None or self._is_test:
            message = (
                f"Unable to retrieve the ops.json file path for the last generated session of the animal "
                f"'{self._animal_name}' and project '{self._project_name}'. Call create_session() method before using "
                f"this property."
            )
            console.error(message=message, error=ValueError)
        return self._local.joinpath(self._project_name, self._animal_name, self._session_name, "raw_data", "ops.json")

    @property
    def frame_invariant_metadata_path(self) -> Path:
        """Returns the full path to the frame_invariant_metadata.json file of the last generated session.

        This path is used to save the metadata shared by all frames in all stacks acquired by the mesoscope
        during the same session. While we do not use this data during processing, it is stored for future reference and
        reproducibility.

        Raises:
            ValueError: If create_session() has not been called to generate the experimental session directory.
        """
        if self._session_name is None or self._is_test:
            message = (
                f"Unable to retrieve the frame_invariant_metadata.json file path for the last generated session of the "
                f"animal '{self._animal_name}' and project '{self._project_name}'. Call create_session() method before "
                f"using this property."
            )
            console.error(message=message, error=ValueError)
        return self._local.joinpath(
            self._project_name, self._animal_name, self._session_name, "raw_data", "frame_invariant_metadata.json"
        )

    @property
    def frame_variant_metadata_path(self) -> Path:
        """Returns the full path to the frame_variant_metadata.npz file of the last generated session.

        This path is used to save the metadata unique for each frame in all stacks acquired by the mesoscope
        during the same session. While we do not use this data during processing, it is stored for future reference and
        reproducibility.

        Notes:
            Unlike frame-invariant metadata, this file is stored as a compressed NumPy archive (NPZ) file to optimize
            storage space usage.

        Raises:
            ValueError: If create_session() has not been called to generate the experimental session directory.
        """
        if self._session_name is None or self._is_test:
            message = (
                f"Unable to retrieve the frame_variant_metadata.npz file path for the last generated session of the "
                f"animal '{self._animal_name}' and project '{self._project_name}'. Call create_session() method before "
                f"using this property."
            )
            console.error(message=message, error=ValueError)
        return self._local.joinpath(
            self._project_name, self._animal_name, self._session_name, "raw_data", "frame_variant_metadata.npz"
        )

    @property
    def mesoscope_frames_path(self) -> Path:
        """Returns the full path to the mesoscope_frames directory of the last generated session.

        This path is used during mesoscope data preprocessing to store compressed mesoscope frames. Aggregating all
        frames inside a dedicated directory optimized further processing steps and visual inspection of the acquired
        data.

        Raises:
            ValueError: If create_session() has not been called to generate the experimental session directory.
        """
        if self._session_name is None or self._is_test:
            message = (
                f"Unable to retrieve the 'mesoscope_frames' folder path for the last generated session of the animal "
                f"'{self._animal_name}' and project '{self._project_name}'. Call create_session() method before using "
                f"this property."
            )
            console.error(message=message, error=ValueError)
        directory_path = self._local.joinpath(
            self._project_name, self._animal_name, self._session_name, "raw_data", "mesoscope_frames"
        )
        # Since this is a directory, we need to ensure it exists before this path is returned to caller
        ensure_directory_exists(directory_path)
        return directory_path

    @property
    def zaber_positions_path(self) -> Path:
        """Returns the full path to the zaber_positions.yaml file of the last generated session.

        This path is used to save the positions for all Zaber motors used in the HeadBar and LickPort assemblies at the
        end of the current experimental session. This allows restoring the motors to those positions during the
        following experimental session(s).

        Raises:
            ValueError: If create_session() has not been called to generate the experimental session directory.
        """
        if self._session_name is None or self._is_test:
            message = (
                f"Unable to retrieve the zaber_positions.yaml file path for the last generated session of the animal "
                f"'{self._animal_name}' and project '{self._project_name}'. Call create_session() method before using "
                f"this property."
            )
            console.error(message=message, error=ValueError)
        return self._local.joinpath(
            self._project_name, self._animal_name, self._session_name, "raw_data", "zaber_positions.yaml"
        )

    @property
    def previous_zaber_positions_path(self) -> Path | None:
        """Returns the full path to the zaber_positions.yaml file of the session that precedes the last generated
        session.

        This method is primarily intended to be called by the MesoscopeExperiment class to extract the positions of
        Zaber motors that control the HeadBar and the LickPort used during the previous session. This allows restoring
        the motors to the same positions during the current session runtime.

        Note:
            If there are no previous sessions available for this project and animal combination, this method will
            return None.

        Raises:
            ValueError: If create_session() has not been called to generate the experimental session directory.
        """
        if self._session_name is None or self._is_test:
            message = (
                f"Unable to retrieve the zaber_positions.yaml file path for the session that precedes the last "
                f"generated session of the animal '{self._animal_name}' and project '{self._project_name}'. Call "
                f"create_session() method before using this property."
            )
            console.error(message=message, error=ValueError)

        # If there are no previous sessions, returns None.
        if self._previous_session_name is None:
            return None

        return self._local.joinpath(
            self._project_name, self._animal_name, self._previous_session_name, "raw_data", "zaber_positions.yaml"
        )

    @property
    def camera_timestamps_path(self) -> Path:
        """Returns the full path to the camera_timestamps.npz file of the last generated session.

        This path is used to save the timestamps associated with each saved frame acquired by each camera used to record
        experimental session runtime. This data is later used during DeepLabCut tracking to generate the tracking
        dataset. In turn, that dataset is eventually used to build the unified session dataset that contains behavioral
        and brain activity data.

        Raises:
            ValueError: If create_session() has not been called to generate the experimental session directory.
        """
        if self._session_name is None or self._is_test:
            message = (
                f"Unable to retrieve the camera_timestamps.npz file path for the last generated session of the "
                f"animal '{self._animal_name}' and project '{self._project_name}'. Call create_session() method before "
                f"using this property."
            )
            console.error(message=message, error=ValueError)
        return self._local.joinpath(
            self._project_name, self._animal_name, self._session_name, "raw_data", "camera_timestamps.npz"
        )

    @property
    def behavioral_data_path(self) -> Path:
        """Returns the full path to the behavioral_data.parquet file of the last generated session.

        This path is used to save the behavioral dataset assembled from the data logged during runtime by the central
        process and the AtaraxisMicroController modules. This dataset is assembled via Polars and stored as a
        Parquet file. It contains all behavioral data other than video-tracking, aligned to the acquired mesoscope
        frames. This data is used for further processing and analysis.

        Raises:
            ValueError: If create_session() has not been called to generate the experimental session directory.
        """
        if self._session_name is None or self._is_test:
            message = (
                f"Unable to retrieve the behavioral_data.parquet file path for the last generated session of the "
                f"animal '{self._animal_name}' and project '{self._project_name}'. Call create_session() method before "
                f"using this property."
            )
            console.error(message=message, error=ValueError)
        return self._local.joinpath(
            self._project_name, self._animal_name, self._session_name, "raw_data", "behavioral_data.parquet"
        )

    def pull_mesoscope_data(self, num_threads: int = 28) -> None:
        """Pulls the frames acquired by the mesoscope from the ScanImage PC to the raw_data directory of the last
        created session.

        This method should be called after the data acquisition runtime to aggregate all recorded data on the VRPC
        before running the preprocessing pipeline. The method expects that the mesoscope frames source directory
        contains only the frames acquired during the current session runtime, in addition to the MotionEstimator.me and
        zstack.mat used for motion registration.

        Notes:
            This method is configured to parallelize data transfer and verification to optimize runtime speeds where
            possible.

        Args:
            num_threads: The number of parallel threads used for transferring the data from ScanImage (mesoscope) PC to
                the local machine. Depending on the connection speed between the PCs, it may be useful to set this
                number to the number of available CPU cores - 4.

        """
        # Prevents calling this method for test sessions and before a valid destination session has been created
        if self._session_name is None or self._is_test:
            message = (
                f"Unable to pull the mesoscope frames into the raw_data directory of the last generated session of the "
                f"animal '{self._animal_name}' and project '{self._project_name}'. Call create_session() method before "
                f"using this property."
            )
            console.error(message=message, error=ValueError)

        # Resolves source and destination paths
        source_directory = self._mesoscope
        destination = self.session_path  # The path to the raw_data subdirectory of the current session

        # Extracts the names of files stored in the source folder
        files = tuple([path for path in source_directory.glob("*")])

        # Prevents 'pulling' an empty folder
        if len(files) == 0:
            message = (
                f"Unable to pull the mesoscope frames into the raw_data directory of the last generated session of the "
                f"animal '{self._animal_name}' and project '{self._project_name}'. The 'transient' ScanImage PC "
                f"directory does not contain any files, indicating that either no frames were acquired during runtime "
                f"or the frames were saved at a different location."
            )
            console.error(message=message, error=RuntimeError)
        if "MotionEstimator.me" not in files:
            message = (
                f"Unable to pull the mesoscope frames into the raw_data directory of the last generated session of the "
                f"animal '{self._animal_name}' and project '{self._project_name}'. The 'transient' ScanImage PC "
                f"directory does not contain the MotionEstimator.me file, which has to be stored with the mesoscope "
                f"frames."
            )
            console.error(message=message, error=RuntimeError)
        if "zstack.mat" not in files:
            message = (
                f"Unable to pull the mesoscope frames into the raw_data directory of the last generated session of the "
                f"animal '{self._animal_name}' and project '{self._project_name}'. The 'transient' ScanImage PC "
                f"directory does not contain the zstack.mat file, which has to be stored with the mesoscope "
                f"frames."
            )
            console.error(message=message, error=RuntimeError)

        # The mesoscope path statically points to the mesoscope_frames folder. It is expected that the folder ONLY
        # contains the frames acquired during this session's runtime. First, generates the checksum for the raw
        # mesoscope frames
        calculate_directory_checksum(directory=self._mesoscope, num_processes=None, save_checksum=True)

        # Generates the path to the local session raw_data folder

        # Transfers the mesoscope frames data from the ScanImage PC to the local machine.
        transfer_directory(source=self._mesoscope, destination=destination, num_threads=num_threads)

        # Removes the checksum file after the transfer is complete. The checksum will be recalculated for the whole
        # session directory during preprocessing, so there is no point in keeping the original mesoscope checksum file.
        destination.joinpath("ax_checksum.txt").unlink(missing_ok=True)

        # After the transfer completes successfully (including integrity verification), recreates the mesoscope_frames
        # folder to clear the transferred images.
        shutil.rmtree(self._mesoscope)
        ensure_directory_exists(self._mesoscope)

    def push_to_destinations(self, parallel: bool = True, num_threads: int = 10) -> None:
        """Pushes the raw_data directory of the last created session to the NAS and the SunLab BioHPC server.

        This method should be called after data acquisition and preprocessing to move the prepared data to the NAS and
        the server. This method generates the xxHash3-128 checksum for the source folder and uses it to verify the
        integrity of transferred data at each destination before removing the source folder.

        Notes:
            This method is configured to run data transfer and checksum calculation in parallel where possible. It is
            advised to minimize the use of the host-machine while it is running this method, as most CPU resources will
            be consumed by the data transfer process.

        Args:
            parallel: Determines whether to parallelize the data transfer. When enabled, the method will transfer the
                data to all destinations at the same time (in-parallel). Note, this argument does not affect the number
                of parallel threads used by each transfer process or the number of threads used to compute the
                xxHash3-128 checksum. This is determined by the 'num_threads' argument (see below).
            num_threads: Determines the number of threads used by each transfer process to copy the files and calculate
                the xxHash3-128 checksums. Since each process uses the same number of threads, it is highly
                advised to set this value so that num_threads * 2 (number of destinations) does not exceed the total
                number of CPU cores - 4.
        """
        # Generates the path to session raw_data folder
        source = self._local.joinpath(self._project_name, self._animal_name, self._session_name, "raw_data")

        # Ensures that the source folder has been checksummed. If not, generates the checksum before executing the
        # transfer operation
        if not source.joinpath("ax_checksum.txt").exists():
            calculate_directory_checksum(directory=source, num_processes=None, save_checksum=True)

        # Generates the paths to the destination folders. Bundles them with descriptive names to enhance progress
        # reporting
        destinations = (
            (self._nas.joinpath(self._session_name, "raw_data"), "NAS"),
            (self._server.joinpath(self._session_name, "raw_data"), "Server"),
        )

        # If the method is configured to transfer files in parallel, submits tasks to a ProcessPoolExecutor
        if parallel:
            with ProcessPoolExecutor(max_workers=len(destinations)) as executor:
                futures = {
                    executor.submit(
                        transfer_directory, source=source, destination=dest[0], num_threads=num_threads
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
                transfer_directory(source=source, destination=destination[0], num_threads=num_threads)

        # After all transfers complete successfully (including integrity verification), removes the source directory
        shutil.rmtree(source)


class MesoscopeExperiment:
    """The base class for all Sun lab mesoscope experiment runtimes.

    This class provides methods for conducting experiments in the Sun lab using the Mesoscope-VR system. This class
    abstracts most low-level interactions with the VR system and the mesoscope via a simple high-level API. In turn, the
    API can be used by all lab members to write custom Experiment class specializations for their projects.

    This class also provides methods for initial preprocessing of the raw data. These preprocessing steps all use
    multiprocessing to optimize runtime speeds and are designed to be executed after each experimental session to
    prepare the data for long-term storage and further processing and analysis.

    Notes:
        Calling the initializer does not start the underlying processes. Use the start() method before issuing other
        commands to properly initialize all remote processes. Depending on the runtime mode, this class reserves up to
        11 CPU cores during runtime.

        See the 'axtl-ports' cli command from the ataraxis-transport-layer-pc library if you need help discovering the
        USB ports used by Ataraxis Micro Controller (AMC) devices.

        See the 'axvs-ids' cli command from ataraxis-video-system if you need help discovering the camera indices used
        by the Harvesters-managed and OpenCV-managed cameras.

        See the 'sl-devices' cli command from this library if you need help discovering the serial ports used by the
        Zaber motion controllers.

        This class statically reserves the id code '1' to label its log entries. Make sure no other Ataraxis class, such
        as the MicroControllerInterface or the VideoSystem uses this id code.

        This class can be configured to perform an experiment runtime or one of the calibration / training runtimes. All
        interactions with the Mesoscope-VR system should be performed through this class instance.

    Args:
        project_data: An instance of the ProjectData class initialized for the animal whose data will be recorded by
            this class. The ProjectData instance encapsulates all data management procedures used to acquire,
            preprocess, and move the experimental session data to long-term storage. Do NOT create a new session before
            passing the ProjectData instance to this class, MesoscopeExperiment handles session creation.
        runtime_mode: Specifies the intended runtime mode. THe class can be used to run experiments, train the
            animal, and execute various maintenance tasks, such as water valve calibration.
        screens_on: Determines whether the VR screens are ON when this class is initialized. Since there is no way of
            getting this information via hardware, the initial screen state has to be supplied by the user as an
            argument. The class will manage and track the state after initialization.
        experiment_state: The integer code that represents the initial state of the experiment. Experiment state codes
            are used to mark different stages of each experiment (such as setup, rest, task 1, task 2, etc.). During
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
        _logger: A DataLogger instance that collects behavior log data from all sources: microcontrollers, video
            cameras, and the MesoscopeExperiment instance.
        _mesoscope_start: The interface that starts mesoscope frame acquisition via TTL pulse.
        _mesoscope_stop: The interface that stops mesoscope frame acquisition via TTL pulse.
        _break: The interface that controls the electromagnetic break attached to the running wheel.
        _reward: The interface that controls the solenoid water valve that delivers water to the animal.
        _screens: The interface that controls the power state of the VR display screens.
        _actor: The main interface for the 'Actor' Ataraxis Micro Controller (AMC) device.
        _mesoscope_frame: The interface that monitors frame acquisition timestamp signals sent by the mesoscope.
        _lick: The interface that monitors animal's interactions with the lick sensor (detects licks).
        _torque: The interface that monitors the torque applied by the animal to the running wheel.
        _sensor: The main interface for the 'Sensor' Ataraxis Micro Controller (AMC) device.
        _wheel_encoder: The interface that monitors the rotation of the running wheel and converts it into the distance
            traveled by the animal.
        _encoder: The main interface for the 'Encoder' Ataraxis Micro Controller (AMC) device.
        _unity: The interface used to directly communicate with the Unity game engine (Gimbl) via the MQTT. Consider
            this the Unity binding interface.
        _face-camera: The interface that captures and saves the frames acquired by the 9MP scientific camera aimed at
            the animal's face and eye from the left side (via a hot mirror).
        _left_camera: The interface that captures and saves the frames acquired by the 1080P security camera aimed on
            the left side of the animal and the right and center VR screens.
        _right_camera: The interface that captures and saves the frames acquired by the 1080P security camera aimed on
            the right side of the animal and the left VR screen.
        _headbar: Stores the Connection class instance that manages the USB connection to a daisy-chain of Zaber devices
            (controllers) that allow repositioning the headbar.
        _headbar_z: Stores the Axis (motor) class that controls the position of the headbar along the Z axis.
        _headbar_pitch: Stores the Axis (motor) class that controls the position of the headbar along the Pitch axis.
        _headbar_roll: Stores the Axis (motor) class that controls the position of the headbar along the Roll axis.
        _lickport: Stores the Connection class instance that manages the USB connection to a daisy-chain of Zaber
            devices (controllers) that allow repositioning the lick tube.
        _lickport_z: Stores the Axis (motor) class that controls the position of the lickport along the Z axis.
        _lickport_x: Stores the Axis (motor) class that controls the position of the lickport along the X axis.
        _lickport_y: Stores the Axis (motor) class that controls the position of the lickport along the Y axis.
        _screen_on: Tracks whether the VR displays are currently ON.
        _vr_state: Stores the current state of the VR system. The MesoscopeExperiment updates this value whenever it is
            instructed to change the VR system state.
        _state_map: Maps the integer state-codes used to represent VR system states to human-readable string-names.
        _experiment_state: Stores the user-defined experiment state. Experiment states are defined by the user and
            are expected to be unique for each project and, potentially, experiment. Different experiment states can
            reuse the same VR state.
        _timestamp-timer: A PrecisionTimer instance used to timestamp log entries generated by the class instance.
        _source_id: Stores the unique identifier code for this class instance. The identifier is used to mark log
            entries sent by this class instance and has to be unique for all sources that log data at the same time,
            such as MicroControllerInterfaces and VideoSystems.
        _project_data: Stores the ProjectData instance used to manage the acquired data.
        _mode: Stores the integer-code for the class runtime mode.

    Raises:
        TypeError: If any of the arguments are not of the expected type.
        ValueError: If any of the arguments are not of the expected value.
    """

    _state_map: dict[int, str] = {0: "Idle", 1: "Rest", 2: "Run", 3: "Lick Train", 4: "Run Train"}

    def __init__(
        self,
        project_data: ProjectData,
        runtime_mode: RuntimeModes,
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
        # As with other pipelines that use intelligent resource termination, presets the _started flag first to avoid
        # leaks if the initialization method fails.
        self._started: bool = False

        # Defines other flags used during runtime:
        self._screen_on: bool = screens_on  # Usually this would be false, but this is not guaranteed
        self._vr_state: int = 0  # Stores the current state of the VR system
        self._experiment_state: int = experiment_state  # Stores user-defined experiment state
        self._timestamp_timer: PrecisionTimer = PrecisionTimer("us")  # A timer used to timestamp local log entries
        self._source_id: np.uint8 = np.uint8(1)  # Reserves source ID code 1 for this class

        # Input verification:
        if not isinstance(project_data, ProjectData):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a ProjectData instance for "
                f"'project_data' argument, but instead encountered {project_data} of type "
                f"{type(project_data).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(runtime_mode, RuntimeModes):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a RuntimeModes field for "
                f"'runtime_mode' argument, but instead encountered {runtime_mode} of type "
                f"{type(runtime_mode).__name__}."
            )
            console.error(message=message, error=ValueError)

        # Saves the ProjectData instance to class attribute so that it can be used from class methods.
        self._project_data: ProjectData = project_data

        # Sets the runtime mode to the value of the passed RuntimeModes enumeration field.
        self._mode: int = runtime_mode.value

        # For training and experiment runtimes, creates a new session directory to store the experimental data
        if self._mode != RuntimeModes.CALIBRATION and self._mode != RuntimeModes.VALVE_CALIBRATION:
            self._project_data.create_session()  # Creates the directory structure for the new experimental session
        else:
            # For calibration runtimes, also creates a session directory but uses the test scheme instead to store
            # the data inside a dedicated TEST folder.
            self._project_data.create_session(is_test=True)

        if not isinstance(valve_calibration_data, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
            for item in valve_calibration_data
        ):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a tuple of 2-element tuples with "
                f"integer or float values for 'valve_calibration_data' argument, but instead encountered "
                f"{valve_calibration_data} of type {type(valve_calibration_data).__name__} with at least one "
                f"incompatible element."
            )
            console.error(message=message, error=TypeError)

        # Initializes the DataLogger instance used to log data from all microcontrollers, camera frame savers and this
        # class instance.
        self._logger: DataLogger = DataLogger(
            output_directory=project_data.session_path,
            instance_name="behavior",
            sleep_timer=0,
            exist_ok=True,
            process_count=1,
            thread_count=10,
        )

        # ACTOR. Actor AMC controls the hardware that needs to be triggered by PC at irregular intervals. Most of such
        # hardware is designed to produce some form of an output: deliver water reward, engage wheel breaks, issue a
        # TTL trigger, etc.

        # Module interfaces:
        self._mesoscope_start: TTLInterface = TTLInterface(module_id=np.uint8(1))  # mesoscope acquisition start
        self._mesoscope_stop: TTLInterface = TTLInterface(module_id=np.uint8(2))  # mesoscope acquisition stop
        self._break = BreakInterface(
            minimum_break_strength=43.2047,  # 0.6 in oz
            maximum_break_strength=1152.1246,  # 16 in oz
            object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
        )  # Wheel break
        self._reward = ValveInterface(valve_calibration_data=valve_calibration_data)  # Reward solenoid valve
        self._screens = ScreenInterface(initially_on=screens_on)  # VR Display On/Off switch

        # Main interface:
        self._actor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=actor_port,
            data_logger=self._logger,
            module_interfaces=(self._mesoscope_start, self._mesoscope_stop, self._break, self._reward, self._screens),
            mqtt_broker_ip=unity_ip,
            mqtt_broker_port=unity_port,
        )

        # SENSOR. Sensor AMC controls the hardware that collects data at regular intervals. This includes lick sensors,
        # torque sensors, and input TTL recorders. Critically, all managed hardware does not rely on hardware interrupt
        # logic to maintain the necessary precision.

        # Module interfaces:
        # Mesoscope frame timestamp recorder. THe class is configured to report detected pulses during runtime to
        # support checking whether mesoscope start trigger correctly starts the frame acquisition process.
        self._mesoscope_frame: TTLInterface = TTLInterface(module_id=np.uint8(1), report_pulses=True)
        self._lick: LickInterface = LickInterface(lick_threshold=1000)  # Lick sensor
        self._torque: TorqueInterface = TorqueInterface(
            baseline_voltage=2046,  # ~1.65 V
            maximum_voltage=2750,  # This was determined experimentally and matches the torque that overcomes break
            sensor_capacity=720.0779,  # 10 in oz
            object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
        )  # Wheel torque sensor

        # Main interface:
        self._sensor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(152),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=sensor_port,
            data_logger=self._logger,
            module_interfaces=(self._mesoscope_frame, self._lick, self._torque),
            mqtt_broker_ip=unity_ip,
            mqtt_broker_port=unity_port,
        )

        # ENCODER. Encoder AMC is specifically designed to interface with a rotary encoder connected to the running
        # wheel. The encoder uses hardware interrupt logic to maintain high precision and, therefore, it is isolated
        # to a separate microcontroller to ensure adequate throughput.

        # Module interfaces:
        self._wheel_encoder: EncoderInterface = EncoderInterface(
            encoder_ppr=8192, object_diameter=15.0333, cm_per_unity_unit=10.0
        )

        # Main interface:
        self._encoder: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(203),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=encoder_port,
            data_logger=self._logger,
            module_interfaces=(self._wheel_encoder,),
            mqtt_broker_ip=unity_ip,
            mqtt_broker_port=unity_port,
        )

        # Also instantiates a separate MQTTCommunication instance to directly communicate with Unity. Primarily, this
        # is used to collect data generated by unity, such as the sequence of VR corridors.
        monitored_topics = ("CueSequence/",)
        self._unity: MQTTCommunication = MQTTCommunication(
            ip=unity_ip, port=unity_port, monitored_topics=monitored_topics
        )

        # FACE CAMERA. This is the high-grade scientific camera aimed at the animal's face using the hot-mirror. It is
        # a 10-gigabit 9MP camera with a red long-pass filter and has to be interfaced through the GeniCam API. Since
        # the VRPC has a 4090 with 2 hardware acceleration chips, we are using the GPU to save all of our frame data.
        self._face_camera: VideoSystem = VideoSystem(
            system_id=np.uint8(51),
            data_logger=self._logger,
            output_directory=project_data.session_path,
            harvesters_cti_path=harvesters_cti_path,
        )
        # The acquisition parameters (framerate, frame dimensions, crop offsets, etc.) are set via the SVCapture64
        # software and written to non-volatile device memory. Generally, all projects in the lab should be using the
        # same parameters.
        self._face_camera.add_camera(
            save_frames=True,
            camera_index=face_camera_index,
            camera_backend=CameraBackends.HARVESTERS,
            output_frames=False,
            display_frames=True,
            display_frame_rate=25,
        )
        self._face_camera.add_video_saver(
            hardware_encoding=True,
            video_format=VideoFormats.MP4,
            video_codec=VideoCodecs.H265,
            preset=GPUEncoderPresets.MEDIUM,
            input_pixel_format=InputPixelFormats.MONOCHROME,
            output_pixel_format=OutputPixelFormats.YUV444,
            quantization_parameter=15,
        )

        # LEFT CAMERA. A 1080P security camera that is mounted on the left side from the mouse's perspective
        # (viewing the left side of the mouse and the right screen). This camera is interfaced with through the OpenCV
        # backend.
        self._left_camera: VideoSystem = VideoSystem(
            system_id=np.uint8(62), data_logger=self._logger, output_directory=project_data.session_path
        )

        # DO NOT try to force the acquisition rate. If it is not 30 (default), the video will not save.
        self._left_camera.add_camera(
            save_frames=True,
            camera_index=left_camera_index,
            camera_backend=CameraBackends.OPENCV,
            output_frames=False,
            display_frames=True,
            display_frame_rate=25,
            color=False,
        )
        self._left_camera.add_video_saver(
            hardware_encoding=True,
            video_format=VideoFormats.MP4,
            video_codec=VideoCodecs.H265,
            preset=GPUEncoderPresets.FASTEST,
            input_pixel_format=InputPixelFormats.MONOCHROME,
            output_pixel_format=OutputPixelFormats.YUV420,
            quantization_parameter=30,
        )

        # RIGHT CAMERA. Same as the left camera, but mounted on the right side from the mouse's perspective.
        self._right_camera: VideoSystem = VideoSystem(
            system_id=np.uint8(73), data_logger=self._logger, output_directory=project_data.session_path
        )
        # Same as above, DO NOT force acquisition rate
        self._right_camera.add_camera(
            save_frames=True,
            camera_index=right_camera_index,  # The only difference between left and right cameras.
            camera_backend=CameraBackends.OPENCV,
            output_frames=False,
            display_frames=True,
            display_frame_rate=25,
            color=False,
        )
        self._right_camera.add_video_saver(
            hardware_encoding=True,
            video_format=VideoFormats.MP4,
            video_codec=VideoCodecs.H265,
            preset=GPUEncoderPresets.FASTEST,
            input_pixel_format=InputPixelFormats.MONOCHROME,
            output_pixel_format=OutputPixelFormats.YUV420,
            quantization_parameter=30,
        )

        # HeadBar controller (zaber). This is an assembly of 3 zaber controllers (devices) that allow moving the
        # headbar attached to the mouse in Z, Roll and Pitch dimensions. Note, this assumes that the chaining order of
        # individual zaber devices is fixed and is always Z-Pitch-Roll.
        self._headbar: ZaberConnection = ZaberConnection(port=headbar_port)
        self._headbar.connect()  # Since this does not reserve additional resources, establishes connection right away
        self._headbar_z: ZaberAxis = self._headbar.get_device(0).axis
        self._headbar_pitch: ZaberAxis = self._headbar.get_device(1).axis
        self._headbar_roll: ZaberAxis = self._headbar.get_device(2).axis

        # Lickport controller (zaber). This is an assembly of 3 zaber controllers (devices) that allow moving the
        # lick tube in Z, X and Y dimensions. Note, this assumes that the chaining order of individual zaber devices is
        # fixed and is always Z-X-Y.
        self._lickport: ZaberConnection = ZaberConnection(port=lickport_port)
        self._lickport.connect()  # Since this does not reserve additional resources, establishes connection right away
        self._lickport_z: ZaberAxis = self._headbar.get_device(0).axis
        self._lickport_x: ZaberAxis = self._headbar.get_device(1).axis
        self._lickport_y: ZaberAxis = self._headbar.get_device(2).axis

        # Also, for the sake of completeness, the mesoscope comes with movement in Z, X, Y and Roll dimensions, but
        # this is controllable exclusively through ThorLabs bindings.

    def start(self) -> None:
        """Sets up all assets used in the runtime mode selected at class initialization.

        This internal method establishes the communication with the microcontrollers, data logger cores, and video
        system processes if these assets are required by the runtime mode. It also verifies the configuration of
        Unity game engine and the mesoscope and activates mesoscope frame

        Notes:
            The particular assets activated during this method depend on the runtime mode of the class. For example,
            mesoscope and Unity are only initialized for 'experiment' modes.

            This process will not execute unless the host PC has access to the necessary number of logical CPU cores
            and other required hardware resources (GPU for video encoding, etc.). This prevents using the class on
            machines that are unlikely to sustain the runtime requirements.

            Zaber devices are connected during the initialization process and not this method runtime to enable
            manipulating Headbar and Lickport before starting the main experiment.

            Calling this method automatically enables Console class (via console variable) if it was not enabled.
            It is expected that this method runs inside the central process managing the runtime at the highest level.

        Raises:
            RuntimeError: If the host PC does not have enough logical CPU cores available.
        """

        # Prevents (re) starting an already started VR process.
        if self._started:
            return

        # Activates the console to display messages to the user
        if not console.enabled:
            console.enable()

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
        # changes to VR and Experiment state during runtime.

        # Constructs a timezone-aware stamp using UTC time. This creates a reference point for all later delta time
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

        # Starts all video systems. Note, this initializes frame acquisition but not saving. Frame saving is controlled
        # via MesoscopeExperiment methods.
        self._face_camera.start()
        self._left_camera.start()
        self._right_camera.start()

        message = "VideoSystems: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts all microcontroller interfaces
        self._actor.start()
        self._actor.unlock_controller()  # Only Actor outputs data, so no need to unlock other controllers.
        self._sensor.start()
        self._encoder.start()

        message = "MicroControllerInterfaces: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        if self._mode == RuntimeModes.EXPERIMENT.value:
            self._unity.connect()  # Directly connects to some Unity communication channels.

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
        else:
            message = "Unity Game Engine: Not used."
            console.echo(message=message, level=LogLevel.WARNING)

        # Configures the encoder to only report forward motion (CW) if the motion exceeds ~ 1 mm of distance.
        self._wheel_encoder.set_parameters(report_cw=False, report_ccw=True, delta_threshold=15)

        # Configures mesoscope start and stop triggers to use 10 ms pulses
        self._mesoscope_start.set_parameters(pulse_duration=np.uint32(10000))
        self._mesoscope_stop.set_parameters(pulse_duration=np.uint32(10000))

        # Configures screen trigger to use 500 ms pulses
        self._screens.set_parameters(pulse_duration=np.uint32(500000))

        # Configures the water valve to deliver ~ 5 uL of water. Also configures the valve calibration method to run the
        # 'reference' calibration for 5 uL rewards used to verify the valve calibration before every experiment.
        self._reward.set_parameters(
            pulse_duration=np.uint32(35590), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
        )

        # Configures the lick sensor to filter out dry touches and only report significant changes in detected voltage
        # (used as a proxy for detecting licks).
        self._lick.set_parameters(
            signal_threshold=np.uint16(300), delta_threshold=np.uint16(300), averaging_pool_size=np.uint8(30)
        )

        # Configures the torque sensor to filter out noise and sub-threshold 'slack' torque signals.
        self._torque.set_parameters(
            report_ccw=True,
            report_cw=True,
            signal_threshold=np.uint16(100),
            delta_threshold=np.uint16(70),
            averaging_pool_size=np.uint8(5),
        )

        # If the runtime mode requires mesoscope, starts frame monitoring and frame acquisition
        if self._mode == RuntimeModes.EXPERIMENT.value:
            # The mesoscope acquires frames at ~10 Hz and sends triggers with the on-phase duration of ~100 ms. We use
            # a polling frequency of ~1000 Hz here to ensure frame acquisition times are accurately detected.
            self._mesoscope_frame.check_state(repetition_delay=np.uint32(1000))

            # Starts mesoscope frame acquisition. This also verifies that the mesoscope responds to triggers and
            # actually starts acquiring frames using the _mesoscope_frame interface above.
            self._start_mesoscope()

            message = "Mesoscope frame acquisition: Started."
            console.echo(message=message, level=LogLevel.SUCCESS)
        else:
            message = "Mesoscope: Not used."
            console.echo(message=message, level=LogLevel.WARNING)

        # Starts monitoring licks. Uses 1000 Hz polling frequency, which should be enough to resolve individual
        # licks of variable duration.
        self._lick.check_state(repetition_delay=np.uint32(1000))

        message = "Hardware module setup: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # If the class is initialized to run an experiment, sets the rest of the subsystems to use the REST state.
        if self._mode == RuntimeModes.EXPERIMENT.value:
            self.vr_rest()

        # For Lick and Run training, there is only one state that configures all used subsystems.
        elif self._mode == RuntimeModes.LICK_TRAINING.value:
            self._vr_lick_train()

        elif self._mode == RuntimeModes.RUN_TRAINING.value:
            self._vr_run_train()

        # The setup procedure is complete.
        self._started = True

        message = "MesoscopeExperiment assets: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self):
        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        # Initializes the timer to enforce the necessary delays
        timer = PrecisionTimer("s")

        self._vr_state = 0  # Resets the VR state to Idle

        # Resets the _started tracker and the VR state
        self._started = False

        # Switches the system into the rest state. Since REST state has most modules set to stop-friendly states,
        # this is used as a shortcut to prepare the VR system for shutdown.
        self.vr_rest()

        # If the mesoscope is still acquiring images, ensures frame acquisition is disabled.
        self.mesoscope_off()
        timer.delay_noblock(2)  # Delays for 2 seconds. This ensures all mesoscope frame triggers are received.

        # Manually prepares remaining modules for shutdown
        # noinspection PyTypeChecker
        self._sensor.send_message(self._mesoscope_frame.dequeue_command)
        # noinspection PyTypeChecker
        self._sensor.send_message(self._lick.dequeue_command)
        # noinspection PyTypeChecker
        self._sensor.send_message(self._torque.dequeue_command)

        # Delays for another 2 seconds to ensure all microcontroller-sent data is received before shutting down the
        # interfaces.
        timer.delay_noblock(2)

        # Stops all interfaces
        self._logger.stop()
        self._actor.stop()
        self._sensor.stop()
        self._encoder.stop()

        # Compresses all logs into a single .npz file. This is done both for long-term storage optimization and to
        # allow parsing the data.
        self._logger.compress_logs(remove_sources=True, verbose=True)
        # TODO ADD VS SECTION HERE

    def __del__(self) -> None:
        """Ensures the instance properly releases all resources before it is garbage-collected."""
        self.stop()

    def vr_rest(self) -> None:
        """Switches the VR system to the rest state.

        In the rest state, the break is engaged to prevent the mouse from moving the wheel. The encoder module is
        disabled, and instead the torque sensor is enabled. The VR screens are switched off, cutting off light emission.
        By default, the VR system starts all experimental runtimes using the REST state.

        Note:
            This command is only executable if the class is running in the experiment mode.

        Raises:
            RuntimeError: If the Mesoscope-VR system is not started, or the class is not in the experiment runtime
                mode.
        """

        if not self._started or not self._mode == RuntimeModes.EXPERIMENT.value:
            message = (
                f"Unable to switch the Mesoscope-VR system to the rest state. Either the start() method of the "
                f"MesoscopeExperiment class has not been called to setup the necessary assets or the current runtime "
                f"mode of the class is not set to the 'experiment' mode."
            )
            console.error(message=message, error=RuntimeError)

        # Engages the break to prevent the mouse from moving the wheel
        self._break.toggle(state=True)

        # Initiates torque monitoring at 1000 Hz. The torque can only be accurately measured when the wheel is locked,
        # as it requires a resistance force to trigger the sensor.
        self._torque.check_state(repetition_delay=np.uint32(1000))

        # Temporarily suspends encoder monitoring. Since the wheel is locked, the mouse should not be able to produce
        # meaningful motion data.
        self._wheel_encoder.reset_command_queue()

        # Toggles the state of the VR screens to be OFF if the VR screens are currently ON. If the screens are OFF,
        # keeps them OFF.
        if self._screen_on:
            self._screens.toggle()
            self._screen_on = False

        # Configures the state tracker to reflect the REST state
        self._change_vr_state(1)

        message = f"VR State: {self._state_map[1]}."
        console.echo(message=message, level=LogLevel.INFO)

    def vr_run(self) -> None:
        """Switches the VR system to the run state.

        In the run state, the break is disengaged to allow the mouse to freely move the wheel. The encoder module is
        enabled to record and share live running data with Unity, and the torque sensor is disabled. The VR screens are
        switched on to render the VR environment.

        Note:
            This command is only executable if the class is running in the experiment mode.

        Raises:
            RuntimeError: If the Mesoscope-VR system is not started, or the class is not in the experiment runtime
                mode.
        """

        if not self._started or not self._mode == RuntimeModes.EXPERIMENT.value:
            message = (
                f"Unable to switch the Mesoscope-VR system to the run state. Either the start() method of the "
                f"MesoscopeExperiment class has not been called to setup the necessary assets or the current runtime "
                f"mode of the class is not set to the 'experiment' mode."
            )
            console.error(message=message, error=RuntimeError)

        # Initializes encoder monitoring at 2 kHz rate. The encoder aggregates wheel data at native speeds; this rate
        # only determines how often the aggregated data is sent to PC and Unity.
        self._wheel_encoder.check_state(repetition_delay=np.uint32(500))

        # Disables torque monitoring. To accurately measure torque, the sensor requires a resistance force provided by
        # the break. During running, measuring torque is not very reliable and adds little value compared to the
        # encoder.
        self._torque.reset_command_queue()

        # Disengages the break to allow the mouse to move the wheel
        self._break.toggle(False)

        # Toggles the state of the VR screens to be ON if the VR screens are currently OFF. If the screens are ON,
        # keeps them ON.
        if not self._screen_on:
            self._screens.toggle()
            self._screen_on = True

        # Configures the state tracker to reflect RUN state
        self._change_vr_state(2)
        message = f"VR State: {self._state_map[self._vr_state]}."
        console.echo(message=message, level=LogLevel.INFO)

    def _vr_lick_train(self) -> None:
        """Switches the VR system into the lick training state.

        In the lick training state, the break is enabled, preventing the mouse from moving the wheel. The screens
        are turned off, Unity and mesoscope are disabled. Torque and Encoder monitoring are also disabled. The only
        working hardware modules are lick sensor and water valve.

        Note:
            This command is only executable if the class is running in the lick training mode.

            This state is set automatically during start() method runtime. It should not be called externally by the
            user.

        Raises:
            RuntimeError: If the Mesoscope-VR system is not started, or the class is not in the lick training runtime
                mode.
        """

        if not self._started or not self._mode == RuntimeModes.LICK_TRAINING.value:
            message = (
                f"Unable to switch the Mesoscope-VR system to the lick training state. Either the start() method of "
                f"the MesoscopeExperiment class has not been called to setup the necessary assets or the current "
                f"runtime mode of the class is not set to the 'lick_training' mode."
            )
            console.error(message=message, error=RuntimeError)

        # Ensures the break is enabled. The mice do not need to move the wheel during the lick training runtime.
        self._break.toggle(True)

        # Disables both torque and encoder. During lick training we are not interested in mouse motion data.
        self._torque.reset_command_queue()
        self._wheel_encoder.reset_command_queue()

        # Toggles the state of the VR screens to be OFF if the VR screens are currently ON. If the screens are OFF,
        # keeps them OFF.
        if self._screen_on:
            self._screens.toggle()
            self._screen_on = False

        # The lick sensor should already be running at 1000 Hz resolution, as we generally keep it on for all our
        # pipelines. Unity and mesoscope should not be enabled.

        # Configures the state tracker to reflect the LICK TRAIN state
        self._change_vr_state(3)
        message = f"VR State: {self._state_map[self._vr_state]}."
        console.echo(message=message, level=LogLevel.INFO)

    def _vr_run_train(self) -> None:
        """Switches the VR system into the run training state.

        In the run training state, the break is disabled, allowing the animal to move the wheel. The encoder module is
        enabled to monitor the running metrics (distance and / or speed). The lick sensor and water valve modules are
        also enabled to conditionally reward the animal for desirable performance. The VR screens are turned off. Unity,
        mesoscope, and the torque module are disabled.

        Note:
            This command is only executable if the class is running in the run training mode.

            This state is set automatically during start() method runtime. It should not be called externally by the
            user.

        Raises:
            RuntimeError: If the Mesoscope-VR system is not started, or the class is not in the run training runtime
                mode.
        """

        if not self._started or not self._mode == RuntimeModes.RUN_TRAINING.value:
            message = (
                f"Unable to switch the Mesoscope-VR system to the run training state. Either the start() method of the "
                f"MesoscopeExperiment class has not been called to setup the necessary assets or the current runtime "
                f"mode of the class is not set to the 'run_training' mode."
            )
            console.error(message=message, error=RuntimeError)

        # Disables both torque sensor.
        self._torque.reset_command_queue()

        # Enables the encoder module to monitor animal's running performance
        self._wheel_encoder.check_state(repetition_delay=np.uint32(500))

        # Toggles the state of the VR screens to be OFF if the VR screens are currently ON. If the screens are OFF,
        # keeps them OFF.
        if self._screen_on:
            self._screens.toggle()
            self._screen_on = False

        # Ensures the break is disabled. This allows the animal to run on the wheel freely.
        self._break.toggle(False)

        # The lick sensor should already be running at 1000 Hz resolution, as we generally keep it on for all our
        # pipelines. Unity and mesoscope should not be enabled.

        # Configures the state tracker to reflect the RUN TRAIN state
        self._change_vr_state(4)
        message = f"VR State: {self._state_map[self._vr_state]}."
        console.echo(message=message, level=LogLevel.INFO)

    def _change_vr_state(self, new_state: int) -> None:
        """Sets the vr_state attribute to the input state value and logs the change to the VR state.

        This method is used internally to update and log stream when the VR state changes.

        Args:
            new_state: The byte-code for the newly activated VR state.
        """
        self._vr_state = new_state  # Updates the VR state

        # Logs the VR state update
        timestamp = self._timestamp_timer.elapsed
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(timestamp),
            serialized_data=np.array([new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def _get_cue_sequence(self) -> NDArray[np.uint8]:
        """Requests Unity game engine to transmit the sequence of virtual reality track wall cues for the current task.

        This method is used as part of the experimental runtime startup process to both get the sequence of cues and
        verify that the Unity game engine is running and configured correctly. The sequence of cues communicates the
        task order of track segments, which is necessary for post-processing the data for some projects.

        Returns:
            The Numpy array that stores the sequence of virtual reality segments as byte (uint8) values.

        Raises:
            RuntimeError: If no response from Unity is received within 2 seconds or if Unity sends a message to an
            unexpected (different) topic other than "CueSequence/" while this method is running.
        """
        # Initializes a second-precise timer to ensure the request is fulfilled within a 2-second timeout
        timeout_timer = PrecisionTimer("s")

        # Sends a request for the task cue (corridor) sequence to Unity GIMBL package.
        self._unity.send_data(topic="CueSequenceTrigger/")

        # Waits at most 2 seconds to receive the response, which should be enough at this stage (no heavy communication
        # traffic).
        timeout_timer.reset()
        while timeout_timer.elapsed < 2:
            # If Unity responds with the cue sequence message, attempts to parse the message
            if self._unity.has_data:
                topic: str
                payload: bytes
                topic, payload = self._unity.get_data()
                if topic == "CueSequence/":
                    # Extracts the sequence of cues that will be used during task runtime.
                    sequence: NDArray[np.uint8] = np.frombuffer(buffer=payload, dtype=np.uint8)
                    return sequence

                else:
                    # If the topic is not "CueSequence", aborts with an error
                    message = (
                        f"Received an unexpected topic {topic} while waiting for Unity to respond to the cue sequence "
                        f"request. Make sure the Unity is not configured to send data to any topics monitored by the "
                        f"MesoscopeExperiment instance until the Cue Sequence is resolved as part of the start() "
                        f"method runtime."
                    )
                    console.error(message=message, error=RuntimeError)

        # If the loop above is escaped, this is due to not receiving any message from Unity. Raises an error.
        message = (
            f"The MesoscopeExperiment has requested the task Cue Sequence by sending the trigger to the "
            f"'CueSequenceTrigger/' topic and received no response for 2 seconds. It is likely that the Unity game "
            f"engine is not running or is not configured to transmit task cue sequences, which is required for the "
            f"MesoscopeExperiment to start."
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
        self._mesoscope_start.send_pulse()

        # Waits at most 2 seconds for the mesoscope to begin sending frame acquisition timestamps to the PC
        timeout_timer.reset()
        while timeout_timer.elapsed < 2:
            # Frame acquisition is confirmed by the frame timestamp recorder class flipping the pulse_status
            # property to True
            if self._mesoscope_frame.pulse_status:
                return

        # If the loop above is escaped, this is due to not receiving the mesoscope frame acquisition pulses. Raises an
        # error.
        message = (
            f"The MesoscopeExperiment has requested the mesoscope to start acquiring frames and received no frame "
            f"acquisition trigger for 2 seconds. It is likely that the mesoscope has not been armed for frame "
            f"acquisition or that the mesoscope trigger or frame timestamp connection is not functional."
        )
        console.error(message=message, error=RuntimeError)

        # This code is here to appease mypy. It should not be reachable
        raise RuntimeError(message)  # pragma: no cover

    def change_experiment_state(self, new_state: int) -> None:
        """Updates and logs the new experiment state.

        Use this method to timestamp and log experiment state (stage) changes, such as transitioning between different
        task versions.

        Args:
            new_state: The integer byte-code for the new experiment state. The code will be serialized as an uint8
                value, so only values between 0 and 255 inclusive are supported.
        """
        self._experiment_state = new_state  # Updates the Experiment state

        # Logs the VR state update
        timestamp = self._timestamp_timer.elapsed
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(timestamp),
            serialized_data=np.array([new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    @property
    def experiment_state(self) -> int:
        """Returns the integer code for the current experiment state.

        Experiment states are set via the change_experiment_state() method. If your specific experiment implementation
        does not update experiment states, this method will always return the default state-code zero.
        """
        return self._experiment_state

    @property
    def vr_state(self) -> str:
        """Returns the current VR system state as a string."""
        return self._state_map[self._vr_state]


def _encoder_cli(encoder: EncoderInterface, polling_delay: int, delta_threshold: int) -> None:
    """Exposes a console-based CLI that interfaces with the encoder connected to the running wheel."""
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "f":  # CW only
            encoder.reset_pulse_count()
            encoder.set_parameters(report_cw=False, report_ccw=True, delta_threshold=delta_threshold)
            encoder.check_state(repetition_delay=np.uint32(polling_delay))
        elif code == "r":  # CCW only
            encoder.reset_pulse_count()
            encoder.set_parameters(report_cw=True, report_ccw=False, delta_threshold=delta_threshold)
            encoder.check_state(repetition_delay=np.uint32(polling_delay))
        elif code == "a":  # Both CW and CCW
            encoder.reset_pulse_count()
            encoder.set_parameters(report_cw=True, report_ccw=True, delta_threshold=delta_threshold)
            encoder.check_state(repetition_delay=np.uint32(polling_delay))


def _mesoscope_ttl_cli(start_trigger: TTLInterface, stop_trigger: TTLInterface, pulse_duration: int) -> None:
    """Exposes a console-based CLI that interfaces with the TTL modules used to start and stop mesoscope frame
    acquisition.
    """
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "b":  # Start / Begin
            start_trigger.set_parameters(pulse_duration=np.uint32(pulse_duration))
            start_trigger.send_pulse()
        elif code == "e":  # Stop / End
            stop_trigger.set_parameters(pulse_duration=np.uint32(pulse_duration))
            stop_trigger.send_pulse()


def _break_cli(wheel_break: BreakInterface) -> None:
    """Exposes a console-based CLI that interfaces with the break connected to the running wheel."""
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "e":  # Engage
            wheel_break.toggle(state=True)
        elif code == "d":  # Disengage
            wheel_break.toggle(state=False)


def _valve_cli(valve: ValveInterface, pulse_duration: int) -> None:
    """Exposes a console-based CLI that interfaces with the water reward delivery valve."""
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "r":  # Deliver reward
            valve.set_parameters(pulse_duration=np.uint32(pulse_duration))
            valve.send_pulse()
        elif code == "c1":
            valve.set_parameters(
                pulse_duration=np.uint32(15000), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
            )  # 15 milliseconds
            valve.calibrate()
        elif code == "c2":
            valve.set_parameters(
                pulse_duration=np.uint32(30000), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
            )  # 30 milliseconds
            valve.calibrate()
        elif code == "c3":
            valve.set_parameters(
                pulse_duration=np.uint32(45000), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
            )  # 45 milliseconds
            valve.calibrate()
        elif code == "c4":
            valve.set_parameters(
                pulse_duration=np.uint32(60000), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
            )  # 60 milliseconds
            valve.calibrate()
        elif code == "o":
            valve.toggle(state=True)
        elif code == "c":
            valve.toggle(state=False)
        elif code == "t":  # Test
            valve.set_parameters(
                pulse_duration=np.uint32(35590), calibration_delay=np.uint32(200000), calibration_count=np.uint16(200)
            )  # 5 ul x 200 times
            valve.calibrate()
        else:
            # noinspection PyBroadException
            try:
                pulse_duration = valve.get_duration_from_volume(float(code))
                valve.set_parameters(pulse_duration=pulse_duration)
                valve.send_pulse()
            except:
                print(f"Unknown command: {code}")


def _screen_cli(screen: ScreenInterface, pulse_duration: int) -> None:
    """Exposes a console-based CLI that interfaces with the HDMI translator boards connected to all three VR screens."""
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "t":  # Toggle
            screen.set_parameters(pulse_duration=np.uint32(pulse_duration))
            screen.toggle()


def _mesoscope_frames_cli(frame_watcher: TTLInterface, polling_delay: int) -> None:
    """Exposes a console-based CLI that interfaces with the TTL module used to receive mesoscope frame acquisition
    timestamps sent by the ScanImage PC.
    """
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "b":  # Start / Begin
            frame_watcher.check_state(repetition_delay=np.uint32(polling_delay))
        elif code == "e":  # Stop / End
            frame_watcher.reset_command_queue()


def _lick_cli(
    lick: LickInterface, polling_delay: int, signal_threshold: int, delta_threshold: int, averaging: int
) -> None:
    """Exposes a console-based CLI that interfaces with the Voltage-based Lick detection sensor."""
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "b":  # Start / Begin
            lick.set_parameters(
                signal_threshold=np.uint16(signal_threshold),
                delta_threshold=np.uint16(delta_threshold),
                averaging_pool_size=np.uint8(averaging),
            )
            lick.check_state(repetition_delay=np.uint32(polling_delay))


def _torque_cli(
    torque: TorqueInterface, polling_delay: int, signal_threshold: int, delta_threshold: int, averaging: int
) -> None:
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "f":  # Forward Torque only
            torque.set_parameters(
                report_cw=False,
                report_ccw=True,
                signal_threshold=np.uint16(signal_threshold),
                delta_threshold=np.uint16(delta_threshold),
                averaging_pool_size=np.uint8(averaging),
            )
            torque.check_state(repetition_delay=np.uint32(polling_delay))
        elif code == "r":  # Forward Torque only
            torque.set_parameters(
                report_cw=True,
                report_ccw=False,
                signal_threshold=np.uint16(signal_threshold),
                delta_threshold=np.uint16(delta_threshold),
                averaging_pool_size=np.uint8(averaging),
            )
            torque.check_state(repetition_delay=np.uint32(polling_delay))
        elif code == "a":  # Both directions
            torque.set_parameters(
                report_cw=True,
                report_ccw=True,
                signal_threshold=np.uint16(signal_threshold),
                delta_threshold=np.uint16(delta_threshold),
                averaging_pool_size=np.uint8(averaging),
            )
            torque.check_state(repetition_delay=np.uint32(polling_delay))


def _camera_cli(camera: VideoSystem):
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "a":  # Start saving
            camera.start_frame_saving()
            print(camera.started)
        elif code == "s":  # Stop saving
            camera.stop_frame_saving()
            print(f"Not Saving")


def calibration() -> None:
    # Output dir
    temp_dir = Path("/home/cybermouse/Desktop/TestOut")
    data_logger = DataLogger(output_directory=temp_dir, instance_name="amc", exist_ok=True)

    # camera: VideoSystem = VideoSystem(
    #     system_id=np.uint8(62), data_logger=data_logger, output_directory=temp_dir
    # )
    # camera.add_camera(
    #     save_frames=True,
    #     camera_index=0,
    #     camera_backend=CameraBackends.OPENCV,
    #     output_frames=False,
    #     display_frames=True,
    #     display_frame_rate=25,
    #     color=False,
    # )
    # camera.add_video_saver(
    #     hardware_encoding=True,
    #     video_format=VideoFormats.MP4,
    #     video_codec=VideoCodecs.H265,
    #     preset=GPUEncoderPresets.FASTEST,
    #     input_pixel_format=InputPixelFormats.MONOCHROME,
    #     output_pixel_format=OutputPixelFormats.YUV420,
    #     quantization_parameter=30
    # )
    #
    # camera.start()
    # _camera_cli(camera)
    # camera.stop()
    # data_logger.compress_logs(remove_sources=True, memory_mapping=False, verbose=True)

    # Defines static assets needed for testing
    valve_calibration_data = (
        (15000, 1.8556),
        (30000, 3.4844),
        (45000, 7.1846),
        (60000, 10.0854),
    )
    actor_id = np.uint8(101)
    sensor_id = np.uint8(152)
    encoder_id = np.uint8(203)
    sensor_usb = "/dev/ttyACM1"
    actor_usb = "/dev/ttyACM0"
    encoder_usb = "/dev/ttyACM2"

    # Add console support for print debugging
    console.enable()

    # Tested module interface
    module = EncoderInterface(debug=True)

    module_1 = TTLInterface(module_id=np.uint8(1), debug=True)
    module_2 = TTLInterface(module_id=np.uint8(2), debug=True)
    module_3 = BreakInterface(debug=True)
    module_4 = ValveInterface(valve_calibration_data=valve_calibration_data, debug=True)
    module_5 = ScreenInterface(initially_on=False, debug=True)

    module_10 = TTLInterface(module_id=np.uint8(1), debug=True)  # Frame TTL
    module_11 = LickInterface(debug=True, lick_threshold=1000)
    module_12 = TorqueInterface(debug=True)

    # Tested AMC interface
    interface = MicroControllerInterface(
        controller_id=actor_id,
        data_logger=data_logger,
        module_interfaces=(module_4,),
        microcontroller_serial_buffer_size=8192,
        microcontroller_usb_port=actor_usb,
        baudrate=115200,
    )

    # Starts interfaces
    data_logger.start()
    interface.start()

    # For ACTOR modules, enables writing to output pins
    interface.unlock_controller()

    # Calls the appropriate CLI to test the target module
    # _encoder_cli(module, 500, 15)
    # _mesoscope_ttl_cli(module_1, module_2, 10000)
    # _break_cli(module_3)
    _valve_cli(module_4, 35590)
    # _screen_cli(module_5, 500000)
    # _mesoscope_frames_cli(module_10, 500)
    # _lick_cli(module_11, 1000, 300, 300, 30)
    # _torque_cli(module_12, 1000, 100, 70, 5)

    # Shutdown
    interface.stop()
    data_logger.stop()

    data_logger.compress_logs(remove_sources=True, memory_mapping=False, verbose=True)


if __name__ == "__main__":
    calibration()
    # session_dir = ProjectData(
    #     project_name="TestProject",
    #     animal_name="TM1",
    #     create=True,
    # )
    # session_dir.create_session()
    # print(session_dir.session_name)
    # print(session_dir.session_path)
    #
    # # Generates 10000 files with random arrays
    # # for i in range(10000):
    # #     arr = np.random.rand(100, 100)  # 100x100 array of random floats
    # #     filename = session_dir.session_path.joinpath(f"test_{i:03d}.npy")
    # #     np.save(filename, arr)
    #
    # # Moves test data from holdout to the generated session
    # root_data = Path("/media/Data/Holdout")
    # calculate_directory_checksum(root_data)
    # transfer_directory(root_data, session_dir.session_path, num_threads=30)
    #
    # session_dir.push_to_destinations()
