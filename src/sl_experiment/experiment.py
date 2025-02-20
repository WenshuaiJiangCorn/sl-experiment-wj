"""This module provides the main MesoscopeExperiment class that abstracts working with Sun lab's mesoscope-VR system
and SessionData class that abstracts working with experimental data."""

import os
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from numpy.lib.npyio import NpzFile
from ataraxis_time import PrecisionTimer

from module_interfaces import (
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

from zaber_bindings import ZaberConnection, ZaberAxis
from transfer_tools import transfer_directory
from packaging_tools import calculate_directory_checksum
from data_preprocessing import interpolate_data, process_mesoscope_directory
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from dataclasses import dataclass
import polars as pl
from typing import Any


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


class SessionData:
    """Provides methods for managing the experimental data acquired during one experimental session performed by the
    input animal as part of the input experimental project.

    This class functions as the central hub for collecting the data from all local PCs involved in the data acquisition
    process and pushing it to the NAS and the BioHPC server. Its primary purpose is to maintain the session data
    structure across all supported destinations and to efficiently and safely move the data to these destinations with
    minimal redundancy and footprint. Additionally, this class generates the paths used during data preprocessing to
    determine where to output the preprocessed data and during runtime to import data from previous sessions.

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
        local_root_directory: Path = Path("/media/Data/Experiments"),
        server_root_directory: Path = Path("/media/cybermouse/Extra Data/server/storage"),
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
    def camera_timestamps_path(self) -> Path:
        """Returns the path to the camera_timestamps.npz file of the managed session.

        This path is used to save the timestamps associated with each saved frame acquired by each camera during
        runtime. This data is later used to align the eye tracking data extracted from camera frames via DeepLabCut with
        other behavioral data stored in the parquet dataset file.
        """
        return self.camera_frames_path.joinpath("camera_timestamps.npz")

    @property
    def behavioral_data_path(self) -> Path:
        """Returns the path to the behavioral_data.parquet file of the managed session.

        This path is used to save the behavioral dataset assembled from the data logged during runtime by the central
        process and the AtaraxisMicroController modules. This dataset is assembled via Polars and stored as a
        parquet file. It contains all behavioral data other than video-tracking, aligned to the acquired mesoscope
        frames. After processing, the dataset is eventually expanded to include all data, including segmented cell
        activity and the video-tracking data.
        """
        return self._local.joinpath("behavioral_data.parquet")

    @property
    def previous_zaber_positions_path(self) -> Path | None:
        """Returns the path to the zaber_positions.yaml file of the previous session.

        The file is stored inside the 'persistent' directory of the project and animal combination. The file is saved to
        the persistent directory when the original session is moved to long-term storage. Loading the file allows
        reusing LickPort and HeadBar motor positions across sessions. The contents of this file are updated after each
        experimental or training session.

        Notes:
            If the file does not exist, returns None. This would be the case, for example, for the very first session
            of each new animal.
        """
        file_path = self._persistent.joinpath("zaber_positions.yaml")
        if not file_path.exists():
            return None
        else:
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

    def pull_mesoscope_data(self, num_threads: int = 28) -> None:
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

        # Generates the checksum for the source folder
        calculate_directory_checksum(directory=source, num_processes=None, save_checksum=True)

        # Transfers the mesoscope frames data from the ScanImage PC to the local machine.
        transfer_directory(source=source, destination=destination, num_threads=num_threads)

        # Removes the checksum file after the transfer is complete. The checksum will be recalculated for the whole
        # session directory during preprocessing, so there is no point in keeping the original mesoscope checksum file.
        destination.joinpath("ax_checksum.txt").unlink(missing_ok=True)

        # After the transfer completes successfully (including integrity verification), recreates the mesoscope_frames
        # folder to remove the transferred data from the ScanImage PC.
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
            num_processes=30,
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

    def push_data(self, parallel: bool = True, num_threads: int = 10) -> None:
        """Pushes the raw_data directory from the VRPC to the NAS and the SunLab BioHPC server.

        This method should be called after data acquisition and preprocessing to move the prepared data to the NAS and
        the server. This method generates the xxHash3-128 checksum for the source folder and uses it to verify the
        integrity of transferred data at each destination before removing the source folder.

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
        """
        # Resolves source and destination paths
        source = self.raw_data_path

        # Destinations include short destination names used for progress reporting
        destinations = (
            (self._nas.joinpath(self._session_name, "raw_data"), "NAS"),
            (self._server.joinpath(self._session_name, "raw_data"), "Server"),
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

        # Ensures that the source folder has been checksummed. If not, generates the checksum before executing the
        # transfer operation
        if not source.joinpath("ax_checksum.txt").exists():
            calculate_directory_checksum(directory=source, num_processes=None, save_checksum=True)

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

        This class is specifically designed to acquire experimental data. To run a training session, use
        BehavioralTraining class. To calibrate hardware modules, including the water valve, use CalibSystemCalibration
        class.

    Args:
        session_data: An initialized SessionData instance. This instance is used to transfer the data between VRPC,
            ScanImagePC, BioHPC server, and the NAS during runtime. Each instance is initialized for the specific
            project, animal, and session combination for which the data is acquired.
        cue_length_map: A dictionary that maps each integer-code associated with a wall cue used in the Virtual Reality
            experiment environment to its length in centimeters. MesoscopeExperiment collect the sequence of wall cues
            from Unity before starting the experiment. Knowing the lengths of these cues allows accurately mapping
            teh distance traveled by the animal during experiments to its location in the VR.
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
        _timestamp_timer: A PrecisionTimer instance used to timestamp log entries generated by the class instance.
        _source_id: Stores the unique identifier code for this class instance. The identifier is used to mark log
            entries sent by this class instance and has to be unique for all sources that log data at the same time,
            such as MicroControllerInterfaces and VideoSystems.
        _cue_map: Stores the dictionary that maps each integer-code associated with each Virtual Reality wall cue with
            its length in centimeters.
        _session_data: Stores the SessionData instance used to manage the acquired data.

    Raises:
        TypeError: If any of the arguments are not of the expected type.
        ValueError: If any of the arguments are not of the expected value.
    """

    # Maps integer VR state codes to human-readable string-names.
    _state_map: dict[int, str] = {0: "Idle", 1: "Rest", 2: "Run", 3: "Lick Train", 4: "Run Train"}

    def __init__(
        self,
        session_data: SessionData,
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
        # Creates the _started flag first to avoid leaks if the initialization method fails.
        self._started: bool = False

        # Input verification:
        if not isinstance(session_data, SessionData):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a SessionData instance for "
                f"'session_data' argument, but instead encountered {session_data} of type "
                f"{type(session_data).__name__}."
            )
            console.error(message=message, error=TypeError)
        if cue_length_map is not None and not isinstance(cue_length_map, dict):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a dictionary or None instance for "
                f"'cue_length_map' argument, but instead encountered {cue_length_map} of type "
                f"{type(cue_length_map).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Defines other flags used during runtime:
        self._screen_on: bool = screens_on  # Usually this would be false, but this is not guaranteed
        self._vr_state: int = 0  # Stores the current state of the VR system
        self._experiment_state: int = experiment_state  # Stores user-defined experiment state
        self._timestamp_timer: PrecisionTimer = PrecisionTimer("us")  # A timer used to timestamp log entries
        self._source_id: np.uint8 = np.uint8(1)  # Reserves source ID code 1 for this class

        # This dictionary is used to convert distance traveled by the animal into the corresponding sequence of
        # traversed cues (corridors).
        self._cue_map: dict[int, float] = {}
        if cue_length_map is not None:
            self._cue_map = cue_length_map

        # Saves the SessionData instance to class attribute so that it can be used from class methods. Since SessionData
        # resolves session directory structure at initialization, the instance is ready to resolve all paths used by
        # the experiment class instance.
        self._session_data: SessionData = session_data

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
        # is used to collect data generated by unity, such as the sequences of VR corridors.
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
            output_directory=session_data.raw_data_path,
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
            system_id=np.uint8(62), data_logger=self._logger, output_directory=session_data.raw_data_path
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
            system_id=np.uint8(73), data_logger=self._logger, output_directory=session_data.raw_data_path
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
        # headbar attached to the mouse in Z, Roll, and Pitch dimensions. Note, this assumes that the chaining order of
        # individual zaber devices is fixed and is always Z-Pitch-Roll.
        self._headbar: ZaberConnection = ZaberConnection(port=headbar_port)
        self._headbar.connect()  # Since this does not reserve additional resources, establishes connection right away
        self._headbar_z: ZaberAxis = self._headbar.get_device(0).axis
        self._headbar_pitch: ZaberAxis = self._headbar.get_device(1).axis
        self._headbar_roll: ZaberAxis = self._headbar.get_device(2).axis

        # Lickport controller (zaber). This is an assembly of 3 zaber controllers (devices) that allow moving the
        # lick tube in Z, X, and Y dimensions. Note, this assumes that the chaining order of individual zaber devices is
        # fixed and is always Z-X-Y.
        self._lickport: ZaberConnection = ZaberConnection(port=lickport_port)
        self._lickport.connect()  # Since this does not reserve additional resources, establishes connection right away
        self._lickport_z: ZaberAxis = self._lickport.get_device(0).axis
        self._lickport_x: ZaberAxis = self._lickport.get_device(1).axis
        self._lickport_y: ZaberAxis = self._lickport.get_device(2).axis

    def start(self) -> None:
        """Sets up all assets used during the experiment.

        This internal method establishes the communication with the microcontrollers, data logger cores, and video
        system processes. It also verifies the configuration of Unity game engine and the mesoscope and activates
        mesoscope frame acquisition.

        Notes:
            This method will not run unless the host PC has access to the necessary number of logical CPU cores
            and other required hardware resources (GPU for video encoding, etc.). This prevents using the class on
            machines that are unlikely to sustain the runtime requirements.

            Zaber devices are connected during the initialization process and do not require a call to this method to
            operate. This design pattern is used to enable manipulating Headbar and Lickport before starting the main
            experiment.

            Calling this method automatically enables Console class (via console variable) if it was not enabled.

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

        # Starts all video systems. Note, this does NOT start frame saving. This is intentional, as it may take the
        # user some time to orient the mouse and the mesoscope.
        self._face_camera.start()
        self._left_camera.start()
        self._right_camera.start()

        message = "VideoSystems: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # This is a fairly long section that involves user feedback. It guides the user through mounting the mouse and
        # adjusting the HeadBar, LickPort, and the mesoscope objective.

        # Forces the user to confirm the system is prepared for zaber motor homing and positioning procedures. This code
        # will get stuck in the 'input' mode until the user confirms.
        message = (
            "Preparing to position the HeadBar motors. Remove the mesoscope objective, swivel out the VR screens, "
            "and make sure the animal is NOT mounted on the rig. Failure to fulfill these steps may DAMAGE the "
            "mesoscope and / or HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Unparks all motors. It is safe to do this for all motors at the same time, as the motors are not moving during
        # this operation. After this step all motors are 'armed' and can be interfaced with using the Zaber UI.
        self._headbar_z.unpark()
        self._headbar_pitch.unpark()
        self._headbar_roll.unpark()
        self._lickport_z.unpark()
        self._lickport_x.unpark()
        self._lickport_y.unpark()

        # First, homes lickport motors. The homing procedure aligns the lickport tube to the top right corner of the
        # running wheel (looking at the wheel from the front of the mesoscope cage). Assuming both HeadBar and LickPort
        # start from the parked position, the LickPort should be able to home without obstruction.
        self._lickport_z.home()
        self._lickport_x.home()
        self._lickport_y.home()

        # Waits for the lickport motors to finish homing. This is essential, since HeadBar homing trajectory intersects
        # LickPort homing trajectory.
        while self._lickport_z.is_busy or self._lickport_x.is_busy or self._lickport_y.is_busy:
            self._timestamp_timer.delay_noblock(delay=1000000)  # Delays for 1 second

        # Homes the HeadBar motors. The HeadBar homing procedure aligns the headbar roughly to the top middle of the
        # running wheel (looking at the wheel from the front of the mesoscope cage).
        self._headbar_z.home()
        self._headbar_pitch.home()
        self._headbar_roll.home()

        # Waits for the HeadBar motors to finish homing.
        while self._headbar_z.is_busy or self._headbar_pitch.is_busy or self._headbar_roll.is_busy:
            self._timestamp_timer.delay_noblock(delay=1000000)  # Delays for 1 second

        # Attempts to restore the HeadBar to the position used during the previous experiment or training session.
        # If restore positions are not available, uses the default mounting position. This HAS to be done before
        # moving the lickport motors to the mounting position
        self._headbar_restore()

        # Moves the LickPort to the mounting position. The mounting position is aligned to the top left corner
        # of the running wheel. This moves the LickPort out of the way the experimenter will use to mount the animal,
        # making it easier to mount the mouse. The trajectory to go from homing position to the mounting position
        # goes underneath the properly restored or mounter HeadBar, so there should be no risk of collision.
        self._lickport_z.move(amount=self._lickport_z.mount_position, absolute=True, native=True)
        self._lickport_x.move(amount=self._lickport_x.mount_position, absolute=True, native=True)
        self._lickport_y.move(amount=self._lickport_y.mount_position, absolute=True, native=True)

        while self._lickport_z.is_busy or self._lickport_x.is_busy or self._lickport_y.is_busy:
            self._timestamp_timer.delay_noblock(delay=1000000)  # Delays for 1 second

        # Gives user time to mount the animal and requires confirmation before proceeding further.
        message = (
            "Preparing to position the LickPort motors. Mount the animal onto the VR rig and install the mesoscope "
            "objetive. Do NOT swivel the VR screens back into position until instructed."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Attempts to restore the LickPort to the position used during the previous experiment or training session.
        # If restore positions are not available, uses the default PARKING position roughly aligned to the animal's
        # mouse.
        self._lickport_restore()

        message = "Zaber motor positioning: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)
        message = (
            "If necessary, adjust the HeadBar and LickPort positions to optimize imaging quality and animal running "
            "performance. Align the mesoscope objective with the cranial window and carry out the imaging preparation "
            "steps. Once everything is ready, swivel the VR screens back into position and arm the Mesoscope and "
            "Unity. This is the last manual checkpoint before the experiment starts."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Starts all microcontroller interfaces
        self._actor.start()
        self._actor.unlock_controller()  # Only Actor outputs data, so no need to unlock other controllers.
        self._sensor.start()
        self._encoder.start()

        message = "MicroControllerInterfaces: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

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

        # The mesoscope acquires frames at ~10 Hz and sends triggers with the on-phase duration of ~100 ms. We use
        # a polling frequency of ~1000 Hz here to ensure frame acquisition times are accurately detected.
        self._mesoscope_frame.check_state(repetition_delay=np.uint32(1000))

        # Starts mesoscope frame acquisition. This also verifies that the mesoscope responds to triggers and
        # actually starts acquiring frames using the _mesoscope_frame interface above.
        self._start_mesoscope()

        message = "Mesoscope frame acquisition: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts monitoring licks. Uses 1000 Hz polling frequency, which should be enough to resolve individual
        # licks of variable duration.
        self._lick.check_state(repetition_delay=np.uint32(1000))

        message = "Hardware module setup: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Sets the rest of the subsystems to use the REST state.
        self.vr_rest()

        # Finally, begins saving camera frames to disk
        self._face_camera.start_frame_saving()
        self._left_camera.start_frame_saving()
        self._right_camera.start_frame_saving()

        # The setup procedure is complete.
        self._started = True

        message = "MesoscopeExperiment assets: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops and terminates the MesoscopeExperiment runtime.

        First, stops acquiring mesoscope and visual camera frames. Then, disables all hardware modules and disconnects
        from cameras, microcontrollers, and data logger cores. Next, prompts the user to remove the animal from the
        VR rig and reset all Zaber motors to parking positions.

        Once all assets are terminated, pulls the data acquired from the mesoscope to the local session directory and
        preprocesses the mesoscope frame data and the behavioral data acquired during runtime. Finally, pushes the
        preprocessed session data directory to the NAS and Server.

        Notes:
            This method aggregates asset termination and data processing. After this method finishes its runtime, the
            experiment is complete, and it is safe to run another animal.
        """

        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        message = "Terminating MesoscopeExperiment runtime..."
        console.echo(message=message, level=LogLevel.INFO)

        # Initializes the timer to enforce the necessary delays
        timer = PrecisionTimer("s")

        # Resets the _started tracker
        self._started = False

        # Switches the system into the rest state. Since REST state has most modules set to stop-friendly states,
        # this is used as a shortcut to prepare the VR system for shutdown.
        self.vr_rest()

        # Instructs the mesoscope to stop acquiring frames
        self._mesoscope_stop.send_pulse()

        message = "Mesoscope stop pulse: sent."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Also instructs all cameras to stop saving frames
        self._face_camera.stop_frame_saving()
        self._left_camera.stop_frame_saving()
        self._right_camera.stop_frame_saving()

        message = "Camera frame saving: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Delays for 10 seconds. This ensures that the mesoscope and cameras have time to stop producing data.
        timer.delay_noblock(10)

        # Shuts down the modules that are still acquiring data. Note, this works in-addition to the VR REST state
        # disabling the encoder monitoring.
        self._lick.reset_command_queue()
        self._torque.reset_command_queue()
        self._mesoscope_frame.reset_command_queue()

        # Stops all microcontroller interfaces
        self._actor.stop()
        self._sensor.stop()
        self._encoder.stop()

        message = "MicroControllers: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops all cameras
        self._face_camera.stop()
        self._left_camera.stop()
        self._right_camera.stop()

        message = "Cameras: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops the data logger instance
        self._logger.stop()

        message = "Data Logger: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Generates the snapshot of the current HeadBar and LickPort positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are reset back to parking position.
        zaber_positions = _ZaberPositions(
            headbar_z=int(self._headbar_z.get_position(native=True)),
            headbar_pitch=int(self._headbar_pitch.get_position(native=True)),
            headbar_roll=int(self._headbar_roll.get_position(native=True)),
            lickport_z=int(self._lickport_z.get_position(native=True)),
            lickport_x=int(self._lickport_x.get_position(native=True)),
            lickport_y=int(self._lickport_y.get_position(native=True)),
        )
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)

        # Gives user time to remove the animal and the mesoscope objective and requires confirmation before proceeding
        # further.
        message = "Preparing to move the LickPort into the mounting position. Swivel the VR screens out."
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Moves the LickPort to the mounting position to assist removing the animal from the rig.
        self._lickport_z.move(amount=self._lickport_z.mount_position, absolute=True, native=True)
        self._lickport_x.move(amount=self._lickport_x.mount_position, absolute=True, native=True)
        self._lickport_y.move(amount=self._lickport_y.mount_position, absolute=True, native=True)

        # Waits for the lickport motors to finish moving.
        while self._lickport_z.is_busy or self._lickport_x.is_busy or self._lickport_y.is_busy:
            self._timestamp_timer.delay_noblock(delay=1000000)  # Delays for 1 second

        # Gives user time to remove the animal and the mesoscope objective and requires confirmation before proceeding
        # further.
        message = (
            "Preparing to reset the HeadBar and LickPort back to the parking position. Uninstall the mesoscope "
            "objective and remove the animal from the VR rig. Failure to do so may DAMAGE the mesoscope objective and "
            "HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Disconnects from all Zaber motors. This moves them into the parking position via the shutdown() method.
        self._headbar.disconnect()
        self._lickport.disconnect()

        message = "HeadBar and LickPort: reset."
        console.echo(message=message, level=LogLevel.SUCCESS)

        message = "Initializing data preprocessing..."
        console.echo(message=message, level=LogLevel.INFO)

        # Compresses all logs into a single .npz file. This is done both for long-term storage optimization and to
        # allow parsing the data.
        self._logger.compress_logs(remove_sources=True, verbose=True)

        # Parses behavioral data from the compressed logs and uses it to generate the behavioral_dataset.parquet file.
        # Also, extracts camera frame timestamps for each camera and saves them as a separate .npz file to optimize
        # further camera frame processing.
        self._process_log_data()

        # Pulls the frames and motion estimation data from the ScanImagePC into the local data directory.
        self._session_data.pull_mesoscope_data()

        # Preprocesses the pulled mesoscope data.
        self._session_data.process_mesoscope_data()

        # Moves all video files from the raw data directory to the camera_frames directory. Also renames the video files
        # to use more descriptive names
        shutil.move(
            self._session_data.raw_data_path.joinpath("51.mp4"),
            self._session_data.camera_frames_path.joinpath("face_camera.mp4"),
        )
        shutil.move(
            self._session_data.raw_data_path.joinpath("62.mp4"),
            self._session_data.camera_frames_path.joinpath("left_camera.mp4"),
        )
        shutil.move(
            self._session_data.raw_data_path.joinpath("73.mp4"),
            self._session_data.camera_frames_path.joinpath("right_camera.mp4"),
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

        Notes:
            This command is only executable if the class is running in the experiment mode.
        """

        # Toggles the state of the VR screens to be OFF if the VR screens are currently ON. If the screens are OFF,
        # keeps them OFF. This is done first to serve as a predictor of the breaks engaging, so that the animal can
        # interrupt active running sequences.
        if self._screen_on:
            self._screens.toggle()
            self._screen_on = False

        # Engages the break to prevent the mouse from moving the wheel
        self._break.toggle(state=True)

        # Temporarily suspends encoder monitoring. Since the wheel is locked, the mouse should not be able to produce
        # meaningful motion data.
        self._wheel_encoder.reset_command_queue()

        # Initiates torque monitoring at 1000 Hz. The torque can only be accurately measured when the wheel is locked,
        # as it requires a resistance force to trigger the sensor.
        self._torque.check_state(repetition_delay=np.uint32(1000))

        # Configures the state tracker to reflect the REST state
        self._change_vr_state(1)

    def vr_run(self) -> None:
        """Switches the VR system to the run state.

        In the run state, the break is disengaged to allow the mouse to freely move the wheel. The encoder module is
        enabled to record and share live running data with Unity, and the torque sensor is disabled. The VR screens are
        switched on to render the VR environment.

        Notes:
            This command is only executable if the class is running in the experiment mode.
        """
        # Initializes encoder monitoring at 2 kHz rate. The encoder aggregates wheel data at native speeds; this rate
        # only determines how often the aggregated data is sent to PC and Unity.
        self._wheel_encoder.check_state(repetition_delay=np.uint32(500))

        # Disables torque monitoring. To accurately measure torque, the sensor requires a resistance force provided by
        # the break. During running, measuring torque is not very reliable and adds little value compared to the
        # encoder.
        self._torque.reset_command_queue()

        # Toggles the state of the VR screens to be ON if the VR screens are currently OFF. If the screens are ON,
        # keeps them ON.
        if not self._screen_on:
            self._screens.toggle()
            self._screen_on = True

        # Disengages the break to allow the mouse to move the wheel
        self._break.toggle(False)

        # Configures the state tracker to reflect RUN state
        self._change_vr_state(2)

    def _headbar_restore(self) -> None:
        """Restores the motor positions for the HeadBar to the states recorded at the end of the previous experiment
        or training session.

        This method is called as part of the start() method runtime to prepare the VR system for mounting the
        animal. It tries to adjust the HeadBar motors to match the position used during the previous session, which
        should be optimal for the animal whose data will be recorded by the MesoscopeExperiment class. Additionally,
        this helps with aligning the cranial window and the mesoscope objective, by ensuring the animal head is oriented
        consistently across sessions.

        Notes:
            If this is the first ever session for the animal, this method will move the HeadBar to the predefined
            mounting position. Although it is not perfect for the given animal, the generic mounting position should
            still be fairly comfortable for the animal to be initially mounted into the VR system.

            This method should be called before calling the _lickport_restore() method.
        """
        # If the positions are not available, warns the user and sets the motors to the 'generic' mount position.
        if self._session_data.previous_zaber_positions_path is None:
            message = (
                "No previous Zaber positions found when attempting to restore HeadBar to the previous session "
                "position. Setting the HeadBar motors to use the default animal mounting positions. Adjust the "
                "positions manually when prompted by the startup runtime to optimize imaging quality and animal's"
                "running performance."
            )
            warnings.warn(message)
            self._headbar_z.move(amount=self._headbar_z.mount_position, absolute=True, native=True)
            self._headbar_pitch.move(amount=self._headbar_pitch.mount_position, absolute=True, native=True)
            self._headbar_roll.move(amount=self._headbar_roll.mount_position, absolute=True, native=True)
        else:
            # Loads previous zaber positions if they were saved
            previous_positions = _ZaberPositions.from_yaml(file_path=self._session_data.previous_zaber_positions_path)

            # Otherwise, restores Zaber positions.
            self._headbar_z.move(amount=previous_positions.headbar_z, absolute=True, native=True)
            self._headbar_pitch.move(amount=previous_positions.headbar_pitch, absolute=True, native=True)
            self._headbar_roll.move(amount=previous_positions.headbar_roll, absolute=True, native=True)

        # Waits for the motors to finish moving before returning to caller.
        while self._headbar_z.is_busy or self._headbar_pitch.is_busy or self._headbar_roll.is_busy:
            self._timestamp_timer.delay_noblock(delay=1000000)

    def _lickport_restore(self) -> None:
        """Restores the motor positions for the LickPort to the states recorded at the end of the previous experiment
        or training session.

        This method is called as part of the start() method runtime after the animal is mounted into the VR system.
        It positions the lickport to be comfortably accessible for the mounted animal.

        Notes:
            Unlike the _headbar_restore() method, this method falls back to the parking, rather than mounting position
            if it is not able to restore the animal to the previous session's position. This is intentional. The
            mounting position for the LickPort aligns it to the bottom left corner of the running wheel, rather than the
            animals' mouth. This provides the experimenter with more space for mounting the animal and will NOT be
            adequate for running experiment or training sessions. The parking position for the LickPort, on the other
            hand, aligns it to be easily accessible by the animal locked in the default 'mounting' HeadBar position.

            This method should be called after the _headbar_restore() method.
        """
        # If the positions are not available, warns the user and sets the motors to the 'generic' mount position.
        if self._session_data.previous_zaber_positions_path is None:
            message = (
                "No previous Zaber positions found when attempting to restore LickPort to the previous session "
                "position. Setting the LickPort motors to use the default parking positions. Adjust the "
                "positions manually when prompted by the startup runtime to optimize animal's running performance."
            )
            warnings.warn(message)
            self._lickport_z.move(amount=self._lickport_z.park_position, absolute=True, native=True)
            self._lickport_x.move(amount=self._lickport_x.park_position, absolute=True, native=True)
            self._lickport_y.move(amount=self._lickport_y.park_position, absolute=True, native=True)
        else:
            # Loads previous zaber positions if they were saved
            previous_positions = _ZaberPositions.from_yaml(file_path=self._session_data.previous_zaber_positions_path)

            # Otherwise, restores Zaber positions.
            self._lickport_z.move(amount=previous_positions.lickport_z, absolute=True, native=True)
            self._lickport_x.move(amount=previous_positions.lickport_x, absolute=True, native=True)
            self._lickport_y.move(amount=previous_positions.lickport_y, absolute=True, native=True)

        # Waits for the motors to finish moving before returning to caller.
        while self._lickport_z.is_busy or self._lickport_x.is_busy or self._lickport_y.is_busy:
            self._timestamp_timer.delay_noblock(delay=1000000)

    def _get_cue_sequence(self) -> NDArray[np.uint8]:
        """Requests Unity game engine to transmit the sequence of virtual reality track wall cues for the current task.

        This method is used as part of the experimental runtime startup process to both get the sequence of cues and
        verify that the Unity game engine is running and configured correctly.

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
                    sequence: NDArray[np.uint8] = np.frombuffer(buffer=payload, dtype=np.uint8)
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
        self._mesoscope_start.send_pulse()

        # Waits at most 2 seconds for the mesoscope to begin sending frame acquisition timestamps to the PC
        timeout_timer.reset()
        while timeout_timer.elapsed < 2:
            # Frame acquisition is confirmed by the frame timestamp recorder class flipping the pulse_status
            # property to True
            if self._mesoscope_frame.pulse_status:
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

    def _extract_logged_data(self) -> dict[int, Any]:
        """Extracts the VR states, Experiment states, and the Virtual Reality cue sequence from the log generated
        by the instance during runtime.

        This method reads the compressed '.npz' archives generated by the MesoscopeExperiment instance during runtime
        and parses all logged data. Unlike extraction methods used by hardware modules and VideoSystem instances, this
        method is not designed to be called by the end-user. Instead, it is called internally as part of the stop()
        method runtime to generate the unified behavioral dataset.

        Returns:
            A dictionary that uses id-codes associated with each type of log entries as keys and stores two-element
            tuples for all keys other than 0. The first element in each tuple is an array of timestamps, where each
            timestamp is a 64-bit unsigned numpy integer and specifies the number of microseconds since the UTC epoch
            onset. The second element of each tuple is an array of 8-bit unsigned numpy integer data-points. Key 0
            stores a byte numpy array that communicates the sequence of wall cues encountered by the animal as it was
            performing the experiment.
        """

        # Generates the log file path using the instance source ID and the data logger instance.
        log_path = self._logger.output_directory.joinpath(f"{self._source_id}_log.npz")

        # Loads the archive into RAM
        archive: NpzFile = np.load(file=log_path)

        # Precreates the dictionary to store the extracted data and temporary lists to store VR and Experiment states
        output_dict: dict[int, Any] = {}
        vr_states = []
        vr_timestamps = []
        experiment_states = []
        experiment_timestamps = []

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

            # If the message is longer than 10,000 bytes, it is a sequence of wall cues. It is very unlikely that we
            # will log any other data with this length, so it is a safe heuristic to use.
            if len(payload) > 10000:
                cue_sequence: NDArray[np.uint8] = payload.view(np.uint8).copy()  # Keeps the original numpy uint8 format
                output_dict[0] = cue_sequence  # Packages the cue sequence into the dictionary

            # If the message has a length of 2 bytes and the first element is 1, the message communicates the VR state
            # code.
            elif len(payload) == 2 and payload[0] == 1:
                vr_state = np.uint8(payload[2])  # Extracts the VR state code from the second byte of the message.
                vr_states.append(vr_state)
                vr_timestamps.append(timestamp)

            elif len(payload) == 2 and payload[0] == 2:
                # Extracts the experiment state code from the second byte of the message.
                experiment_state = np.uint8(payload[2])
                experiment_states.append(experiment_state)
                experiment_timestamps.append(timestamp)

        # Closes the archive to free up memory
        archive.close()

        # Converts lists to arrays and packages them into the dictionary as a 2-element tuple
        output_dict[1] = (np.array(vr_timestamps, dtype=np.uint64), np.array(vr_states, dtype=np.uint8))
        output_dict[2] = (np.array(experiment_timestamps, dtype=np.uint64), np.array(experiment_states, dtype=np.uint8))

        # Returns the extracted data dictionary
        return output_dict

    def _process_log_data(self) -> None:
        """Extracts the data logged during runtime from compressed .npz archives and uses it to generate the initial
        behavioral_dataset.parquet file and the temporary camera_timestamps.npz file.

        This method is called during the stop() method runtime to extract, align, and output the initial behavioral
        dataset. All data processed as part of our cell registration and video registration pipelines will later be
        aligned to and appended to this dataset to form the final dataset used for data analysis. Both the
        .parquet dataset and the .npz archive that stores the timestamps for the frames saved by each camera during
        runtime are saved to the session folder.
        """
        # First extracts the timestamps for the mesoscope frames. These timestamps are used as seeds to which all other
        # data sources are aligned during preprocessing and post-processing of the data.
        seeds = self._mesoscope_frame.parse_logged_data()

        # Iteratively goes over all hardware modules and extracts the data recorded by each module during runtime.
        # Uses discrete or continuous interpolation to align the data to the seed timestamps:
        # ENCODER
        timestamps, data = self._wheel_encoder.parse_logged_data()
        encoder_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=False)

        # TORQUE
        timestamps, data = self._torque.parse_logged_data()
        torque_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=False)

        # LICKS
        timestamps, data = self._lick.parse_logged_data()
        lick_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # WATER REWARD
        timestamps, data = self._reward.parse_logged_data()
        reward_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # SCREENS
        timestamps, data = self._screens.parse_logged_data()
        screen_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # BREAK
        timestamps, data = self._break.parse_logged_data()
        break_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # Extracts the VR states, Experiment states, and Virtual Reality queue sequence logged by the main process.
        extracted_data = self._extract_logged_data()

        # Processes VR states and Experiment states similar to hardware modules data. They are both discrete states:
        # VR STATE
        timestamps, data = extracted_data[1]
        vr_states = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # EXPERIMENT STATE
        timestamps, data = extracted_data[2]
        experiment_states = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # CUE SEQUENCE
        # Unlike other extracted data, we do not have the time points for each cue in the sequence, as this data is
        # pre-generated when the Unity task is initialized. However, since we know the distance associated with each
        # cue, we can use teh cumulative traveled distance extracted from the encoder to know the cue experienced by the
        # mouse at each time-point.
        cue_sequence: NDArray[np.uint8] = extracted_data[0]

        # Uses the dictionary provided at class initialization to compute the length of each cue in the sequence. Then
        # computes the cumulative distance the animal needs to travel to reach each cue in the sequence. The first cue
        # is associated with distance of 0 (the animal starts at this cue), the distance to each following cue is the
        # sum of all previous cue lengths
        distance_sequence = np.zeros(len(cue_sequence), dtype=np.float64)
        distance_sequence[1:] = np.cumsum([self._cue_map[int(code)] for code in cue_sequence[:-1]])

        # Finds indices of cues for each encoder distance point, ensuring 0 maps to first cue (index 0). This gives us
        # the cues the animal experiences at each seed timestamp (insofar as each cue is tied to the cumulative distance
        # covered by the animal at each timestamp).
        indices = np.maximum(0, np.searchsorted(distance_sequence, encoder_data, side="right") - 1)
        cue_data = cue_sequence[indices]  # Extracts the cue experienced by the mouse at each seed timestamp.

        # Assembles the aligned behavioral dataset from the data processed above:
        # Define the schema with proper types and units. The schema determines the datatypes and column names used by
        # the dataset
        schema = {
            "time_us": pl.UInt64,
            "distance_cm": pl.Float64,
            "torque_N_cm": pl.Float64,
            "lick_on": pl.UInt8,
            "dispensed_water_μL": pl.Float64,
            "screens_on": pl.UInt8,
            "break_on": pl.UInt8,
            "vr_state": pl.UInt8,
            "experiment_state": pl.UInt8,
            "cue": pl.UInt8,
        }

        # Creates a mapping of our data arrays to schema columns
        data_mapping = {
            "time_us": seeds,
            "distance_cm": encoder_data,
            "torque_N_cm": torque_data,
            "lick_on": lick_data,
            "dispensed_water_μL": reward_data,
            "screens_on": screen_data,
            "break_on": break_data,
            "vr_state": vr_states,
            "experiment_state": experiment_states,
            "cue": cue_data,
        }

        # Creates Polars dataframe with schema
        dataframe = pl.DataFrame(data_mapping, schema=schema)

        # Saves the dataset as a zstd compressed parquet file
        dataframe.write_parquet(
            file=self._session_data.behavioral_data_path,
            compression="zstd",
            compression_level=22,
            use_pyarrow=True,
            statistics=True,
        )

        # Also extracts the timestamps for the frames saved by each camera. This data will be used during DeepLabCut
        # processing to align extracted data to the mesoscope frame timestamps (seeds). For now this data is kept as
        # an .npz archive.
        face_stamps = self._face_camera.extract_logged_data()
        left_stamps = self._left_camera.extract_logged_data()
        right_stamps = self._right_camera.extract_logged_data()

        # Saves extracted data as an .npz archive
        np.savez_compressed(
            file=self._session_data.camera_timestamps_path,
            face_timestamps=face_stamps,
            left_timestamps=left_stamps,
            right_timestamps=right_stamps,
        )


# class BehavioralTraining:
#     pass
#
#     def _vr_lick_train(self) -> None:
#         """Switches the VR system into the lick training state.
#
#         In the lick training state, the break is enabled, preventing the mouse from moving the wheel. The screens
#         are turned off, Unity and mesoscope are disabled. Torque and Encoder monitoring are also disabled. The only
#         working hardware modules are lick sensor and water valve.
#
#         Notes:
#             This command is only executable if the class is running in the lick training mode.
#
#             This state is set automatically during start() method runtime. It should not be called externally by the
#             user.
#
#         Raises:
#             RuntimeError: If the Mesoscope-VR system is not started, or the class is not in the lick training runtime
#                 mode.
#         """
#
#         if not self._started or not self._mode == RuntimeModes.LICK_TRAINING.value:
#             message = (
#                 f"Unable to switch the Mesoscope-VR system to the lick training state. Either the start() method of "
#                 f"the MesoscopeExperiment class has not been called to setup the necessary assets or the current "
#                 f"runtime mode of the class is not set to the 'lick_training' mode."
#             )
#             console.error(message=message, error=RuntimeError)
#
#         # Ensures the break is enabled. The mice do not need to move the wheel during the lick training runtime.
#         self._break.toggle(True)
#
#         # Disables both torque and encoder. During lick training we are not interested in mouse motion data.
#         self._torque.reset_command_queue()
#         self._wheel_encoder.reset_command_queue()
#
#         # Toggles the state of the VR screens to be OFF if the VR screens are currently ON. If the screens are OFF,
#         # keeps them OFF.
#         if self._screen_on:
#             self._screens.toggle()
#             self._screen_on = False
#
#         # The lick sensor should already be running at 1000 Hz resolution, as we generally keep it on for all our
#         # pipelines. Unity and mesoscope should not be enabled.
#
#         # Configures the state tracker to reflect the LICK TRAIN state
#         self._change_vr_state(3)
#         message = f"VR State: {self._state_map[self._vr_state]}."
#         console.echo(message=message, level=LogLevel.INFO)
#
#     def _vr_run_train(self) -> None:
#         """Switches the VR system into the run training state.
#
#         In the run training state, the break is disabled, allowing the animal to move the wheel. The encoder module is
#         enabled to monitor the running metrics (distance and / or speed). The lick sensor and water valve modules are
#         also enabled to conditionally reward the animal for desirable performance. The VR screens are turned off. Unity,
#         mesoscope, and the torque module are disabled.
#
#         Notes:
#             This command is only executable if the class is running in the run training mode.
#
#             This state is set automatically during start() method runtime. It should not be called externally by the
#             user.
#
#         Raises:
#             RuntimeError: If the Mesoscope-VR system is not started, or the class is not in the run training runtime
#                 mode.
#         """
#
#         if not self._started or not self._mode == RuntimeModes.RUN_TRAINING.value:
#             message = (
#                 f"Unable to switch the Mesoscope-VR system to the run training state. Either the start() method of the "
#                 f"MesoscopeExperiment class has not been called to setup the necessary assets or the current runtime "
#                 f"mode of the class is not set to the 'run_training' mode."
#             )
#             console.error(message=message, error=RuntimeError)
#
#         # Disables both torque sensor.
#         self._torque.reset_command_queue()
#
#         # Enables the encoder module to monitor animal's running performance
#         self._wheel_encoder.check_state(repetition_delay=np.uint32(500))
#
#         # Toggles the state of the VR screens to be OFF if the VR screens are currently ON. If the screens are OFF,
#         # keeps them OFF.
#         if self._screen_on:
#             self._screens.toggle()
#             self._screen_on = False
#
#         # Ensures the break is disabled. This allows the animal to run on the wheel freely.
#         self._break.toggle(False)
#
#         # The lick sensor should already be running at 1000 Hz resolution, as we generally keep it on for all our
#         # pipelines. Unity and mesoscope should not be enabled.
#
#         # Configures the state tracker to reflect the RUN TRAIN state
#         self._change_vr_state(4)
#         message = f"VR State: {self._state_map[self._vr_state]}."
#         console.echo(message=message, level=LogLevel.INFO)


class SystemCalibration:
    """The base class for all Sun lab calibration runtimes.

    This class is used to test and calibrate various elements of the Mesoscope-VR system before conducting experiment
    and training runtimes. It is heavily used during initial system construction and configuration. Once the system is
    built, this class is primarily used for water valve setup (filling / emptying) and calibration, which has to be
    carried out frequently between experimental days.

    This class exposes many methods that address individual components in the system. These methods should be used
    to issue commands to the calibrated systems. This is similar to the 'state' setter methods of MesoscopeExperiment
    and BehavioralTraining classes, but provides a considerably finer control over modules.

    Notes:
        Calling the initializer does not start the underlying processes. Use the start() method before issuing other
        commands to properly initialize all remote processes.

        See the 'axtl-ports' cli command from the ataraxis-transport-layer-pc library if you need help discovering the
        USB ports used by Ataraxis Micro Controller (AMC) devices.

        See the 'axvs-ids' cli command from ataraxis-video-system if you need help discovering the camera indices used
        by the Harvesters-managed and OpenCV-managed cameras.

        See the 'sl-devices' cli command from this library if you need help discovering the serial ports used by the
        Zaber motion controllers.

        This class is specifically designed to test Mesoscope-VR system components. To run a training session, use
        BehavioralTraining class. To run experiments, use MesoscopeExperiment class.

    Args:
        screens_on: Determines whether the VR screens are ON when this class is initialized. Since there is no way of
            getting this information via hardware, the initial screen state has to be supplied by the user as an
            argument. The class will manage and track the state after initialization.
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
        _delay_timer: Stores the PrecisionTimer used to delay execution where necessary.
        _test_folder: Stores the path to the root test directory used to store the data generated during calibration
            runtime.
        _zaber_positions_path: Stores the path to the YAML file that contains the Zaber motor positions used during
            previous calibration runtimes. This is used to simulate the behavior of experiment and training classes with
            respect to restoring Zaber positions across experimental sessions. If the file does not exist, this will
            be set to None.

    Raises:
        TypeError: If any of the arguments are not of the expected type.
        ValueError: If any of the arguments are not of the expected value.
    """

    def __init__(
        self,
        test_root: Path = Path("/media/Data/Experiments/Test/test_data"),
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

        # Initializes the start state tracker first
        self._started: bool = False

        # Initializes a timer to delay code execution. Primarily, this is used during Zaber motor manipulation.
        self._delay_timer = PrecisionTimer('s')

        # Resets the test folder and saves its path to class attribute
        shutil.rmtree(test_root, ignore_errors=True)
        ensure_directory_exists(test_root)
        self._test_folder: Path = test_root

        # Generates the path to the zaber yaml file. If the file does not exist, this will be set to None.
        self._zaber_positions_path: Path | None = Path(self._test_folder.parent.joinpath("zaber_positions.yaml"))
        if not self._zaber_positions_path.exists():
            self._zaber_positions_path = None

        if not isinstance(valve_calibration_data, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
            for item in valve_calibration_data
        ):
            message = (
                f"Unable to initialize the SystemCalibration class. Expected a tuple of 2-element tuples with "
                f"integer or float values for 'valve_calibration_data' argument, but instead encountered "
                f"{valve_calibration_data} of type {type(valve_calibration_data).__name__} with at least one "
                f"incompatible element."
            )
            console.error(message=message, error=TypeError)

        # Initializes the DataLogger instance used to log data from all microcontrollers, camera frame savers, and this
        # class instance.
        self._logger: DataLogger = DataLogger(
            output_directory=self._test_folder,
            instance_name="system",
            sleep_timer=0,
            exist_ok=True,
            process_count=1,
            thread_count=10,
        )

        # ACTOR. Actor AMC controls the hardware that needs to be triggered by PC at irregular intervals. Most of such
        # hardware is designed to produce some form of an output: deliver water reward, engage wheel breaks, issue a
        # TTL trigger, etc.

        # Module interfaces:
        self._mesoscope_start: TTLInterface = TTLInterface(module_id=np.uint8(1), debug=True)
        self._mesoscope_stop: TTLInterface = TTLInterface(module_id=np.uint8(2), debug=True)
        self._break = BreakInterface(
            minimum_break_strength=43.2047,  # 0.6 in oz
            maximum_break_strength=1152.1246,  # 16 in oz
            object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
            debug=True,
        )
        self._reward = ValveInterface(valve_calibration_data=valve_calibration_data, debug=True)
        self._screens = ScreenInterface(initially_on=screens_on, debug=True)

        # Main interface:
        self._actor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=actor_port,
            data_logger=self._logger,
            module_interfaces=(self._mesoscope_start, self._mesoscope_stop, self._break, self._reward, self._screens),
        )

        # SENSOR. Sensor AMC controls the hardware that collects data at regular intervals. This includes lick sensors,
        # torque sensors, and input TTL recorders. Critically, all managed hardware does not rely on hardware interrupt
        # logic to maintain the necessary precision.

        # Module interfaces:
        # Mesoscope frame timestamp recorder. THe class is configured to report detected pulses during runtime to
        # support checking whether mesoscope start trigger correctly starts the frame acquisition process.
        self._mesoscope_frame: TTLInterface = TTLInterface(module_id=np.uint8(1), debug=True)
        self._lick: LickInterface = LickInterface(lick_threshold=1000, debug=True)  # Lick sensor
        self._torque: TorqueInterface = TorqueInterface(
            baseline_voltage=2046,  # ~1.65 V
            maximum_voltage=2750,  # This was determined experimentally and matches the torque that overcomes break
            sensor_capacity=720.0779,  # 10 in oz
            object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
            debug=True,
        )

        # Main interface:
        self._sensor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(152),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=sensor_port,
            data_logger=self._logger,
            module_interfaces=(self._mesoscope_frame, self._lick, self._torque),
        )

        # ENCODER. Encoder AMC is specifically designed to interface with a rotary encoder connected to the running
        # wheel. The encoder uses hardware interrupt logic to maintain high precision and, therefore, it is isolated
        # to a separate microcontroller to ensure adequate throughput.

        # Module interfaces:
        self._wheel_encoder: EncoderInterface = EncoderInterface(
            encoder_ppr=8192, object_diameter=15.0333, cm_per_unity_unit=10.0, debug=True
        )

        # Main interface:
        self._encoder: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(203),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=encoder_port,
            data_logger=self._logger,
            module_interfaces=(self._wheel_encoder,),
        )

        # FACE CAMERA. This is the high-grade scientific camera aimed at the animal's face using the hot-mirror. It is
        # a 10-gigabit 9MP camera with a red long-pass filter and has to be interfaced through the GeniCam API. Since
        # the VRPC has a 4090 with 2 hardware acceleration chips, we are using the GPU to save all of our frame data.
        self._face_camera: VideoSystem = VideoSystem(
            system_id=np.uint8(51),
            data_logger=self._logger,
            output_directory=self._test_folder,
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
            system_id=np.uint8(62),
            data_logger=self._logger,
            output_directory=self._test_folder,
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
            system_id=np.uint8(73),
            data_logger=self._logger,
            output_directory=self._test_folder,
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
        # headbar attached to the mouse in Z, Roll, and Pitch dimensions. Note, this assumes that the chaining order of
        # individual zaber devices is fixed and is always Z-Pitch-Roll.
        self._headbar: ZaberConnection = ZaberConnection(port=headbar_port)
        self._headbar.connect()  # Since this does not reserve additional resources, establishes connection right away
        self._headbar_z: ZaberAxis = self._headbar.get_device(0).axis
        self._headbar_pitch: ZaberAxis = self._headbar.get_device(1).axis
        self._headbar_roll: ZaberAxis = self._headbar.get_device(2).axis

        # Lickport controller (zaber). This is an assembly of 3 zaber controllers (devices) that allow moving the
        # lick tube in Z, X, and Y dimensions. Note, this assumes that the chaining order of individual zaber devices is
        # fixed and is always Z-X-Y.
        self._lickport: ZaberConnection = ZaberConnection(port=lickport_port)
        self._lickport.connect()  # Since this does not reserve additional resources, establishes connection right away
        self._lickport_z: ZaberAxis = self._lickport.get_device(0).axis
        self._lickport_x: ZaberAxis = self._lickport.get_device(1).axis
        self._lickport_y: ZaberAxis = self._lickport.get_device(2).axis

    def start(self) -> None:
        """Sets up all assets used during calibration.

        This internal method establishes the communication with the microcontrollers, data logger cores, and video
        system processes. After this method's runtime, it is possible to use most other class methods to issue commands
        to all Mesoscope-VR systems addressable through this class.

        Notes:
            This method will not run unless the host PC has access to the necessary number of logical CPU cores
            and other required hardware resources (GPU for video encoding, etc.). This prevents using the class on
            machines that are unlikely to sustain the runtime requirements.

            Zaber devices are connected during the initialization process and do not require a call to this method to
            operate. This design pattern is used to enable manipulating Headbar and Lickport before starting the main
            experiment.

            Calling this method automatically enables Console class (via console variable) if it was not enabled.

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

        message = "Initializing SystemCalibration assets..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts the data logger
        self._logger.start()

        message = "DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts all video systems. Note, this does NOT start frame saving. This is intentional, as it may take the
        # user some time to orient the mouse and the mesoscope.
        self._face_camera.start()
        self._left_camera.start()
        self._right_camera.start()

        message = "VideoSystems: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # This is a fairly long section that involves user feedback. It guides the user through mounting the mouse and
        # adjusting the HeadBar, LickPort, and the mesoscope objective.

        # Forces the user to confirm the system is prepared for zaber motor homing and positioning procedures. This code
        # will get stuck in the 'input' mode until the user confirms.
        message = (
            "Preparing to position the HeadBar motors. Remove the mesoscope objective, swivel out the VR screens, "
            "and make sure the animal is NOT mounted on the rig. Failure to fulfill these steps may DAMAGE the "
            "mesoscope and / or HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Attempts to restore the HeadBar to the position used during the previous experiment or training session.
        # If restore positions are not available, uses the default mounting position. This HAS to be done before
        # moving the lickport motors to the mounting position
        self.headbar_restore()

        # Moves the LickPort to the mounting position. The mounting position is aligned to the top left corner
        # of the running wheel. This moves the LickPort out of the way the experimenter will use to mount the animal,
        # making it easier to mount the mouse. The trajectory to go from homing position to the mounting position
        # goes underneath the properly restored or mounter HeadBar, so there should be no risk of collision.
        self._lickport_z.move(amount=self._lickport_z.mount_position, absolute=True, native=True)
        self._lickport_x.move(amount=self._lickport_x.mount_position, absolute=True, native=True)
        self._lickport_y.move(amount=self._lickport_y.mount_position, absolute=True, native=True)

        while self._lickport_z.is_busy or self._lickport_x.is_busy or self._lickport_y.is_busy:
            self._delay_timer.delay_noblock(delay=1)  # Delays for 1 second

        # Gives user time to mount the animal and requires confirmation before proceeding further.
        message = (
            "Preparing to position the LickPort motors. Mount the animal onto the VR rig and install the mesoscope "
            "objetive. Do NOT swivel the VR screens back into position until instructed."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Attempts to restore the LickPort to the position used during the previous experiment or training session.
        # If restore positions are not available, uses the default PARKING position roughly aligned to the animal's
        # mouse.
        self.lickport_restore()

        message = "Zaber motor positioning: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)
        message = (
            "If necessary, adjust the HeadBar and LickPort positions to optimize imaging quality and animal running "
            "performance. Align the mesoscope objective with the cranial window and carry out the imaging preparation "
            "steps. Once everything is ready, swivel the VR screens back into position and arm the Mesoscope and "
            "Unity. This is the last manual checkpoint before the runtime starts."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Starts all microcontroller interfaces
        self._actor.start()
        self._actor.unlock_controller()  # Only Actor outputs data, so no need to unlock other controllers.
        self._sensor.start()
        self._encoder.start()

        message = "SystemCalibration: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

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

        # The mesoscope acquires frames at ~10 Hz and sends triggers with the on-phase duration of ~100 ms. We use
        # a polling frequency of ~1000 Hz here to ensure frame acquisition times are accurately detected.
        self._mesoscope_frame.check_state(repetition_delay=np.uint32(1000))

        # Starts monitoring licks. Uses 1000 Hz polling frequency, which should be enough to resolve individual
        # licks of variable duration.
        self._lick.check_state(repetition_delay=np.uint32(1000))

        message = "Hardware module setup: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Finally, begins saving camera frames to disk
        self._face_camera.start_frame_saving()
        self._left_camera.start_frame_saving()
        self._right_camera.start_frame_saving()

        # The setup procedure is complete.
        self._started = True

        message = "SystemCalibration assets: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops and terminates the SystemCalibration runtime.

        This method shuts down all data collectors and disconnects from all cameras, microcontrollers, and the data
        logger. It then runs the behavioral data log processing algorithm on the data collected during calibration and
        displays the results to the user. Overall, this method simulates the stop() method behavior of the
        MesoscopeExperiment and BehavioralTraining classes.

        Notes:
            This method aggregates asset termination and data processing. After this method finishes its runtime, the
            calibration is complete, and it is safe to start another runtime.
        """

        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        message = "Terminating SystemCalibration runtime..."
        console.echo(message=message, level=LogLevel.INFO)

        # Resets the _started tracker
        self._started = False

        # Instructs all cameras to stop saving frames
        self._face_camera.stop_frame_saving()
        self._left_camera.stop_frame_saving()
        self._right_camera.stop_frame_saving()

        message = "Camera frame saving: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Delays for 10 seconds. This ensures that the cameras have time to stop producing data.
        self._delay_timer.delay_noblock(10)

        # Stops all microcontroller interfaces. This directly shuts down all individual modules.
        self._actor.stop()
        self._sensor.stop()
        self._encoder.stop()

        message = "MicroControllers: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops all cameras
        self._face_camera.stop()
        self._left_camera.stop()
        self._right_camera.stop()

        message = "Cameras: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops the data logger instance
        self._logger.stop()

        message = "Data Logger: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Generates the snapshot of the current HeadBar and LickPort positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are reset back to parking position.
        zaber_positions = _ZaberPositions(
            headbar_z=int(self._headbar_z.get_position(native=True)),
            headbar_pitch=int(self._headbar_pitch.get_position(native=True)),
            headbar_roll=int(self._headbar_roll.get_position(native=True)),
            lickport_z=int(self._lickport_z.get_position(native=True)),
            lickport_x=int(self._lickport_x.get_position(native=True)),
            lickport_y=int(self._lickport_y.get_position(native=True)),
        )
        zaber_positions.to_yaml(file_path=self._zaber_positions_path)

        # Gives user time to remove the animal and the mesoscope objective and requires confirmation before proceeding
        # further.
        message = "Preparing to move the LickPort into the mounting position. Swivel the VR screens out."
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Moves the LickPort to the mounting position to assist removing the animal from the rig.
        self._lickport_z.move(amount=self._lickport_z.mount_position, absolute=True, native=True)
        self._lickport_x.move(amount=self._lickport_x.mount_position, absolute=True, native=True)
        self._lickport_y.move(amount=self._lickport_y.mount_position, absolute=True, native=True)

        # Waits for the lickport motors to finish moving.
        while self._lickport_z.is_busy or self._lickport_x.is_busy or self._lickport_y.is_busy:
            self._delay_timer.delay_noblock(delay=1)  # Delays for 1 second

        # Gives user time to remove the animal and the mesoscope objective and requires confirmation before proceeding
        # further.
        message = (
            "Preparing to reset the HeadBar and LickPort back to the parking position. Uninstall the mesoscope "
            "objective and remove the animal from the VR rig. Failure to do so may DAMAGE the mesoscope objective and "
            "HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Disconnects from all Zaber motors. This moves them into the parking position via the shutdown() method.
        self._headbar.disconnect()
        self._lickport.disconnect()

        message = "HeadBar and LickPort: reset."
        console.echo(message=message, level=LogLevel.SUCCESS)

        message = "SystemCalibration runtime: terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def headbar_restore(self) -> None:
        """Restores the motor positions for the HeadBar to the states recorded at the end of the previous calibration
        runtime.

        This method is called as part of the start() method runtime to simulate the animal mounting procedure performed
        during MesoscopeExperiment and BehavioralTraining runtimes.

        Notes:
            If the position log file is not available, this method will move the HeadBar to the predefined mounting
            position.

            This method should be called before calling the _lickport_restore() method.
        """
        # If the positions are not available, warns the user and sets the motors to the 'generic' mount position.
        if self._zaber_positions_path is None:
            message = (
                "No previous Zaber positions found when attempting to restore HeadBar to the previous session "
                "position. Setting the HeadBar motors to use the default animal mounting positions. Adjust the "
                "positions manually when prompted by the startup runtime to optimize imaging quality and animal's"
                "running performance."
            )
            warnings.warn(message)
            self._headbar_z.move(amount=self._headbar_z.mount_position, absolute=True, native=True)
            self._headbar_pitch.move(amount=self._headbar_pitch.mount_position, absolute=True, native=True)
            self._headbar_roll.move(amount=self._headbar_roll.mount_position, absolute=True, native=True)
        else:
            # Loads previous zaber positions if they were saved
            previous_positions = _ZaberPositions.from_yaml(file_path=self._zaber_positions_path)

            # Otherwise, restores Zaber positions.
            self._headbar_z.move(amount=previous_positions.headbar_z, absolute=True, native=True)
            self._headbar_pitch.move(amount=previous_positions.headbar_pitch, absolute=True, native=True)
            self._headbar_roll.move(amount=previous_positions.headbar_roll, absolute=True, native=True)

        # Waits for the motors to finish moving before returning to caller.
        while self._headbar_z.is_busy or self._headbar_pitch.is_busy or self._headbar_roll.is_busy:
            self._delay_timer.delay_noblock(delay=1)

    def lickport_restore(self) -> None:
        """Restores the motor positions for the LickPort to the states recorded at the end of the previous calibration
        runtime.

        This method is called as part of start() method runtime to simulate the animal mounting procedure performed
        during MesoscopeExperiment and BehavioralTraining runtimes.

        Notes:
            If the position log file is not available, this method will move the LickPort to the predefined parking
            position.

            This method should be called after the _headbar_restore() method.
        """
        # If the positions are not available, warns the user and sets the motors to the 'generic' mount position.
        if self._zaber_positions_path is None:
            message = (
                "No previous Zaber positions found when attempting to restore LickPort to the previous session "
                "position. Setting the LickPort motors to use the default parking positions. Adjust the "
                "positions manually when prompted by the startup runtime to optimize animal's running performance."
            )
            warnings.warn(message)
            self._lickport_z.move(amount=self._lickport_z.park_position, absolute=True, native=True)
            self._lickport_x.move(amount=self._lickport_x.park_position, absolute=True, native=True)
            self._lickport_y.move(amount=self._lickport_y.park_position, absolute=True, native=True)
        else:
            # Loads previous zaber positions if they were saved
            previous_positions = _ZaberPositions.from_yaml(file_path=self._zaber_positions_path)

            # Otherwise, restores Zaber positions.
            self._lickport_z.move(amount=previous_positions.lickport_z, absolute=True, native=True)
            self._lickport_x.move(amount=previous_positions.lickport_x, absolute=True, native=True)
            self._lickport_y.move(amount=previous_positions.lickport_y, absolute=True, native=True)

        # Waits for the motors to finish moving before returning to caller.
        while self._lickport_z.is_busy or self._lickport_x.is_busy or self._lickport_y.is_busy:
            self._delay_timer.delay_noblock(delay=1)

    def home_zaber_motors(self) -> None:
        """Unparks and homes all Zaber motors controlling the HeadBar and LickPort.

        This method is used as part of the start() method runtime to ensure Zaber motors are unparked and homed before
        they are used to position the HeadBar and Lickport for the task (calibration, mounting, etc.).
        """

        # Unparks all motors. It is safe to do this for all motors at the same time, as the motors are not moving during
        # this operation. After this step all motors are 'armed' and can be interfaced with using the Zaber UI.
        self._headbar_z.unpark()
        self._headbar_pitch.unpark()
        self._headbar_roll.unpark()
        self._lickport_z.unpark()
        self._lickport_x.unpark()
        self._lickport_y.unpark()

        # First, homes lickport motors. The homing procedure aligns the lickport tube to the top right corner of the
        # running wheel (looking at the wheel from the front of the mesoscope cage). Assuming both HeadBar and LickPort
        # start from the parked position, the LickPort should be able to home without obstruction.
        self._lickport_z.home()
        self._lickport_x.home()
        self._lickport_y.home()

        # Waits for the lickport motors to finish homing. This is essential, since HeadBar homing trajectory intersects
        # LickPort homing trajectory.
        while self._lickport_z.is_busy or self._lickport_x.is_busy or self._lickport_y.is_busy:
            self._delay_timer.delay_noblock(delay=1)  # Delays for 1 second

        # Homes the HeadBar motors. The HeadBar homing procedure aligns the headbar roughly to the top middle of the
        # running wheel (looking at the wheel from the front of the mesoscope cage).
        self._headbar_z.home()
        self._headbar_pitch.home()
        self._headbar_roll.home()

        # Waits for the HeadBar motors to finish homing.
        while self._headbar_z.is_busy or self._headbar_pitch.is_busy or self._headbar_roll.is_busy:
            self._delay_timer.delay_noblock(delay=1)  # Delays for 1 second

    def enable_encoder_monitoring(self, polling_delay: int = 500) -> None:
        """Enables wheel encoder monitoring with the provided polling delay in microseconds."""
        self._wheel_encoder.reset_pulse_count()
        self._wheel_encoder.check_state(repetition_delay=np.uint32(polling_delay))

    def disable_encoder_monitoring(self) -> None:
        """Stops monitoring the wheel encoder."""
        self._wheel_encoder.reset_command_queue()

    def start_mesoscope(self) -> None:
        """Sends the acquisition start TTL pulse to the mesoscope."""
        self._mesoscope_start.send_pulse()

    def stop_mesoscope(self) -> None:
        """Sends the acquisition stop TTL pulse to the mesoscope."""
        self._mesoscope_stop.send_pulse()

    def enable_break(self) -> None:
        """Engages wheel break, preventing the animal from running on the wheel."""
        self._break.toggle(state=True)

    def disable_break(self) -> None:
        """Disengages wheel break, enabling the animal to run on the wheel."""
        self._break.toggle(state=False)

    def toggle_screens(self) -> None:
        """Changes (flips) the current state of the VR screens (On / Off)."""
        self._screens.toggle()

    def enable_mesoscope_frame_monitoring(self, polling_delay: int) -> None:
        """Enables monitoring the TTL pulses sent by the mesoscope to communicate when it is scanning a frame with the
        provided polling delay in microseconds."""
        self._mesoscope_frame.check_state(repetition_delay=np.uint32(polling_delay))

    def disable_mesoscope_frame_monitoring(self) -> None:
        """Stops monitoring the TTL pulses sent by the mesoscope to communicate when it is scanning a frame."""
        self._mesoscope_frame.reset_command_queue()

    def enable_lick_monitoring(self, polling_delay: int) -> None:
        """Enables lick sensor monitoring with the provided polling delay in microseconds."""
        self._lick.check_state(repetition_delay=np.uint32(polling_delay))

    def disable_lick_monitoring(self) -> None:
        """Disables lick sensor monitoring."""
        self._lick.reset_command_queue()

    def enable_torque_monitoring(self, polling_delay: int) -> None:
        """Enables torque sensor monitoring with the provided polling delay in microseconds."""
        self._torque.check_state(repetition_delay=np.uint32(polling_delay))

    def disable_torque_monitoring(self) -> None:
        """Disables torque sensor monitoring."""
        self._torque.reset_command_queue()

    def open_valve(self) -> None:
        """Opens the solenoid valve to enable water flow."""
        self._reward.toggle(state=True)

    def close_valve(self) -> None:
        """Closes the solenoid valve to restrict water flow."""
        self._reward.toggle(state=False)

    def deliver_reward(self, volume: float = 5.0) -> None:
        """Pulses the valve for the duration of time necessary to deliver the provided volume of water.

        Note, the way the requested volume is translated into valve open time depends on the calibration data
        provided to the class at initialization.
        """
        self._reward.set_parameters(pulse_duration=self._reward.get_duration_from_volume(volume))
        self._reward.send_pulse()

    def reference_valve(self) -> None:
        """Runs the 'reference' valve calibration procedure.

        The reference calibration HAS to be run with the water line being primed, deaerated, and the holding syringe
        filled exactly to the 5 mL mark. This procedure is designed to dispense 5 uL of water 200 times, which should
        overall deliver ~ 1 ml of water.

        Notes:
            This method repositions HeadBar and LickPort motors to optimize collecting dispensed water before the
            procedure and move them back to the parking positions when the referencing is over.

            Use one of the conical tubes stored next to the mesoscope to collect the dispensed water. We advise using
            both the visual confirmation (looking at the syringe water level drop) and the weight confirmation
            (weighing the water dispensed into the collection tube). This provides the most accurate referencing result.
        """
        self._reward.set_parameters(
            pulse_duration=np.uint32(35590), calibration_delay=np.uint32(200000), calibration_count=np.uint16(200)
        )  # 5 ul x 200 times
        self._reward.calibrate()

    def calibrate_valve(self) -> None:
        self._reward.set_parameters(
            pulse_duration=np.uint32(15000), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
        )  # 15 milliseconds
        self._reward.calibrate()

        self._reward.set_parameters(
            pulse_duration=np.uint32(30000), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
        )  # 30 milliseconds
        self._reward.calibrate()

        self._reward.set_parameters(
            pulse_duration=np.uint32(45000), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
        )  # 45 milliseconds
        self._reward.calibrate()

        self._reward.set_parameters(
            pulse_duration=np.uint32(60000), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
        )  # 60 milliseconds
        self._reward.calibrate()


if __name__ == "__main__":
    test_class = SystemCalibration()
    test_class.start()
    while input("Calibration runtime goes brrr. Enter 'exit' to exit.") != "exit":
        continue
    test_class.stop()

    # calibration()

    # session_dir = SessionData(
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
