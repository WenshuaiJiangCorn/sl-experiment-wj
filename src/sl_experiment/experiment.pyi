from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from pynput import keyboard
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_data_structures import DataLogger, YamlConfig, SharedMemoryArray
from ataraxis_communication_interface import MQTTCommunication

from .visualizers import BehaviorVisualizer as BehaviorVisualizer
from .binding_classes import (
    HeadBar as HeadBar,
    LickPort as LickPort,
    VideoSystems as VideoSystems,
    ZaberPositions as ZaberPositions,
    MicroControllerInterfaces as MicroControllerInterfaces,
)
from .module_interfaces import (
    BreakInterface as BreakInterface,
    ValveInterface as ValveInterface,
)
from .data_preprocessing import (
    RuntimeHardwareConfiguration as RuntimeHardwareConfiguration,
    preprocess_session_directory as preprocess_session_directory,
)
from .google_sheet_tools import (
    SurgeryData as SurgeryData,
    SurgerySheet as SurgerySheet,
    WaterSheetData as WaterSheetData,
)

@dataclass()
class _LickTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to lick training sessions as a .yaml file."""

    experimenter: str
    mouse_weight_g: float
    session_type: str = ...
    dispensed_water_volume_ul: float = ...
    average_reward_delay_s: int = ...
    maximum_deviation_from_average_s: int = ...
    maximum_water_volume_ml: float = ...
    maximum_training_time_m: int = ...
    experimenter_notes: str = ...
    experimenter_given_water_volume_ml: float = ...

@dataclass()
class _RunTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to run training sessions as a .yaml file."""

    experimenter: str
    mouse_weight_g: float
    session_type: str = ...
    dispensed_water_volume_ul: float = ...
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
class _MesoscopeExperimentDescriptor(YamlConfig):
    """This class is used to save the description information specific to experiment sessions as a .yaml file."""

    experimenter: str
    mouse_weight_g: float
    session_type: str = ...
    dispensed_water_volume_ul: float = ...
    experimenter_notes: str = ...
    experimenter_given_water_volume_ml: float = ...

@dataclass()
class ExperimentState:
    """Encapsulates the information used to set and maintain the Mesoscope-VR state-machine according to the desired
    experiment state.

    Primarily, experiment runtime logic is resolved by the Unity game engine. However, the Mesoscope-VR system
    configuration may need to change throughout the experiment, for example, between the run and rest configurations.
    Overall, the Mesoscope-VR system functions like a state-machine, with multiple statically configured states that
    can be activated and maintained throughout the experiment. During runtime, the main function expects a sequence of
    ExperimentState instances that will be traversed, start-to-end, to determine the flow of the experiment runtime.
    """

    experiment_state_code: int
    vr_state_code: int
    state_duration_s: float

class _KeyboardListener:
    """Monitors the keyboard input for various runtime control signals and changes internal flags to communicate
    detected signals.

    This class is used during all training runtimes to allow the user to manually control some aspects of the
    Mesoscope-VR system and runtime. For example, it is used to abort the training runtime early and manually deliver
    rewards via the lick-tube.

    Notes:
        This monitor may pick up keyboard strokes directed at other applications during runtime. While our unique key
        combination is likely to not be used elsewhere, exercise caution when using other applications alongside the
        runtime code.

        The monitor runs in a separate process (on a separate core) and sends the data to the main process via
        shared memory arrays. This prevents the listener from competing for resources with the runtime logic and the
        visualizer class.

    Attributes:
        _data_array: A SharedMemoryArray used to store the data recorded by the remote listener process.
        _currently_pressed: Stores the keys that are currently being pressed.
        _keyboard_process: The Listener instance used to monitor keyboard strokes. The listener runs in a remote
            process.
        _started: A static flag used to prevent the __del__ method from shutting down an already terminated instance.
    """

    _data_array: Incomplete
    _currently_pressed: set[str]
    _keyboard_process: Incomplete
    _started: bool
    def __init__(self) -> None: ...
    def __del__(self) -> None:
        """Ensures all class resources are released before the instance is destroyed.

        This is a fallback method, using shutdown() should be standard.
        """
    def shutdown(self) -> None:
        """This method should be called at the end of runtime to properly release all resources and terminate the
        remote process."""
    _listener: keyboard.Listener
    def _run_keyboard_listener(self) -> None:
        """The main function that runs in the parallel process to monitor keyboard inputs."""
    def _on_press(self, key: Any) -> None:
        """Adds newly pressed keys to the storage set and determines whether the pressed key combination matches the
        shutdown combination.

        This method is used as the 'on_press' callback for the Listener instance.
        """
    def _on_release(self, key: Any) -> None:
        """Removes no longer pressed keys from the storage set.

        This method is used as the 'on_release' callback for the Listener instance.
        """
    @property
    def exit_signal(self) -> bool:
        """Returns True if the listener has detected the runtime abort keys combination (ESC + q) being pressed.

        This indicates that the user has requested the runtime to gracefully abort.
        """
    @property
    def reward_signal(self) -> bool:
        """Returns True if the listener has detected the water reward delivery keys combination (ESC + r) being
        pressed.

        This indicates that the user has requested the system to deliver 5uL water reward.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
    @property
    def speed_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running speed threshold.

        This is used during run training to manually update the running speed threshold.
        """
    @property
    def duration_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running epoch duration threshold.

        This is used during run training to manually update the running epoch duration threshold.
        """

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
        and BehaviorTraining classes. The only reason the class is defined as public is to support reconfiguring major
        source / destination paths (NAS, BioHPC, etc.).

        It is expected that the server, nas, and mesoscope data directories are mounted on the host-machine via the
        SMB or equivalent protocol. All manipulations with these destinations are carried out with the assumption that
        the OS has full access to these directories and filesystems.

        This class is specifically designed for working with raw data from a single animal participating in a single
        experimental project session. Processed data is managed by the processing library methods and classes.

        This class generates an xxHash-128 checksum stored inside the ax_checksum.txt file at the root of each
        experimental session 'raw_data' directory. The checksum verifies the data of each file and the paths to each
        file relative to the 'raw_data' root directory.

        This class also works with Sun lab Google Sheet logs to write water restriction data and read surgery data for
        processed animals. Overall, this is aimed at centralizing all raw data processing under a single binding and to
        collect all data in the same directory structure.

    Args:
        project_name: The name of the project managed by the class.
        animal_name: The name of the animal managed by the class.
        generate_mesoscope_paths: Determines whether the managed session uses ScanImage (mesoscope) PC. Training
            sessions that do not use the Mesoscope do not need to resolve paths to mesoscope data folders and storage
            directories.
        surgery_sheet_id: The ID for the Google Sheet file used to store surgery information for the animals used in the
            currently managed project. This is used to parse and write the surgery data for each managed animal into
            its 'metadata' folder.
        water_log_sheet_id: The ID for the Google Sheet file used to store water restriction information for the animals
            used in the currently managed project. This is used to automatically update water restriction logs for each
            animal during training or experiment sessions.
        credentials_path: The path to the locally stored .JSON file that stores the service account credentials used to
            read and write Google Sheet data.
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
        _surgery_sheet_id: Stores the ID for the Google Sheet file used to store surgery information for the animals
            used in the currently managed project.
        _water_log_sheet_id: Stores the ID for the Google Sheet file used to store water restriction information for
           the animals in the currently managed project.
        _credentials_path: The path to the .JSON file that stores the service account credentials used to read and
            write data from the surgery and water restriction Google Sheets.
        _local: The path to the host-machine directory for the managed project, animal, and session combination.
            This path points to the 'raw_data' subdirectory that stores all acquired and preprocessed session data.
        _server: The path to the root BioHPC server directory.
        _nas: The path to the root Synology NAS directory.
        _mesoscope: The path to the root ScanImage PC (mesoscope) data directory. This directory is shared by
            all projects, animals, and sessions.
        _persistent: The path to the host-machine directory used to retain persistent data from previous session(s) of
            the managed project and animal combination. For example, this directory is used to persist Zaber positions
            and mesoscope motion estimator files when the original session directory is moved to nas and server.
        _metadata: The path to the host-machine directory used to store the metadata information about the managed
            project and animal combination. This information typically does not change across sessions, but it is also
            exported to the nas and server, like the raw data.
            directory. This directory ios used to persist motion estimator files between experimental sessions.
        _project_name: Stores the name of the project whose data is managed by the class.
        _animal_name: Stores the name of the animal whose data is managed by the class.
        _session_name: Stores the name of the session directory whose data is managed by the class.
        _mesoscope_frames_exist: A boolean flag used to trigger additional mesoscope_frame directory preprocessing.
            This data preprocessing is not needed for runtimes that do not acquire mesoscope frames.
    """

    _surgery_sheet_id: str
    _water_log_sheet_id: str
    _credentials_path: Path
    _local: Path
    _persistent: Path
    _metadata: Path
    _project_name: str
    _animal_name: str
    _session_name: str
    _mesoscope: Path
    _server: Path
    _nas: Path
    _mesoscope_frames_exist: bool
    def __init__(
        self,
        project_name: str,
        animal_name: str,
        surgery_sheet_id: str,
        water_log_sheet_id: str,
        generate_mesoscope_paths: bool = True,
        credentials_path: Path = ...,
        local_root_directory: Path = ...,
        server_root_directory: Path = ...,
        nas_root_directory: Path = ...,
        mesoscope_data_directory: Path = ...,
    ) -> None: ...
    @property
    def raw_data_path(self) -> Path:
        """Returns the path to the 'raw_data' directory of the managed session.

        The raw_data is the root directory for aggregating all acquired and preprocessed data. This path is primarily
        used by MesoscopeExperiment class to determine where to save captured videos, data logs, and other acquired data
        formats. After runtime, SessionData pulls all mesoscope frames into this directory.
        """
    @property
    def camera_frames_path(self) -> Path:
        """Returns the path to the camera_frames directory of the managed session.

        This path is used during camera data preprocessing to store the videos and extracted frame timestamps for all
        video cameras used to record behavior.
        """
    @property
    def zaber_positions_path(self) -> Path:
        """Returns the path to the zaber_positions.yaml file of the managed session.

        This path is used to save the positions for all Zaber motors of the HeadBar and LickPort controllers at the
        end of the experimental session. This allows restoring the motors to those positions during the following
        experimental session(s).
        """
    @property
    def session_descriptor_path(self) -> Path:
        """Returns the path to the session_descriptor.yaml file of the managed session.

        This path is used to save important session information to be viewed by experimenters post-runtime and to use
        for further processing. This includes the type of the session (e.g. 'lick_training') and the total volume of
        water delivered during runtime (important for water restriction).
        """
    @property
    def hardware_configuration_path(self) -> Path:
        """Returns the path to the hardware_configuration.yaml file of the managed session.

        This file stores hardware module parameters used to read and parse .npz log files during data processing.
        """
    @property
    def previous_zaber_positions_path(self) -> Path:
        """Returns the path to the zaber_positions.yaml file of the previous session.

        The file is stored inside the 'persistent' directory of the project and animal combination. The file is saved to
        the persistent directory when the original session is moved to long-term storage. Loading the file allows
        reusing LickPort and HeadBar motor positions across sessions. The contents of this file are updated after each
        experimental or training session.
        """
    def write_water_restriction_data(self, experimenter_id: str, animal_weight: float, water_volume: float) -> None:
        """Updates the water restriction log tab for the currently managed project and animal with the provided data.

        This method should be called at the end of each training and experiment runtime to automatically update the
        water restriction log with the data provided by experimenter and gathered automatically during runtime.
        Primarily, this method is intended to streamline experiments by synchronizing the Google Sheet logs with the
        data acquired during runtime.

        Args:
            experimenter_id: The ID of the experimenter who collected the data.
            animal_weight: The weight of the animal in grams at the beginning of the runtime.
            water_volume: The total volume of water delivered during the experimental session in milliliters. This
                includes the water dispensed during runtime and the water given manually by the experimenter after
                runtime.
        """
    def write_surgery_data(self) -> None:
        """Extracts the surgery data for the currently managed project and animal combination from the surgery log
        Google Sheet and saves it to the local, server and the NAS metadata directories.

        This method is used to actualize the surgery data stored in the 'metadata' directories after each training or
        experiment runtime. Although it is not necessary to always overwrite the metadata, since this takes very little
        time, the current mode of operation is to always update this data.
        """
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
        descriptor: Stores the session descriptor instance.
        _logger: A DataLogger instance that collects behavior log data from all sources: microcontrollers, video
            cameras, and the MesoscopeExperiment instance.
        _microcontrollers: Stores the MicroControllerInterfaces instance that interfaces with all MicroController
            devices used during runtime.
        _cameras: Stores the VideoSystems instance that interfaces with video systems (cameras) used during
            runtime.
        HeadBar: Stores the HeadBar class instance that interfaces with all HeadBar manipulator motors.
        LickPort: Stores the LickPort class instance that interfaces with all LickPort manipulator motors.
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

    _state_map: dict[int, str]
    _started: bool
    descriptor: _MesoscopeExperimentDescriptor
    _vr_state: int
    _experiment_state: int
    _timestamp_timer: PrecisionTimer
    _source_id: np.uint8
    _cue_map: dict[int, float]
    _session_data: SessionData
    _logger: DataLogger
    _microcontrollers: MicroControllerInterfaces
    _unity: MQTTCommunication
    _cameras: VideoSystems
    HeadBar: HeadBar
    LickPort: LickPort
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
        harvesters_cti_path: Path = ...,
    ) -> None: ...
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
    def stop(self) -> None:
        """Stops and terminates the MesoscopeExperiment runtime.

        This method achieves two main purposes. First, releases the hardware resources used during the experiment
        runtime by various system components. Second, it pulls all collected data to the VRPC and runs the preprocessing
        pipeline on the data to prepare it for long-term storage and further processing.
        """
    def vr_rest(self) -> None:
        """Switches the VR system to the rest state.

        In the rest state, the break is engaged to prevent the mouse from moving the wheel. The encoder module is
        disabled, and instead the torque sensor is enabled. The VR screens are switched off, cutting off light emission.
        By default, the VR system starts all experimental runtimes using the REST state.
        """
    def vr_run(self) -> None:
        """Switches the VR system to the run state.

        In the run state, the break is disengaged to allow the mouse to freely move the wheel. The encoder module is
        enabled to record and share live running data with Unity, and the torque sensor is disabled. The VR screens are
        switched on to render the VR environment.
        """
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
    def _start_mesoscope(self) -> None:
        """Sends the frame acquisition start TTL pulse to the mesoscope and waits for the frame acquisition to begin.

        This method is used internally to start the mesoscope frame acquisition as part of the experiment startup
        process. It is also used to verify that the mesoscope is available and properly configured to acquire frames
        based on the input triggers.

        Raises:
            RuntimeError: If the mesoscope does not confirm frame acquisition within 2 seconds after the
                acquisition trigger is sent.
        """
    def _change_vr_state(self, new_state: int) -> None:
        """Updates and logs the new VR state.

        This method is used internally to timestamp and log VR state (stage) changes, such as transitioning between
        rest and run VR states.

        Args:
            new_state: The byte-code for the newly activated VR state.
        """
    def change_experiment_state(self, new_state: int) -> None:
        """Updates and logs the new experiment state.

        Use this method to timestamp and log experiment state (stage) changes, such as transitioning between different
        task goals.

        Args:
            new_state: The integer byte-code for the new experiment state. The code will be serialized as an uint8
                value, so only values between 0 and 255 inclusive are supported.
        """
    @property
    def trackers(self) -> tuple[SharedMemoryArray, SharedMemoryArray, SharedMemoryArray]:
        """Returns the tracker SharedMemoryArrays for (in this order) the LickInterface, ValveInterface, and
        EncoderInterface.

        These arrays should be passed to the BehaviorVisualizer class to monitor the lick, valve and running speed data
        in real time during the experiment session.
        """

class BehaviorTraining:
    """The base class for all behavior training runtimes.

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
        descriptor: Stores the session descriptor instance.
        _logger: A DataLogger instance that collects behavior log data from all sources: microcontrollers and video
            cameras.
        _microcontrollers: Stores the MicroControllerInterfaces instance that interfaces with all MicroController
            devices used during runtime.
        _cameras: Stores the VideoSystems instance that interfaces with video systems (cameras) used during
            runtime.
        HeadBar: Stores the HeadBar class instance that interfaces with all HeadBar manipulator motors.
        LickPort: Stores the LickPort class instance that interfaces with all LickPort manipulator motors.
        _screen_on: Tracks whether the VR displays are currently ON.
        _session_data: Stores the SessionData instance used to manage the acquired data.

    Raises:
        TypeError: If session_data argument has an invalid type.
    """

    _started: bool
    _lick_training: bool
    descriptor: _LickTrainingDescriptor | _RunTrainingDescriptor
    _screen_on: bool
    _session_data: SessionData
    _logger: DataLogger
    _microcontrollers: MicroControllerInterfaces
    _cameras: VideoSystems
    HeadBar: HeadBar
    LickPort: LickPort
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
        harvesters_cti_path: Path = ...,
    ) -> None: ...
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
    def stop(self) -> None:
        """Stops and terminates the BehaviorTraining runtime.

        This method achieves two main purposes. First, releases the hardware resources used during the training runtime
        by various system components. Second, it runs the preprocessing pipeline on the data to prepare it for long-term
        storage and further processing.
        """
    def lick_train_state(self) -> None:
        """Configures the VR system for running the lick training.

        In the rest state, the break is engaged to prevent the mouse from moving the wheel. The encoder module is
        disabled, and the torque sensor is enabled. The VR screens are switched off, cutting off light emission.
        The lick sensor monitoring is on to record animal licking data.
        """
    def run_train_state(self) -> None:
        """Configures the VR system for running the run training.

        In the rest state, the break is disengaged, allowing the mouse to run on the wheel. The encoder module is
        enabled, and the torque sensor is disabled. The VR screens are switched off, cutting off light emission.
        The lick sensor monitoring is on to record animal licking data.
        """
    def deliver_reward(self, reward_size: float = 5.0) -> None:
        """Uses the solenoid valve to deliver the requested volume of water in microliters.

        This method is used by the training runtimes to reward the animal with water as part of the training process.
        """
    @property
    def trackers(self) -> tuple[SharedMemoryArray, SharedMemoryArray, SharedMemoryArray]:
        """Returns the tracker SharedMemoryArrays for (in this order) the LickInterface, ValveInterface, and
        EncoderInterface.

        These arrays should be passed to the BehaviorVisualizer class to monitor the lick, valve and running speed data
        in real time during training.
        """

def lick_training_logic(
    project: str,
    animal: str,
    experimenter: str,
    animal_weight: float,
    surgery_log_id: str,
    water_restriction_log_id: str,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
    average_reward_delay: int = 12,
    maximum_deviation_from_mean: int = 6,
    maximum_water_volume: float = 1.0,
    maximum_training_time: int = 20,
) -> None:
    """Encapsulates the logic used to train animals how to operate the lick port.

    The lick training consists of delivering randomly spaced 5 uL water rewards via the solenoid valve to teach the
    animal that water comes out of the lick port. Each reward is delivered after a pseudorandom delay. Reward delay
    sequence is generated before training runtime by sampling a uniform distribution centered at 'average_reward_delay'
    with lower and upper bounds defined by 'maximum_deviation_from_mean'. The training continues either until the valve
    delivers the 'maximum_water_volume' in milliliters or until the 'maximum_training_time' in minutes is reached,
    whichever comes first.

    Notes:
        This function acts on top of the BehaviorTraining class and provides the overriding logic for the lick
        training process. During experiments, runtime logic is handled by Unity game engine, so specialized control
        functions are only required when training the animals without Unity.

        This function contains all necessary logic to set up, execute, and terminate the training. This includes data
        acquisition, preprocessing, and distribution. All lab projects should implement a CLI that calls this function
        to run lick training with parameters specific to each project.

    Args:
        project: The name of the project the animal belongs to.
        animal: The id (name) of the animal being trained.
        experimenter: The id of the experimenter conducting the training.
        animal_weight: The weight of the animal, in grams, at the beginning of the training session.
        surgery_log_id: The ID for the Google Sheet file used to store surgery information for the trained animal.
        water_restriction_log_id: The ID for the Google Sheet file used to store water restriction information for the
            trained animal.
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters. This is used to determine how long to keep the valve open to deliver the specific volume of
            water used during training and experiments to reward the animal.
        average_reward_delay: The average time, in seconds, that separates two reward deliveries. This is used to
            generate the reward delay sequence as the center of the uniform distribution from which delays are sampled.
        maximum_deviation_from_mean: The maximum deviation from the average reward delay, in seconds. This determines
            the upper and lower boundaries for the data sampled from the uniform distribution centered at the
            average_reward_delay.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered during this runtime.
        maximum_training_time: The maximum time, in minutes, to run the training.
    """

def vr_maintenance_logic(valve_calibration_data: tuple[tuple[int | float, int | float], ...]) -> None:
    """Encapsulates the logic used to interface with the hardware modules to maintain the solenoid valve and the
    running wheel.

    This runtime allows interfacing with the water valve and the wheel break outside training and experiment runtime
    contexts. Usually, at the beginning of each experiment or training day the valve is filled with water and
    'referenced' to verify it functions as expected. At the end of each day, the valve is emptied. Similarly, the wheel
    is cleaned after each session and the wheel surface wrap is replaced on a weekly or monthly interval.

    Args:
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters.

    Notes:
        This runtime will position the Zaber motors to facilitate working with the valve and the wheel.
    """

def run_train_logic(
    project: str,
    animal: str,
    experimenter: str,
    animal_weight: float,
    surgery_log_id: str,
    water_restriction_log_id: str,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
    initial_speed_threshold: float = 0.05,
    initial_duration_threshold: float = 0.05,
    speed_increase_step: float = 0.05,
    duration_increase_step: float = 0.05,
    increase_threshold: float = 0.1,
    maximum_speed_threshold: float = 10.0,
    maximum_duration_threshold: float = 10.0,
    maximum_water_volume: float = 1.0,
    maximum_training_time: int = 40,
) -> None:
    """Encapsulates the logic used to train animals how to run on the VR wheel.

    The run training consists of making the animal run on the wheel with a desired speed, in centimeters per second,
    maintained for the desired duration of seconds. Each time the animal satisfies the speed and duration threshold, it
    receives 5 uL of water reward, and the speed and durations trackers reset for the next 'epoch'. If the animal
    performs well and receives many water rewards, the speed and duration thresholds increase to make the task more
    challenging. This is used to progressively train the animal to run better and to prevent the animal from completing
    the training too early. Overall, the goal of the training is to both teach the animal to run decently fast and to
    train its stamina to sustain hour-long experiment sessions.

    Notes:
        Primarily, this function is designed to increment the initial speed and duration thresholds by the
        speed_increase_step and duration_increase_step each time the animal receives increase_threshold milliliters of
        water. The speed and duration thresholds incremented in this way cannot exceed the maximum_speed_threshold and
        maximum_duration_threshold. The training ends either when the training time exceeds the maximum_training_time,
        or when the animal receives the maximum_water_volume of water, whichever happens earlier. During runtime, it is
        possible to manually increase or decrease both thresholds via 'ESC' and arrow keys.

        This function is highly configurable and can be adapted to a wide range of training scenarios. It acts on top
        of the BehaviorTraining class and provides the overriding logic for the run training process. During
        experiments, runtime logic is handled by Unity game engine, so specialized control functions are only required
        when training the animals without Unity.

        This function contains all necessary logic to set up, execute, and terminate the training. This includes data
        acquisition, preprocessing, and distribution. All lab projects should implement a CLI that calls this function
        to run lick training with parameters specific to each project.

    Args:
        project: The name of the project the animal belongs to.
        animal: The id (name) of the animal being trained.
        experimenter: The id of the experimenter conducting the training.
        animal_weight: The weight of the animal, in grams, at the beginning of the training session.
        surgery_log_id: The ID for the Google Sheet file used to store surgery information for the trained animal.
        water_restriction_log_id: The ID for the Google Sheet file used to store water restriction information for the
            trained animal.
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters. This is used to determine how long to keep the valve open to deliver the specific volume of
            water used during training and experiments to reward the animal.
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
        maximum_duration_threshold: The maximum duration threshold, in seconds, that the animal must maintain
            above-threshold running speed to receive water rewards. Once this threshold is reached, it will not be
            increased further regardless of how much water is delivered to the animal.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered during this runtime.
        maximum_training_time: The maximum time, in minutes, to run the training.
    """

def run_experiment_logic(
    project: str,
    animal: str,
    experimenter: str,
    animal_weight: float,
    surgery_log_id: str,
    water_restriction_log_id: str,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
    cue_length_map: dict[int, float],
    experiment_state_sequence: tuple[ExperimentState, ...],
) -> None:
    """Encapsulates the logic used to run experiments via the Mesoscope-VR system.

    This function can be used to execute any valid experiment using the Mesoscope-VR system. Each experiment should be
    broken into one or more experiment states (phases), such as 'baseline', 'task' and 'cooldown'. Furthermore, each
    experiment state can use one or more VR system states. Currently, the VR system has two states: rest (1) and run
    (2). The states are used to broadly configure the Mesoscope-VR system, and they determine which systems are active
    and what data is collected (see library ReadMe for more details on VR states).

    Primarily, this function is concerned with iterating over the states stored inside the experiment_state_sequence
    tuple. Each experiment and VR state combination is maintained for the requested duration of seconds. Once all states
    have been executed, the experiment runtime ends. Under this design pattern, each experiment is conceptualized as
    a sequence of states.

    Notes:
        During experiment runtimes, the task logic and the Virtual Reality world are resolved via the Unity game engine.
        This function itself does not resolve the task logic, it is only concerned with iterating over experiment
        states, controlling the VR system, and monitoring user command issued via keyboard.

        Similar to all other runtime functions, this function contains all necessary bindings to set up, execute, and
        terminate an experiment runtime. Custom projects should implement a cli that calls this function with
        project-specific parameters.

    Args:
        project: The name of the project the animal belongs to.
        animal: The id (name) of the animal running the experiment.
        experimenter: The id of the experimenter conducting the experiment.
        animal_weight: The weight of the animal, in grams, at the beginning of the training session.
        surgery_log_id: The ID for the Google Sheet file used to store surgery information for the trained animal.
        water_restriction_log_id: The ID for the Google Sheet file used to store water restriction information for the
            trained animal.
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters. This is used to determine how long to keep the valve open to deliver the specific volume of
            water used during training and experiments to reward the animal.
        cue_length_map: A dictionary that maps each integer-code associated with a wall cue used in the Virtual Reality
            experiment environment to its length in real-world centimeters. Ity is used to map each VR cue to the
            distance the animal needs to travel to fully traverse the wall cue region from start to end.
        experiment_state_sequence: A tuple of ExperimentState instances, each representing a phase of the experiment.
            The function executes each experiment state provided via this tuple in the order they appear. Once the last
            state has been executed, the experiment runtime ends.
    """
