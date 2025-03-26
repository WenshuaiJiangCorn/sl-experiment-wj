"""This module provides classes that abstract working with Sun lab's Mesoscope-VR system to acquire training or
experiment data."""

import os
import copy
import json
from typing import Any
from pathlib import Path
import tempfile
from dataclasses import dataclass
from multiprocessing import Process

from tqdm import tqdm
import numpy as np
from pynput import keyboard
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, LogPackage, SharedMemoryArray
from ataraxis_time.time_helpers import get_timestamp
from ataraxis_communication_interface import MQTTCommunication, MicroControllerInterface

from .visualizers import BehaviorVisualizer
from .binding_classes import HeadBar, LickPort, VideoSystems, ZaberPositions, MicroControllerInterfaces
from .module_interfaces import BreakInterface, ValveInterface
from .data_preprocessing import (
    SessionData,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    RuntimeHardwareConfiguration,
    MesoscopeExperimentDescriptor,
)


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
    """The integer code of the experiment state. Experiment states do not have a predefined meaning, Instead, each 
    project is expected to define and follow its own experiment state code mapping. Typically, the experiment state 
    code is used to denote major experiment stages, such as 'baseline', 'task1', 'cooldown', etc. Note, the same 
    experiment state code can be used by multiple sequential ExperimentState instances to change the VR system states 
    while maintaining the same experiment state."""
    vr_state_code: int
    """One of the supported VR system state-codes. Currently, the Mesoscope-VR system supports 2 state codes. State 
    code 1 denotes 'REST' state and code 2 denotes 'RUN' state. In the rest state, the running wheel is locked and the
    screens are off. In the run state, the screens are on and the wheel is unlocked. Note, multiple consecutive 
    ExperimentState instances with different experiment state codes can reuse the same VR state code."""
    state_duration_s: float
    """The time, in seconds, to maintain the current combination of the experiment and VR states."""


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

    def __init__(self) -> None:
        self._data_array = SharedMemoryArray.create_array(
            name="keyboard_listener", prototype=np.zeros(shape=5, dtype=np.int32), exist_ok=True
        )
        self._currently_pressed: set[str] = set()

        # Starts the listener process
        self._keyboard_process = Process(target=self._run_keyboard_listener, daemon=True)
        self._keyboard_process.start()
        self._started = True

    def __del__(self) -> None:
        """Ensures all class resources are released before the instance is destroyed.

        This is a fallback method, using shutdown() should be standard.
        """
        if self._started:
            self.shutdown()

    def shutdown(self) -> None:
        """This method should be called at the end of runtime to properly release all resources and terminate the
        remote process."""
        if self._keyboard_process.is_alive():
            self._data_array.write_data(index=0, data=np.int32(1))  # Termination signal
            self._keyboard_process.terminate()
            self._keyboard_process.join(timeout=1.0)
        self._data_array.disconnect()
        self._data_array.destroy()
        self._started = False

    def _run_keyboard_listener(self) -> None:
        """The main function that runs in the parallel process to monitor keyboard inputs."""

        # Sets up listeners for both press and release
        self._listener: keyboard.Listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

        # Connects to the shared memory array from the remote process
        self._data_array.connect()

        # Initializes the timer used to delay the process. This reduces the CPU load
        delay_timer = PrecisionTimer("ms")

        # Keeps the process alive until it receives the shutdown command via the SharedMemoryArray instance
        while self._data_array.read_data(index=0, convert_output=True) == 0:
            delay_timer.delay_noblock(delay=10, allow_sleep=True)  # 10 ms delay

        # If the loop above is escaped, this indicates that the listener process has been terminated. Disconnects from
        # the shared memory array and exits
        self._data_array.disconnect()

    def _on_press(self, key: Any) -> None:
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
                self._data_array.write_data(index=1, data=np.int32(1))

            # Reward combination: ESC + r
            if "'r'" in self._currently_pressed:
                self._data_array.write_data(index=2, data=np.int32(1))

            # Speed control: ESC + Up/Down arrows
            if "Key.up" in self._currently_pressed:
                previous_value = self._data_array.read_data(index=3, convert_output=False)
                previous_value += 1
                self._data_array.write_data(index=3, data=previous_value)

            if "Key.down" in self._currently_pressed:
                previous_value = self._data_array.read_data(index=3, convert_output=False)
                previous_value -= 1
                self._data_array.write_data(index=3, data=previous_value)

            # Duration control: ESC + Left/Right arrows
            if "Key.right" in self._currently_pressed:
                previous_value = self._data_array.read_data(index=4, convert_output=False)
                previous_value -= 1
                self._data_array.write_data(index=4, data=previous_value)

            if "Key.left" in self._currently_pressed:
                previous_value = self._data_array.read_data(index=4, convert_output=False)
                previous_value += 1
                self._data_array.write_data(index=4, data=previous_value)

    def _on_release(self, key: Any) -> None:
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
        return bool(self._data_array.read_data(index=1, convert_output=True))

    @property
    def reward_signal(self) -> bool:
        """Returns True if the listener has detected the water reward delivery keys combination (ESC + r) being
        pressed.

        This indicates that the user has requested the system to deliver 5uL water reward.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
        reward_flag = bool(self._data_array.read_data(index=2, convert_output=True))
        self._data_array.write_data(index=2, data=np.int32(0))
        return reward_flag

    @property
    def speed_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running speed threshold.

        This is used during run training to manually update the running speed threshold.
        """
        return int(self._data_array.read_data(index=3, convert_output=True))

    @property
    def duration_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running epoch duration threshold.

        This is used during run training to manually update the running epoch duration threshold.
        """
        return int(self._data_array.read_data(index=4, convert_output=True))


class _MesoscopeExperiment:
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

    # Maps integer VR state codes to human-readable string-names.
    _state_map: dict[int, str] = {0: "Idle", 1: "Rest", 2: "Run"}

    def __init__(
        self,
        session_data: SessionData,
        descriptor: MesoscopeExperimentDescriptor,
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
        self.descriptor: MesoscopeExperimentDescriptor = descriptor

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

        # Saves the SessionData instance to an attribute so that it can be used from class methods. Since SessionData
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
        self._microcontrollers: MicroControllerInterfaces = MicroControllerInterfaces(
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
        self._cameras: VideoSystems = VideoSystems(
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
            "running before proceeding further. If ZaberLauncher is not running, you WILL NOT be able to manually "
            "control the HeadBar and LickPort motor positions until you reset the runtime."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Initializes the binding classes for the HeadBar and LickPort manipulator motors.
        self.HeadBar: HeadBar = HeadBar(
            headbar_port=headbar_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )
        self.LickPort: LickPort = LickPort(
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
        cpu_count = os.cpu_count()
        if cpu_count is None or not cpu_count >= 11:
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
            source_id=self._source_id, time_stamp=np.uint64(0), serialized_data=onset
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
        input("Enter anything to continue: ")

        # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not intersect
        # with each other, so it is safe to move both assemblies at the same time.
        self.HeadBar.prepare_motors(wait_until_idle=False)
        self.LickPort.prepare_motors(wait_until_idle=True)
        self.HeadBar.wait_until_idle()

        # Sets the motors into the mounting position. The HeadBar is either restored to the previous session position or
        # is set to the default mounting position stored in non-volatile memory. The LickPort is moved to a position
        # optimized for putting the animal on the VR rig.
        self.HeadBar.restore_position(wait_until_idle=False)
        self.LickPort.mount_position(wait_until_idle=True)
        self.HeadBar.wait_until_idle()

        message = "HeadBar: Positioned."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Gives user time to mount the animal and requires confirmation before proceeding further.
        message = (
            "Preparing to move the LickPort into position. Mount the animal onto the VR rig and install the mesoscope "
            "objetive. If necessary, adjust the HeadBar position to make sure the animal can comfortably run the task."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Restores the lickPort to the previous session's position or to the default parking position. This positions
        # the LickPort in a way that is easily accessible by the animal.
        self.LickPort.restore_position()
        message = "LickPort: Positioned."
        console.echo(message=message, level=LogLevel.SUCCESS)

        message = (
            "If necessary, adjust LickPort position to be easily reachable by the animal and position the mesoscope "
            "objective above the imaging field. Take extra care when moving the LickPort towards the animal! Run any "
            "mesoscope preparation procedures, such as motion correction, before proceeding further. This is the last "
            "manual checkpoint, entering 'y' after this message will begin the experiment."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Generates a snapshot of all zaber positions. This serves as an early checkpoint in case the runtime has to be
        # aborted in a non-graceful way (without running the stop() sequence). This way, next runtime will restart with
        # the calibrated zaber positions.
        head_bar_positions = self.HeadBar.get_positions()
        lickport_positions = self.LickPort.get_positions()
        zaber_positions = ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        self._session_data.previous_zaber_positions_path.unlink(missing_ok=True)  # Removes the previous persisted file
        # Saves the newly generated file both to the persistent folder adn to the session folder
        zaber_positions.to_yaml(file_path=self._session_data.previous_zaber_positions_path)
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)
        message = "HeadBar and LickPort positions: Saved."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Generates a snapshot of the runtime hardware configuration. In turn, this data is used to parse the .npz log
        # files during processing.
        hardware_configuration = RuntimeHardwareConfiguration(
            cue_map=self._cue_map,
            cm_per_pulse=float(self._microcontrollers.wheel_encoder.cm_per_pulse),
            maximum_break_strength=float(self._microcontrollers.wheel_break.maximum_break_strength),
            minimum_break_strength=float(self._microcontrollers.wheel_break.minimum_break_strength),
            lick_threshold=int(self._microcontrollers.lick.lick_threshold),
            scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
            nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
            torque_per_adc_unit=float(self._microcontrollers.torque.torque_per_adc_unit),
            initially_on=self._microcontrollers.screens.initially_on,
            has_ttl=True,
        )
        hardware_configuration.to_yaml(self._session_data.hardware_configuration_path)
        message = "Hardware configuration snapshot: Generated."
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
        self.descriptor.dispensed_water_volume_ml = round(delivered_water / 1000, ndigits=3)  # Converts from uL to ml
        self.descriptor.to_yaml(file_path=self._session_data.session_descriptor_path)

        # Generates the snapshot of the current HeadBar and LickPort positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are reset back to parking position.
        head_bar_positions = self.HeadBar.get_positions()
        lickport_positions = self.LickPort.get_positions()
        zaber_positions = ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        self._session_data.previous_zaber_positions_path.unlink(missing_ok=True)  # Removes the previous persisted file
        # Saves the newly generated file both to the persistent folder adn to the session folder
        zaber_positions.to_yaml(file_path=self._session_data.previous_zaber_positions_path)
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)

        # Moves the LickPort to the mounting position to assist removing the animal from the rig.
        self.LickPort.mount_position()

        # Notifies the user about the volume of water dispensed during runtime, so that they can ensure the mouse
        # get any leftover daily water limit.
        message = (
            f"During runtime, the system dispensed ~{delivered_water} uL of water to the animal. "
            f"If the animal is on water restriction, make sure it receives any additional water, if the dispensed "
            f"volume does not cover the daily water limit for that animal."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Prompts the user to add their notes to the appropriate section of the descriptor file. This has to be done
        # before processing so that the notes are properly transferred to the NAS and server. Also, this makes it more
        # obvious to the user when it is safe to start preparing for the next session and leave the current one
        # processing the data.
        message = (
            f"Data acquisition: Complete. Open the session descriptor file stored in session's raw_data folder and "
            f"update the notes session with the notes taken during runtime. Then, uninstall the mesoscope objective "
            f"and remove the animal from the VR rig. Failure to do so may DAMAGE the mesoscope objective and HARM the "
            f"animal. This is the last manual checkpoint, once you hit 'y,' it is safe to start preparing for the next "
            f"session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Verifies and blocks in-place until the user has updated the session descriptor file with experimenter notes.
        descriptor: MesoscopeExperimentDescriptor = self.descriptor.from_yaml(  # type: ignore
            file_path=self._session_data.session_descriptor_path
        )
        while "Replace this with your notes." in descriptor.experimenter_notes:
            message = (
                "Failed to verify that the session_descriptor.yaml file stored inside the session raw_data directory "
                "has been updated to include experimenter notes. Manually edit the session_descriptor.yaml file and "
                "replaced the default text under the 'experimenter_notes' field with the notes taken during the "
                "experiment. Make sure to save the changes to the file by using 'CTRL+S' combination."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to continue: ")

            # Reloads the descriptor from disk each time to ensure experimenter notes have been modified.
            descriptor = self.descriptor.from_yaml(file_path=self._session_data.session_descriptor_path)  # type: ignore

        # Parks both controllers and then disconnects from their Connection classes. Note, the parking is performed
        # in-parallel
        self.HeadBar.park_position(wait_until_idle=False)
        self.LickPort.park_position(wait_until_idle=True)
        self.HeadBar.wait_until_idle()
        self.HeadBar.disconnect()
        self.LickPort.disconnect()

        message = "HeadBar and LickPort motors: Reset."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Preprocesses the session data
        self._session_data.preprocess_session_data()

        message = "MesoscopeExperiment runtime: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def vr_rest(self) -> None:
        """Switches the VR system to the rest state.

        In the rest state, the break is engaged to prevent the mouse from moving the wheel. The encoder module is
        disabled, and instead the torque sensor is enabled. The VR screens are switched off, cutting off light emission.
        By default, the VR system starts all experimental runtimes using the REST state.
        """

        # Prevents changing the VR state if the VR system is already in REST state.
        if self._vr_state == 1:
            return

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

        # Prevents changing the VR state if the VR system is already in RUN state.
        if self._vr_state == 2:
            return

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
                topic, payload = self._unity.get_data()  # type: ignore
                if topic == "CueSequence/":
                    # Extracts the sequence of cues that will be used during task runtime.
                    sequence: NDArray[np.uint8] = np.array(
                        json.loads(payload.decode("utf-8"))["cue_sequence"], dtype=np.uint8
                    )
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

        # Prevents changing the experiment state if the experiment is already in the desired state.
        if self._experiment_state == new_state:
            return

        self._experiment_state = new_state  # Updates the tracked experiment state value

        # Logs the VR state update. Uses header-code 2 to indicate that the logged value is the experiment state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([2, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

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


class _BehaviorTraining:
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

    def __init__(
        self,
        session_data: SessionData,
        descriptor: LickTrainingDescriptor | RunTrainingDescriptor,
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
        self.descriptor: LickTrainingDescriptor | RunTrainingDescriptor = descriptor

        # Input verification:
        if not isinstance(session_data, SessionData):
            message = (
                f"Unable to initialize the BehaviorTraining class. Expected a SessionData instance for "
                f"'session_data' argument, but instead encountered {session_data} of type "
                f"{type(session_data).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Defines other flags used during runtime:
        self._screen_on: bool = screens_on  # Usually this would be false, but this is not guaranteed

        # Saves the SessionData instance to an attribute so that it can be used from class methods. Since SessionData
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
        self._microcontrollers: MicroControllerInterfaces = MicroControllerInterfaces(
            data_logger=self._logger,
            actor_port=actor_port,
            sensor_port=sensor_port,
            encoder_port=encoder_port,
            valve_calibration_data=valve_calibration_data,
            debug=False,
        )

        # Initializes the binding class for all VideoSystems.
        self._cameras: VideoSystems = VideoSystems(
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
            "running before proceeding further. If ZaberLauncher is not running, you WILL NOT be able to manually "
            "control the HeadBar and LickPort motor positions until you reset the runtime."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Initializes the binding classes for the HeadBar and LickPort manipulator motors.
        self.HeadBar: HeadBar = HeadBar(
            headbar_port=headbar_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )
        self.LickPort: LickPort = LickPort(
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
        cpu_count = os.cpu_count()
        if cpu_count is None or not cpu_count >= 11:
            message = (
                f"Unable to start the BehaviorTraining runtime. The host PC must have at least 11 logical CPU "
                f"cores available for this class to work as expected, but only {os.cpu_count()} cores are "
                f"available."
            )
            console.error(message=message, error=RuntimeError)

        message = "Initializing BehaviorTraining assets..."
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
        input("Enter anything to continue: ")

        # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not intersect
        # with each other, so it is safe to move both assemblies at the same time.
        self.HeadBar.prepare_motors(wait_until_idle=False)
        self.LickPort.prepare_motors(wait_until_idle=True)
        self.HeadBar.wait_until_idle()

        # Sets the motors into the mounting position. The HeadBar is either restored to the previous session position or
        # is set to the default mounting position stored in non-volatile memory. The LickPort is moved to a position
        # optimized for putting the animal on the VR rig.
        self.HeadBar.restore_position(wait_until_idle=False)
        self.LickPort.mount_position(wait_until_idle=True)
        self.HeadBar.wait_until_idle()

        message = "HeadBar: Positioned."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Gives user time to mount the animal and requires confirmation before proceeding further.
        message = (
            "Preparing to move the LickPort into position. Mount the animal onto the VR rig. If necessary, adjust the "
            "HeadBar position to make sure the animal can comfortably run the task."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Restores the lickPort to the previous session's position or to the default parking position. This positions
        # the LickPort in a way that is easily accessible by the animal.
        self.LickPort.restore_position()

        message = "LickPort: Positioned."
        console.echo(message=message, level=LogLevel.SUCCESS)

        message = (
            "If necessary, adjust LickPort position to be easily reachable by the animal. Take extra care when moving "
            "the LickPort towards the animal! This is the last manual checkpoint, entering 'y' after this message will "
            "begin the training."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Generates a snapshot of all zaber positions. This serves as an early checkpoint in case the runtime has to be
        # aborted in a non-graceful way (without running the stop() sequence). This way, next runtime will restart with
        # the calibrated zaber positions.
        head_bar_positions = self.HeadBar.get_positions()
        lickport_positions = self.LickPort.get_positions()
        zaber_positions = ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        self._session_data.previous_zaber_positions_path.unlink(missing_ok=True)  # Removes the previous persisted file
        # Saves the newly generated file both to the persistent folder adn to the session folder
        zaber_positions.to_yaml(file_path=self._session_data.previous_zaber_positions_path)
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)
        message = "HeadBar and LickPort positions: Saved."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Generates a snapshot of the runtime hardware configuration. In turn, this data is used to parse the .npz log
        # files during processing. Note, lick training does not use the encoder and run training does not use the torque
        # sensor.
        if self._lick_training:
            hardware_configuration = RuntimeHardwareConfiguration(
                torque_per_adc_unit=float(self._microcontrollers.torque.torque_per_adc_unit),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
                has_ttl=False,
            )
        else:
            hardware_configuration = RuntimeHardwareConfiguration(
                cm_per_pulse=float(self._microcontrollers.wheel_encoder.cm_per_pulse),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
                has_ttl=False,
            )
        hardware_configuration.to_yaml(self._session_data.hardware_configuration_path)
        message = "Hardware configuration snapshot: Generated."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Enables body cameras. Starts frame saving for all cameras
        self._cameras.start_body_cameras()
        self._cameras.save_face_camera_frames()
        self._cameras.save_body_camera_frames()

        # Initializes communication with the microcontrollers
        self._microcontrollers.start()

        # The setup procedure is complete.
        self._started = True

        message = "BehaviorTraining assets: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops and terminates the BehaviorTraining runtime.

        This method achieves two main purposes. First, releases the hardware resources used during the training runtime
        by various system components. Second, it runs the preprocessing pipeline on the data to prepare it for long-term
        storage and further processing.
        """

        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        message = "Terminating BehaviorTraining runtime..."
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
        # Overwrites the delivered water volume with the volume recorded over the runtime.
        self.descriptor.dispensed_water_volume_ml = round(delivered_water / 1000, ndigits=3)  # Converts from uL to ml
        self.descriptor.to_yaml(file_path=self._session_data.session_descriptor_path)

        # Generates the snapshot of the current HeadBar and LickPort positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are reset back to parking position.
        head_bar_positions = self.HeadBar.get_positions()
        lickport_positions = self.LickPort.get_positions()
        zaber_positions = ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        self._session_data.previous_zaber_positions_path.unlink(missing_ok=True)  # Removes the previous persisted file
        # Saves the newly generated file both to the persistent folder adn to the session folder
        zaber_positions.to_yaml(file_path=self._session_data.previous_zaber_positions_path)
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)

        # Moves the LickPort to the mounting position to assist removing the animal from the rig.
        self.LickPort.mount_position()

        # Notifies the user about the volume of water dispensed during runtime, so that they can ensure the mouse
        # get any leftover daily water limit.
        message = (
            f"During runtime, the system dispensed ~{delivered_water} uL of water to the animal. "
            f"If the animal is on water restriction, make sure it receives any additional water, if the dispensed "
            f"volume does not cover the daily water limit for that animal."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Prompts the user to add their notes to the appropriate section of the descriptor file. This has to be done
        # before processing so that the notes are properly transferred to the NAS and server. Also, this makes it more
        # obvious to the user when it is safe to start preparing for the next session and leave the current one
        # processing the data.
        message = (
            f"Data acquisition: Complete. Open the session descriptor file stored in session's raw_data folder and "
            f"update the notes session with the notes taken during runtime. Then, uninstall the mesoscope objective "
            f"and remove the animal from the VR rig. Failure to do so may DAMAGE the mesoscope objective and HARM the "
            f"animal. This is the last manual checkpoint, once you hit 'y,' it is safe to start preparing for the next "
            f"session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Verifies and blocks in-place until the user has updated the session descriptor file with experimenter notes.
        descriptor: LickTrainingDescriptor | RunTrainingDescriptor = self.descriptor.from_yaml(  # type: ignore
            file_path=self._session_data.session_descriptor_path
        )
        while "Replace this with your notes." in descriptor.experimenter_notes:
            message = (
                "Failed to verify that the session_descriptor.yaml file stored inside the session raw_data directory "
                "has been updated to include experimenter notes. Manually edit the session_descriptor.yaml file and "
                "replaced the default text under the 'experimenter_notes' field with the notes taken during the "
                "experiment. Make sure to save the changes to the file by using 'CTRL+S' combination."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to continue: ")

            # Reloads the descriptor from disk each time to ensure experimenter notes have been modified.
            descriptor = self.descriptor.from_yaml(file_path=self._session_data.session_descriptor_path)  # type: ignore

        # Parks both controllers and then disconnects from their Connection classes. Note, the parking is performed
        # in-parallel
        self.HeadBar.park_position(wait_until_idle=False)
        self.LickPort.park_position(wait_until_idle=True)
        self.HeadBar.wait_until_idle()
        self.HeadBar.disconnect()
        self.LickPort.disconnect()

        message = "HeadBar and LickPort motors: Reset."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Preprocesses the session data
        self._session_data.preprocess_session_data()

        message = "Data preprocessing: Complete. BehaviorTraining runtime: Terminated."
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


def lick_training_logic(
    experimenter: str,
    animal_weight: float,
    session_data: SessionData,
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
        experimenter: The id of the experimenter conducting the training.
        animal_weight: The weight of the animal, in grams, at the beginning of the training session.
        session_data: The SessionData instance that manages the data acquired during training.
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
    # Enables the console
    if not console.enabled:
        console.enable()

    message = f"Initializing lick training runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Dumps the SessionData information to the raw_data folder as a session_data.yaml file. This is a prerequisite for
    # being able to run preprocessing on the data if runtime fails.
    session_data.to_path()

    # Pre-generates the SessionDescriptor class and populates it with training data.
    descriptor = LickTrainingDescriptor(
        average_reward_delay_s=average_reward_delay,
        maximum_deviation_from_average_s=maximum_deviation_from_mean,
        maximum_training_time_m=maximum_training_time,
        maximum_water_volume_ml=maximum_water_volume,
        experimenter=experimenter,
        mouse_weight_g=animal_weight,
    )

    # Initializes the main runtime interface class. Note, most class parameters are statically configured to work for
    # the current VRPC setup and may need to be adjusted as that setup evolves over time.
    runtime = _BehaviorTraining(
        session_data=session_data,
        descriptor=descriptor,
        actor_port="/dev/ttyACM0",
        sensor_port="/dev/ttyACM1",
        encoder_port="/dev/ttyACM2",
        headbar_port="/dev/ttyUSB0",
        lickport_port="/dev/ttyUSB1",
        face_camera_index=0,
        left_camera_index=0,
        right_camera_index=2,
        harvesters_cti_path=Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"),
        screens_on=False,
        valve_calibration_data=valve_calibration_data,
    )

    # Initializes the timer used to enforce reward delays
    delay_timer = PrecisionTimer("us")

    # Uses runtime tracker extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers

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

    # Since we preset the descriptor class before determining the time necessary to deliver the maximum allowed water
    # volume, the maximum training time may actually not be accurate. This would be the case if the training runtime is
    # limited by the maximum allowed water delivery volume and not time. In this case, updates the training time to
    # reflect the factual training time. This would be the case if the reward_delays array size is the same as the
    # cumulative time array size, indicating no slicing was performed due to session time constraints.
    if len(reward_delays) == len(cumulative_time):
        # Actual session time is the accumulated delay converted from seconds to minutes at the last index.
        runtime.descriptor.maximum_training_time_m = np.ceil(cumulative_time[-1] / 60)

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Visualizer initialization HAS to happen after the runtime start to avoid interfering with cameras.
    visualizer = BehaviorVisualizer(
        lick_tracker=lick_tracker, valve_tracker=valve_tracker, distance_tracker=speed_tracker
    )

    # Configures all system components to support lick training
    runtime.lick_train_state()

    # Initializes the listener instance used to detect training abort signals and manual reward trigger signals sent
    # via the keyboard.
    listener = _KeyboardListener()

    message = (
        f"Initiating lick training procedure. Press 'ESC' + 'q' to immediately abort the training at any "
        f"time. Press 'ESC' + 'r' to deliver 5 uL of water to the animal."
    )
    console.echo(message=message, level=LogLevel.INFO)

    # This tracker is used to terminate the training if manual abort command is sent via the keyboard
    terminate = False

    # Loops over all delays and delivers reward via the lick tube as soon as the delay expires.
    try:
        delay_timer.reset()
        for delay in tqdm(
            reward_delays,
            desc="Running lick training",
            unit="reward",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} rewards [{elapsed}]",
        ):
            # This loop is executed while the code is waiting for the delay to pass. Anything that needs to be done
            # during the delay has to go here
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
                    "Lick training abort signal detected. Aborting the lick training with a graceful shutdown "
                    "procedure."
                )
                console.echo(message=message, level=LogLevel.ERROR)
                break  # Breaks the for loop

            # Once the delay is up, triggers the solenoid valve to deliver water to the animal and starts timing the
            # next reward delay
            runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water
            delay_timer.reset()

        # Ensures the animal has time to consume the last reward before the LickPort is moved out of its range.
        delay_timer.delay_noblock(lower_bound * 1000000)  # Converts to microseconds before delaying

    except Exception as e:
        message = (
            f"Training runtime has encountered an error and had to be terminated early. Attempting to gracefully "
            f"shutdown all assets and preserve as much of the data as possible. The encountered error message: "
            f"{str(e)}"
        )
        console.echo(message=message, level=LogLevel.ERROR)

    # Shutdown sequence:
    message = f"Training runtime: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Terminates the listener
    listener.shutdown()

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()


def vr_maintenance_logic(
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
) -> None:
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
    # Enables the console
    if not console.enabled:
        console.enable()

    message = f"Initializing valve interface runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Initializes a timer used to optimize console printouts for using the valve in debug mode (which also posts
    # things to console).
    delay_timer = PrecisionTimer("s")

    message = f"Initializing interface assets..."
    console.echo(message=message, level=LogLevel.INFO)

    # Runs all calibration procedures inside a temporary directory which is deleted at the end of runtime.
    with tempfile.TemporaryDirectory(prefix="sl_maintenance_") as output_dir:
        output_path: Path = Path(output_dir)

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

        # Initializes HeadBar and LickPort binding classes
        headbar = HeadBar("/dev/ttyUSB0", output_path.joinpath("zaber_positions.yaml"))
        lickport = LickPort("/dev/ttyUSB1", output_path.joinpath("zaber_positions.yaml"))

        message = f"Zaber controllers: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Initializes the Actor MicroController with the valve and break modules. Ignores all other modules at this
        # time.
        valve: ValveInterface = ValveInterface(valve_calibration_data=valve_calibration_data, debug=True)
        wheel: BreakInterface = BreakInterface(debug=True)
        controller: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port="/dev/ttyACM0",
            data_logger=logger,
            module_interfaces=(valve, wheel),
        )
        controller.start()
        controller.unlock_controller()

        # Delays for 1 second for the valve to initialize and send the state message. This avoids the visual clash
        # with the zaber positioning dialog
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
        input("Enter anything to continue: ")

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
            "Supported valve commands: open, close, close_10, reference, reward, calibrate_15, calibrate_30, "
            "calibrate_45, calibrate_60. Supported break (wheel) commands: lock, unlock. Use 'q' command to terminate "
            "the runtime."
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

            if command == "close_10":
                message = f"Closing the valve after a 10-second delay..."
                console.echo(message=message, level=LogLevel.INFO)
                start = delay_timer.elapsed
                previous_time = delay_timer.elapsed
                while delay_timer.elapsed - start < 10:
                    if previous_time != delay_timer.elapsed:
                        previous_time = delay_timer.elapsed
                        console.echo(
                            message=f"Remaining time: {10 - (delay_timer.elapsed - start)} seconds.",
                            level=LogLevel.INFO,
                        )
                valve.toggle(state=False)  # Closes the valve after a 10-second delay

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

            if command == "lock":
                message = f"Locking wheel break."
                console.echo(message=message, level=LogLevel.INFO)
                wheel.toggle(state=True)

            if command == "unlock":
                message = f"Unlocking wheel break."
                console.echo(message=message, level=LogLevel.INFO)
                wheel.toggle(state=False)

            if command == "q":
                message = f"Terminating valve calibration runtime."
                console.echo(message=message, level=LogLevel.INFO)
                break

        # Instructs the user to remove the objective and the animal before resetting all zaber motors.
        message = (
            "Preparing to reset the HeadBar and LickPort motors. Remove all objects used during calibration, such as "
            "water collection flasks, from the Mesoscope-VR cage."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

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

        message = f"Runtime: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # The logs will be cleaned up by deleting the temporary directory when this runtime exits.


def run_train_logic(
    experimenter: str,
    animal_weight: float,
    session_data: SessionData,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
    initial_speed_threshold: float = 0.05,
    initial_duration_threshold: float = 0.05,
    speed_increase_step: float = 0.05,
    duration_increase_step: float = 0.0,
    increase_threshold: float = 0.2,
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
        experimenter: The id of the experimenter conducting the training.
        animal_weight: The weight of the animal, in grams, at the beginning of the training session.
        session_data: The SessionData instance that manages the data acquired during training.
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
    # Enables the console
    if not console.enabled:
        console.enable()

    # Dumps the SessionData information to the raw_data folder as a session_data.yaml file. This is a prerequisite for
    # being able to run preprocessing on the data if runtime fails.
    session_data.to_path()

    message = f"Initializing run training runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Pre-generates the SessionDescriptor class and populates it with training data
    descriptor = RunTrainingDescriptor(
        initial_running_speed_cm_s=initial_speed_threshold,
        initial_speed_duration_s=initial_duration_threshold,
        increase_threshold_ml=increase_threshold,
        increase_running_speed_cm_s=speed_increase_step,
        increase_speed_duration_s=duration_increase_step,
        maximum_running_speed_cm_s=maximum_speed_threshold,
        maximum_speed_duration_s=maximum_duration_threshold,
        maximum_training_time_m=maximum_training_time,
        maximum_water_volume_ml=maximum_water_volume,
        experimenter=experimenter,
        mouse_weight_g=animal_weight,
    )

    # Initializes the main runtime interface class. Note, most class parameters are statically configured to work for
    # the current VRPC setup and may need to be adjusted as that setup evolves over time.
    runtime = _BehaviorTraining(
        session_data=session_data,
        descriptor=descriptor,
        actor_port="/dev/ttyACM0",
        sensor_port="/dev/ttyACM1",
        encoder_port="/dev/ttyACM2",
        headbar_port="/dev/ttyUSB0",
        lickport_port="/dev/ttyUSB1",
        face_camera_index=0,
        left_camera_index=0,
        right_camera_index=2,
        harvesters_cti_path=Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"),
        screens_on=False,
        valve_calibration_data=valve_calibration_data,
    )

    # Initializes the timers used during runtime
    runtime_timer = PrecisionTimer("s")
    speed_timer = PrecisionTimer("ms")

    # Converts all arguments used to determine the speed and duration threshold over time into numpy variables to
    # optimize main loop runtime speed:
    initial_speed = np.float64(initial_speed_threshold)  # In centimeters per second
    maximum_speed = np.float64(maximum_speed_threshold)  # In centimeters per second
    speed_step = np.float64(speed_increase_step)  # In centimeters per second

    initial_duration = np.float64(initial_duration_threshold * 1000)  # In milliseconds
    maximum_duration = np.float64(maximum_duration_threshold * 1000)  # In milliseconds
    duration_step = np.float64(duration_increase_step * 1000)  # In milliseconds

    # The way 'increase_threshold' is used requires it to be greater than 0. So if a threshold of 0 is passed, the
    # system sets it to a very small number instead. which functions similar to it being 0, but does not produce an
    # error.
    if increase_threshold < 0:
        increase_threshold = 0.000000000001

    water_threshold = np.float64(increase_threshold * 1000)  # In microliters
    maximum_volume = np.float64(maximum_water_volume * 1000)  # In microliters

    # Converts the training time from minutes to seconds to make it compatible with the timer precision.
    training_time = maximum_training_time * 60

    # Uses runtime trackers extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Visualizer initialization HAS to happen after the runtime start to avoid interfering with cameras.
    visualizer = BehaviorVisualizer(
        lick_tracker=lick_tracker, valve_tracker=valve_tracker, distance_tracker=speed_tracker
    )

    # Updates the threshold lines to use the initial speed and duration values
    visualizer.update_speed_thresholds(speed_threshold=initial_speed, duration_threshold=initial_duration)

    # Configures all system components to support run training
    runtime.run_train_state()

    # Initializes the listener instance used to enable keyboard-driven training runtime control.
    listener = _KeyboardListener()

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
    previous_speed_threshold = copy.copy(initial_speed)
    previous_duration_threshold = copy.copy(initial_duration)

    # Also pre-initializes the speed and duration trackers
    speed_threshold: np.float64 = np.float64(0)
    duration_threshold: np.float64 = np.float64(0)

    # Initializes the main training loop. The loop will run either until the total training time expires, the maximum
    # volume of water is delivered or the loop is aborted by the user.
    try:
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

            # Determines the speed and duration thresholds for each cycle. This factors in the user input via keyboard.
            # Note, user input has a static resolution of 0.1 cm/s per step and 50 ms per step.
            speed_threshold = np.clip(
                a=initial_speed + (increase_steps * speed_step) + (listener.speed_modifier * 0.05),
                a_min=0.1,  # Minimum value
                a_max=maximum_speed,  # Maximum value
            )
            duration_threshold = np.clip(
                a=initial_duration + (increase_steps * duration_step) + (listener.duration_modifier * 50),
                a_min=50,  # Minimum value (0.05 seconds == 50 milliseconds)
                a_max=maximum_duration,  # Maximum value
            )

            # If any of the threshold changed relative to the previous loop iteration, updates the visualizer and
            # previous threshold trackers with new data.
            if duration_threshold != previous_duration_threshold or previous_speed_threshold != speed_threshold:
                visualizer.update_speed_thresholds(speed_threshold, duration_threshold)
                previous_speed_threshold = speed_threshold
                previous_duration_threshold = duration_threshold

            # Reads the animal's running speed from the visualizer. The visualizer uses the distance tracker to
            # calculate the running speed of the animal over 100 millisecond windows. This accesses the result of this
            # computation and uses it to determine whether the animal is performing above the threshold.
            current_speed = visualizer.running_speed

            # If the speed is above the speed threshold, and the animal has been maintaining the above-threshold speed
            # for the required duration, delivers 5 uL of water. If the speed is above threshold, but the animal has
            # not yet maintained the required duration, the loop will keep cycling and accumulating the timer count.
            # This is done until the animal either reaches the required duration or drops below the speed threshold.
            if current_speed >= speed_threshold and speed_timer.elapsed >= duration_threshold:
                runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water

                # Also resets the timer. While mice typically stop to consume water rewards, which would reset the
                # timer, this guards against animals that carry on running without consuming water rewards.
                speed_timer.reset()

            # If the current speed is below the speed threshold, resets the speed timer.
            elif current_speed < speed_threshold:
                speed_timer.reset()

            # Updates the progress bar with each elapsed second. Note, this is technically not safe in case multiple
            # seconds pass between reaching this conditional, but we know empirically that the loop will run at
            # millisecond intervals, so it is not a concern.
            if runtime_timer.elapsed > previous_time:
                previous_time = runtime_timer.elapsed  # Updates the previous time for the next progress bar update
                progress_bar.update(1)

            # Updates the visualizer plot
            visualizer.update()

            # If the total volume of water dispensed during runtime exceeds the maximum allowed volume, aborts the
            # training early with a success message.
            if dispensed_water_volume >= maximum_volume:
                message = (
                    f"Run training has delivered the maximum allowed volume of water ({maximum_volume} uL). Aborting "
                    f"the training process."
                )
                console.echo(message=message, level=LogLevel.SUCCESS)
                break

            # If the listener detects a reward delivery signal, delivers the reward to the animal.
            if listener.reward_signal:
                runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water

            # If the user sent the abort command, terminates the training early with an error message.
            if listener.exit_signal:
                message = (
                    "Run training abort signal detected. Aborting the training with a graceful shutdown procedure."
                )
                console.echo(message=message, level=LogLevel.ERROR)
                break

        # Close the progress bar
        progress_bar.close()

    except Exception as e:
        message = (
            f"Training runtime has encountered an error and had to be terminated early. Attempting to gracefully "
            f"shutdown all assets and preserve as much of the data as possible. The encountered error message: "
            f"{str(e)}"
        )
        console.echo(message=message, level=LogLevel.ERROR)

    # Shutdown sequence:
    message = f"Training runtime: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Directly overwrites the final running speed and duration thresholds in the descriptor instance stored in the
    # runtime attributes. This ensures the descriptor properly reflects the final thresholds used at the end of
    # the training.
    if not isinstance(runtime.descriptor, LickTrainingDescriptor):  # This is to appease mypy
        runtime.descriptor.final_running_speed_cm_s = float(speed_threshold)
        runtime.descriptor.final_speed_duration_s = float(duration_threshold / 1000)  # Converts from s to ms

    # Terminates the listener
    listener.shutdown()

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()


def run_experiment_logic(
    experimenter: str,
    animal_weight: float,
    session_data: SessionData,
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
        experimenter: The id of the experimenter conducting the experiment.
        animal_weight: The weight of the animal, in grams, at the beginning of the training session.
        session_data: The SessionData instance that manages the data acquired during the experiment.
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

    # Enables the console
    if not console.enabled:
        console.enable()

    # Dumps the SessionData information to the raw_data folder as a session_data.yaml file. This is a prerequisite for
    # being able to run preprocessing on the data if runtime fails.
    session_data.to_path()

    message = f"Initializing lick training runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Generates the runtime class and other assets
    # Pre-generates the SessionDescriptor class and populates it with experiment data.
    descriptor = MesoscopeExperimentDescriptor(
        experimenter=experimenter,
        mouse_weight_g=animal_weight,
    )

    # Initializes the main runtime interface class. Note, most class parameters are statically configured to work for
    # the current VRPC setup and may need to be adjusted as that setup evolves over time.
    runtime = _MesoscopeExperiment(
        session_data=session_data,
        descriptor=descriptor,
        cue_length_map=cue_length_map,
        actor_port="/dev/ttyACM0",
        sensor_port="/dev/ttyACM1",
        encoder_port="/dev/ttyACM2",
        headbar_port="/dev/ttyUSB0",
        lickport_port="/dev/ttyUSB1",
        face_camera_index=0,
        left_camera_index=0,
        right_camera_index=2,
        harvesters_cti_path=Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"),
        screens_on=False,
        valve_calibration_data=valve_calibration_data,
    )

    runtime_timer = PrecisionTimer("s")  # Initializes the timer to enforce experiment state durations

    # Uses runtime trackers extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Visualizer initialization HAS to happen after the runtime start to avoid interfering with cameras.
    visualizer = BehaviorVisualizer(
        lick_tracker=lick_tracker, valve_tracker=valve_tracker, distance_tracker=speed_tracker
    )

    # Initializes the keyboard listener to support aborting test runtimes.
    listener = _KeyboardListener()

    # Main runtime loop. It loops over all submitted experiment states and ends the runtime after executing the last
    # state
    try:
        for state in experiment_state_sequence:
            runtime_timer.reset()  # Resets the timer

            # Sets the Experiment state
            runtime.change_experiment_state(state.experiment_state_code)

            # Resolves and sets the VR state
            if state.vr_state_code == 1:
                runtime.vr_rest()
            elif state.vr_state_code == 2:
                runtime.vr_run()
            else:
                warning_text = (
                    f"Invalid VR state code {state.vr_state_code} encountered when executing experiment runtime. "
                    f"Currently, only codes 1 (rest) and 2 (run) are supported. Skipping the unsupported state."
                )
                console.echo(message=warning_text, level=LogLevel.ERROR)
                continue

            # Creates a tqdm progress bar for the current experiment state
            with tqdm(
                total=state.state_duration_s,
                desc=f"Executing experiment state {state.experiment_state_code}",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s",
            ) as pbar:
                previous_seconds = 0

                while runtime_timer.elapsed < state.state_duration_s:
                    visualizer.update()  # Continuously updates the visualizer

                    # Updates the progress bar every second. While the current implementation is not safe, we know that
                    # the loop will cycle much faster than 1 second, so it should not be possible for the delta to ever
                    # exceed 1 second.
                    if runtime_timer.elapsed != previous_seconds:
                        pbar.update(1)
                        previous_seconds = runtime_timer.elapsed

                    # If the user sent the abort command, terminates the runtime early with an error message.
                    if listener.exit_signal:
                        message = "Experiment runtime: aborted due to user request."
                        console.echo(message=message, level=LogLevel.ERROR)
                        break

    except Exception as e:
        message = (
            f"Experiment runtime has encountered an error and had to be terminated early. Attempting to gracefully "
            f"shutdown all assets and preserve as much of the data as possible. The encountered error message: "
            f"{str(e)}"
        )
        console.echo(message=message, level=LogLevel.ERROR)

    # Shutdown sequence:
    message = f"Experiment runtime: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()
