"""This module provides the main VR class that abstracts working with Sun lab's mesoscope-VR system"""

import os
from pathlib import Path

from ataraxis_communication_interface.communication import MQTTCommunication
from tqdm import tqdm
import numpy as np
from ataraxis_time import PrecisionTimer
from packaging_tools import calculate_directory_checksum
from module_interfaces import (
    TTLInterface,
    LickInterface,
    BreakInterface,
    ValveInterface,
    TorqueInterface,
    EncoderInterface,
    ScreenInterface,
)
from ataraxis_base_utilities import console, ensure_directory_exists
from mesoscope_preprocessing import process_mesoscope_directory, _fix_mesoscope_frames
from ataraxis_data_structures import DataLogger, LogPackage
from ataraxis_time.time_helpers import get_timestamp
from ataraxis_communication_interface import MicroControllerInterface


class MesoscopeExperiment:
    """The base class for all Sun lab Mesoscope experiment runtimes.

    This class provides methods for conducting experiments in the Sun lab using the Mesoscope-VR system. This class
    abstracts most low-level interactions with the VR system and the mesoscope via a simple high-level API. In turn, the
    API can be used by all lab members to write custom Experiment class specializations for their projects.

    This class also provides methods for initial preprocessing of the raw data. These preprocessing steps all use
    multiprocessing to optimize runtime speeds and are designed to be executed after each experimental session to
    prepare the data for long-term storage and further processing and analysis.

    Notes:
        Calling the initializer does not start the underlying processes. Use start() method before issuing other
        commands to properly initialize all remote processes. In the current configuration, this class reserves ~10
        CPU cores during runtime.

    Args:
        output_directory: The directory where all experiment data should be saved. Typically, this is the output of the
            ProjectData class create_session() method.
        screens_off: Determines whether the VR screens are OFF or ON when this class is called. This is used to
            determine the initial state of the VR screens, which is essential for this class to be able to track the
            state of the VR screens during runtime.
        experiment_state: The integer code that represents the initial state of the experiment. Experiment state codes
            are used to mark different stages of each experiment (such as setup, rest, task 1, task 2, etc.). During
            analysis, these codes can be used to segment experimental data into sections.
        actor_port: The USB port used by the actor Microcontroller.
        sensor_port: The USB port used by the sensor Microcontroller.
        encoder_port: The USB port used by the encoder Microcontroller.
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is teh volume of dispensed water, in
            microliters. It is expected that the valve will be frequently replaced and recalibrated, so this parameter
            is kept addressable.
        unity_ip: The IP address of the MQTT broker used to communicate with Unity game engine.
        unity_port: The port number of the MQTT broker used to communicate with Unity game engine.

    Attributes:
        _started: A flag indicating whether the VR system and experiment runtime are currently running.
        _amc_logger: A DataLogger instance that collects data from all microcontrollers.
        _mesoscope_trigger: The interface that triggers Mesoscope frame acquisition via TTL signaling.
        _break: The interface that controls the electromagnetic break attached to the running wheel.
        _reward: The interface that controls the solenoid water valve to deliver water rewards to the animal.
        _screens: The interface that allows turning VR displays on and off.
        _actor: The main actor AMC interface.
        _mesoscope_frame: The interface that receives frame acquisition timestamp signals from the Mesoscope.
        _lick: The interface that monitors and records animal's interactions with the lick sensor.
        _torque: The interface that monitors and records the torque applied by the animal to the running wheel.
        _sensor: The main sensor AMC interface.
        _wheel_encoder: The interface that monitors the rotation of the running wheel and converts it into the distance
            traveled by the animal.
        _encoder: The main encoder AMC interface.
        _screen_off: Tracks whether the VR displays are currently ON or OFF.
        _mesoscope_off: Tracks whether the mesoscope is currently acquiring images.
        _vr_state: Stores the current state of the VR system. The MesoscopeExperiment updates this value whenever it is
            instructed to change the VR system state.
        _state_map: Maps the integer state-codes used to represent VR system states to human-readable string-names.
        _experiment_state: Stores the user-defined experiment state. Experiment states are defined by the user and
            are expected to be unique for each project and, potentially, experiment. Different experiment states can
            reuse the same VR state.
        _timestamp-timer: A PrecisionTimer instance used to timestamp log entries generated by the experiment class
            instance.

    Raises:
        TypeError: If any of the arguments are not of the expected type.
        ValueError: If any of the arguments are not of the expected value.
    """

    _state_map: dict[int, str] = {0: "Idle", 1: "Rest", 2: "Run", 3: "Lick Train", 4: "Run Train"}

    def __init__(
        self,
        output_directory: Path,
        screens_off: bool,
        experiment_state: int = 0,
        actor_port: str = "/dev/ttyACM0",
        sensor_port: str = "/dev/ttyACM1",
        encoder_port: str = "/dev/ttyACM2",
        unity_ip: str = "127.0.0.1",
        unity_port: int = 1883,
        valve_calibration_data: tuple[tuple[int | float, int | float], ...] = ((10, 10),),
    ) -> None:
        # As with other pipelines that use intelligent resource termination, presets the _started flags first to avoid
        # leaks if the initialization method fails.
        self._started: bool = False

        # Defines other flags used during runtime:
        self._screen_off = screens_off
        self._mesoscope_off = True  # Expects the mesoscope to NOT be acquiring images
        self._vr_state: int = 0  # Stores the current state of the VR system
        self._experiment_state = experiment_state  # Stores user-defined experiment state
        self._timestamp_timer = PrecisionTimer("us")  # A timer to log changes to VR and Experiment states

        # Input verification:
        if not isinstance(output_directory, Path):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a Path instance for 'output_directory' "
                f"argument, but instead encountered {output_directory} of type {type(output_directory).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not output_directory.is_dir():
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected the path to a directory for "
                f"'output_directory argument, but instead encountered a file path {output_directory}."
            )
            console.error(message=message, error=ValueError)

        # Ensures that the output directory exists if it passed the verification above
        ensure_directory_exists(output_directory)

        if not isinstance(valve_calibration_data, tuple) or not all(
            isinstance(item, tuple)
            and len(item)
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float)) == 2
            for item in valve_calibration_data
        ):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a tuple of 2-element tuples with "
                f"integer or float values for 'valve_calibration_data' argument, but instead encountered "
                f"{valve_calibration_data} of type {type(valve_calibration_data).__name__}."
            )
            console.error(message=message, error=TypeError)

        # The rest of the arguments are verified directly by the Communication and Interface classes.

        # Initializes the microcontroller data logger. This datalogger works exclusively with microcontroller-generated
        # data and aggregates the data streams for all used microcontrollers.
        self._amc_logger: DataLogger = DataLogger(output_directory=output_directory, instance_name="amc", sleep_timer=0)

        # ACTOR. Actor AMC controls the hardware that needs to be triggered by PC at irregular intervals. Most of such
        # hardware is designed to produce some form of an output: deliver water reward, engage wheel breaks, issue a
        # TTL trigger, etc.

        # Module interfaces:
        self._mesoscope_trigger: TTLInterface = TTLInterface()  # Mesoscope trigger
        self._break = BreakInterface(
            minimum_break_strength=43.2047,  # 0.6 in oz
            maximum_break_strength=1152.1246,  # 16 in oz
            object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
        )  # Wheel break
        self._reward = ValveInterface(valve_calibration_data=valve_calibration_data)  # Reward solenoid valve
        self._screens = ScreenInterface()  # VR Display On/Off switch

        # Main interface:
        self._actor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=actor_port,
            data_logger=self._amc_logger,
            module_interfaces=(self._mesoscope_trigger, self._break, self._reward, self._screens),
            mqtt_broker_ip=unity_ip,
            mqtt_broker_port=unity_port,
        )

        # SENSOR. Sensor AMC controls the hardware that collects data at regular intervals. This includes lick sensors,
        # torque sensors, and input TTL recorders. Critically, all managed hardware does not rely on hardware interrupt
        # logic to maintain the necessary precision.

        # Module interfaces:
        self._mesoscope_frame: TTLInterface = TTLInterface()  # Mesoscope frame timestamp recorder
        self._lick: LickInterface = LickInterface(lick_threshold=200)  # Main lick sensor
        self._torque: TorqueInterface = TorqueInterface(
            baseline_voltage=2046,  # ~1.65 V
            maximum_voltage=4095,  # ~3.3 V  # TODO adjust based on calibration data
            sensor_capacity=720.0779,  # 10 in oz
            object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
        )  # Wheel torque sensor

        # Main interface:
        self._sensor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(152),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=sensor_port,
            data_logger=self._amc_logger,
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
            data_logger=self._amc_logger,
            module_interfaces=(self._wheel_encoder,),
            mqtt_broker_ip=unity_ip,
            mqtt_broker_port=unity_port,
        )

        # Also instantiates a separate MQTTCommunication instance to directly communicate with Unity. Primarily, this
        # is used to support direct VR screen manipulation (for example, to black them out) and to collect data
        # generated by unity, such as the sequence of VR corridors.
        monitored_topics = ("CueSequence/",)
        self._unity: MQTTCommunication = MQTTCommunication(
            ip=unity_ip, port=unity_port, monitored_topics=monitored_topics
        )

    def __del__(self) -> None:
        """Ensures the instance properly releases all resources before it is garbage-collected."""
        self.stop()

    def start(self) -> None:
        """Sets up all assets used to support realtime experiment control and data acquisition.

        This method establishes the communication with the microcontrollers, data logger cores, and video system
        processes. Until this method is called, the instance will not be able to carry out any commands, as it will not
        have access to the necessary resources.

        Notes:
             This process will not execute unless the host PC has access to the necessary number of logical CPU cores
             and any other hardware resources. This prevents using the class on machines that are unlikely to sustain
             the runtime requirements.

        Raises:
            RuntimeError: If the host PC does not have enough logical CPU cores available.
        """

        # Prevents (re) starting an already started VR process.
        if self._started:
            return

        # 3 cores for microcontrollers, 2 cores for data loggers, 3 cores for the current video_system
        # configuration (2 producers, 1 consumer), 1 core for the central process calling this method. 9 cores
        # total.
        if not os.cpu_count() >= 9:
            message = (
                f"Unable to start the MesoscopeExperiment runtime. The host PC must have at least 9 logical CPU "
                f"cores available for this class to work as expected, but only {os.cpu_count()} cores are "
                f"available."
            )
            console.error(message=message, error=RuntimeError)

        # Ensures that shared memory buffers used by the managed classes are available for instantiation
        self._amc_logger._vacate_shared_memory_buffer()
        self._actor.vacate_shared_memory_buffer()
        self._sensor.vacate_shared_memory_buffer()
        self._encoder.vacate_shared_memory_buffer()

        # Starts the data logger
        self._amc_logger.start()

        # Generates and logs the onset timestamp for the VR system as a whole. The MesoscopeExperiment class logs
        # changes to VR and Experiment state during runtime. Since we maintain two logs, one for microcontroller data
        # and one for video data, this information is saved to both log files:

        # Constructs a timezone-aware stamp using UTC time. This creates a reference point for all later delta time
        # readouts. The time is returned as an array of bytes.
        onset: NDArray[np.uint8] = get_timestamp(as_bytes=True)  # type: ignore
        self._timestamp_timer.reset()  # Immediately resets the timer to make it as close as possible to the onset time

        # Logs the onset timestamp. All further timestamps will be treated as integer time deltas (in microseconds)
        # relative to the onset timestamp. Note, ID of 0 is used to mark the main experiment system.
        package = LogPackage(0, 0, onset)  # Packages the id, timestamp, and data.
        self._amc_logger.input_queue.put(package)

        # TODO ADD VS SECTION HERE

        # Starts all microcontroller interfaces
        self._actor.start()
        self._actor.unlock_controller()  # Only Actor outputs data, so no need to unlock other controllers.
        self._sensor.start()
        self._encoder.start()
        self._unity.connect()  # Also connects to the Unity communication channels.

        # Engages the modules that need to stay on during the entire experiment duration:

        # The mesoscope acquires frames at ~10 Hz and sends triggers with the on-phase duration of ~50 ms, so a
        # 100 Hz polling frequency should be enough to detect all triggers.
        # noinspection PyTypeChecker
        self._sensor.send_message(self._mesoscope_frame.check_state(repetition_delay=np.uint32(10000)))

        # Starts monitoring licks. Uses 100 Hz polling frequency, since mice are expected to lick at ~10 Hz rate.
        # This resolution should be sufficient to resolve individual licks of variable duration and lick bouts
        # (bursts).
        # noinspection PyTypeChecker
        self._sensor.send_message(self._lick.check_state(repetition_delay=np.uint32(10000)))

        # Sets up other VR systems according to REST state specifications.
        self.vr_rest()

        # Water Valve receives triggers directly from unity, so we do not need to manipulate the valve state
        # manually. Mesoscope is triggered via a dedicated instance method.
        self._started = True

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
        # this is used as a shortcut o prepare the VR system for shutdown.
        self.vr_rest()

        # If the mesoscope is still acquiring images, ensures frame acquisition is disabled.
        if not self._mesoscope_off:
            self.mesoscope_off()
            timer.delay_noblock(2)  # Delays for 2 seconds. This ensures all mesoscope frame triggers are received.

        # Manually prepares remaining modules for shutdown
        # noinspection PyTypeChecker
        self._sensor.send_message(self._mesoscope_frame.dequeue_command)
        # noinspection PyTypeChecker
        self._sensor.send_message(self._lick.dequeue_command)
        # noinspection PyTypeChecker
        self._sensor.send_message(self._torque.dequeue_command)

        # Delays for another 2 seconds, to ensure all microcontroller-sent data is received before shutting down the
        # interfaces.
        timer.delay_noblock(2)

        # Stops all interfaces
        self._amc_logger.stop()
        self._actor.stop()
        self._sensor.stop()
        self._encoder.stop()

        # Compresses all logs into a single .npz file. This is done both for long-term storage optimization and to
        # allow parsing the data.
        self._amc_logger.compress_logs(remove_sources=True, verbose=True)
        # TODO ADD VS SECTION HERE

    def vr_rest(self) -> None:
        """Switches the VR system to the REST state.

        In the rest state, the break is engaged to prevent the mouse from moving the wheel. The encoder module is
        disabled and instead the torque sensor is enabled. The VR screens are switched off, cutting off light emission.
        By default, the VR system starts all experimental runtimes using the REST state.
        """

        # Engages the break to prevent the mouse from moving the wheel
        # noinspection PyTypeChecker
        self._actor.send_message(self._break.toggle(True))

        # Initiates torque monitoring at 100 Hz. The torque can only be accurately measured when the wheel is locked,
        # as it requires a resistance force to trigger the sensor. Since we downsample all data to the mesoscope
        # acquisition rate of ~10 Hz, and do not use torque data in real time, the sampling rate is seto to a 100 Hz.
        # noinspection PyTypeChecker
        self._sensor.send_message(self._torque.check_state(repetition_delay=np.uint32(10000)))

        # Temporarily suspends encoder monitoring. Since the wheel is locked, the mouse should not be able to produce
        # meaningful motion data.
        # noinspection PyTypeChecker
        self._encoder.send_message(self._wheel_encoder.dequeue_command)

        # Toggles the state of the VR screens to be OFF if the VR screens are currently ON. If the screens are OFF,
        # keeps them OFF.
        if not self._screen_off:
            # noinspection PyTypeChecker
            self._actor.send_message(self._screens.toggle())
            self._screen_off = True

        # Configures the state tracker to reflect REST state
        self._change_vr_state(1)

    def vr_run(self) -> None:
        """Switches the VR system to the RUN state.

        In the run state, the break is disengaged to allow the mouse to freely move the wheel. The encoder module is
        enabled to record and share live running data with Unity and the torque sensor is disabled. The VR screens are
        switched on to render the VR environment.
        """

        # Initializes encoder monitoring at 1 kHz rate. The encoder aggregates wheel data at native speeds; this rate
        # only determines how often the aggregated data is sent to PC and Unity.
        # noinspection PyTypeChecker
        self._encoder.send_message(self._wheel_encoder.check_state(repetition_delay=np.uint32(1000)))

        # Disables torque monitoring. To accurately measure torque, the sensor requires a resistance force provided by
        # the break. During running, measuring torque is not very reliable and adds little in addition to the
        # encoder.
        # noinspection PyTypeChecker
        self._sensor.send_message(self._torque.dequeue_command)

        # Disengages the break to allow the mouse to move the wheel
        # noinspection PyTypeChecker
        self._actor.send_message(self._break.toggle(False))

        # Toggles the state of the VR screens to be ON if the VR screens are currently OFF. If the screens are ON,
        # keeps them ON.
        if self._screen_off:
            # noinspection PyTypeChecker
            self._actor.send_message(self._screens.toggle())
            self._screen_off = False

        # Configures the state tracker to reflect RUN state
        self._change_vr_state(2)

    def mesoscope_on(self) -> None:
        """Instructs the mesoscope to continuously acquire frames.

        Notes:
            This relies on the ScanImage being configured to recognize and accept triggers via TTL and for the
            ScanImage to be armed before receiving the trigger.
        """

        # Toggles the mesoscope acquisition trigger to continuously deliver a HIGH signal. The mesoscope will
        # continuously acquire frames as long as the trigger is HIGH.
        if self._mesoscope_off:  # Guards against multiple start calls.
            # noinspection PyTypeChecker
            self._actor.send_message(self._mesoscope_trigger.toggle(True))
            self._mesoscope_off = False

    def mesoscope_off(self) -> None:
        """Instructs the mesoscope to stop acquiring frames.

        Notes:
            This relies on the ScanImage being configured to recognize and accept triggers via TTL and for the
            ScanImage to be armed before receiving the trigger.
        """

        # Toggles the mesoscope acquisition trigger to continuously deliver a LOW signal. When the trigger is LOW,
        # the mesoscope will not acquire frames.
        if not self._mesoscope_off:
            # noinspection PyTypeChecker
            self._actor.send_message(self._mesoscope_trigger.toggle(False))
            self._mesoscope_off = True

    @property
    def mesoscope_state(self) -> bool:
        """Returns True if the mesoscope is currently acquiring frames, False otherwise."""
        return self._mesoscope_off

    @property
    def vr_state(self) -> str:
        """Returns the current VR state as a string."""
        return self._state_map[self._vr_state]

    def _change_vr_state(self, new_state: int) -> None:
        """Sets the vr_state attribute to the input value and logs the change to the VR state.

        This method is used internally to update and log new VR states.
        """
        self._vr_state = new_state  # Updates the VR state

        # Logs the VR state update
        timestamp = self._timestamp_timer.elapsed
        log_package = LogPackage(
            source_id=0, time_stamp=timestamp, serialized_data=np.array([new_state], dtype=np.uint8)
        )
        self._amc_logger.input_queue.put(log_package)
        # TODO Add VS logger

    @property
    def experiment_state(self) -> int:
        """Returns the current experiment state as an integer."""
        return self._experiment_state

    def change_experiment_state(self, new_state: int) -> None:
        """Updates the experiment state tracker and logs the change to the experiment state.

        Use this method to timestamp and log experiment state (stage) changes. The state change is logged to both the
        amc and the video loggers.
        """
        self._experiment_state = new_state  # Updates the Experiment state

        # Logs the VR state update
        timestamp = self._timestamp_timer.elapsed
        log_package = LogPackage(
            source_id=0, time_stamp=timestamp, serialized_data=np.array([new_state], dtype=np.uint8)
        )
        self._amc_logger.input_queue.put(log_package)
        # TODO Add VS logger


class ProjectData:
    """Provides methods for managing the experimental data acquired by all Sun lab pipelines.

    This class functions as the central hub for managing the data across all destinations: The PC(s) that acquire the
    data, the NAS, and the data analysis server(s). Its primary purpose is to maintain the project data structure
    across all supported destinations and efficiently and safely move the data between these destinations with minimal
    redundancy and footprint.

    Note:
        The class is written to be data-agnostic. It does not contain methods for preprocessing or analyzing the data
        and does not expect any particular dta format beyond adhering to the project-animal-session structure employed
        by the Sun lab.

    Args:
        project_directory: The path to the root directory managed the project.
        animal_name: The name of the animal whose' data will be managed by this class.
    """

    def __init__(self, project_directory: Path, animal_name: str) -> None:
        # Combines project directory and animal name to get the animal directory path
        self._animal_directory = project_directory.joinpath(animal_name)

        # Loops over the input directory and resolves available session folders. If the directory does not exist,
        # sets the variable to an empty tuple and generates the animal directory
        self._sessions: tuple[str, ...]
        if self._animal_directory.exists():
            session: Path
            self._sessions = tuple([str(session.stem) for session in project_directory.glob("*") if session.is_dir()])
        else:
            ensure_directory_exists(self._animal_directory)
            self._sessions = tuple()

    def create_session(self) -> Path:
        """Creates a new session directory within the broader project-animal data structure.

        Uses the current timestamp down to microseconds as the session folder name, which ensures that each session
        name within the broader project-animal structure has a unique name that accurately preserves the order of the
        sessions.

        Notes:
            You can use the 'stem' property of the returned path to get the session name.

        Returns:
            The Path to the newly created session directory.
        """
        # Generates a unique name for the session, based on the current timestamp. Since the timestamp is accurate
        # to microseconds, this method is guaranteed to provide unique timestamp-based names across all our
        # use cases.
        session_name: str = get_timestamp(time_separator="-")
        session_path = self._animal_directory.joinpath(session_name)

        # Precreates the 'raw' and the 'processed' directories within the session directory
        ensure_directory_exists(session_path.joinpath("raw"))
        ensure_directory_exists(session_path.joinpath("processed"))

        self._sessions += (session_name,)  # Updates the sessions tuple with the new session name
        return session_path


def convert_old_data(
    root_directory: Path, remove_sources: bool = False, num_processes: int = None, verify_integrity=True
) -> None:
    """A temporary function used to convert old Tyche data to our latest data format."""
    # Resolves the number of processes to use for processing the data.
    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 4)  # Keeps 4 cores free for other tasks

    # Recursively finds all subdirectories inside the root directory, excluding 'mesoscope_frames'
    subdirectories = [
        directory
        for directory in root_directory.rglob("*")
        if directory.is_dir() and directory.name != "mesoscope_frames"  # Exclude already processed frames.
    ]

    # Collects the paths to each directory with TIFF files and to each directory that has a mesoscope_frames
    # subdirectory. This is used as a heuristic to discover folders with Tyche cohort session data.
    session_directories = set()  # Directories that have not been processed will have raw tiff stacks
    old_directories = set()  # Directories processed by the pipeline will have mesoscope_frames folder in them
    all_directories = set()  # Stores all directories that may contain mesoscope frames
    for subdirectory in subdirectories:
        # Discovers all tiff and tif files inside each candidate subdirectory
        tiff_files_in_subdir = list(subdirectory.glob("*.tif")) + list(subdirectory.glob("*.tiff"))

        # Also builds the path to the mesoscope_frames directory, which may be located inside the evaluated directory
        mesoscope_directory = subdirectory / "mesoscope_frames"

        # Checks for mesoscope_frames subdirectory. This directory would exist in already processed directories and
        # may contain frame data using the old format.
        has_mesoscope_frames = mesoscope_directory.exists()

        # Any directory that does not have mesoscope_frames and has tiff files is marked to be processed as a new
        # directory. This processing type will compress existing stacks with LERC and extract metadata.
        if not has_mesoscope_frames and tiff_files_in_subdir:
            session_directories.add(subdirectory)
        # Conversely, directories that contain mesoscope_frames can either be processed with the modern or old pipeline.
        # Directories processed with the old pipeline will have thousands of frames inside the mesoscope_frames. These
        # directories need to be reprocessed to reinstate the stack format used by the modern pipeline.
        if (
            has_mesoscope_frames
            and len(list(mesoscope_directory.glob("*.tif")) + list(mesoscope_directory.glob("*.tiff"))) > 200
        ):
            old_directories.add(subdirectory)

        # This final set is used to ensure all directories are checksummed, regardless of whether they are unprocessed
        # or reprocessed.
        if has_mesoscope_frames or tiff_files_in_subdir:
            all_directories.add(subdirectory)

    # First processes new session directories
    if len(session_directories) > 0:
        for directory in tqdm(session_directories, desc="Processing mesoscope frame directories", unit="directory"):
            process_mesoscope_directory(
                image_directory=directory,
                num_processes=num_processes,
                remove_sources=remove_sources,
                batch=True,
                verify_integrity=verify_integrity,
            )
    else:
        print("No new mesoscope frame directories to process.")

    # Next, handles any old directories that require reprocessing
    if len(old_directories) > 0:
        for directory in tqdm(old_directories, desc="Reprocessing old mesoscope frame directories", unit="directory"):
            _fix_mesoscope_frames(
                mesoscope_directory=directory,
                num_processes=num_processes,
                remove_sources=remove_sources,
                stack_size=500,
                batch=True,
                verify_integrity=verify_integrity,
            )
    else:
        print("No old mesoscope frame directories to reprocess.")

    # Calculates the checksum for all mesoscope-related directories
    for directory in tqdm(all_directories, desc="Calculating checksums for processed directories", unit="directory"):
        calculate_directory_checksum(directory, num_processes, batch=True, save_checksum=True)


# def test_runtime() -> None:
#     temp_dir = Path(tempfile.mkdtemp())
#     data_logger = DataLogger(output_directory=temp_dir, instance_name="test_logger")
#
#     console.enable()
#
#     encoder = EncoderInterface()
#     brake = BreakInterface()
#     lick = LickInterface()
#     valve = ValveInterface(valve_calibration_data=((1000, 0.001), (5000, 0.005), (10000, 0.01)))
#
#     actor_interfaces = (brake, valve)
#
#     encoder_interfaces = (encoder,)
#
#     sensor_interfaces = (lick,)
#
#     actor_interface = MicroControllerInterface(
#         controller_id=np.uint8(101),
#         data_logger=data_logger,
#         module_interfaces=actor_interfaces,
#         microcontroller_serial_buffer_size=8192,
#         microcontroller_usb_port="/dev/ttyACM1",
#         baudrate=115200,
#     )
#
#     encoder_interface = MicroControllerInterface(
#         controller_id=np.uint8(203),
#         data_logger=data_logger,
#         module_interfaces=encoder_interfaces,
#         microcontroller_serial_buffer_size=8192,
#         microcontroller_usb_port="/dev/ttyACM2",
#         baudrate=115200,
#     )
#
#     sensor_interface = MicroControllerInterface(
#         controller_id=np.uint8(152),
#         data_logger=data_logger,
#         module_interfaces=sensor_interfaces,
#         microcontroller_serial_buffer_size=8192,
#         microcontroller_usb_port="/dev/ttyACM0",
#         baudrate=115200,
#     )
#
#     timer = PrecisionTimer("s")
#
#     # cue_topic = 'CueSequence/'
#     # unity_communication = UnityCommunication(monitored_topics=cue_topic)
#     # unity_communication.connect()
#
#     # unity_communication.send_data("Display/Blank/")
#     # timer.delay_noblock(5)
#     # unity_communication.send_data("Display/Show/")
#
#     # unity_communication.send_data("CueSequenceTrigger/")
#
#     # while not unity_communication.has_data:
#     #     pass
#
#     # sequence: bytes = unity_communication.get_data()
#
#     data_logger._vacate_shared_memory_buffer()
#     actor_interface.vacate_shared_memory_buffer()
#     sensor_interface.vacate_shared_memory_buffer()
#     encoder_interface.vacate_shared_memory_buffer()
#
#     data_logger.start()
#
#     actor_interface.start()
#     encoder_interface.start()
#     sensor_interface.start()
#
#     actor_interface.unlock_controller()
#
#     actor_interface.send_message(brake.toggle(state=False))  # Disengage the break
#
#     # Calibrate the encoder to only report CCW movement and start position recording
#     encoder_interface.send_message(encoder.set_parameters(report_cw=False, report_ccw=True, delta_threshold=20))
#     encoder_interface.send_message(encoder.check_state(repetition_delay=np.uint32(100)))
#
#     # Starts lick monitoring
#     sensor_interface.send_message(lick.check_state(repetition_delay=np.uint32(20000)))
#
#     # Starts reward delivery
#     actor_interface.send_message(
#         valve.set_parameters(
#             pulse_duration=np.uint32(4000000), calibration_delay=np.uint32(10000), calibration_count=np.uint16(100)
#         )
#     )  # 1 second
#
#     timer.delay_noblock(delay=120)
#
#     actor_interface.stop()
#     encoder_interface.stop()
#     sensor_interface.stop()
#     data_logger.stop()

if __name__ == "__main__":
    target = Path("/media/Data/2022_01_25")
    convert_old_data(root_directory=target, remove_sources=True, verify_integrity=True)
