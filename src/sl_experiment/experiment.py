"""This module provides the main VR class that abstracts working with Sun lab's mesoscope-VR system"""

import os
import warnings
from pathlib import Path

import numpy as np
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
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import DataLogger, LogPackage
from ataraxis_time.time_helpers import get_timestamp
from ataraxis_communication_interface import MicroControllerInterface, MQTTCommunication
from ataraxis_video_system import VideoSystem
from transfer_tools import transfer_directory
from packaging_tools import calculate_directory_checksum
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil


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
        face_camera_index: int = 0,
        left_camera_index: int = 0,
        right_camera_index: int = 1,
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
        # TODO Add VS logger`


class ProjectData:
    """Provides methods for managing the experimental data acquired by all Sun lab pipelines for a given project and
    animal combination.

    This class functions as the central hub for collecting the data from all local PCs involved in the acquisition
    process and pushing it to the NAS and the data analysis server(s). Its primary purpose is to maintain the project
    data structure across all supported destinations and to efficiently and safely move the data to these
    destinations with minimal redundancy and footprint.

    Note:
        It is expected that the server, nas, and mesoscope data directories are mounted on the host-machine via the
        SMB or equivalent protocol. All manipulations with these destinations are carried out with the assumption that
        the OS has full access to these directories and filesystems.

        This class is specifically designed for working with raw data for a single animal used in the managed project.
        Processed data is managed by the processing pipeline classes.

        This class generates an xxHash-128 checksum stored inside the ax_checksum.txt file at the root of each session
        directory.

        If this class is instantiated with the 'create' flag for an already existing project and / or animal, it will
        NOT recreate the project and animal directories. Instead, the instance will use the path to the already existing
        directories.

    Args:
        project_name: The name of the project whose data will be managed by the class.
        animal_name: The name of the animal whose data will be managed by the class.
        create: A boolean flag indicating whether to create the project directory if it does not exist.
        local_root_directory: The path to the root directory where all projects are stored on the host-machine. Usually,
            this is the 'Experiments' folder on the 16TB volume of the VRPC machine.
        server_root_directory: The path to the root directory where all projects are stored on the server-machine.
            Usually, this is the 'storage' (RAID 6) volume of the BioHPC server.
        nas_root_directory: The path to the root directory where all projects are stored on the Synology NAS. Usually,
            this is a non-compressed 'raw_data' directory on one of the slow (RAID 6) volumes.
        mesoscope_data_directory: The path to the directory where the mesoscope saves acquired frame data. Usually, this
            directory is on the fast Data volume (NVME) and is cleared of data after each acquisition runtime.

    Attributes:
        _local: The absolute path to the host-machine animal directory for the managed project.
        _nas: The absolute path to the Synology NAS animal directory for the managed project.
        _server: The absolute path to the BioHPC server-machine animal directory for the managed project.
        _mesoscope: The absolute path to the mesoscope data directory.
        _project_name: Stores the name of the project whose data is currently manage by the class.
        _animal_name: Stores the name of the animal whose data is currently manage by the class.
        _session_name: Stores the name of the last generated session directory.

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

        # If requested, creates the directories on all destinations and locally
        if create:
            ensure_directory_exists(local_project_directory)
            ensure_directory_exists(nas_project_directory)
            ensure_directory_exists(server_project_directory)

            self._local = local_project_directory
            self._nas = nas_project_directory
            self._server = server_project_directory

        # If 'create' flag is off, raises an error if the target directory does not exist.
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

    def create_session(self) -> None:
        """Creates a new session directory within the broader project-animal data structure.

        Uses the current timestamp down to microseconds as the session folder name, which ensures that each session
        name within the project-animal structure has a unique name that accurately preserves the order of the sessions.

        Notes:
            Most other class methods require this method to be called at least once before they can be used.

            To retrieve the name of the generated session, use 'session_name' property. To retrieve the full path to the
            session raw_data directory, use 'session_path' property.'

        Returns:
            The Path to the root session directory.
        """
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

        if counter > 0:
            message = (
                f"Session name conflict occurred for animal '{self._animal_name}' of project '{self._project_name}' "
                f"when adding the new session. The newly created session directory uses a '_{counter}' postfix to "
                f"distinguish itself from the already existing session directory."
            )
            warnings.warn(message=message)

        # Creates the session directory and the raw_data subdirectory
        ensure_directory_exists(raw_session_path.joinpath("raw_data"))
        self._session_name = raw_session_path.stem

    @property
    def session_name(self) -> str:
        """Returns the name of the last generated session directory."""
        if self._session_name is None:
            message = (
                f"Unable to retrieve the last generates session name for the animal '{self._animal_name}' of project "
                f"'{self._project_name}'. Call create_session() method before using this property."
            )
            console.error(message=message, error=ValueError)
        return self._session_name

    @property
    def session_path(self) -> Path:
        """Returns the full path to the last generated session raw_data directory."""
        if self._session_name is None:
            message = (
                f"Unable to retrieve the 'raw_data' folder path for the last generated session of the animal "
                f"'{self._animal_name}' of project '{self._project_name}'. Call create_session() method before using "
                f"this property."
            )
            console.error(message=message, error=ValueError)
        return self._local.joinpath(self._session_name, "raw_data")

    def pull_mesoscope_data(self, num_threads: int = 28) -> None:
        """Pulls the frames acquired by the Mesoscope from the ScanImage PC to the raw_data directory of the last
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
        # The mesoscope path statically points to the mesoscope_frames folder. It is expected that the folder ONLY
        # contains the frames acquired during this session's runtime. First, generates the checksum for the raw
        # mesoscope frames
        calculate_directory_checksum(directory=self._mesoscope, num_processes=None, save_checksum=True)

        # Generates the path to the local session raw_data folder
        destination = self._local.joinpath(self._session_name, "raw_data")

        # Transfers the mesoscope frames data from the Mesoscope PC to the local machine.
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
            This method is configured to run data transfer and checksum calculation in-parallel where possible. It is
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
        source = self._local.joinpath(self._session_name, "raw_data")

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

        # If the method is configured to transfer files in-parallel, submits tasks to a ProcessPoolExecutor
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

        # Otherwise, runs the transfers sequentially. Note, transferring individual files is still done in-parallel, but
        # the transfer is performed for each destination sequentially.
        else:
            for destination in destinations:
                transfer_directory(source=source, destination=destination[0], num_threads=num_threads)

        # After all transfers complete successfully (including integrity verification), removes the source directory
        shutil.rmtree(source)


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
                pulse_duration=np.uint32(25000), calibration_delay=np.uint32(100000), calibration_count=np.uint16(500)
            )  # 25 milliseconds
            valve.calibrate()
        elif code == "c2":
            valve.set_parameters(
                pulse_duration=np.uint32(50000), calibration_delay=np.uint32(100000), calibration_count=np.uint16(500)
            )  # 50 milliseconds
            valve.calibrate()
        elif code == "c3":
            valve.set_parameters(
                pulse_duration=np.uint32(75000), calibration_delay=np.uint32(100000), calibration_count=np.uint16(500)
            )  # 75 milliseconds
            valve.calibrate()
        elif code == "c4":
            valve.set_parameters(
                pulse_duration=np.uint32(100000), calibration_delay=np.uint32(100000), calibration_count=np.uint16(500)
            )  # 100 milliseconds
            valve.calibrate()
        elif code == "c5":
            valve.set_parameters(
                pulse_duration=np.uint32(125000), calibration_delay=np.uint32(100000), calibration_count=np.uint16(500)
            )  # 125 milliseconds
            valve.calibrate()
        elif code == "o":
            valve.toggle(state=True)
        elif code == "c":
            valve.toggle(state=False)
        elif code == "t":
            valve.set_parameters(
                pulse_duration=np.uint32(17841), calibration_delay=np.uint32(100000), calibration_count=np.uint16(1000)
            )  # 4 ul x 1000 times
            valve.calibrate()
        elif code.isnumeric():
            pulse_duration = valve.get_duration_from_volume(float(code))
            print(pulse_duration)
            # valve.set_parameters(pulse_duration=pulse_duration)
            # valve.send_pulse()


def _screen_cli(screen: ScreenInterface, pulse_duration: int) -> None:
    """Exposes a console-based CLI that interfaces with the HDMI translator boards connected to all three VR screens."""
    while True:
        code = input()  # Sneaky UI
        if code == "q":
            break
        elif code == "t":  # Toggle
            screen.set_parameters(pulse_duration=np.uint32(pulse_duration))
            screen.toggle()


def calibration() -> None:
    # Output dir
    temp_dir = Path("/home/cybermouse/Desktop/TestOut")
    data_logger = DataLogger(output_directory=temp_dir, instance_name="amc", exist_ok=True)

    # Defines static assets needed for testing
    valve_calibration_data = (
        (0, 0),  # Need to add this logical constraint to prevent irrational computations, such as negative intercept
        (25000, 2.36),
        (50000, 6.44),
        (75000, 12.47),
        (100000, 17.85),
        (125000, 28.73),
    )
    actor_id = np.uint8(101)
    sensor_id = np.uint8(152)
    encoder_id = np.uint8(203)
    usb = "/dev/ttyACM0"

    # Add console support for print debugging
    console.enable()

    # Tested module interface
    module = EncoderInterface(debug=True)

    module_1 = TTLInterface(module_id=np.uint8(1), debug=True)
    module_2 = TTLInterface(module_id=np.uint8(2), debug=True)
    module_3 = BreakInterface(debug=True)
    module_4 = ValveInterface(valve_calibration_data=valve_calibration_data, debug=True)
    module_5 = ScreenInterface(initially_on=False, debug=True)

    # Tested AMC interface
    interface = MicroControllerInterface(
        controller_id=actor_id,
        data_logger=data_logger,
        module_interfaces=(module_4,),
        microcontroller_serial_buffer_size=8192,
        microcontroller_usb_port=usb,
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
    _valve_cli(module_4, 800000)
    # _screen_cli(module_5, 500000)

    # Shutdown
    interface.stop()
    data_logger.stop()

    data_logger.compress_logs(remove_sources=True)

    # Checks log parsing
    # stamps, water = module_4.parse_logged_data()
    #
    # print(f"Log data:")
    # print(f"Timestamps: {stamps}")
    # print(f"Water: {water}")


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
