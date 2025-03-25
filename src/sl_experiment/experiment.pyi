from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from pynput import keyboard
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_data_structures import DataLogger, SharedMemoryArray
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
    SessionData as SessionData,
    RunTrainingDescriptor as RunTrainingDescriptor,
    LickTrainingDescriptor as LickTrainingDescriptor,
    RuntimeHardwareConfiguration as RuntimeHardwareConfiguration,
    MesoscopeExperimentDescriptor as MesoscopeExperimentDescriptor,
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

    _state_map: dict[int, str]
    _started: bool
    descriptor: MesoscopeExperimentDescriptor
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

    _started: bool
    _lick_training: bool
    descriptor: LickTrainingDescriptor | RunTrainingDescriptor
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
