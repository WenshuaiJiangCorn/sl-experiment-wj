"""This module provides classes that abstract working with Sun lab's Mesoscope-VR system to acquire training or
experiment data or maintain the system modules. Primarily, this includes the runtime management classes
(highest level of internal data acquisition and processing API) and specialized runtime logic functions
(user-facing external API functions).
"""

import os
import copy
import json
from json import dumps
import shutil as sh
from pathlib import Path
import tempfile

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from sl_shared_assets import (
    SessionData,
    MesoscopePositions,
    ProjectConfiguration,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    MesoscopeHardwareState,
    MesoscopeExperimentDescriptor,
    MesoscopeExperimentConfiguration,
)
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, LogPackage
from ataraxis_time.time_helpers import convert_time, get_timestamp
from ataraxis_communication_interface import MQTTCommunication, MicroControllerInterface

from .tools import MesoscopeData, RuntimeControlUI, get_system_configuration
from .visualizers import BehaviorVisualizer
from .binding_classes import ZaberMotors, VideoSystems, MicroControllerInterfaces
from ..shared_components import WaterSheet, SurgerySheet, BreakInterface, ValveInterface, write_version_data
from .data_preprocessing import purge_failed_session, preprocess_session_data

# Statically defines the names used by supported session types to ensure that the name is used consistently across the
# entire module.
_experiment: str = "mesoscope experiment"
_run: str = "run training"
_lick: str = "lick training"
_window: str = "window checking"


class _MesoscopeVRSystem:
    """The base class for all Mesoscope-VR system runtimes.

    This class provides methods for conducting training and experiment sessions using the Mesoscope-VR system.
    It abstracts most low-level interactions with the mesoscope (2P-RAM) and Virtual Reality (VR) system components via
    a shared high-level state API.

    Notes:
        Calling this initializer only instantiates a minimal subset of all Mesoscope-VR assets. Use the start() method
        before issuing other commands to properly initialize all required runtime assets and remote processes.

        This class statically reserves the id code '1' to label its log entries. Make sure no other Ataraxis class,
        such as MicroControllerInterface or VideoSystem, uses this id code.

    Args:
        session_data: An initialized SessionData instance used to control the flow of data during acquisition and
            preprocessing. Each instance is initialized for the specific project, animal, and session combination for
            which the data is acquired.
        session_descriptor: A partially initialized LickTrainingDescriptor, RunTrainingDescriptor, or
            MesoscopeExperimentDescriptor instance. This instance is used to store session-specific information in a
            human-readable format.
        experiment_configuration: Only for experiment sessions. An initialized MesoscopeExperimentConfiguration instance
            that specifies experiment configuration and experiment state sequence. Keep this set to None for behavior
            training sessions.

    Attributes:
        _state_map: Maps the integer state-codes used to represent VR system states to human-readable string-names.
        _unity_termination_topic: Stores the MQTT topic used by Unity task environment to announce when it terminates
            the active game state (stops 'playing' the task environment).
        _cue_sequence_topic: Stores the MQTT topic used by Unity task environment to respond to the request of the VR
            wall cue sequence sent to the '_cue_sequence_request_topic'.
        _cue_sequence_request_topic: Stores the MQTT topic used to request the Unity virtual task to send the sequence
            of wall cues used during runtime. The data is sent to the topic specified by '_cue_sequence_topic'.
        _disable_guidance_topic: Stores the MQTT topic used to switch the virtual task into unguided mode (animal must
            lick to receive water).
        _enable_guidance_topic: Stores the MQTT topic used to switch the virtual task into guided mode (the water
            dispenses automatically when animal enters reward zone).
        _show_reward_zone_boundary_topic: Stores the MQTT topic used to show the reward zone collision boundary to
            the animal. The collision boundary is the typically hidden virtual wall the animal must collide with to
            trigger automated (guided) water reward delivery.
        _hide_reward_zone_boundary_topic: Stores the MQTT topic used to hide the reward zone collision boundary from
            the animal.
        _source_id: Stores the unique identifier code for this class instance. The identifier is used to mark log
            entries made by this class instance and has to be unique across all sources that log data at the same time,
            such as MicroControllerInterfaces and VideoSystems.
        _started: Tracks whether the VR system and experiment runtime are currently running. Primarily, this is used
            to support releasing hardware resources in the case of an unexpected runtime termination.
        _terminated: Tracks whether the user has terminated the runtime.
        _paused: Tracks whether the user has paused the runtime.
        descriptor: Stores the session descriptor instance of the managed session.
        _experiment_configuration: Stores the MesoscopeExperimentConfiguration instance of the managed session, if the
            session is of the 'mesoscope experiment' type.
        _session_data: Stores the SessionData instance of the managed session.
        _mesoscope_data: Stores the MesoscopeData instance of the managed session.
        _system_state: Stores the current state of the Mesoscope-VR system. The instance updates this value whenever it
            is instructed to change the system state.
        _runtime_state: Stores the stage of the managed runtime (experiment or training). Experiment states are defined
            by the user as part of the MesoscopeExperimentConfiguration data. Training runtimes have two states:
            0 (idle) and 255 (running).
        _timestamp_timer: A PrecisionTimer instance used to timestamp log entries generated by the class instance.
        _position: Stores the absolute position of the animal, in Unity units, used during the last communication cycle
            between this instance and Unity.
        _lick_count: Stores the lick count used during the last communication cycle between this class and Unity.
        _cue_sequence: Stores the Virtual Reality wall cue sequence used by the currently active Unity task environment.
        _unconsumed_reward_count: Tracks the number of rewards delivered to the animal without the animal
            consuming them.
        _enable_guidance: Determines whether the runtime is currently executed in the guided mode.
        _show_reward_zone_boundary: Determines whether the reward zone collision boundary is currently visible.
        _reward_volume: Stores the current volume of water dispensed to the animal as part of each reward delivery.
        _logger: Stores the DataLogger instance that collects behavior log data from all sources: microcontrollers,
            video cameras, and the MesoscopeExperiment instance.
        _microcontrollers: Stores the MicroControllerInterfaces instance that interfaces with all MicroController
            devices used during runtime.
        _cameras: Stores the VideoSystems instance that interfaces with video systems (cameras) used during
            runtime.
        _zaber_motors: Stores the ZaberMotors class instance that interfaces with HeadBar, LickPort, and Wheel motors.
        _ui: After start() method runtime, stores the RuntimeControl UI instance used during runtime to allow the user
            to interface with the Mesoscope-VR system via QT6 GUI.
        _visualizer: After start() method runtime, stores the BehaviorVisualizer instance used by this instance to
            visualize certain runtime data.
    """

    # Maps integer VR state codes to human-readable string-names.
    _state_map: dict[str, int] = {"idle": 0, "rest": 1, "run": 2, "training": 3}

    # Stores the names of MQTT topics that need to be monitored for incoming messages sent by Unity game engine.
    _unity_termination_topic: str = "Gimbl/Session/Stop"
    _cue_sequence_topic: str = "CueSequence/"
    _cue_sequence_request_topic: str = "CueSequenceTrigger/"
    _disable_guidance_topic: str = "MustLick/True/"
    _enable_guidance_topic: str = "MustLick/False/"
    _show_reward_zone_boundary_topic: str = "VisibleRewardWall/True/"
    _hide_reward_zone_boundary_topic: str = "VisibleRewardWall/False/"

    # Reserves logging source ID code 1 for this class
    _source_id: np.uint8 = np.uint8(1)

    def __init__(
        self,
        session_data: SessionData,
        session_descriptor: MesoscopeExperimentDescriptor,
        experiment_configuration: MesoscopeExperimentConfiguration | None = None,
    ) -> None:
        # Creates the _started flag first to avoid memory leaks if the initialization method fails.
        self._started: bool = False

        # Creates other important runtime flow control flags
        self._terminated: bool = False
        self._paused: bool = False

        # Caches SessionDescriptor and MesoscopeExperimentConfiguration instances to class attributes.
        self.descriptor: MesoscopeExperimentDescriptor = session_descriptor
        self._experiment_configuration: MesoscopeExperimentConfiguration | None = experiment_configuration

        # Caches the descriptor to disk. Primarily, this is required for preprocessing the data if the session runtime
        # terminates unexpectedly.
        self.descriptor.to_yaml(file_path=Path(session_data.raw_data.session_descriptor_path))

        # Saves SessionData to class attribute and uses it to initialize the MesoscopeData instance. MesoscopeData works
        # similar to SessionData, but only stores the paths used by the Mesoscope-VR system while managing the session's
        # data acquisition. These paths only exist on the VRPC filesystem.
        self._session_data: SessionData = session_data
        self._mesoscope_data: MesoscopeData = MesoscopeData(session_data)

        # Defines other flags used during runtime:
        # VR and runtime states are initialized to 0 (idle state) by default. Note, initial VR and runtime states are
        # logged when the instance first calls the idle() method as part of start() method runtime.
        self._system_state: int = 0
        self._runtime_state: int = 0
        self._timestamp_timer: PrecisionTimer = PrecisionTimer("us")  # A timer used to timestamp log entries

        # Initializes additional tracker variables used to cyclically handle various data updates during runtime.
        self._position: float = 0.0
        self._lick_count: int = 0
        self._cue_sequence: NDArray[np.uint8] = np.zeros(shape=(0,), dtype=np.uint8)
        self._unconsumed_reward_count: int = 0
        self._enable_guidance: bool = False
        self._show_reward_zone_boundary: bool = False
        self._reward_volume: float = 5.0

        # Initializes the DataLogger instance used to log data from all microcontrollers, camera frame savers, and this
        # class instance.
        self._logger: DataLogger = DataLogger(
            output_directory=Path(session_data.raw_data.raw_data_path),
            instance_name="behavior",  # Creates behavior_log subfolder under raw_data
            sleep_timer=0,
            exist_ok=True,
            process_count=1,
            thread_count=10,
        )

        # Initializes the binding class for all MicroController Interfaces.
        self._microcontrollers: MicroControllerInterfaces = MicroControllerInterfaces(data_logger=self._logger)

        # Instantiates an MQTTCommunication instance to directly communicate with Unity. Currently, ALL Unity
        # communication is carried out through this instance.
        monitored_topics = (
            self._cue_sequence_topic,  # Used as part of Unity (re) initialization
            self._unity_termination_topic,  # Used to detect Unity shutdown events
            self._microcontrollers.valve.mqtt_topic,  # Allows Unity to operate the valve
        )

        # Only initializes MQTTCommunication instance if the managed runtime is a mesoscope experiment
        self._unity: MQTTCommunication | None
        if self._session_data.session_type == _experiment:
            self._unity = MQTTCommunication(monitored_topics=monitored_topics)
        else:
            self._unity = None

        # Initializes the binding class for all VideoSystems.
        self._cameras: VideoSystems = VideoSystems(
            data_logger=self._logger,
            output_directory=Path(self._session_data.raw_data.camera_data_path),
        )

        # While we can connect to ports managed by ZaberLauncher, ZaberLauncher cannot connect to ports managed via
        # software. Therefore, we have to make sure ZaberLauncher is running before connecting to motors.
        message = (
            "Preparing to connect to all Zaber motor controllers. Make sure that ZaberLauncher app is running before "
            "proceeding further. If ZaberLauncher is not running, you WILL NOT be able to manually control Zaber motor "
            "positions until you reset the runtime."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        self._zaber_motors: ZaberMotors = ZaberMotors(
            zaber_positions_path=self._mesoscope_data.vrpc_persistent_data.zaber_positions_path
        )

        # Defines critical runtime assets initialized during start() method runtime.
        self._ui: RuntimeControlUI | None = None
        self._visualizer: BehaviorVisualizer | None = None

    def start(self) -> None:
        """Initializes and configures all microcontrollers, cameras, and Zaber motors used during the runtime.

        This method establishes the communication with the microcontrollers, data logger cores, and video system
        processes. It also executes the runtime preparation sequence, which includes positioning all Zaber motors
        to support the runtime and generating data files that store runtime metadata. Critically, it does not
        initialize external runtime assets, such as the Mesoscope and Unity game engine, as these tasks are carried
        out by the dedicated start_runtime() method.

        Notes:
            This method will not run unless the host PC has access to the necessary number of logical CPU cores
            and other required hardware resources (GPU for video encoding, etc.). This prevents using the class on
            machines that are unlikely to sustain the runtime requirements.

            As part of its runtime, this method attempts to set all Zaber motors to the optimal runtime position for the
            participating animal. Exercise caution and always monitor the system when it is running this method,
            as motor motion can damage the mesoscope or harm the animal. It is the responsibility of the user to ensure
            that the execution of all Zaber motor commands is safe.

        Raises:
            RuntimeError: If the host PC does not have enough logical CPU cores available.
        """
        # Prevents (re) starting an already started VR process.
        if self._started:
            return

        # 3 cores for microcontrollers, 1 core for the data logger, 6 cores for the current video_system
        # configuration (3 producers, 3 consumer), 1 core for the central process calling this method, 1 core for
        # the main GUI: 12 cores total. Note, the system may use additional cores if they are requested from various
        # C / C++ extensions used by our source code.
        cpu_count = os.cpu_count()
        if cpu_count is None or not cpu_count >= 12:
            message = (
                f"Unable to start the Mesoscope-VR system runtime. The host PC must have at least 12 logical CPU "
                f"cores available for this runtime to work as expected, but only {cpu_count} cores are "
                f"available."
            )
            console.error(message=message, error=RuntimeError)

        message = "Initializing Mesoscope-VR system assets..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts the data logger
        self._logger.start()

        # Generates and logs the onset timestamp for the VR system as a whole. The MesoscopeVRSystem class generates
        # log entries similar to other data acquisition classes, so it too requires a temporal reference point.

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

        # Begins acquiring and displaying frames with the face camera. Does not start body cameras and does not save
        # face camera frames to disk at this time to conserve disk space.
        self._cameras.start_face_camera()

        # Starts all microcontroller interfaces
        self._microcontrollers.start()

        # If necessary, carries out the Zaber motor setup and animal mounting sequence. Once this call returns, the
        # runtime assumes that the animal is mounted in the mesoscope enclosure.
        self._setup_zaber_motors()

        # Generates a snapshot of all zaber motor positions. This serves as an early checkpoint in case the runtime has
        # to be aborted in a non-graceful way (without running the stop() sequence). This way, next runtime will restart
        # with the calibrated zaber positions. The snapshot includes any adjustment to the HeadBar positions performed
        # during red-dot alignment.
        self._generate_zaber_snapshot()

        # Generates a snapshot of the runtime hardware configuration. In turn, this data is used to parse the .npz log
        # files during processing.
        self._generate_hardware_state_snapshot()

        # Saves the MesoscopeExperimentConfiguration instance to the session folder.
        if self._experiment_configuration is not None:
            self._experiment_configuration.to_yaml(Path(self._session_data.raw_data.experiment_configuration_path))
            message = "Experiment configuration snapshot: Generated."
            console.echo(message=message, level=LogLevel.SUCCESS)

        # Sets the runtime into the Idle state before instructing the user to finalize runtime preparations.
        self.idle()

        # The setup procedure is complete.
        self._started = True

        message = "Mesoscope-VR system: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def start_runtime(self) -> None:

        # Makes it impossible to start the runtime unless class assets have been initialized (started).
        if not self._started:
            message = (
                "Unable to start the Mesoscope-VR system runtime as runtime assets have not been initialized. Call the "
                "MesoscopeVRSystem class start() method before calling this method."
            )
            console.error(message=message, error=RuntimeError)
            return  # Fallback to appease mypy

        # Initializes a second-precise timer used to enforce various delays used by the
        delay_timer = PrecisionTimer("s")

        # Starts acquiring data from body cameras. Does not start camera frame saving at this point to avoid generating
        # unnecessary data
        self._cameras.start_body_cameras()

        # Initializes the runtime control UI.
        self._ui = RuntimeControlUI()

        # Initializes the runtime visualizer. This HAS to be initialized after cameras and the UI to prevent collisions
        # in the QT backend, which is used by all three assets.
        self._visualizer = BehaviorVisualizer(
            lick_tracker=self._microcontrollers.lick_tracker,
            valve_tracker=self._microcontrollers.valve_tracker,
            distance_tracker=self._microcontrollers.distance_tracker,
        )

        # Queries certain runtime configuration parameters from GUI
        self._enable_guidance = self._ui.enable_guidance
        self._show_reward_zone_boundary = self._ui.show_reward

        # Initializes external assets. Currently, these assets are only used as part of the experiment runtime, so this
        # section is skipped for all other runtime types.
        if self._session_data.session_type == _experiment:

            if self._unity is not None:
                # Establishes communication with the MQTT broker.
                self._unity.connect()

                # Instructs the user to configure the VR scene and verify that it properly interfaces with the VR
                # screens.
                self._setup_unity()

                # Queries the task cue (segment) sequence from Unity. This also acts as a check for whether Unity is
                # running and is configured appropriately. The extracted sequence data is logged as a sequence of byte
                # values.
                self._get_cue_sequence()

            # Configures the VR task to match GUI state
            self._toggle_lick_guidance(enable_guidance=self._enable_guidance)
            self._toggle_show_reward(show_reward=self._show_reward_zone_boundary)

            # Instructs the user to prepare the mesoscope for data acquisition.
            self._setup_mesoscope()

        # Final manual checkpoint
        message = (
            f"Runtime preparation: Complete. Carry out all final checks and adjustments, such as priming the water "
            f"delivery valve. When you are ready to start the runtime, use the UI to 'resume' it."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)

        # At this point, the user can use the GUI and the Zaber UI to freely manipulate all components of the
        # mesoscope-VR system.
        while self._ui.pause_runtime:
            if self._ui.reward_signal:
                self.deliver_reward(reward_size=self._ui.reward_volume)

            if self._ui.open_valve:
                self._microcontrollers.open_valve()

            if self._ui.close_valve:
                self._microcontrollers.close_valve()

            # Switches the guidance status in response to user requests
            if self._ui.enable_guidance != self._enable_guidance:
                self._enable_guidance = self._ui.enable_guidance
                self._toggle_lick_guidance(enable_guidance=self._enable_guidance)

            # Switches the reward boundary visibility in response to user requests
            if self._ui.show_reward != self._show_reward_zone_boundary:
                self._show_reward_zone_boundary = self._ui.show_reward
                self._toggle_show_reward(show_reward=self._show_reward_zone_boundary)

        # Ensures the valve is closed before continuing.
        self._microcontrollers.close_valve()

        # Resets the valve tracker array before proceeding. This allows the user to use the section above to debug the
        # valve, potentially dispensing a lot of water in the process. If the tracker is not reset, this may immediately
        # terminate the runtime and lead to an inaccurate tracking of the water volume received by the animal.
        self._microcontrollers.reset_valve_tracker()

        message = f"Initiating the runtime..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts saving frames from al cameras
        self._cameras.save_face_camera_frames()
        self._cameras.save_body_camera_frames()

        # Starts mesoscope frame acquisition if the runtime is a mesoscope experiment.
        if self._session_data.session_type == _experiment:
            # Enables mesoscope frame monitoring
            self._microcontrollers.enable_mesoscope_frame_monitoring()
            delay_timer.delay_noblock(delay=1)  # Ensures that the frame monitoring starts before acquisition.

            # Starts mesoscope frame acquisition
            self._start_mesoscope()

        message = f"Runtime: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops and terminates all Mesoscope-VR system components, external assets, and ends the runtime.

        This method releases the hardware resources used during runtime by various system components by triggering
        appropriate graceful shutdown procedures for all components. Then, it generates a set of files that store
        various runtime metadata. Finally, it calls the data preprocessing pipeline to efficiently package the data and
        safely transfer it ot the long-term storage destinations.
        """

        # Prevents stopping an already stopped process.
        if not self._started:
            return

        # Resets the _started tracker
        self._started = False

        message = "Terminating Mesoscope-VR system runtime..."
        console.echo(message=message, level=LogLevel.INFO)

        # Shuts down the UI and the visualizer, if these assets are used by the managed runtime
        if self._ui is not None:
            self._ui.shutdown()

        if self._visualizer is not None:
            self._visualizer.close()

        # Switches the system into the IDLE state. Since IDLE state has most modules set to stop-friendly states,
        # this is used as a shortcut to prepare the VR system for shutdown. Also, this clearly marks the end of the
        # main runtime period.
        self.idle()

        # Stops all cameras.
        self._cameras.stop()

        # Stops mesoscope frame acquisition and monitoring, if the runtime is a Mesoscope experiment.
        if self._session_data.session_type == _experiment:
            self._stop_mesoscope()
            self._microcontrollers.disable_mesoscope_frame_monitoring()

        # Stops all microcontroller interfaces
        self._microcontrollers.stop()

        # Stops the data logger instance
        self._logger.stop()

        message = "Data Logger: Stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Updates internally stored SessionDescriptor instance with runtime data, saves it to disk, and instructs the
        # user to add experimenter notes and other user-defined information to the descriptor file.
        self._generate_session_descriptor()

        # For Mesoscope experiment runtimes, generates the snapshot of the current mesoscope objective position. This
        # has to be done before the objective is lifted to remove the animal from the Mesoscope enclosure. This data
        # is reused during the following experiment session to restore the imaging field to the same state as during
        # this session.
        if self._session_data.session_type == _experiment:
            self._generate_mesoscope_position_snapshot()

        # Generates the snapshot of the current Zaber motor positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are potentially reset back to parking position.
        self._generate_zaber_snapshot()

        # Optionally resets Zaber motors by moving them to the dedicated parking position before shutting down Zaber
        # connection.
        self._reset_zaber_motors()

        # Disconnects from Zaber motors. This does not change motor positions, but does lock (park) all motors before
        # disconnecting.
        self._zaber_motors.disconnect()

        # Notifies the user that the acquisition is complete.
        console.echo(message=f"Data acquisition: Complete.", level=LogLevel.SUCCESS)

        # Determines whether to carry out data preprocessing or purging.
        message = (
            f"Do you want to carry out data preprocessing or purge the data? CRITICAL! Only enter 'purge_session' if "
            f"you want to permanently DELETE the session data. All valid data REQUIRES preprocessing to ensure safe "
            f"storage."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while True:
            answer = input("Enter 'yes', 'no' or 'purge_session': ")

            # Default case: preprocesses the data. For experiment runtimes, this may take between 15 and 20 minutes.
            if answer.lower() == "yes":
                preprocess_session_data(session_data=self._session_data)
                break

            # Does not carry out data preprocessing or purging. In certain scenarios, it may be necessary to skip data
            # preprocessing in favor of faster animal turnover. Although highly discouraged, this is nonetheless a valid
            # runtime termination option.
            elif answer.lower() == "no":
                break

            # Exclusively for failed runtimes: removes all session data from all destinations.
            elif answer.lower() == "purge_session":
                purge_failed_session(session_data=self._session_data)
                break

        message = "Mesoscope-VR system runtime: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _setup_zaber_motors(self) -> None:
        """If necessary, carries out the Zaber motor setup and positioning sequence.

        This method is used as part of the start() method execution for most runtimes to prepare the Zaber motors for
        the specific animal that participates in the runtime.
        """

        # Determines whether to carry out the Zaber motor positioning sequence.
        message = (
            f"Do you want to carry out the Zaber motor setup sequence for this animal? Only enter 'no' if the animal "
            f"is already positioned inside the Mesoscope enclosure."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Blocks until a valid answer is received from the user
        while True:
            answer = input("Enter 'yes' or 'no': ")

            if answer.lower() == "no":
                # Aborts method runtime, as no further Zaber setup is required.
                return

            if answer.lower() == "yes":
                # Proceeds with the setup sequence.
                break

        # Since it is now possible to shut down Zaber motors without fixing HeadBarRoll position, requests the user
        # to verify this manually
        message = (
            "Check that the HeadBarRoll motor has a positive (>0) angle. If the angle is negative (<0), the motor will "
            "collide with the stopper during homing, which will DAMAGE the motor."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is
        # safe to proceed with motor movements.
        message = (
            "Preparing to move Zaber motors into mounting position. Remove the mesoscope objective, swivel out the "
            "VR screens, and make sure the animal is NOT mounted on the rig."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not
        # intersect with each other, so it is safe to move all motor assemblies at the same time.
        self._zaber_motors.prepare_motors()

        # Sets the motors into the mounting position. The HeadBar and Wheel are either restored to the previous
        # session's position or are set to the default mounting position stored in non-volatile memory. The
        # LickPort is moved to a position optimized for putting the animal on the VR rig (positioned away from the
        # HeadBar).
        self._zaber_motors.mount_position()

        message = "Motor Positioning: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Gives user time to mount the animal and requires confirmation before proceeding further.
        message = (
            "Preparing to move the motors into the imaging position. Mount the animal onto the VR rig. Do NOT "
            "adjust any motors manually at this time. Do NOT install the mesoscope objective."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Primarily, this restores the LickPort to the previous session's position or default parking position. The
        # HeadBar and Wheel should not move, as they are already 'restored'. However, if the user did move them
        # manually, they too will be restored to default positions.
        self._zaber_motors.restore_position()

        message = "Motor Positioning: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _reset_zaber_motors(self) -> None:
        """Optionally resets Zaber motors back to the hardware-defined parking positions.

        This method is called as part of the stop() method runtime to achieve two major goals. First, it facilitates
        removing the animal from the Mesoscope enclosure by retracting the lick-port away from its head. Second, it
        positions Zaber motors in a way that ensures that the motors can be safely homed after potential power cycling.
        """

        # Determines whether to carry out the Zaber motor shutdown sequence.
        message = (
            f"Do you want to carry out Zaber motor shutdown sequence? If ending a successful runtime, enter 'yes'. "
            f"Entering 'yes' does NOT itself move any motors, so it is SAFE. If terminating a failed runtime to "
            f"restart it, enter 'no'."
        )
        console.echo(message=message, level=LogLevel.INFO)
        while True:
            answer = input("Enter 'yes' or 'no': ")

            # Continues with the rest of the shutdown runtime
            if answer.lower() == "yes":
                break

            # Ends the runtime, as there is no need to move Zaber motors.
            elif answer.lower() == "no":
                return

        # Helps with removing the animal from the enclosure by retracting the lick-port in the Y-axis (moving it away
        # from the animal).
        message = f"Retracting the lick-port away from the animal..."
        console.echo(message=message, level=LogLevel.INFO)

        self._zaber_motors.unmount_position()

        message = "Motor Positioning: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)

        message = "Uninstall the mesoscope objective and REMOVE the animal from the VR rig."
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        self._zaber_motors.park_position()

        message = "Zaber motors: Reset."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _setup_unity(self) -> None:
        """Pauses runtime until Unity sends a termination message and instructs the user to check that the virtual
        environment displays on the VR screens as intended.

        This method is used to ensure that the user starts and stops Unity game engine at least once. This forces each
        user to check that their virtual task properly displays on the VR screens to avoid issues with experiment
        runtimes that rely on delivering visual feedback to the animal via the VR screens.

        Raises:
            RuntimeError: If Unity does not send a termination message within 10 minutes of this runtime starting and
                the user chooses to abort the runtime.
        """

        # Does not do anything if Unity communication class is not initialized.
        if self._unity is None:
            return

        # Activates the VR screens so that the user can check whether the Unity task displays as expected.
        self._microcontrollers.enable_vr_screens()

        # Delays the runtime for 2 seconds to ensure that the VR screen controllers receive the activation pulse and
        # activate the screens before prompting the user to cycle Unity task states.
        delay_timer = PrecisionTimer("s")
        delay_timer.delay_noblock(delay=2)

        # Discards all data received from Unity up to this point. This is done to ensure that the user carries out the
        # power up and shutdown cycle as instructed and avoids any previous cycles cached in communication class memory
        # inadvertently aborting the runtime.
        while self._unity.has_data:
            _ = self._unity.get_data()

        # Instructs the user to check the displays.
        message = (
            f"Start Unity game engine and load your experiment task (scene). Ensure that the task properly displays on "
            f"the VR screens inside the Mesoscope enclosure. The runtime will not advance until the task is started "
            f"(played) and STOPPED at least once."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Blocks until Unity sends the task termination message or until the user manually aborts the runtime.
        outcome = ""
        while outcome != "abort":
            # Blocks for at most 10 minutes at a time.
            delay_timer.reset()
            while delay_timer.elapsed < 600:
                # Parses all data received from Unity game engine.
                if self._unity.has_data:
                    topic: str
                    topic, _ = self._unity.get_data()

                    # If received data is a termination message, breaks the loop
                    if topic == self._unity_termination_topic:
                        # Disables the VR screens before returning.
                        self._microcontrollers.disable_vr_screens()

                        message = "VR Display Check: Passed."
                        console.echo(message=message, level=LogLevel.SUCCESS)

                        # Instructs the user to restart the task (re-arm Unity).
                        message = (
                            f"Arm (start) the Virtual Reality task. Do not stop the task until the end of the runtime."
                        )
                        console.echo(message=message, level=LogLevel.INFO)
                        input("Enter anything to continue: ")

                        return

            # If the loop above is escaped, this is due to not receiving any message from Unity for 10 minutes.
            message = (
                f"The Mesoscope-VR system did not receive a Unity runtime termination message after waiting for 10 "
                f"minutes. It is likely that the Unity game engine is not running or is not configured to work with "
                f"Mesoscope-VR system. Make sure Unity game engine is started and configured before continuing."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            outcome = input("Enter 'abort' to abort with an error. Enter anything else to retry: ").lower()

        message = f"Runtime aborted due to user request."
        console.error(message=message, error=RuntimeError)
        raise RuntimeError(message)  # Fallback to appease mypy, should not be reachable

    def _setup_mesoscope(self) -> None:
        """Prompts the user to carry out steps to prepare the mesoscope for acquiring brain activity data.

        This method is used as part of the start_runtime() method execution for experiment runtimes to ensure the
        mesoscope is ready for data acquisition. It guides the user through all mesoscope preparation steps and, at
        the end of runtime, ensures the mesoscope is ready for acquisition.
        """

        # Step 0: Prepare ScanImagePC and laser
        message = (
            f"Launch ScanImage library on the ScanImagePC by calling 'scanimage' in Matlab command line interface and "
            f"activate the laser. Critically, make sure that the mesoscope has the 'external triggers' checkbox "
            f"enabled."
        )
        console.echo(message=message, level=LogLevel.INFO)
        input("Enter anything to continue: ")

        # Step 1: Find the imaging plane and confirm there are no bubbles
        # If previous session's mesoscope positions were saved, loads the objective coordinates and uses them to
        # augment the message to the user.
        if Path(self._mesoscope_data.vrpc_persistent_data.mesoscope_positions_path).exists():
            previous_positions: MesoscopePositions = MesoscopePositions.from_yaml(  # type: ignore
                file_path=Path(self._mesoscope_data.vrpc_persistent_data.mesoscope_positions_path)
            )
            # Gives user time to mount the animal and requires confirmation before proceeding further.
            message = (
                f"Attach the light-shield to the ring of the headbar, install the mesoscope objective, clean the "
                f"window, and find the imaging plane. Do NOT tape the light-shield until you find the imaging "
                f"plane and confirm no bubbles are present above the imaging plane. "
                f"Previous mesoscope coordinates were: x={previous_positions.mesoscope_x}, "
                f"y={previous_positions.mesoscope_y}, roll={previous_positions.mesoscope_roll}, "
                f"z={previous_positions.mesoscope_z}, fast_z={previous_positions.mesoscope_fast_z}, "
                f"tip={previous_positions.mesoscope_tip}, tilt={previous_positions.mesoscope_tilt}."
            )
        else:
            # While it is somewhat unlikely that imaging plane is not established at this time, this is not impossible.
            # In that case, instructs the user to choose the imaging plane instead of restoring it to the state used
            # during the previous runtime.
            message = (
                f"Attach the light-shield to the ring of the headbar, install the mesoscope objective, clean the "
                f"window, and choose the imaging plane. Do NOT tape the light-shield until you confirm no bubbles "
                f"are present above the chosen imaging plane. Note, all future imaging sessions will use the same "
                f"imaging plane as established during this runtime!"
            )
        console.echo(message=message, level=LogLevel.INFO)
        input("Enter anything to continue: ")

        # Step 2: Carry out the red-dot alignment process
        message = (
            "Withdraw the objective ~1000 um away from the imaging plane and tape up the light shield. Confirm that "
            "there are no bubbles present above the imaging plane and carry out the red-dot alignment. Then, move the "
            "objective back to the imaging plane and re-align it to the reference MotionEstimator file state (if one "
            "exists)."
        )
        console.echo(message=message, level=LogLevel.INFO)
        input("Enter anything to continue: ")

        # Step 3: Generate the screenshot of the red-dot alignment and the cranial window
        message = (
            "Generate the screenshot of the red-dot alignment, the imaging plane state (cell activity), and the "
            "ScanImage acquisition parameters using 'Win + PrtSc' combination."
        )
        console.echo(message=message, level=LogLevel.INFO)
        input("Enter anything to continue: ")

        # Forces the user to create the dot-alignment and cranial window screenshot on the ScanImage PC before
        # continuing.
        screenshots = [
            screenshot for screenshot in Path(self._mesoscope_data.scanimagepc_data.meso_data_path).glob("*.png")
        ]
        while len(screenshots) != 1:
            message = (
                f"Unable to retrieve the screenshot from the ScanImage PC. Specifically, expected a single .png file "
                f"to be stored in the root mesoscope data (mesodata) folder of the ScanImagePC, but instead found "
                f"{len(screenshots)} candidates. If multiple candidates are present, remove any extra screenshots "
                f"stored in the folder before proceeding."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to continue: ")
            screenshots = [
                screenshot for screenshot in Path(self._mesoscope_data.scanimagepc_data.meso_data_path).glob("*.png")
            ]

        # Transfers the screenshot to the mesoscope_frames folder of the session's raw_data folder
        screenshot_path = Path(self._session_data.raw_data.window_screenshot_path)
        source_path: Path = screenshots[0]
        sh.move(source_path, screenshot_path)  # Moves the screenshot from the ScanImagePC to the VRPC

        # Also saves the screenshot to the animal's persistent data folder, so that it can be reused during the next
        # runtime.
        sh.copy2(screenshot_path, self._mesoscope_data.vrpc_persistent_data.window_screenshot_path)

        # Step 4: Generate the new MotionEstimator file and arm mesoscope for acquisition
        message = (
            "Create a new MotionEstimator file by calling the 'setupZstackALL' command in Matlab command line "
            "interface. Use the Motion Estimation user interface to save the newly generated MotionEstimator file to "
            "the mesoscope_data folder. Then, arm the mesoscope for acquisition by activating the 'Loop' mode."
        )
        console.echo(message=message, level=LogLevel.INFO)
        input("Enter anything to continue: ")

        console.echo(message="Mesoscope Preparation: Complete.", level=LogLevel.SUCCESS)

    def _generate_zaber_snapshot(self) -> None:
        """Creates a snapshot of current Zaber motor positions and saves them to the session raw_dat folder and the
        persistent folder of the animal that participates in the runtime."""

        # Generates the snapshot
        zaber_positions = self._zaber_motors.generate_position_snapshot()

        # Saves the newly generated file both to the persistent folder and to the session folder. Note, saving to the
        # persistent data directory automatically overwrites any existing positions file.
        zaber_positions.to_yaml(file_path=Path(self._mesoscope_data.vrpc_persistent_data.zaber_positions_path))
        zaber_positions.to_yaml(file_path=Path(self._session_data.raw_data.zaber_positions_path))

        message = "Zaber motor positions: Saved."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _generate_hardware_state_snapshot(self) -> None:
        """Resolves and generates the snapshot of hardware configuration parameters used by the Mesoscope-VR system
        modules.

        This method determines which modules are used by the executed runtime (session) type and caches them into
        a HardwareStates object stored inside the session's raw_data folder.
        """

        # Experiment runtimes use all available hardware modules
        if self._experiment_configuration:
            hardware_state = MesoscopeHardwareState(
                cue_map=self._experiment_configuration.cue_map,
                cm_per_pulse=float(self._microcontrollers.wheel_encoder.cm_per_pulse),
                maximum_break_strength=float(self._microcontrollers.wheel_break.maximum_break_strength),
                minimum_break_strength=float(self._microcontrollers.wheel_break.minimum_break_strength),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                valve_scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                valve_nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
                torque_per_adc_unit=float(self._microcontrollers.torque.torque_per_adc_unit),
                screens_initially_on=self._microcontrollers.screens.initially_on,
                recorded_mesoscope_ttl=True,
            )
        # Lick training runtimes use a subset of hardware, including torque sensor
        elif self._session_data.session_type == _lick:
            hardware_state = MesoscopeHardwareState(
                torque_per_adc_unit=float(self._microcontrollers.torque.torque_per_adc_unit),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                valve_scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                valve_nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
            )
        # Run training runtimes use the same subset of hardware as the rest training runtime, except instead of torque
        # sensor, they monitor the encoder.
        elif self._session_data.session_type == _run:
            hardware_state = MesoscopeHardwareState(
                cm_per_pulse=float(self._microcontrollers.wheel_encoder.cm_per_pulse),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                valve_scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                valve_nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
            )
        else:
            # It should be impossible to satisfy this error clause, but is kept for safety reasons
            message = (
                f"Unsupported session type: {self._session_data.session_type} encountered when resolving the "
                f"Mesoscope-VR runtime hardware state."
            )
            console.error(message=message, error=ValueError)
            raise ValueError(message)  # A fall-back to appease mypy

        # Caches the hardware state to disk
        hardware_state.to_yaml(Path(self._session_data.raw_data.hardware_state_path))
        message = "Mesoscope-VR hardware state snapshot: Generated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _generate_mesoscope_position_snapshot(self) -> None:
        """Generates a precursor mesoscope_positions.yaml file and optionally forces the user to update it to reflect
        the current Mesoscope objective position coordinates.

        This utility method is used during the stop() method runtime to generate a snapshot of Mesoscope objective
        positions that will be reused during the next session to restore the imaging field.
        """

        # Generates a precursor MesoscopePositions file and dumps it to the session raw_data folder.
        # If a previous set of mesoscope position coordinates is available, overwrites the 'default' mesoscope
        # coordinates with the positions loaded from the snapshot stored inside the persistent_data folder of the
        # animal.
        force_mesoscope_positions_update: bool = False
        if Path(self._mesoscope_data.vrpc_persistent_data.mesoscope_positions_path).exists():
            sh.copy(
                self._mesoscope_data.vrpc_persistent_data.mesoscope_positions_path,
                self._session_data.raw_data.mesoscope_positions_path,
            )
            # Loads the previous position data into memory
            previous_mesoscope_positions: MesoscopePositions = MesoscopePositions.from_yaml(  # type: ignore
                file_path=self._session_data.raw_data.mesoscope_positions_path
            )

            # Asks the user whether they want to update position data. If not, then there is no need to update the data
            # inside the precursor .YAML file.
            if not force_mesoscope_positions_update:
                message = (
                    f"Do you want to update the mesoscope objective position data stored inside the "
                    f"mesoscope_positions.yaml file loaded from the previous session?"
                )
                console.echo(message=message, level=LogLevel.INFO)
                while True:
                    answer = input("Enter 'yes' or 'no': ")

                    # If the answer is 'yes', breaks the loop and executes the data update sequence.
                    if answer.lower() == "yes":
                        break

                    # If the answer is 'no', ends method runtime,a s there is no need to update the data inside the
                    # file.
                    elif answer.lower() == "no":
                        return

        # If previous position data is not available, creates a new MesoscopePositions instance with default (0)
        # position values.
        else:
            previous_mesoscope_positions = MesoscopePositions()

            # Caches the precursor file to the raw_data session directory.
            previous_mesoscope_positions.to_yaml(file_path=Path(self._session_data.raw_data.mesoscope_positions_path))

        # Notifies the user that the precursor file is ready for modification.
        message = "Mesoscope objective position precursor: Created."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Forces the user to update the mesoscope positions file with current mesoscope data
        message = (
            "Update the data inside the mesoscope_positions.yaml file stored inside the raw_data session directory "
            "to reflect the current mesoscope objective position."
        )
        console.echo(message=message, level=LogLevel.INFO)
        input("Enter anything to continue: ")

        # Reads the current mesoscope positions data cached inside the session's mesoscope_positions.yaml file into
        # memory.
        mesoscope_positions: MesoscopePositions = MesoscopePositions.from_yaml(  # type: ignore
            file_path=Path(self._session_data.raw_data.mesoscope_positions_path),
        )

        # Ensures that the user has updated the position data.
        while (
            mesoscope_positions.mesoscope_x == previous_mesoscope_positions.mesoscope_x
            and mesoscope_positions.mesoscope_y == previous_mesoscope_positions.mesoscope_y
            and mesoscope_positions.mesoscope_z == previous_mesoscope_positions.mesoscope_z
            and mesoscope_positions.mesoscope_roll == previous_mesoscope_positions.mesoscope_roll
            and mesoscope_positions.mesoscope_fast_z == previous_mesoscope_positions.mesoscope_fast_z
            and mesoscope_positions.mesoscope_tip == previous_mesoscope_positions.mesoscope_tip
            and mesoscope_positions.mesoscope_tilt == previous_mesoscope_positions.mesoscope_tilt
        ):
            message = (
                "Failed to verify that the mesoscope_positions.yaml file stored inside the session raw_data "
                "directory has been updated to include the mesoscope objective positions used during runtime. "
                "Manually edit the mesoscope_positions.yaml file to update the position fields with coordinates "
                "displayed in ScanImage software or ThorLabs pad. Make sure to save the changes to the file by "
                "using 'CTRL+S' combination."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to continue: ")

            # Reloads the positions file each time to ensure positions have been modified.
            mesoscope_positions: MesoscopePositions = MesoscopePositions.from_yaml(  # type: ignore
                file_path=Path(self._session_data.raw_data.mesoscope_positions_path),
            )

        # Copies the updated mesoscope positions data into the animal's persistent directory.
        sh.copy2(
            src=self._session_data.raw_data.mesoscope_positions_path,
            dst=self._mesoscope_data.vrpc_persistent_data.mesoscope_positions_path,
        )

    def _generate_session_descriptor(self) -> None:
        """Updates the contents of the locally stored session descriptor file with runtime data and caches it to
        session's raw_data directory.

        This utility method is used as part of the stop() method runtime to generate the session_descriptor.yaml file.
        Since this file combines both runtime-generated and user-generated data, this method also ensures that the
        user updates the descriptor file to include experimenter notes taken during runtime.
        """

        # Updates the contents of the pregenerated descriptor file and dumps it as a .yaml into the root raw_data
        # session directory. This needs to be done after the microcontrollers and loggers have been stopped to ensure
        # that the reported dispensed_water_volume_ul is accurate.
        delivered_water = self._microcontrollers.total_delivered_volume

        # Note, although the class supports various descriptor file formats, the data written by this method is uniform
        # (shared) by all supported descriptor types.

        # Overwrites the delivered water volume with the volume recorded over the runtime.
        self.descriptor.dispensed_water_volume_ml = round(delivered_water / 1000, ndigits=3)  # Converts from uL to ml
        self.descriptor.incomplete = False  # If the runtime reaches this point, the session is likely complete.

        # Dumps the updated descriptor as a .yaml, so that the user can edit it with user-generated data.
        self.descriptor.to_yaml(file_path=Path(self._session_data.raw_data.session_descriptor_path))
        console.echo(message=f"Session descriptor precursor file: Created.", level=LogLevel.SUCCESS)

        # Prompts the user to add their notes to the appropriate section of the descriptor file. This has to be done
        # before processing so that the notes are properly transferred to long-term storage destinations.
        message = (
            f"Open the session descriptor file stored in the session's raw_data folder and update it with the notes "
            f"taken during runtime."
        )
        console.echo(message=message, level=LogLevel.INFO)
        input("Enter anything to continue: ")

        # Verifies and blocks in-place until the user updates the session descriptor file with experimenter notes.
        self.descriptor = self.descriptor.from_yaml(  # type: ignore
            file_path=Path(self._session_data.raw_data.session_descriptor_path)
        )
        while "Replace this with your notes." in self.descriptor.experimenter_notes:
            message = (
                "Failed to verify that the session_descriptor.yaml file stored inside the session raw_data directory "
                "has been updated to include experimenter notes. Manually edit the session_descriptor.yaml file and "
                "replace the default text under the 'experimenter_notes' field with the notes taken during the "
                "runtime. Make sure to save the changes to the file by using 'CTRL+S' combination."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to continue: ")

            # Reloads the descriptor from disk each time to ensure experimenter notes have been modified.
            self.descriptor = self.descriptor.from_yaml(  # type: ignore
                file_path=Path(self._session_data.raw_data.session_descriptor_path),
            )

        # If the descriptor has passed the verification, backs it up to the animal's persistent directory. This is a
        # feature primarily used during training to restore the training parameters between training sessions of the
        # same type. The MesoscopeData resolves the paths to the persistent descriptor files in a way that
        # allows to keep a copy of each supported descriptor without interfering with other descriptor types.
        sh.copy2(
            src=self._session_data.raw_data.session_descriptor_path,
            dst=self._mesoscope_data.vrpc_persistent_data.session_descriptor_path,
        )

    def _get_cue_sequence(self) -> None:
        """Queries the sequence of virtual reality track wall cues for the current task from Unity.

        This method is used to both get the static VR task cue sequence and to verify that the Unity task is currently
        running. It is called as part of the start_runtime() method and to recover from unexpected Unity shutdowns that
        occur during runtime.

        Notes:
            This method contains an infinite loop that allows retrying failed connection attempts. This prevents the
            runtime from aborting unless the user purposefully chooses the hard abort option.

            Upon receiving the cue sequence data, the method caches the data into the private _cue_sequence class
            attribute. The attribute can be used to handle 'blackout teleportation' runtime events entirely through
            Python, bypassing the need for specialized Unity logic.

        Raises:
            RuntimeError: If the user chooses to abort the runtime when the method does not receive a response from
                Unity in 20 seconds.
        """

        # Does not do anything if Unity communication class is not initialized.
        if self._unity is None:
            return

        # Initializes a second-precise timer to ensure the request is fulfilled within a 20-second timeout
        timeout_timer = PrecisionTimer("s")

        # The procedure repeats until it succeeds or until the user chooses to abort the runtime.
        outcome = ""
        while outcome != "abort":
            # Sends a request for the task cue (corridor) sequence to Unity GIMBL package.
            self._unity.send_data(topic=self._cue_sequence_request_topic)

            # Waits at most 20 seconds to receive the response
            timeout_timer.reset()
            while timeout_timer.elapsed < 20:
                # Repeatedly queries and checks incoming messages from Unity.
                if self._unity.has_data:
                    topic: str
                    payload: bytes
                    topic, payload = self._unity.get_data()  # type: ignore

                    # If the message contains cue sequence data, parses it and finishes method runtime. Discards all
                    # other messages.
                    if topic == self._cue_sequence_topic:
                        # Extracts the sequence of cues that will be used during task runtime.
                        sequence: NDArray[np.uint8] = np.array(
                            json.loads(payload.decode("utf-8"))["cue_sequence"], dtype=np.uint8
                        )

                        # Logs the received sequence and also caches it into class attribute for later use
                        package = LogPackage(
                            source_id=self._source_id,
                            time_stamp=np.uint64(self._timestamp_timer.elapsed),
                            serialized_data=sequence,
                        )
                        self._logger.input_queue.put(package)
                        self._cue_sequence = sequence

                        # Ends the runtime
                        message = "VR cue sequence: Received."
                        console.echo(message=message, level=LogLevel.SUCCESS)
                        return

            # If the loop above is escaped, this is due to not receiving any message from Unity. Raises an error.
            message = (
                f"The Mesoscope-VR system has requested the Virtual task wall cue sequence by sending the trigger to "
                f"the {self._cue_sequence_request_topic}' topic and received no response in 20 seconds. It is likely "
                f"that the Unity game engine is not running or is not configured to work with Mesoscope-VR system. "
                f"Make sure Unity game engine is started and configured before continuing."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            outcome = input("Enter 'abort' to abort with an error. Enter anything else to retry: ").lower()

        message = f"Runtime aborted due to user request."
        console.error(message=message, error=RuntimeError)
        raise RuntimeError(message)  # Fallback to appease mypy, should not be reachable

    def _start_mesoscope(self) -> None:
        """Sends the frame acquisition start TTL pulse to the mesoscope and waits for the frame acquisition to begin.

        This method is used internally to start the mesoscope frame acquisition as part of the runtime startup
        process and to verify that the mesoscope is available and properly configured to acquire frames
        based on the input triggers. Also, it is used to recover from unexpected mesoscope frame acquisition failures
        that occur during runtime.

        Notes:
            This method contains an infinite loop that allows retrying the failed mesoscope acquisition start. This
            prevents the runtime from aborting unless the user purposefully chooses the hard abort option.

        Raises:
            RuntimeError: If the mesoscope does not confirm frame acquisition within 20 seconds after the
                acquisition trigger is sent and the user chooses to abort the runtime.
        """

        # Initializes a second-precise timer to ensure the request is fulfilled within a 2-second timeout
        timeout_timer = PrecisionTimer("s")

        # Keeps retrying to activate mesoscope acquisition until success or until the user aborts the acquisition
        outcome = ""
        while outcome != "abort":
            # Sends mesoscope frame acquisition trigger
            self._microcontrollers.reset_mesoscope_frame_count()  # Resets the frame counter
            self._microcontrollers.start_mesoscope()

            # Ensures that the frame acquisition starts as expected
            message = "Mesoscope acquisition trigger: Sent. Waiting for the mesoscope frame acquisition to start..."
            console.echo(message=message, level=LogLevel.INFO)

            # Waits at most 10 seconds for the mesoscope to acquire at least 10 frames. At ~ 10 Hz, it should take ~ 1
            # second of downtime.
            timeout_timer.reset()
            while timeout_timer.elapsed < 10:
                if self._microcontrollers.mesoscope_frame_count < 10:
                    # Ends the runtime
                    message = "Mesoscope frame acquisition: Started."
                    console.echo(message=message, level=LogLevel.SUCCESS)
                    return

            # If the loop above is escaped, this is due to not receiving the mesoscope frame acquisition pulses.
            message = (
                f"The Mesoscope-VR system has requested the mesoscope to start acquiring frames and failed to receive "
                f"10 frame acquisition triggers over 10 seconds. It is likely that the mesoscope has not been armed "
                f"for externally-triggered frame acquisition or that the mesoscope trigger or frame monitoring modules "
                f"are not functional. Make sure the Mesoscope is configured for data acquisition before continuing and "
                f"retry the mesoscope activation."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            outcome = input("Enter 'abort' to abort with an error. Enter anything else to retry: ").lower()

        message = f"Runtime aborted due to user request."
        console.error(message=message, error=RuntimeError)
        raise RuntimeError(message)  # Fallback to appease mypy, should not be reachable

    def _stop_mesoscope(self) -> None:
        """Sends the frame acquisition stop TTL pulse to the mesoscope and waits for the frame acquisition to stop.

        This method is used internally to stop the mesoscope frame acquisition as part of the stop() method runtime.

        Notes:
            This method contains an infinite loop that waits for the mesoscope to stop generating frame acquisition
            triggers.
        """

        # Sends acquisition stop trigger to the mesoscope.
        self._microcontrollers.stop_mesoscope()

        # Blocks until the Mesoscope stops sending frame acquisition pulses to the microcontroller.
        message = "Waiting for the Mesoscope to stop acquiring frames..."
        console.echo(message=message, level=LogLevel.INFO)
        previous_frame_count = self._microcontrollers.mesoscope_frame_count
        while True:
            # Delays for 2 seconds. Mesoscope acquires frames at 10 Hz, so if there are no incoming triggers for that
            # period of time, it is safe to assume that the acquisition has stopped.
            self._timestamp_timer.delay_noblock(delay=2000000)
            if previous_frame_count == self._microcontrollers.mesoscope_frame_count:
                break  # Breaks the loop
            else:
                previous_frame_count = self._microcontrollers.mesoscope_frame_count

    def _toggle_lick_guidance(self, enable_guidance: bool) -> None:
        """Sets the VR task to either require the animal to lick in the reward zone to get water or to get rewards
        automatically upon entering the reward zone.

        Args:
            enable_guidance: Determines whether the animal must lick (False) to get water rewards or whether it will
                receive rewards automatically when entering the zone (True).
        """

        # Returns without doing anything if the current runtime does not communicate with the Unity game engine.
        if self._unity is None:
            return

        if not enable_guidance:
            self._unity.send_data(topic=self._disable_guidance_topic)
        else:
            self._unity.send_data(topic=self._enable_guidance_topic)

        # Logs the current lick guidance state. Statically uses header-code 3 to indicate that the logged value is
        # the lick guidance state.
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([3, enable_guidance], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def _toggle_show_reward(self, show_reward: bool) -> None:
        """Sets the VR task to either show or hide the reward zone boundary (wall) with which the animal needs to
        collide to receive guided water rewards.

        To receive water rewards in the guided mode, the animal has to collide with the invisible wall, typically
        located in the middle of the reward zone. This method is used to optionally make that wall visible to the
        animal, which may be desirable for certain tasks.

        Args:
            show_reward: Determines whether the reward zone collision wall is visible to the animal (True) or hidden
                from the animal (False).
        """

        # Returns without doing anything if the current runtime does not communicate with the Unity game engine.
        if self._unity is None:
            return

        if not show_reward:
            self._unity.send_data(topic=self._hide_reward_zone_boundary_topic)
        else:
            self._unity.send_data(topic=self._show_reward_zone_boundary_topic)

        # Logs the current reward boundary visibility state. Statically uses header-code 4 to indicate that the logged
        # value is the reward zone boundary visibility state.
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([4, show_reward], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def _change_system_state(self, new_state: int) -> None:
        """Updates and logs the new Mesoscope-VR system state.

        This method is used internally to timestamp and log system state changes, such as transitioning between
        rest and run states during experiment runtimes.

        Args:
            new_state: The byte-code for the newly activated Mesoscope-VR system state.
        """
        self._system_state = new_state  # Updates the Mesoscope-VR system state

        # Logs the system state update. Uses header-code 1 to indicate that the logged value is the system state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([1, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def change_runtime_state(self, new_state: int) -> None:
        """Updates and logs the new runtime state (stage).

        Use this method to timestamp and log runtime state (stage) changes, such as transitioning between different
        task goals or experiment phases.

        Args:
            new_state: The integer byte-code for the new runtime state. The code will be serialized as an uint8
                value, so only values between 0 and 255 inclusive are supported.
        """
        self._runtime_state = new_state  # Updates the tracked runtime state value

        # Logs the runtime state update. Uses header-code 2 to indicate that the logged value is the runtime state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([2, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def idle(self) -> None:
        """Switches the Mesoscope-VR system to the idle state.

        In the idle state, the break is engaged to prevent the animal from moving the wheel and the screens are turned
        off. Both torque and encoder monitoring is disabled. Note, idle state is designed to be used exclusively during
        periods where the runtime is paused and no valid data is generated.

        Notes:
            Unlike the other VR states, setting the system to 'idle' also automatically changes the runtime state to
            0 (idle).

            Idle Mesoscope-VR state is hardcoded as '0'.
        """

        # Switches runtime state to 0
        self.change_runtime_state(new_state=0)

        # Blackens the VR screens
        self._microcontrollers.disable_vr_screens()

        # Engages the break
        self._microcontrollers.enable_break()

        # Disables all sensor monitoring
        self._microcontrollers.disable_encoder_monitoring()
        self._microcontrollers.disable_torque_monitoring()
        self._microcontrollers.disable_lick_monitoring()

        # Sets system state to 0
        self._change_system_state(0)

    def rest(self) -> None:
        """Switches the Mesoscope-VR system to the rest state.

        In the rest state, the break is engaged to prevent the animal from moving the wheel. The encoder module is
        disabled, and instead the torque sensor is enabled. The VR screens are switched off, cutting off light emission.

        Notes:
            Rest Mesoscope-VR state is hardcoded as '1'.
        """

        # Enables lick monitoring
        self._microcontrollers.enable_lick_monitoring()

        # Blackens the VR screens
        self._microcontrollers.disable_vr_screens()

        # Engages the break
        self._microcontrollers.enable_break()

        # Suspends encoder monitoring.
        self._microcontrollers.disable_encoder_monitoring()

        # Enables torque monitoring.
        self._microcontrollers.enable_torque_monitoring()

        # Sets system state to 1
        self._change_system_state(1)

    def run(self) -> None:
        """Switches the Mesoscope-VR system to the run state.

        In the run state, the break is disengaged to allow the animal to freely move the wheel. The encoder module is
        enabled to record motion data, and the torque sensor is disabled. The VR screens are switched on to render the
        VR environment.

        Notes:
            Run Mesoscope-VR state is hardcoded as '2'.
        """

        # Enables lick monitoring
        self._microcontrollers.enable_lick_monitoring()

        # Initializes encoder monitoring.
        self._microcontrollers.enable_encoder_monitoring()

        # Disables torque monitoring.
        self._microcontrollers.disable_torque_monitoring()

        # Activates VR screens.
        self._microcontrollers.enable_vr_screens()

        # Disengages the break
        self._microcontrollers.disable_break()

        # Sets system state to 2
        self._change_system_state(2)

    def lick_train(self) -> None:
        """Switches the Mesoscope-VR system to the lick training state.

        In this state, the break is engaged to prevent the animal from moving the wheel. The encoder module is
        disabled, and the torque sensor is enabled. The VR screens are switched off, cutting off light emission.

        Notes:
            Lick training Mesoscope-VR state is hardcoded as '3'.

            Calling this method automatically switches the runtime state to 255 (active training).
        """

        # Switches runtime state to 255 (active)
        self.change_runtime_state(new_state=255)

        # Blackens the VR screens
        self._microcontrollers.disable_vr_screens()

        # Engages the break
        self._microcontrollers.enable_break()

        # Disables encoder monitoring
        self._microcontrollers.disable_encoder_monitoring()

        # Initiates torque monitoring
        self._microcontrollers.enable_torque_monitoring()

        # Initiates lick monitoring
        self._microcontrollers.enable_lick_monitoring()

        # Sets system state to 3
        self._change_system_state(3)

    def run_train(self) -> None:
        """Switches the Mesoscope-VR system to the run training state.

        In this state, the break is disengaged, allowing the animal to run on the wheel. The encoder module is
        enabled, and the torque sensor is disabled. The VR screens are switched off, cutting off light emission.

         Notes:
            Run training Mesoscope-VR state is hardcoded as '4'.

            Calling this method automatically switches the runtime state to 255 (active training).
        """

        # Switches runtime state to 255 (active)
        self.change_runtime_state(new_state=255)

        # Blackens the VR screens
        self._microcontrollers.disable_vr_screens()

        # Disengages the break.
        self._microcontrollers.disable_break()

        # Ensures that encoder monitoring is enabled
        self._microcontrollers.enable_encoder_monitoring()

        # Ensures torque monitoring is disabled
        self._microcontrollers.disable_torque_monitoring()

        # Initiates lick monitoring
        self._microcontrollers.enable_lick_monitoring()

        # Sets system state to 4
        self._change_system_state(4)

    def deliver_reward(self, reward_size: float = 5.0) -> None:
        """Uses the solenoid valve to deliver the requested volume of water in microliters.

        This method is used by external runtime logic functions to deliver water rewards to the animal.
        """
        self._microcontrollers.deliver_reward(volume=reward_size)

    def simulate_reward(self) -> None:
        """Uses the buzzer controlled by the valve module to deliver an audible tone without delivering any water
        reward.

        This method is used when the animal refuses to consume water rewards during training or experiment runtimes. The
        tone notifies the animal that it performs the task as expected, while simultaneously minimizing water reward
        wasting.
        """
        self._microcontrollers.simulate_reward()

    # TODO finish these

    def _unity_cycle(self) -> None:
        """Handles bidirectional communication between the Virtual Reality task in Unity and the main runtime logic and
        hardware module interfaces in Python.

        The VR system broadly consists of two components: Unity game engine (virtual task) and a set of microcontrollers
        managing the physical task environment. This function sends the data expected by Unity to the virtual reality
        and receives and executes commands sent by Unity to the microcontrollers. To do so, it leverages the
        MQTTCommunication class and acts as a persistent bidirectional interface between Unity and the rest of the
        Mesoscope-VR system.

        Notes:
            This method has been introduced in version 2.0.0 to aggregate all Unity communication (via MQTT) at the
            highest level of the runtime hierarchy (the main runtime management class). This prevents an error with the
            Mosquitto MQTT broker, where the broker arbitrarily disconnects clients running in remote processes.

            During runtime, this method attempts to receive data from Unity. During each runtime cycle, the method
            receives and parses exactly one message stored in the MQTTCommunication class buffer. Also, as part of each
            cycle the method sends updated distance and lick data to Unity.
        """

        # If the mouse has changed its position since the previous cycle, updates the position and sends the data to
        # Unity. Since Unity expects to be sent the delta (change) in mouse position, rather than the current absolute
        # position, converts the absolute position readout to a delta value.
        current_position = self._microcontrollers.distance_tracker.read_data(index=1, convert_output=True)

        # Subtracting previous position from current correctly maps positive deltas to moving forward and negative
        # deltas to moving backward
        position_delta = current_position - self._position

        # If position changed, sends the change to Unity
        if position_delta != 0:
            # Overwrites the cached position with the loaded data
            self._position = current_position

            # Encodes the motion data into the format expected by the GIMBL Unity module and serializes it into a
            # byte-string.
            json_string = dumps(obj={"movement": position_delta})
            byte_array = json_string.encode("utf-8")

            # Publishes the motion to the appropriate MQTT topic.
            self._unity.send_data(topic=self._microcontrollers.wheel_encoder.mqtt_topic, payload=byte_array)

        # If the lick tracker indicates that the sensor has detected new licks, sends a binary lick indicator to
        # Unity.
        lick_count = self._microcontrollers.lick_tracker.read_data(index=0, convert_output=True)
        if lick_count > self._lick_count:
            self._lick_count = lick_count  # Updates the local lick counter

            # Whenever the animal licks the water delivery tube, it is consuming any available rewards. Resets the
            # unconsumed count whenever new licks are detected.
            self._unconsumed_reward_count = 0

            self._unity.send_data(topic=self._microcontrollers.lick.mqtt_topic, payload=None)

        # If Unity sends updates to the Mesoscope-VR system, receives and processes the data. Note, this will discard
        # all unexpected data
        if self._unity.has_data:
            topic: str
            topic, _ = self._unity.get_data()  # type: ignore

            # Uses the reward volume specified during startup (5.0) or via the UI (Anything from 1 to 20).
            if topic == self._microcontrollers.valve.mqtt_topic:
                # Only delivers water rewards if the current unconsumed count value is below the user-defined threshold.
                if self._unconsumed_reward_count < self.descriptor.maximum_unconsumed_rewards:
                    self._microcontrollers.deliver_reward(ignore_parameters=True)

                # Otherwise, simulates water reward by sounding the buzzer without delivering any water
                else:
                    self.simulate_reward()

            # If Unity runtime (game mode) terminates, Unity sends a message to the termination topic. In turn, the
            # runtime uses this as an indicator to reset the task logic.
            if topic == self._unity_termination_topic:
                # Automatically transitions into the 'idle' VR AND experiment state. Note, the expectation here is that
                # the runtime logic always checks the state of 'unity_terminated' tracker once this method returns and
                # applies the global runtime 'pause' state. Transitioning into 'idle' mode is designed to rapidly cut
                # the incoming wheel motion (distance change) stream.
                self.idle()

                # Reads the current position, in Unity units. Since this is done after cutting off the wheel motion
                # stream, there should be minimal deviation of the read position and the physical position of the
                # animal.
                traveled_distance: float = self._microcontrollers.distance_tracker.read_data(
                    index=1, convert_output=True
                )
                # Converts float to a byte array using little-endian format
                distance_bytes = np.array([traveled_distance], dtype="<i8").view(np.uint8)

                # Generates a new log entry with message ID code 4. This code is statically used to indicate that the
                # Unity runtime has been terminated. The message includes the current position of the animal in Unity
                # units, stored as a byte array (8 bytes). The position stamp is then used during behavior data
                # processing to artificially 'fuse' multiple cue sequences together if the user chooses to restart the
                # unity task and resume the runtime.
                log_package = LogPackage(
                    source_id=self._source_id,
                    time_stamp=np.uint64(self._timestamp_timer.elapsed),
                    serialized_data=np.concatenate([np.array([4], dtype=np.uint8), distance_bytes]),
                )
                self._logger.input_queue.put(log_package)

                # Updates the VR system termination flag
                self.unity_terminated = True

    def _ui_cycle(self):
        """Queries the state of various GUI components and adjusts the runtime behavior accordingly.

        This utility method cycles through various user-addressable runtime components and, depending on corresponding
        UI states, executes the necessary functionality or updates associated parameters. In essence, calling this
        method synchronizes the runtime with the state of the runtime control GUI.

        Notes:
            This method is designed to be called repeatedly as part of the main runtime cycle loop (via the user-facing
            runtime_cycle() method).
        """

        # This is mostly a fall-back to appease mypy, but also simplifies working with runtimes that do not need a GUI,
        # such as 'window checking'.
        if self._ui is None:
            # If the UI is not initialized, the cycle immediately returns to caller.
            return

        # If the ui detects a pause command, enters a pause loop. This effectively locks the runtime into the 'pause'
        # state, ceasing all runtime logic execution until the user resumes the runtime or terminates it.
        if self._ui.pause_runtime:
            self._pause_runtime()
        else:
            # If the user sends a resume command, resumes the runtime and adjusts certain class attributes to help
            # runtime logic functions discount (ignore) the time spent in paused state.
            self._resume_runtime()

        # If the user sent the abort command, terminates the runtime early with an error message.
        if self._ui.exit_signal:
            self._terminate_runtime()

            # If the user confirms runtime termination, braks the ui cycle to expediter runtime shut down sequence.
            if self._terminated:
                return

        # If the user updates the reward volume in the GUI, adjusts the volume used by the instance to match the
        # GUI state and configures the valve module to deliver that much water for each reward command.
        if self._ui.reward_volume != self._reward_volume:
            self._reward_volume = self._ui.reward_volume
            self._microcontrollers.configure_reward_parameters(volume=self._reward_volume)

        # If the user toggles manual reward delivery via the UI, delivers a water reward to the animal.
        if self._ui.reward_signal:
            self.deliver_reward(reward_size=self._reward_volume)  # Delivers 5 uL of water

        # If the user changes the guidance state via the UI, instructs Unity to update the state to match GUI setting.
        if self._ui.enable_guidance != self._enable_guidance:
            self._enable_guidance = self._ui.enable_guidance
            self._toggle_lick_guidance(enable_guidance=self._enable_guidance)

        # If the user changes the reward collision wall visibility state via the UI, instructs Unity to update the
        # state to match GUI setting.
        if self._ui.show_reward != self._show_reward_zone_boundary:
            self._show_reward_zone_boundary = self._ui.show_reward
            self._toggle_show_reward(show_reward=self._show_reward_zone_boundary)

    def runtime_cycle(self):

        # If the managed runtime exposes a GUI, synchronizes the runtime state with the state of the user-facing GUI
        if self._ui is not None:
            self._ui_cycle()

        # If the managed runtime communicates with Unity, synchronizes the state of the Unity virtual task with the
        # state of the runtime (and the GUI).
        if self._unity is not None:
            self._unity_cycle()

        if self._terminated:
            return

    def _pause_runtime(self) -> None:
        # Notifies the user that the runtime has been paused
        message = "Mesoscope-VR runtime: Paused."
        console.echo(message=message, level=LogLevel.WARNING)

        # Ensures that the GUI reflects that the runtime is paused. While most paused states will originate from the
        # GUI, certain events may cause the main process to activate the paused state bypassing the GUI.
        if not ui.pause_runtime:
            ui.set_pause_state(paused=True)

        # Caches valve data to discard any valve data accumulated during the pause period. This is done to prevent
        # user-driven valve manipulation from influence the runtime if the user chooses to resume it.
        cached_valve_pulse_count = self._microcontrollers.valve_tracker.read_data(index=0)
        cached_water_volume = self._microcontrollers.valve_tracker.read_data(index=1)

        # Blocks in-place until the user either unpauses or aborts the runtime.
        while ui.pause_runtime:
            visualizer.update()  # Continuously updates the visualizer

            # Keeps unity communication loop running. Primarily, this is done to support receiving
            # unity termination messages if the user chooses to pause the runtime first and then terminate
            # the Unity
            runtime._unity_cycle()

            # If the ui detects a reward delivery signal, delivers the reward to the animal
            if ui.reward_signal:
                runtime.deliver_reward(reward_size=ui.reward_volume)  # Delivers 5 uL of water

            # Adjusts the reward volume each time it is updated in the GUI
            if ui.reward_volume != previous_reward_volume:
                self._microcontrollers.configure_reward_parameters(volume=ui.reward_volume)

            # Switches the guidance status in response to user requests
            if ui.enable_guidance != previous_guidance_state:
                previous_guidance_state = ui.enable_guidance
                runtime._toggle_lick_guidance(enable_guidance=previous_guidance_state)

            # If the user requests for the paused stage to be aborted, terminates the runtime.
            if ui.exit_signal:
                message = "Experiment runtime abort signal: received. Are you sure you want to abort the runtime?"
                console.echo(message=message, level=LogLevel.WARNING)
                while True:
                    answer = input("Enter 'yes' or 'no': ")

                    if answer.lower() == "yes":
                        terminate_runtime = True  # Sets the terminate flag
                        break  # Breaks the while loop

                    elif answer.lower() == "no":
                        # Returns to running the runtime
                        break

                if terminate_runtime:
                    break  # Escapes the pause 'while' loop
        else:
            # Discards any valve data accumulated during the idle period
            valve_tracker.write_data(index=0, data=cached_valve_pulse_count)
            valve_tracker.write_data(index=1, data=cached_water_volume)

            # Updates the 'additional time' value to reflect the time spent inside the 'paused' state.
            # This increases the experiment stage duration to counteract the duration of the 'paused'
            # state.
            additional_time += runtime_timer.elapsed - pause_start

            # Re-queries the cue sequence as Unity re-initialization always generates a new sequence.
            # Also ensures that Unity has been properly re-initialized.
            if runtime.unity_terminated:
                runtime._get_cue_sequence()
                # Resets the unity termination tracker
                runtime.unity_terminated = False

            # Restores the runtime state
            if state.system_state_code == 1:
                runtime.rest()
            elif state.system_state_code == 2:
                runtime.run()

    def _resume_runtime(self) -> None:
        pass

    def _terminate_runtime(self) -> None:
        """Verifies that the user intends to abort the runtime via terminal prompt and, if so, sets the runtime into
        the termination mode.
        """

        # Verifies that the user intends to abort the runtime to avoid 'misclick' terminations.
        message = "Runtime abort signal: Received. Are you sure you want to abort the runtime?"
        console.echo(message=message, level=LogLevel.WARNING)
        while True:
            answer = input("Enter 'yes' or 'no': ")

            # Sets the runtime into the termination state, which aborts all instance cycles and the outer logic function
            # cycle.
            if answer.lower() == "yes":
                self._terminated = True
                return

            # Returns without terminating the runtime
            elif answer.lower() == "no":
                return


def lick_training_logic(
    experimenter: str,
    project_name: str,
    animal_id: str,
    animal_weight: float,
    minimum_reward_delay: int = 6,
    maximum_reward_delay: int = 18,
    maximum_water_volume: float = 1.0,
    maximum_training_time: int = 20,
    maximum_unconsumed_rewards: int = 1,
    load_previous_parameters: bool = False,
) -> None:
    """Encapsulates the logic used to train animals to operate the lick port.

    The lick training consists of delivering randomly spaced 5 uL water rewards via the solenoid valve to teach the
    animal that water comes out of the lick port. Each reward is delivered after a pseudorandom delay. Reward delay
    sequence is generated before training runtime by sampling a uniform distribution that ranges from
    'minimum_reward_delay' to 'maximum_reward_delay'. The training continues either until the valve
    delivers the 'maximum_water_volume' in milliliters or until the 'maximum_training_time' in minutes is reached,
    whichever comes first.

    Args:
        experimenter: The ID (net-ID) of the experimenter conducting the training.
        project_name: The name of the project to which the trained animal belongs.
        animal_id: The numeric ID of the animal being trained.
        animal_weight: The weight of the animal, in grams, at the beginning of the training session.
        minimum_reward_delay: The minimum time, in seconds, that has to pass between delivering two consecutive rewards.
        maximum_reward_delay: The maximum time, in seconds, that can pass between delivering two consecutive rewards.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered during this runtime.
        maximum_training_time: The maximum time, in minutes, to run the training.
        maximum_unconsumed_rewards: The maximum number of rewards that can be delivered without the animal consuming
            them, before reward delivery (but not the training!) pauses until the animal consumes available rewards.
            If this is set to a value below 1, the unconsumed reward limit will not be enforced. A value of 1 means
            the animal has to consume each reward before getting the next reward.
        load_previous_parameters: Determines whether to override all input runtime-defining parameters with the
            parameters used during the previous session. If this is set to True, the function will ignore most input
            parameters and will instead load them from the cached session descriptor of the previous session. If the
            descriptor is not available, the function will fall back to using input parameters.
    """
    message = f"Initializing lick training runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters
    system_configuration = get_system_configuration()

    # Verifies that the target project exists
    project_folder = system_configuration.paths.root_directory.joinpath(project_name)
    if not project_folder.exists():
        message = (
            f"Unable to execute the lick training for the animal {animal_id} of project {project_name}. The target "
            f"project does not exist on the local machine. Use the 'sl-create-project' command to create the project "
            f"before running training or experiment sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Initializes data-management classes for the runtime. Note, SessionData creates the necessary session directory
    # hierarchy as part of this initialization process
    session_data = SessionData.create(project_name=project_name, animal_id=animal_id, session_type=_lick)
    mesoscope_data = MesoscopeData(session_data=session_data)

    # Caches current sl-experiment and Python versions to disk as a version_data.yaml file.
    write_version_data(session_data)

    # Verifies that the Water Restriction log and the Surgery log Google Sheets are accessible. To do so, instantiates
    # both classes to run through the init checks. The classes are later re-instantiated during session data
    # preprocessing
    project_configuration: ProjectConfiguration = ProjectConfiguration.from_yaml(  # type: ignore
        file_path=session_data.raw_data.project_configuration_path
    )
    _ = WaterSheet(
        animal_id=int(animal_id),
        session_date=session_data.session_name,
        credentials_path=system_configuration.paths.google_credentials_path,
        sheet_id=project_configuration.water_log_sheet_id,
    )
    _ = SurgerySheet(
        project_name=project_name,
        animal_id=int(animal_id),
        credentials_path=system_configuration.paths.google_credentials_path,
        sheet_id=project_configuration.surgery_sheet_id,
    )

    # If the managed animal has cached data from a previous lick training session and the function is
    # configured to load previous data, replaces all runtime-defining parameters passed to the function with data
    # loaded from the previous session's descriptor file
    previous_descriptor_path = mesoscope_data.vrpc_persistent_data.session_descriptor_path
    if previous_descriptor_path.exists() and load_previous_parameters:
        previous_descriptor: LickTrainingDescriptor = LickTrainingDescriptor.from_yaml(  # type: ignore
            file_path=previous_descriptor_path
        )
        maximum_reward_delay = previous_descriptor.maximum_reward_delay_s
        minimum_reward_delay = previous_descriptor.minimum_reward_delay

    # Pre-generates the SessionDescriptor class and populates it with training data.
    descriptor = LickTrainingDescriptor(
        maximum_reward_delay_s=maximum_reward_delay,
        minimum_reward_delay=minimum_reward_delay,
        maximum_training_time_m=maximum_training_time,
        maximum_water_volume_ml=maximum_water_volume,
        experimenter=experimenter,
        mouse_weight_g=animal_weight,
        dispensed_water_volume_ml=0.00,
        maximum_unconsumed_rewards=maximum_unconsumed_rewards,
        incomplete=True,  # Has to be initialized to True, so that if session aborts, it is marked as incomplete
    )

    # Initializes the main runtime interface class.
    runtime = _BehaviorTraining(
        session_data=session_data,
        session_descriptor=descriptor,
    )

    # Initializes the timer used to enforce reward delays
    delay_timer = PrecisionTimer("us")

    # Uses runtime tracker extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers

    message = f"Generating the pseudorandom reward delay sequence..."
    console.echo(message=message, level=LogLevel.INFO)

    # Converts maximum volume to uL and divides it by 5 uL (reward size) to get the number of delays to sample from
    # the delay distribution
    num_samples = np.floor((maximum_water_volume * 1000) / 5).astype(np.uint64)

    # Generates samples from a uniform distribution within delay bounds
    samples = np.random.uniform(minimum_reward_delay, maximum_reward_delay, num_samples)

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
        runtime.descriptor.maximum_training_time_m = int(np.ceil(cumulative_time[-1] / 60))

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Initializes the runtime control UI. Similar to cameras, HAS to be initialized before the visualizer
    ui = RuntimeControlUI()

    # Visualizer initialization HAS to happen after the runtime start to avoid interfering with cameras.
    visualizer = BehaviorVisualizer(
        lick_tracker=lick_tracker, valve_tracker=valve_tracker, distance_tracker=speed_tracker
    )

    # Final checkpoint
    message = (
        f"Runtime preparation: Complete. Carry out all final checks and adjustments, such as priming the water "
        f"delivery valve at this time. When you are ready to start the runtime, use the UI to 'resume' it."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Allows the user to manipulate the valve via the UI. Since Zaber interface should also be active, the user can also
    # manually reposition all Zaber motors.
    while ui.pause_runtime:
        # Updates the visualizer plot ~every 30 ms. This should be enough to reliably capture all events of
        # interest and appear visually smooth to human observers.
        visualizer.update()

        if ui.reward_signal:
            runtime.deliver_reward(reward_size=ui.reward_volume)

        if ui.open_valve:
            runtime.toggle_valve(True)

        if ui.close_valve:
            runtime.toggle_valve(False)

    # Ensures the valve is closed before continuing
    runtime.toggle_valve(False)

    # Resets the valve tracker array before proceeding. This allows the user to use the section above to debug the
    # valve, potentially dispensing a lot of water in the process. If the tracker is not reset, this may immediately
    # terminate the runtime and lead to an inaccurate tracking of the water volume received by the animal.
    # noinspection PyTypeChecker
    valve_tracker.write_data(index=0, data=0)
    # noinspection PyTypeChecker
    valve_tracker.write_data(index=1, data=0)

    message = f"Initiating lick training procedure..."
    console.echo(message=message, level=LogLevel.INFO)

    # This tracker is used to terminate the training if manual abort command is sent via the keyboard
    terminate = False

    # Initializes assets used to ensure that the animal consumes delivered water rewards.
    if maximum_unconsumed_rewards < 1:
        # If the maximum unconsumed reward count is below 1, disables the feature by setting the number to match the
        # number of rewards to be delivered.
        maximum_unconsumed_rewards = len(reward_delays)

    unconsumed_count = 0
    previous_licks = 0
    previous_reward_volume = 0

    # Configures all system components to support lick training
    runtime.lick_train()

    # Loops over all delays and delivers reward via the lick tube as soon as the delay expires.
    delay_timer.reset()
    for delay in tqdm(
        reward_delays,
        desc="Delivered water rewards",
        unit="reward",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} rewards [{elapsed}]",
    ):
        # This loop is executed while the code is waiting for the delay to pass. Anything that needs to be done
        # during the delay has to go here
        while delay_timer.elapsed < delay:
            # Updates the visualizer plot ~every 30 ms. This should be enough to reliably capture all events of
            # interest and appear visually smooth to human observers.
            visualizer.update()

            # If the animal licks during the delay period, this is interpreted as the animal consuming the previous
            # and any other leftover rewards.
            if previous_licks < visualizer.lick_count:
                previous_licks = visualizer.lick_count
                unconsumed_count = 0

            # If the ui detects the default abort sequence, terminates the runtime.
            if ui.exit_signal:
                message = "Lick training runtime abort signal: received. Are you sure you want to abort the runtime?"
                console.echo(message=message, level=LogLevel.WARNING)
                while True:
                    answer = input("Enter 'yes' or 'no': ")

                    if answer.lower() == "yes":
                        terminate = True  # Sets the terminate flag
                        break  # Breaks the while loop

                    elif answer.lower() == "no":
                        # Returns to running the runtime
                        break

            # If the ui detects a reward delivery signal, delivers the reward to the animal
            if ui.reward_signal:
                runtime.deliver_reward(reward_size=ui.reward_volume)  # Delivers the requested volume of water

            # Adjusts the reward volume each time it is updated in the GUI
            if ui.reward_volume != previous_reward_volume:
                runtime.configure_reward_parameters(reward_size=ui.reward_volume)

            # If the ui detects a pause command, enters a holding loop.
            if ui.pause_runtime:
                message = (
                    "Lick training runtime: paused due to user request. Note, sensor readout monitoring is suspended."
                )
                console.echo(message=message, level=LogLevel.WARNING)

                # Switches runtime into the idle state
                runtime.idle()

                # Caches valve data to discard any valve data accumulated during the idle period.
                cached_valve_pulse_count = valve_tracker.read_data(index=0)
                cached_water_volume = valve_tracker.read_data(index=1)

                # Blocks in-place until the user either unpauses or aborts the training.
                while ui.pause_runtime:
                    visualizer.update()  # Continuously updates the visualizer

                    # If the ui detects a reward delivery signal, delivers the reward to the animal
                    if ui.reward_signal:
                        runtime.deliver_reward(reward_size=ui.reward_volume)  # Delivers the requested volume of water

                    # Adjusts the reward volume each time it is updated in the GUI
                    if ui.reward_volume != previous_reward_volume:
                        runtime.configure_reward_parameters(reward_size=ui.reward_volume)

                    # If the user requests for the paused runtime to be aborted, terminates the runtime.
                    if ui.exit_signal:
                        message = (
                            "Lick training runtime abort signal: received. Are you sure you want to abort the runtime?"
                        )
                        console.echo(message=message, level=LogLevel.WARNING)
                        while True:
                            answer = input("Enter 'yes' or 'no': ")

                            if answer.lower() == "yes":
                                terminate = True  # Sets the terminate flag
                                break  # Breaks the while loop

                            elif answer.lower() == "no":
                                # Returns to the pause state
                                break

                        if terminate:
                            break  # Escapes the pause 'while' loop if the user chose to terminate the runtime
                else:
                    # Discards any valve data accumulated during the idle period
                    valve_tracker.write_data(index=0, data=cached_valve_pulse_count)
                    valve_tracker.write_data(index=1, data=cached_water_volume)

                    # Restores the runtime state
                    runtime.lick_train()

                    # Resets unconsumed reward tracker
                    unconsumed_count = 0

                # Escapes the outer (reward delay) 'while' loop
                if terminate:
                    break

        # If the user sent the abort command, terminates the training early
        if terminate:
            message = (
                "Lick training abort signal detected. Aborting the lick training with a graceful shutdown procedure."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            break  # Breaks the for loop

        # Once the delay is up, triggers the solenoid valve to deliver water to the animal and starts timing the
        # next reward delay, unless unconsumed reward guard kicks in.
        if unconsumed_count < maximum_unconsumed_rewards:
            # If the animal did not accumulate the critical number of unconsumed rewards, delivers the reward.
            runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water

            # Increments the unconsumed reward count each time a reward is delivered
            unconsumed_count += 1

        # If the animal does not consume rewards, still issues auditory tones, but does not deliver water
        # rewards.
        else:
            runtime.simulate_reward()

        delay_timer.reset()

    # Ensures the animal has time to consume the last reward before the LickPort is moved out of its range.
    delay_timer.delay_noblock(minimum_reward_delay * 1000000)  # Converts to microseconds before delaying

    # Shutdown sequence:
    message = f"Training runtime: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Terminates the UI
    ui.shutdown()

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()


def run_training_logic(
    experimenter: str,
    project_name: str,
    animal_id: str,
    animal_weight: float,
    initial_speed_threshold: float = 0.50,
    initial_duration_threshold: float = 0.50,
    speed_increase_step: float = 0.05,
    duration_increase_step: float = 0.05,
    increase_threshold: float = 0.1,
    maximum_water_volume: float = 1.0,
    maximum_training_time: int = 40,
    maximum_idle_time: float = 0.5,
    maximum_unconsumed_rewards: int = 1,
    load_previous_parameters: bool = False,
) -> None:
    """Encapsulates the logic used to train animals to run on the wheel treadmill while being head-fixed.

    The run training consists of making the animal run on the wheel with a desired speed, in centimeters per second,
    maintained for the desired duration of time, in seconds. Each time the animal satisfies the speed and duration
    thresholds, it receives 5 uL of water reward, and the speed and durations trackers reset for the next training
    'epoch'. Each time the animal receives 'increase_threshold' of water, the speed and duration thresholds increase to
    make the task progressively more challenging. The training continues either until the training time exceeds the
    'maximum_training_time', or the animal receives the 'maximum_water_volume' of water, whichever happens earlier.

    Args:
        experimenter: The id of the experimenter conducting the training.
        project_name: The name of the project to which the trained animal belongs.
        animal_id: The numeric ID of the animal being trained.
        animal_weight: The weight of the animal, in grams, at the beginning of the training session.
        initial_speed_threshold: The initial running speed threshold, in centimeters per second, that the animal must
            maintain to receive water rewards.
        initial_duration_threshold: The initial duration threshold, in seconds, that the animal must maintain
            above-threshold running speed to receive water rewards.
        speed_increase_step: The step size, in centimeters per second, by which to increase the speed threshold each
            time the animal receives 'increase_threshold' milliliters of water.
        duration_increase_step: The step size, in seconds, by which to increase the duration threshold each time the
            animal receives 'increase_threshold' milliliters of water.
        increase_threshold: The volume of water received by the animal, in milliliters, after which the speed and
            duration thresholds are increased by one step. Note, the animal will at most get 'maximum_water_volume' of
            water, so this parameter effectively controls how many increases will be made during runtime, assuming the
            maximum training time is not reached.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered during this runtime.
        maximum_training_time: The maximum time, in minutes, to run the training.
        maximum_idle_time: The maximum time, in seconds, the animal's speed can be below the speed threshold to
            still receive water rewards. This parameter is designed to help animals with a distinct 'step' pattern to
            not lose water rewards due to taking many large steps, rather than continuously running at a stable speed.
            This parameter allows the speed to dip below the threshold for at most this number of seconds, for the
            'running epoch' to not be interrupted.
        maximum_unconsumed_rewards: The maximum number of rewards that can be delivered without the animal consuming
            them, before reward delivery (but not the training!) pauses until the animal consumes available rewards.
            If this is set to a value below 1, the unconsumed reward limit will not be enforced. A value of 1 means
            the animal has to consume all rewards before getting the next reward.
        load_previous_parameters: Determines whether to override all input runtime-defining parameters with the
            parameters used during the previous session. If this is set to True, the function will ignore most input
            parameters and will instead load them from the cached session descriptor of the previous session. If the
            descriptor is not available, the function will fall back to using input parameters.
    """
    message = f"Initializing run training runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters
    system_configuration = get_system_configuration()

    # Verifies that the target project exists
    project_folder = system_configuration.paths.root_directory.joinpath(project_name)
    if not project_folder.exists():
        message = (
            f"Unable to execute the run training for the animal {animal_id} of project {project_name}. The target "
            f"project does not exist on the local machine. Use the 'sl-create-project' command to create the project "
            f"before running training or experiment sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Initializes data-management classes for the runtime. Note, SessionData creates the necessary session directory
    # hierarchy as part of this initialization process
    session_data = SessionData.create(project_name=project_name, animal_id=animal_id, session_type=_run)
    mesoscope_data = MesoscopeData(session_data=session_data)

    # Caches current sl-experiment and Python versions to disk as a version_data.yaml file.
    write_version_data(session_data)

    # Verifies that the Water Restriction log and the Surgery log Google Sheets are accessible. To do so, instantiates
    # both classes to run through the init checks. The classes are later re-instantiated during session data
    # preprocessing
    project_configuration: ProjectConfiguration = ProjectConfiguration.from_yaml(  # type: ignore
        file_path=session_data.raw_data.project_configuration_path
    )
    _ = WaterSheet(
        animal_id=int(animal_id),
        session_date=session_data.session_name,
        credentials_path=system_configuration.paths.google_credentials_path,
        sheet_id=project_configuration.water_log_sheet_id,
    )
    _ = SurgerySheet(
        project_name=project_name,
        animal_id=int(animal_id),
        credentials_path=system_configuration.paths.google_credentials_path,
        sheet_id=project_configuration.surgery_sheet_id,
    )

    # If the managed animal has cached data from a previous run training session and the function is
    # configured to load previous data, replaces all runtime-defining parameters passed to the function with data
    # loaded from the previous session's descriptor file
    previous_descriptor_path = mesoscope_data.vrpc_persistent_data.session_descriptor_path
    if previous_descriptor_path.exists() and load_previous_parameters:
        previous_descriptor: RunTrainingDescriptor = RunTrainingDescriptor.from_yaml(  # type: ignore
            file_path=previous_descriptor_path
        )

        # Sets initial speed and duration thresholds to the FINAL thresholds from the previous session. This way, each
        # consecutive run training session begins where the previous one has ended.
        initial_speed_threshold = previous_descriptor.final_run_speed_threshold_cm_s
        initial_duration_threshold = previous_descriptor.final_run_duration_threshold_s

    # Pre-generates the SessionDescriptor class and populates it with training data
    descriptor = RunTrainingDescriptor(
        dispensed_water_volume_ml=0.0,
        final_run_speed_threshold_cm_s=initial_speed_threshold,
        final_run_duration_threshold_s=initial_duration_threshold,
        initial_run_speed_threshold_cm_s=initial_speed_threshold,
        initial_run_duration_threshold_s=initial_duration_threshold,
        increase_threshold_ml=increase_threshold,
        run_speed_increase_step_cm_s=speed_increase_step,
        run_duration_increase_step_s=duration_increase_step,
        maximum_training_time_m=maximum_training_time,
        maximum_water_volume_ml=maximum_water_volume,
        maximum_unconsumed_rewards=maximum_unconsumed_rewards,
        maximum_idle_time_s=maximum_idle_time,
        experimenter=experimenter,
        mouse_weight_g=animal_weight,
        incomplete=True,  # Has to be initialized to True, so that if session aborts, it is marked as incomplete
    )

    # Initializes the main runtime interface class. Note, most class parameters are statically configured to work for
    # the current VRPC setup and may need to be adjusted as that setup evolves over time.
    runtime = _BehaviorTraining(
        session_data=session_data,
        session_descriptor=descriptor,
    )

    # Initializes the timers used during runtime
    runtime_timer = PrecisionTimer("s")
    speed_timer = PrecisionTimer("ms")

    # Initializes assets used to guard against interrupting run epochs for mice that take many large steps. For mice
    # with a distinct walking pattern of many very large steps, the speed transiently dips below the threshold for a
    # very brief moment of time, flagging the epoch as unrewarded. To avoid this issue, instead of interrupting the
    # epoch outright, we now allow the speed to be below the threshold for a short period of time. These assets
    # help with that task pattern.
    epoch_timer = PrecisionTimer("ms")
    epoch_timer_engaged: bool = False
    maximum_idle_time = max(0.0, maximum_idle_time)  # Ensures positive values or zero
    maximum_idle_time *= 1000  # Converts to milliseconds

    # Initializes assets used to ensure that the animal consumes delivered water rewards.
    if maximum_unconsumed_rewards < 1:
        # If the maximum unconsumed reward count is below 1, disables the feature by setting the number to match the
        # maximum number of rewards that can possibly be delivered during runtime.
        maximum_unconsumed_rewards = int(np.ceil(maximum_water_volume / 0.005))
    previous_licks = 0
    unconsumed_count = 0

    # Converts all arguments used to determine the speed and duration threshold over time into numpy variables to
    # optimize main loop runtime speed:
    initial_speed = np.float64(initial_speed_threshold)  # In centimeters per second
    maximum_speed = np.float64(5)  # In centimeters per second
    speed_step = np.float64(speed_increase_step)  # In centimeters per second

    initial_duration = np.float64(initial_duration_threshold * 1000)  # In milliseconds
    maximum_duration = np.float64(5000)  # In milliseconds
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

    # Initializes the runtime control UI. Similar to cameras, HAS to be initialized before the visualizer
    ui = RuntimeControlUI()

    # Visualizer initialization HAS to happen after the runtime start to avoid interfering with cameras.
    visualizer = BehaviorVisualizer(
        lick_tracker=lick_tracker, valve_tracker=valve_tracker, distance_tracker=speed_tracker
    )

    # Updates the threshold lines to use the initial speed and duration values
    visualizer.update_speed_thresholds(speed_threshold=initial_speed, duration_threshold=initial_duration)

    message = f"Initiating run training procedure..."
    console.echo(message=message, level=LogLevel.INFO)

    # Tracks the data necessary to update the training progress bar
    previous_time = 0

    # Tracks when speed and / or duration thresholds are updated. This is necessary to redraw the threshold lines in
    # the visualizer plot
    previous_speed_threshold = copy.copy(initial_speed)
    previous_duration_threshold = copy.copy(initial_duration)

    # Also pre-initializes the speed and duration trackers
    speed_threshold: np.float64 = np.float64(0)
    duration_threshold: np.float64 = np.float64(0)

    # Final checkpoint
    message = (
        f"Runtime preparation: Complete. Carry out all final checks and adjustments, such as priming the water "
        f"delivery valve at this time. When you are ready to start the runtime, use the UI to 'resume' it."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Allows the user to manipulate the valve via the UI. Since Zaber interface should also be active, the user can also
    # manually reposition all Zaber motors.
    while ui.pause_runtime:
        # Updates the visualizer plot ~every 30 ms. This should be enough to reliably capture all events of
        # interest and appear visually smooth to human observers.
        visualizer.update()

        if ui.reward_signal:
            runtime.deliver_reward(reward_size=ui.reward_volume)

        if ui.open_valve:
            runtime.toggle_valve(True)

        if ui.close_valve:
            runtime.toggle_valve(False)

    # Ensures the valve is closed before continuing
    runtime.toggle_valve(False)

    # Resets the valve tracker array before proceeding. This allows the user to use the section above to debug the
    # valve, potentially dispensing a lot of water in the process. If the tracker is not reset, this may immediately
    # terminate the runtime and lead to an inaccurate tracking of the water volume received by the animal.
    # noinspection PyTypeChecker
    valve_tracker.write_data(index=0, data=0)
    # noinspection PyTypeChecker
    valve_tracker.write_data(index=1, data=0)

    message = f"Initiating lick training procedure..."
    console.echo(message=message, level=LogLevel.INFO)

    # Initializes the main training loop. The loop will run either until the total training time expires, the maximum
    # volume of water is delivered or the loop is aborted by the user.
    previous_reward_volume = 0

    # If the runtime is paused, this is used to extend the training runtime to account for the time spent in the
    # paused state.
    additional_time = 0

    # Configures all system components to support run training
    runtime.run_train()

    # Creates a tqdm progress bar that tracks the overall training progress by communicating the total volume of water
    # delivered to the animal
    progress_bar = tqdm(
        total=round(maximum_water_volume, ndigits=3),
        desc="Delivered water volume",
        unit="ml",
        bar_format="{l_bar}{bar}| {n:.3f}/{total:.3f} {postfix}",
    )

    runtime_timer.reset()
    speed_timer.reset()  # It is critical to reset BOTh timers at the same time.
    while runtime_timer.elapsed < (training_time + additional_time):
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
            a=initial_speed + (increase_steps * speed_step) + (ui.speed_modifier * 0.01),
            a_min=0.1,  # Minimum value
            a_max=maximum_speed,  # Maximum value
        )
        duration_threshold = np.clip(
            a=initial_duration + (increase_steps * duration_step) + (ui.duration_modifier * 10),
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

        # If the animal licks during the period that separates two rewards, this is interpreted as the animal
        # consuming the previous and any other leftover rewards.
        if previous_licks < visualizer.lick_count:
            previous_licks = visualizer.lick_count
            unconsumed_count = 0

        # If the speed is above the speed threshold, and the animal has been maintaining the above-threshold speed
        # for the required duration, delivers 5 uL of water. If the speed is above threshold, but the animal has
        # not yet maintained the required duration, the loop will keep cycling and accumulating the timer count.
        # This is done until the animal either reaches the required duration or drops below the speed threshold.
        if current_speed >= speed_threshold and speed_timer.elapsed >= duration_threshold:
            # Only issues the rewards if the unconsumed reward counter is below the threshold.
            if unconsumed_count < maximum_unconsumed_rewards:
                runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water

                # 5 uL == 0.005 ml
                # Updates the progress bar whenever the animal receives (automated) rewards. The progress bar
                # purposefully does not track 'manual' water rewards.
                progress_bar.update(0.005)

                # Increments the unconsumed reward count each time a reward is delivered
                unconsumed_count += 1

            # If the animal does not consume rewards, still issues auditory tones, but does not deliver water
            # rewards.
            else:
                runtime.simulate_reward()

            # Also resets the timer. While mice typically stop to consume water rewards, which would reset the
            # timer, this guards against animals that carry on running without consuming water rewards.
            speed_timer.reset()

            # If the epoch timer was active for the current epoch, resets the timer
            epoch_timer_engaged = False

        # If the current speed is below the speed threshold, acts depending on whether the runtime is configured to
        # allow dipping below the threshold
        elif current_speed < speed_threshold:
            # If the user did not allow dipping below the speed threshold, resets the run duration timer.
            if maximum_idle_time == 0:
                speed_timer.reset()

            # If the user has enabled brief dips below the speed threshold, starts the epoch timer to ensure the
            # animal recovers the speed in the allotted time.
            elif not epoch_timer_engaged:
                epoch_timer.reset()
                epoch_timer_engaged = True

            # If epoch timer is enabled, checks whether the animal has failed to recover its running speed in time.
            # If so, resets the run duration timer.
            elif epoch_timer.elapsed >= maximum_idle_time:
                speed_timer.reset()
                epoch_timer_engaged = False

        # If the animal is maintaining the required speed and the epoch timer was activated by the animal dipping
        # below the speed threshold, deactivates the timer. This is essential for ensuring the 'step discount'
        # time is applied to each case of speed dipping below the speed threshold, rather than the entire run epoch.
        elif epoch_timer_engaged and current_speed >= speed_threshold and speed_timer.elapsed < duration_threshold:
            epoch_timer_engaged = False

        # Updates the time display when each second passes. This updates the 'suffix' of the progress bar to keep
        # track of elapsed training time. Accounts for any additional time spent in the 'paused' state.
        elapsed_time = runtime_timer.elapsed - additional_time
        if elapsed_time > previous_time:
            previous_time = elapsed_time  # Updates previous time

            # Updates the time display without advancing the progress bar
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)
            progress_bar.set_postfix_str(
                f"Time: {elapsed_minutes:02d}:{elapsed_seconds:02d}/{maximum_training_time:02d}:00"
            )

            # Refreshes the display to show updated time without changing progress
            progress_bar.refresh()

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

        # If the ui detects a reward delivery signal, delivers the reward to the animal.
        if ui.reward_signal:
            runtime.deliver_reward(reward_size=ui.reward_volume)  # Delivers 5 uL of water

        # Adjusts the reward volume each time it is updated in the GUI
        if ui.reward_volume != previous_reward_volume:
            runtime.configure_reward_parameters(reward_size=ui.reward_volume)

        # If the user sent the abort command, terminates the training early with an error message.
        if ui.exit_signal:
            terminate = False
            message = "Run training runtime abort signal: received. Are you sure you want to abort the runtime?"
            console.echo(message=message, level=LogLevel.WARNING)
            while True:
                answer = input("Enter 'yes' or 'no': ")

                if answer.lower() == "yes":
                    terminate = True  # Sets the terminate flag
                    break  # Breaks the while loop

                elif answer.lower() == "no":
                    # Returns to running the runtime
                    break

            if terminate:
                message = "Aborting the training with a graceful shutdown procedure."
                console.echo(message=message, level=LogLevel.ERROR)
                break

        # If the ui detects a pause command, enters a holding loop.
        if ui.pause_runtime:
            pause_start = runtime_timer.elapsed
            message = "Run training runtime: paused due to user request. Note, sensor readout monitoring is suspended."
            console.echo(message=message, level=LogLevel.WARNING)

            # Switches runtime to the idle state
            runtime.idle()

            # Caches valve data to discard any valve data accumulated during the idle period.
            cached_valve_pulse_count = valve_tracker.read_data(index=0)
            cached_water_volume = valve_tracker.read_data(index=1)

            # Blocks in-place until the user either unpauses or aborts the training.
            abort_stage: bool = False
            while ui.pause_runtime:
                visualizer.update()  # Continuously updates the visualizer

                # If the ui detects a reward delivery signal, delivers the reward to the animal
                if ui.reward_signal:
                    runtime.deliver_reward(reward_size=ui.reward_volume)  # Delivers the requested volume of water

                # Adjusts the reward volume each time it is updated in the GUI
                if ui.reward_volume != previous_reward_volume:
                    runtime.configure_reward_parameters(reward_size=ui.reward_volume)

                # If the user requests for the paused runtime to be aborted, terminates the runtime.
                if ui.exit_signal:
                    message = "Run training runtime abort signal: received. Are you sure you want to abort the runtime?"
                    console.echo(message=message, level=LogLevel.WARNING)
                    while True:
                        answer = input("Enter 'yes' or 'no': ")

                        if answer.lower() == "yes":
                            abort_stage = True
                            break  # Breaks the while loop

                        elif answer.lower() == "no":
                            break  # Returns to the pause state

                    # Escapes the pause loop if the user chose to abort the runtime
                    if abort_stage:
                        break
            else:
                # Discards any valve data accumulated during the idle period
                valve_tracker.write_data(index=0, data=cached_valve_pulse_count)
                valve_tracker.write_data(index=1, data=cached_water_volume)

                # Restores the runtime state
                runtime.run_train()

                # Resets unconsumed reward tracker
                unconsumed_count = 0

            # Updates the 'additional time' value to reflect the time spent inside the 'paused' state. This
            # increases the training time to counteract the duration of the 'paused' state.
            additional_time += runtime_timer.elapsed - pause_start

            # Escapes the outer (experiment state) 'while loop'
            if abort_stage:
                message = f"Run training runtime: aborted due to user request."
                console.echo(message=message, level=LogLevel.ERROR)
                break

    # Close the progress bar
    progress_bar.close()

    # Shutdown sequence:
    message = f"Training runtime: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Directly overwrites the final running speed and duration thresholds in the descriptor instance stored in the
    # runtime attributes. This ensures the descriptor properly reflects the final thresholds used at the end of
    # the training.
    if not isinstance(runtime.descriptor, LickTrainingDescriptor):  # This is to appease mypy
        runtime.descriptor.final_run_speed_threshold_cm_s = float(speed_threshold)
        runtime.descriptor.final_run_duration_threshold_s = float(duration_threshold / 1000)  # Converts from s to ms

    # Terminates the ui
    ui.shutdown()

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()


def experiment_logic(
    experimenter: str,
    project_name: str,
    experiment_name: str,
    animal_id: str,
    animal_weight: float,
    maximum_unconsumed_rewards: int = 1,
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
        project_name: The name of the project for which the experiment is conducted.
        experiment_name: The name or ID of the experiment to be conducted.
        animal_id: The numeric ID of the animal participating in the experiment.
        animal_weight: The weight of the animal, in grams, at the beginning of the experiment session.
        maximum_unconsumed_rewards: The maximum number of rewards that can be delivered without the animal consuming
            them, before reward delivery (but not the experiment!) pauses until the animal consumes available rewards.
            If this is set to a value below 1, the unconsumed reward limit will not be enforced. A value of 1 means
            the animal has to consume each reward before getting the next reward.
    """
    message = f"Initializing {experiment_name} experiment runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters
    system_configuration = get_system_configuration()

    # Verifies that the target project exists
    project_folder = system_configuration.paths.root_directory.joinpath(project_name)
    if not project_folder.exists():
        message = (
            f"Unable to execute the {experiment_name} experiment for the animal {animal_id} of project {project_name}. "
            f"The target project does not exist on the local machine. Use the 'sl-create-project' command to create "
            f"the project before running training or experiment sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Initializes the SessionData and creates the necessary session directory hierarchy as part of this initialization
    # process
    session_data = SessionData.create(
        project_name=project_name,
        animal_id=animal_id,
        session_type=_experiment,
        experiment_name=experiment_name,
    )

    # Caches current sl-experiment and Python versions to disk as a version_data.yaml file.
    write_version_data(session_data)

    # Verifies that the Water Restriction log and the Surgery log Google Sheets are accessible. To do so, instantiates
    # both classes to run through the init checks. The classes are later re-instantiated during session data
    # preprocessing
    project_configuration: ProjectConfiguration = ProjectConfiguration.from_yaml(  # type: ignore
        file_path=session_data.raw_data.project_configuration_path
    )
    _ = WaterSheet(
        animal_id=int(animal_id),
        session_date=session_data.session_name,
        credentials_path=system_configuration.paths.google_credentials_path,
        sheet_id=project_configuration.water_log_sheet_id,
    )
    _ = SurgerySheet(
        project_name=project_name,
        animal_id=int(animal_id),
        credentials_path=system_configuration.paths.google_credentials_path,
        sheet_id=project_configuration.surgery_sheet_id,
    )

    # Uses initialized SessionData instance to load the experiment configuration data
    experiment_config: MesoscopeExperimentConfiguration = MesoscopeExperimentConfiguration.from_yaml(  # type: ignore
        file_path=Path(session_data.raw_data.experiment_configuration_path)
    )

    # Verifies that all Mesoscope-VR states used during experiments are valid
    valid_states = {1, 2}
    for state in experiment_config.experiment_states.values():
        if state.system_state_code not in valid_states:
            message = (
                f"Invalid Mesoscope-VR system state code {state.system_state_code} encountered when verifying "
                f"{experiment_name} experiment configuration. Currently, only codes 1 (rest) and 2 (run) are supported "
                f"for the Mesoscope-VR system."
            )
            console.error(message=message, error=ValueError)

    # Generates the session descriptor class
    descriptor = MesoscopeExperimentDescriptor(
        experimenter=experimenter,
        mouse_weight_g=animal_weight,
        dispensed_water_volume_ml=0.0,
        maximum_unconsumed_rewards=maximum_unconsumed_rewards,
    )

    # Initializes the main runtime interface class.
    runtime = _MesoscopeExperiment(
        experiment_configuration=experiment_config,
        session_data=session_data,
        session_descriptor=descriptor,
    )

    runtime_timer = PrecisionTimer("s")  # Initializes the timer to enforce experiment state durations

    # Uses runtime trackers extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Initializes the runtime control UI. Has to be done before the visualizer
    ui = RuntimeControlUI()

    # Visualizer initialization HAS to happen after the runtime start to avoid interfering with cameras.
    visualizer = BehaviorVisualizer(
        lick_tracker=lick_tracker, valve_tracker=valve_tracker, distance_tracker=speed_tracker
    )

    # To avoid photobleaching during potentially lengthy preparation stage, only starts the Mesoscope once the
    # experimenter unpauses the runtime.
    runtime.start_runtime(ui=ui)

    # Main runtime loop. It loops over all submitted experiment states and ends the runtime after executing the last
    # state
    previous_reward_volume = 0
    terminate_runtime: bool = False
    for state in experiment_config.experiment_states.values():
        runtime_timer.reset()  # Resets the timer

        # Sets the Experiment state
        runtime.change_runtime_state(state.experiment_state_code)

        # Resolves and sets the Mesoscope-VR system state
        if state.system_state_code == 1:
            runtime.rest()
        elif state.system_state_code == 2:
            runtime.run()

        # Creates a tqdm progress bar for the current experiment state
        with tqdm(
            total=state.state_duration_s,
            desc=f"Executing experiment state {state.experiment_state_code}",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s",
        ) as pbar:
            previous_seconds = 0

            # If the runtime is paused, this is used to extend the experiment state duration to account for the time
            # spent in the paused state.
            additional_time = 0

            while runtime_timer.elapsed < (state.state_duration_s + additional_time):
                visualizer.update()  # Continuously updates the visualizer

                # Handles Unity-MQTT communication
                runtime._unity_cycle()

                # Updates the progress bar every second. While the current implementation is technically not safe,
                # we know that the loop will cycle much faster than 1 second, so it should not be possible for the
                # delta to ever exceed 1 second. Note, discounts any time spent inside the paused state.
                if (runtime_timer.elapsed - additional_time) > previous_seconds:
                    pbar.update(1)
                    previous_seconds = runtime_timer.elapsed - additional_time

                # If the ui detects a pause command, enters a holding loop. Pauses are handled before all other
                # states to comply with recovering from receiving Unity termination messages
                if ui.pause_runtime:
                    pause_start = runtime_timer.elapsed
                    message = "Experiment runtime: paused due to user request."
                    console.echo(message=message, level=LogLevel.WARNING)

                    # Caches valve data to discard any valve data accumulated during the idle period.
                    cached_valve_pulse_count = valve_tracker.read_data(index=0)
                    cached_water_volume = valve_tracker.read_data(index=1)

                    # Blocks in-place until the user either unpauses or aborts the runtime.
                    while ui.pause_runtime:
                        visualizer.update()  # Continuously updates the visualizer

                        # Keeps unity communication loop running. Primarily, this is done to support receiving
                        # unity termination messages if the user chooses to pause the runtime first and then terminate
                        # the Unity
                        runtime._unity_cycle()

                        # If the ui detects a reward delivery signal, delivers the reward to the animal
                        if ui.reward_signal:
                            runtime.deliver_reward(reward_size=ui.reward_volume)  # Delivers 5 uL of water

                        # Adjusts the reward volume each time it is updated in the GUI
                        if ui.reward_volume != previous_reward_volume:
                            runtime.configure_reward_parameters(reward_size=ui.reward_volume)

                        # Switches the guidance status in response to user requests
                        if ui.enable_guidance != previous_guidance_state:
                            previous_guidance_state = ui.enable_guidance
                            runtime._toggle_lick_guidance(enable_guidance=previous_guidance_state)

                        # If the user requests for the paused stage to be aborted, terminates the runtime.
                        if ui.exit_signal:
                            message = (
                                "Experiment runtime abort signal: received. Are you sure you want to abort the runtime?"
                            )
                            console.echo(message=message, level=LogLevel.WARNING)
                            while True:
                                answer = input("Enter 'yes' or 'no': ")

                                if answer.lower() == "yes":
                                    terminate_runtime = True  # Sets the terminate flag
                                    break  # Breaks the while loop

                                elif answer.lower() == "no":
                                    # Returns to running the runtime
                                    break

                            if terminate_runtime:
                                break  # Escapes the pause 'while' loop
                    else:
                        # Discards any valve data accumulated during the idle period
                        valve_tracker.write_data(index=0, data=cached_valve_pulse_count)
                        valve_tracker.write_data(index=1, data=cached_water_volume)

                        # Updates the 'additional time' value to reflect the time spent inside the 'paused' state.
                        # This increases the experiment stage duration to counteract the duration of the 'paused'
                        # state.
                        additional_time += runtime_timer.elapsed - pause_start

                        # Re-queries the cue sequence as Unity re-initialization always generates a new sequence.
                        # Also ensures that Unity has been properly re-initialized.
                        if runtime.unity_terminated:
                            runtime._get_cue_sequence()
                            # Resets the unity termination tracker
                            runtime.unity_terminated = False

                        # Restores the runtime state
                        if state.system_state_code == 1:
                            runtime.rest()
                        elif state.system_state_code == 2:
                            runtime.run()

                    # Escapes the outer (experiment state) 'while loop if the user chose to terminate the runtime
                    if terminate_runtime:
                        break

                # If the user sent the abort command, terminates the runtime early with an error message.
                if ui.exit_signal:
                    message = "Experiment runtime abort signal: received. Are you sure you want to abort the runtime?"
                    console.echo(message=message, level=LogLevel.WARNING)
                    while True:
                        answer = input("Enter 'yes' or 'no': ")

                        if answer.lower() == "yes":
                            terminate_runtime = True  # Sets the terminate flag
                            break  # Breaks the while loop

                        elif answer.lower() == "no":
                            # Returns to running the runtime
                            break

                    if terminate_runtime:
                        break

                # If the ui detects a reward delivery signal, delivers the reward to the animal
                if ui.reward_signal:
                    runtime.deliver_reward(reward_size=ui.reward_volume)  # Delivers 5 uL of water

                # Adjusts the reward volume each time it is updated in the GUI
                if ui.reward_volume != previous_reward_volume:
                    runtime.configure_reward_parameters(reward_size=ui.reward_volume)

                # If the user switches the guidance state, adjusts the runtime guidance parameter
                if ui.enable_guidance != previous_guidance_state:
                    previous_guidance_state = ui.enable_guidance
                    runtime._toggle_lick_guidance(enable_guidance=previous_guidance_state)

            if terminate_runtime:
                message = f"Experiment runtime: aborted due to user request."
                console.echo(message=message, level=LogLevel.ERROR)
                break  # Escapes the experiment 'for' loop

    # Shutdown sequence:
    message = f"Experiment runtime: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Shuts down the UI
    ui.shutdown()

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()


def window_checking_logic(
    project_name: str,
    animal_id: str,
) -> None:
    """Encapsulates the logic used to verify the surgery quality (cranial window) and generate the initial snapshot of
    the Mesoscope-VR system configuration for a newly added animal of the target project.

    This function is used when new animals are added to the project, before any other training or experiment runtime.
    Primarily, it is used to verify that the surgery went as expected and the animal is fit for providing high-quality
    scientific data. As part of this process, the function also generates the snapshot of zaber motor positions and the
    mesoscope objective position to be reused by future sessions.

    Notes:
        This function largely behaves similar to all other training and experiment session runtimes. However, it does
        not use most of the Mesoscope-VR components and does not make most of the runtime data files typically generated
        by other sessions. All window checking sessions are automatically marked as 'incomplete' and excluded from
        automated data processing.

    Args:
        project_name: The name of the project to which the checked animal belongs.
        animal_id: The numeric ID of the animal whose cranial window is being checked.
    """
    message = f"Initializing window checking runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters.
    system_configuration = get_system_configuration()

    # Verifies that the target project exists
    project_folder = system_configuration.paths.root_directory.joinpath(project_name)
    if not project_folder.exists():
        message = (
            f"Unable to execute the window checking for the animal {animal_id} of project {project_name}. The target "
            f"project does not exist on the local machine. Use the 'sl-create-project' command to create the project "
            f"before running training or experiment sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Initializes the SessionData and MesoscopeData classes for the session.
    session_data = SessionData.create(
        project_name=project_name,
        animal_id=animal_id,
        session_type=_window,
    )
    mesoscope_data = MesoscopeData(session_data=session_data)

    # Caches current sl-experiment and Python versions to disk as a version_data.yaml file.
    write_version_data(session_data)

    # Verifies that the Surgery log Google Sheet is accessible. To do so, instantiates its interface class to run
    # through the init checks. The class is later re-instantiated during session data preprocessing
    project_configuration: ProjectConfiguration = ProjectConfiguration.from_yaml(  # type: ignore
        file_path=session_data.raw_data.project_configuration_path
    )
    _ = SurgerySheet(
        project_name=project_name,
        animal_id=int(animal_id),
        credentials_path=system_configuration.paths.google_credentials_path,
        sheet_id=project_configuration.surgery_sheet_id,
    )

    message = f"Initializing interface classes..."
    console.echo(message=message, level=LogLevel.INFO)

    # Initializes the data logger. This initialization follows the same procedure as the BehaviorTraining or
    # MesoscopeExperiment classes
    logger: DataLogger = DataLogger(
        output_directory=Path(session_data.raw_data.raw_data_path),
        instance_name="behavior",  # Creates behavior_log subfolder under raw_data
        sleep_timer=0,
        exist_ok=True,
        process_count=1,
        thread_count=10,
    )
    logger.start()

    # Initializes the face camera. Body cameras are not used during window checking.
    cameras = VideoSystems(data_logger=logger, output_directory=session_data.raw_data.camera_data_path)
    cameras.start_face_camera()
    message = f"Face camera display: Started."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # While we can connect to ports managed by ZaberLauncher, ZaberLauncher cannot connect to ports managed via
    # software. Therefore, we have to make sure ZaberLauncher is running before connecting to motors.
    message = (
        "Preparing to connect to all Zaber motor controllers. Make sure that ZaberLauncher app is running before "
        "proceeding further. If ZaberLauncher is not running, you WILL NOT be able to manually control Zaber motor "
        "positions until you reset the runtime."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    input("Enter anything to continue: ")

    # Initializes the Zaber motors interface
    zaber_motors: ZaberMotors = ZaberMotors(
        zaber_positions_path=mesoscope_data.vrpc_persistent_data.zaber_positions_path
    )

    # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is safe to
    # proceed with motor movements.
    message = (
        "Preparing to move Zaber motors into mounting position. Remove the mesoscope objective, swivel out the VR "
        "screens, and make sure the animal is NOT mounted on the rig. Failure to fulfill these steps may DAMAGE "
        "the mesoscope and / or HARM the animal."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    input("Enter anything to continue: ")

    # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not intersect
    # with each other, so it is safe to move both assemblies at the same time.
    zaber_motors.prepare_motors()

    # Sets the motors into the mounting position. Since there should not be any previous positions, all motors should
    # be set to the default mounting position.
    zaber_motors.mount_position()

    message = "Motor Positioning: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # This section is where most manual manipulations take place, as the user needs to move the objective to the imaging
    # plane and check the quality of surgery.
    message = (
        "Position the mesoscope objective above the imaging field to asses the animal surgery and cranial window "
        "implantation quality. Exercise caution when moving HeadBar Roll and Pitch axes motors. Make sure you are "
        "satisfied with the imaging quality before proceeding further."
    )
    console.echo(message=message, level=LogLevel.INFO)
    input("Enter anything to continue: ")

    # Generates the mesoscope positions file precursor in the raw_data folder of the managed session and forces the
    # user to update it with the current mesoscope objective positions.
    mesoscope_positions = MesoscopePositions()
    mesoscope_positions.to_yaml(file_path=Path(session_data.raw_data.mesoscope_positions_path))
    message = f"Mesoscope positions precursor file: Generated."
    console.echo(message=message, level=LogLevel.SUCCESS)

    message = (
        "Generate the cranial window screenshot and record the mesoscope objective positions in the precursor "
        "mesoscope_positions file."
    )
    console.echo(message=message, level=LogLevel.INFO)
    input("Enter anything to continue: ")

    # Retrieves current motor positions and packages them into a ZaberPositions object.
    zaber_positions = zaber_motors.generate_position_snapshot()

    # Dumps zaber data into the raw_data folder of the new session and the persistent_data folder of the animal
    zaber_positions.to_yaml(file_path=Path(session_data.raw_data.zaber_positions_path))
    zaber_positions.to_yaml(file_path=Path(mesoscope_data.vrpc_persistent_data.zaber_positions_path))

    message = f"Zaber motor position snapshot: Saved."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Forces the user to always have a single cranial window screenshot and does not allow proceeding until the
    # screenshot is generated.
    mesodata_path = Path(mesoscope_data.scanimagepc_data.meso_data_path)
    screenshots = [screenshot for screenshot in mesodata_path.glob("*.png")]
    while len(screenshots) != 1:
        message = (
            f"Unable to retrieve the screenshot of the cranial window and the dot-alignment from the "
            f"ScanImage PC. Specifically, expected a single .png file to be stored in the root mesoscope "
            f"data folder of the ScanImagePC, but instead found {len(screenshots)} candidates. Generate a "
            f"single screenshot of the cranial window and the dot-alignment on the ScanImagePC by "
            f"positioning them side-by-side and using 'Win + PrtSc' combination. Remove any extra "
            f"screenshots stored in the folder before proceeding."
        )
        console.echo(message=message, level=LogLevel.ERROR)
        input("Enter anything to continue: ")
        screenshots = [screenshot for screenshot in mesodata_path.glob("*.png")]

    # Moves the screenshot to the raw_data session folder
    screenshot_path: Path = screenshots[0]
    sh.move(src=screenshot_path, dst=Path(session_data.raw_data.window_screenshot_path))
    message = f"Cranial window and dot-alignment screenshot: Saved."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Forces the user to update the mesoscope positions file with current mesoscope data.
    mesoscope_positions = MesoscopePositions.from_yaml(  # type: ignore
        file_path=Path(session_data.raw_data.mesoscope_positions_path)
    )
    while (
        mesoscope_positions.mesoscope_x == 0.0
        and mesoscope_positions.mesoscope_y == 0.0
        and mesoscope_positions.mesoscope_z == 0.0
        and mesoscope_positions.mesoscope_roll == 0.0
        and mesoscope_positions.mesoscope_fast_z == 0.0
        and mesoscope_positions.mesoscope_tip == 0.0
        and mesoscope_positions.mesoscope_tilt == 0.0
    ):
        message = (
            "Failed to verify that the mesoscope_positions.yaml file stored inside the session raw_data directory "
            "has been updated to include the mesoscope objective positions used during runtime. Manually edit the "
            "mesoscope_positions.yaml file and replace the default text under the necessary mesoscope axis position "
            "fields with coordinates displayed in the ScanImage software or the ThorLabs pad. Make sure to save the "
            "changes to the file by using 'CTRL+S' combination."
        )
        console.echo(message=message, level=LogLevel.ERROR)
        input("Enter anything to continue: ")

        # Reloads the mesoscope positions data each time to verify whether the user ahs edited the data.
        mesoscope_positions = MesoscopePositions.from_yaml(  # type: ignore
            file_path=Path(session_data.raw_data.mesoscope_positions_path)
        )

    # Dumps the updated data into the persistent_data folder of the animal
    mesoscope_positions.to_yaml(file_path=Path(mesoscope_data.vrpc_persistent_data.mesoscope_positions_path))

    message = f"Mesoscope-VR and cranial window state snapshot: Generated."
    console.echo(message=message, level=LogLevel.SUCCESS)

    message = f"Retracting the lick-port away from the animal..."
    console.echo(message=message, level=LogLevel.INFO)

    # Helps with removing the animal from the rig by retracting the lick-port in the Y-axis (moving it away from the
    # animal).
    zaber_motors.unmount_position()

    message = "Motor Positioning: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Instructs the user to remove all objects that may interfere with moving the motors.
    message = (
        "REMOVE the animal and the mesoscope objective from the VR rig. Failure to do so may HARM the animal and "
        "DAMAGE the mesoscope. This is the last manual checkpoint, once you progress past this point, the "
        "Microscope-VR system will reset Zaber motor positions and start data preprocessing."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    input("Enter anything to continue: ")

    # Shuts down zaber bindings
    zaber_motors.park_position()
    zaber_motors.disconnect()

    # Terminates the face camera
    cameras.stop()

    # Stops the data logger
    logger.stop()

    # Triggers preprocessing pipeline. In this case, since there is no data to preprocess, the pipeline primarily just
    # copies the session raw_data folder to the NAS and BioHPC server.
    preprocess_session_data(session_data=session_data)

    # Ends the runtime
    message = f"Window checking runtime: Terminated."
    console.echo(message=message, level=LogLevel.SUCCESS)


def maintenance_logic() -> None:
    """Encapsulates the logic used to maintain various components of the Mesoscope-VR system.

    This runtime is primarily used to verify and, if necessary, recalibrate the water valve between training or
    experiment days and to maintain the surface material of the running wheel.
    """

    message = f"Initializing Mesoscope-VR system maintenance runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters. This runtime only needs access to the base acquisition
    # system configuration data, as it does not generate any new data by itself.
    system_configuration = get_system_configuration()

    # Initializes a timer used to optimize console printouts for using the valve in debug mode (which also posts
    # things to console).
    delay_timer = PrecisionTimer("s")

    message = f"Initializing interface classes..."
    console.echo(message=message, level=LogLevel.INFO)

    # All calibration procedures are executed in a temporary directory deleted after runtime.
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

        # While we can connect to ports managed by ZaberLauncher, ZaberLauncher cannot connect to ports managed via
        # software. Therefore, we have to make sure ZaberLauncher is running before connecting to motors.
        message = (
            "Preparing to connect to all Zaber motor controllers. Make sure that ZaberLauncher app is running before "
            "proceeding further. If ZaberLauncher is not running, you WILL NOT be able to manually control Zaber motor "
            "positions until you reset the runtime."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Providing the class with an invalid path makes sure it falls back to using default positions cached in
        # non-volatile memory of each device.
        zaber_motors: ZaberMotors = ZaberMotors(zaber_positions_path=output_path.joinpath("invalid_path.yaml"))

        # Initializes the interface for the Actor MicroController that manages the valve and break modules.
        valve: ValveInterface = ValveInterface(
            valve_calibration_data=system_configuration.microcontrollers.valve_calibration_data,  # type: ignore
            debug=True,  # Hardcoded to True during maintenance
        )
        wheel: BreakInterface = BreakInterface(
            minimum_break_strength=system_configuration.microcontrollers.minimum_break_strength_g_cm,
            maximum_break_strength=system_configuration.microcontrollers.maximum_break_strength_g_cm,
            object_diameter=system_configuration.microcontrollers.wheel_diameter_cm,
            debug=True,  # Hardcoded to True during maintenance
        )
        controller: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),  # Hardcoded
            microcontroller_serial_buffer_size=8192,  # Hardcoded
            microcontroller_usb_port=system_configuration.microcontrollers.actor_port,
            data_logger=logger,
            module_interfaces=(valve, wheel),
        )
        controller.start()
        controller.unlock_controller()  # Unlocks actor controller to allow manipulating managed hardware

        # Delays for 1 second for the valve to initialize and send the state message. This avoids the visual clash
        # with the zaber positioning dialog
        delay_timer.delay_noblock(delay=1)

        message = f"Actor MicroController interface: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

        message = (
            "Preparing to move Zaber motors into maintenance position. Remove the mesoscope objective, swivel out the "
            "VR screens, and make sure the animal is NOT mounted on the rig. Failure to fulfill these steps may DAMAGE "
            "the mesoscope and / or HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        zaber_motors.prepare_motors()  # homes all motors
        zaber_motors.maintenance_position()  # Moves all motors to maintenance position

        message = f"Zaber motors: Positioned for Mesoscope-VR system maintenance."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Notifies the user about supported calibration commands
        message = (
            "Supported valve commands: open, close, close_10, reference, reward, calibrate_15, calibrate_30, "
            "calibrate_45, calibrate_60. Supported break (wheel) commands: lock, unlock. Use 'q' command to terminate "
            "the runtime."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Precomputes correct auditory tone duration from Mesoscope-VR configuration
        tone_duration: float = convert_time(  # type: ignore
            from_units="ms", to_units="us", time=system_configuration.microcontrollers.auditory_tone_duration_ms
        )

        while True:
            command = input()  # Silent input to avoid visual spam.

            if command == "open":
                message = f"Opening the valve..."
                console.echo(message=message, level=LogLevel.INFO)
                valve.toggle(state=True)

            if command == "close":
                message = f"Closing the valve..."
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
                            message=f"Remaining time: {10 - (delay_timer.elapsed - start)} seconds...",
                            level=LogLevel.INFO,
                        )
                valve.toggle(state=False)  # Closes the valve after a 10-second delay

            if command == "reward":
                message = f"Delivering 5 uL water reward..."
                console.echo(message=message, level=LogLevel.INFO)
                pulse_duration = valve.get_duration_from_volume(target_volume=5.0)
                valve.set_parameters(
                    pulse_duration=pulse_duration,
                    calibration_delay=np.uint32(300000),  # Hardcoded for safety reasons
                    calibration_count=np.uint16(system_configuration.microcontrollers.valve_calibration_pulse_count),
                    tone_duration=np.uint32(tone_duration),
                )
                valve.send_pulse()

            if command == "reference":
                message = f"Running the reference valve calibration procedure..."
                console.echo(message=message, level=LogLevel.INFO)
                message = f"Expecting to dispense 1 ml of water (200 pulses x 5 uL each)..."
                console.echo(message=message, level=LogLevel.INFO)
                pulse_duration = valve.get_duration_from_volume(target_volume=5.0)
                valve.set_parameters(
                    pulse_duration=pulse_duration,  # Hardcoded to 5 uL for consistent behavior
                    calibration_delay=np.uint32(300000),  # Hardcoded for safety reasons
                    calibration_count=np.uint16(200),  # Hardcoded to 200 pulses for consistent behavior
                    tone_duration=np.uint32(tone_duration),
                )
                valve.calibrate()

            if command == "calibrate_15":
                message = f"Running 15 ms pulse duration valve calibration..."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(
                    pulse_duration=np.uint32(15000),  # 15 ms in us
                    calibration_delay=np.uint32(300000),  # Hardcoded for safety reasons
                    calibration_count=np.uint16(system_configuration.microcontrollers.valve_calibration_pulse_count),
                    tone_duration=np.uint32(tone_duration),
                )
                valve.calibrate()

            if command == "calibrate_30":
                message = f"Running 30 ms pulse valve calibration..."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(
                    pulse_duration=np.uint32(30000),  # 30 ms in us
                    calibration_delay=np.uint32(300000),  # Hardcoded for safety reasons
                    calibration_count=np.uint16(system_configuration.microcontrollers.valve_calibration_pulse_count),
                    tone_duration=np.uint32(tone_duration),
                )
                valve.calibrate()

            if command == "calibrate_45":
                message = f"Running 45 ms pulse valve calibration..."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(
                    pulse_duration=np.uint32(45000),  # 45 ms in us
                    calibration_delay=np.uint32(300000),  # Hardcoded for safety reasons
                    calibration_count=np.uint16(system_configuration.microcontrollers.valve_calibration_pulse_count),
                    tone_duration=np.uint32(tone_duration),
                )
                valve.calibrate()

            if command == "calibrate_60":
                message = f"Running 60 ms pulse valve calibration..."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(
                    pulse_duration=np.uint32(60000),  # 60 ms in us
                    calibration_delay=np.uint32(300000),  # Hardcoded for safety reasons
                    calibration_count=np.uint16(system_configuration.microcontrollers.valve_calibration_pulse_count),
                    tone_duration=np.uint32(tone_duration),
                )
                valve.calibrate()

            if command == "lock":
                message = f"Locking wheel break..."
                console.echo(message=message, level=LogLevel.INFO)
                wheel.toggle(state=True)

            if command == "unlock":
                message = f"Unlocking wheel break..."
                console.echo(message=message, level=LogLevel.INFO)
                wheel.toggle(state=False)

            if command == "q":
                message = f"Terminating Mesoscope-VR maintenance runtime..."
                console.echo(message=message, level=LogLevel.INFO)
                break

        # Instructs the user to remove all objects that may interfere with moving the motors.
        message = (
            "Preparing to reset all Zaber motors. Remove all objects used during Mesoscope-VR maintenance, such as "
            "water collection flasks, from the Mesoscope-VR cage."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        input("Enter anything to continue: ")

        # Shuts down zaber bindings
        zaber_motors.park_position()
        zaber_motors.disconnect()

        # Shuts down microcontroller interfaces
        controller.stop()

        message = f"Actor MicroController interface: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops the data logger
        logger.stop()

        message = f"Mesoscope-VR system maintenance runtime: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)
