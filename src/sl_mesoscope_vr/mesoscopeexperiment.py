"""This module provides the main VR class that abstracts working with Sun lab's mesoscope-VR system"""

from ataraxis_data_structures import DataLogger
from ataraxis_communication_interface import MicroControllerInterface, UnityCommunication
from module_interfaces import (
    TTLInterface,
    EncoderInterface,
    BreakInterface,
    ValveInterface,
    LickInterface,
    TorqueInterface,
)
from pathlib import Path
import numpy as np
from packaging_tools import calculate_directory_checksum
from mesoscope_preprocessing import extract_frames_from_stack
from tqdm import tqdm
import os
from ataraxis_time import PrecisionTimer
from ataraxis_time.time_helpers import get_timestamp
import tempfile
from ataraxis_base_utilities import ensure_directory_exists, console, LogLevel


class MesoscopeExperiment:
    """The base class for all Sun lab Mesoscope experiment runtimes.

    This class provides methods for conducting experiments in the Sun lab using the Mesoscope-VR system. This class
    abstracts most low-level interactions with the VR system and the mesoscope and enables lab members to use a simple
    high-level API for writing their own experiment procedures without worrying about specific hardware interactions.

    Args:
        output_directory: The directory where all experiment data should be saved. Typically, this is the output of the
            ExperimentData class create_session() method.
    """
    def __init__(self, output_directory: Path) -> None:

        self._started: bool = False

        # Initializes the microcontroller data logger. This datalogger works exclusively with microcontroller-generated
        # data.
        self.amc_logger: DataLogger = DataLogger(output_directory=output_directory, instance_name="vr", sleep_timer=0)

        # Actor AMC controls the hardware that needs to be triggered by PC at irregular intervals. Most of such hardware
        # is designed to produce some form of an output: deliver water reward, engage wheel breaks, issue a TTL trigger,
        # etc.
        self._actor_interfaces: tuple[TTLInterface, BreakInterface, ValveInterface] = (
            TTLInterface(),  # Mesoscope trigger
            BreakInterface(
                minimum_break_strength=43.2047,  # 0.6 in oz
                maximum_break_strength=1152.1246,  # 16 in oz
                object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
            ),  # Wheel break
            ValveInterface(valve_calibration_data=((10, 10), (20, 20))),  # Reward solenoid valve
        )
        self._actor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port="/dev/ttyACM1",
            data_logger=self.amc_logger,
            module_interfaces=self._actor_interfaces,
        )

        # Sensor AMC controls the hardware that collects data at regular intervals. This includes lick sensors, torque
        # sensors, and input TTL recorders. Critically, all managed hardware does not rely on hardware interrupt logic
        # to maintain the necessary precision.
        self._sensor_interfaces: tuple[TTLInterface, LickInterface, TorqueInterface] = (
            TTLInterface(),  # Mesoscope frame recorder
            LickInterface(lick_threshold=2000),  # Main lick sensor
            TorqueInterface(
                baseline_voltage=2046,  # ~1.65 V
                maximum_voltage=4095,  # ~3.3 V
                sensor_capacity=720.0779,  # 10 in oz
                object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
            ),  # Wheel torque sensor
        )
        self._sensor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(152),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port="/dev/ttyACM0",
            data_logger=self.amc_logger,
            module_interfaces=self._sensor_interfaces,
        )

        # Encoder AMC is specifically designed to interface with a rotary encoder connected to the running wheel. The
        # encoder uses hardware interrupt logic to maintain high precision and, therefore, it is isolated to a
        # separate microcontroller to ensure adequate throughput.
        self._encoder_interfaces: tuple[EncoderInterface] = (
            EncoderInterface(encoder_ppr=8192, object_diameter=15.0333, cm_per_unity_unit=10.0),
        )
        self._encoder: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(203),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port="/dev/ttyACM2",
            data_logger=self.amc_logger,
            module_interfaces=self._encoder_interfaces,
        )

    def __del__(self):
        self.stop()

    def start(self):
        if not self._started:
            self.amc_logger.start()
            self._actor.start()
            self._actor.unlock_controller()  # Only Actor outputs data, so no need to unlock other controllers.
            self._sensor.start()
            self._encoder.start()

            # Engages the modules that need to stay on during the entire experiment:

            # The mesoscope acquires frames at ~10 Hz and sends triggers with the on-phase duration of ~50 ms, so a
            # 100 Hz polling frequency should be enough to detect all triggers.
            # noinspection PyTypeChecker
            self._sensor.send_message(self._sensor_interfaces[0].check_state(repetition_delay=np.uint32(10000)))

            # Starts monitoring licks. Uses 100 Hz polling frequency, since mice are expected to lick at ~10 Hz rate.
            # noinspection PyTypeChecker
            self._sensor.send_message(self._sensor_interfaces[1].check_state(repetition_delay=np.uint32(10000)))

            # Water Valve receives triggers directly from unity, so we do not need to manipulate the valve state
            # manually. Mesoscope is triggered via a dedicated method.

            self._started = True

    def stop(self):
        if self._started:
            self._started = False
            self.amc_logger.stop()
            self._actor.stop()
            self._sensor.stop()
            self._encoder.stop()

    def vr_rest(self):
        # Engages the break to prevent the mouse from moving the wheel
        # noinspection PyTypeChecker
        self._actor.send_message(self._actor_interfaces[1].toggle(True))

        # Initiates torque monitoring at 100 Hz. The torque can only be accurately measured when the wheel is locked,
        # as it requires a resistance force to trigger the sensor. Since we downsample all data to the mesoscope
        # acquisition rate of ~10 Hz, and do not use torque data in real time, the sampling rate is seto to a 100 Hz.
        # noinspection PyTypeChecker
        self._sensor.send_message(self._sensor_interfaces[2].check_state(repetition_delay=np.uint32(10000)))

        # Temporarily suspends encoder monitoring. Since the wheel is locked, the mouse should not be able to produce
        # meaningful motion data.
        # noinspection PyTypeChecker
        self._encoder.send_message(self._encoder_interfaces[0].dequeue_command)

    def vr_run(self):
        # Initializes encoder monitoring at 1 kHz rate. The encoder aggregates wheel data at native speeds; this rate
        # only determines how often the aggregated data is sent to PC and Unity.
        # noinspection PyTypeChecker
        self._encoder.send_message(self._encoder_interfaces[0].check_state(repetition_delay=np.uint32(1000)))

        # Disables torque monitoring. To accurately measure torque, the sensor requires a resistance force provided by
        # the break. During running, measuring torque is not very reliable and adds little in addition to the
        # encoder.
        # noinspection PyTypeChecker
        self._sensor.send_message(self._sensor_interfaces[2].dequeue_command)

        # Disengages the break to allow the mouse to move the wheel
        # noinspection PyTypeChecker
        self._actor.send_message(self._actor_interfaces[1].toggle(False))

    def mesoscope_on(self):

        # Toggles the mesoscope acquisition trigger to continuously deliver a HIGH signal. The mesoscope will
        # continuously acquire frames as long as the trigger is HIGH.
        # noinspection PyTypeChecker
        self._actor.send_message(self._actor_interfaces[0].toggle(True))

    def mesoscope_off(self):

        # Toggles the mesoscope acquisition trigger to continuously deliver a LOW signal. When the trigger is LOW,
        # the mesoscope will not acquire frames.
        # noinspection PyTypeChecker
        self._actor.send_message(self._actor_interfaces[0].toggle(False))


class ExperimentData:
    """Provides methods for managing the experimental data acquired by all Sun lab pipelines.

    This class functions as the central hub for managing the data across all destinations: The PCs tht acquire the data,
    the NAS, and the data analysis server(s).
    """

    def __init__(self, project_directory: Path, animal_name: str) -> None:

        # Combines project directory and animal name to get the animal directory path
        self._animal_directory = project_directory.joinpath(animal_name)

        # Loops over the input directory and resolves available session folders. If the directory does not exist,
        # sets the variable to an empty tuple and generates the animal directory
        self._sessions: tuple[str, ...]
        if self._animal_directory.exists():
            session: Path
            self._sessions = tuple(
                [str(session.stem) for session in project_directory.glob("*") if session.is_dir()]
            )
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
        ensure_directory_exists(session_path.joinpath('raw'))
        ensure_directory_exists(session_path.joinpath('processed'))

        self._sessions += (session_name,)  # Updates the sessions tuple with the new session name
        return session_path


def convert_old_data(root_directory: Path, remove_sources: bool = False, num_processes: int = None) -> None:
    """A temporary function used to convert old Tyche data to our new format."""

    # Resolves the number of processes to use for processing the data.
    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 4)  # Keeps 4 cores free for other tasks

    # Recursively finds all subdirectories inside the root directory, excluding 'mesoscope_frames'
    subdirectories = [
        directory
        for directory in root_directory.rglob("*")
        if directory.is_dir() and directory.name != "mesoscope_frames"  # Exclude already processed frames.
    ]

    # Collects the paths to each directory with TIFF files OR that has a mesoscope_frames subdirectory. This is used as
    # a heuristic to discover folders with Tyche cohort session data.
    session_directories = set()
    for subdirectory in subdirectories:
        tiff_files_in_subdir = list(subdirectory.glob("*.tif")) + list(subdirectory.glob("*.tiff"))

        # Check for mesoscope_frames subdirectory. This directory would exist in already processed session directories
        has_mesoscope_frames = (subdirectory / "mesoscope_frames").exists()

        if tiff_files_in_subdir or has_mesoscope_frames:
            session_directories.add(subdirectory)

    # Processes each directory in parallel
    for directory in tqdm(session_directories, desc="Processing mesoscope frames for session directories", unit="dir"):
        extract_frames_from_stack(
            directory,
            num_processes,
            remove_sources,
            batch=True,
        )

    # Calculates the checksum for each processed directory
    for directory in tqdm(session_directories, desc="Calculating checksums for session directories", unit="dir"):
        calculate_directory_checksum(
            directory,
            num_processes,
            batch=True,
        )


def test_runtime() -> None:
    temp_dir = Path(tempfile.mkdtemp())
    data_logger = DataLogger(output_directory=temp_dir, instance_name="test_logger")

    from ataraxis_base_utilities import console

    console.enable()

    encoder = EncoderInterface()
    brake = BreakInterface()
    lick = LickInterface()
    valve = ValveInterface(valve_calibration_data=((1000, 0.001), (5000, 0.005), (10000, 0.01)))

    actor_interfaces = (brake, valve)

    encoder_interfaces = (encoder,)

    sensor_interfaces = (lick,)

    actor_interface = MicroControllerInterface(
        controller_id=np.uint8(101),
        data_logger=data_logger,
        module_interfaces=actor_interfaces,
        microcontroller_serial_buffer_size=8192,
        microcontroller_usb_port="/dev/ttyACM1",
        baudrate=115200,
    )

    encoder_interface = MicroControllerInterface(
        controller_id=np.uint8(203),
        data_logger=data_logger,
        module_interfaces=encoder_interfaces,
        microcontroller_serial_buffer_size=8192,
        microcontroller_usb_port="/dev/ttyACM2",
        baudrate=115200,
    )

    sensor_interface = MicroControllerInterface(
        controller_id=np.uint8(152),
        data_logger=data_logger,
        module_interfaces=sensor_interfaces,
        microcontroller_serial_buffer_size=8192,
        microcontroller_usb_port="/dev/ttyACM0",
        baudrate=115200,
    )

    timer = PrecisionTimer("s")

    # cue_topic = 'CueSequence/'
    # unity_communication = UnityCommunication(monitored_topics=cue_topic)
    # unity_communication.connect()

    # unity_communication.send_data("Display/Blank/")
    # timer.delay_noblock(5)
    # unity_communication.send_data("Display/Show/")

    # unity_communication.send_data("CueSequenceTrigger/")

    # while not unity_communication.has_data:
    #     pass

    # sequence: bytes = unity_communication.get_data()

    data_logger._vacate_shared_memory_buffer()
    actor_interface.vacate_shared_memory_buffer()
    sensor_interface.vacate_shared_memory_buffer()
    encoder_interface.vacate_shared_memory_buffer()

    data_logger.start()

    actor_interface.start()
    encoder_interface.start()
    sensor_interface.start()

    actor_interface.unlock_controller()

    actor_interface.send_message(brake.toggle(state=False))  # Disengage the break

    # Calibrate the encoder to only report CCW movement and start position recording
    encoder_interface.send_message(encoder.set_parameters(report_cw=False, report_ccw=True, delta_threshold=20))
    encoder_interface.send_message(encoder.check_state(repetition_delay=np.uint32(100)))

    # Starts lick monitoring
    sensor_interface.send_message(lick.check_state(repetition_delay=np.uint32(20000)))

    # Starts reward delivery
    actor_interface.send_message(
        valve.set_parameters(
            pulse_duration=np.uint32(4000000), calibration_delay=np.uint32(10000), calibration_count=np.uint16(100)
        )
    )  # 1 second

    timer.delay_noblock(delay=120)

    actor_interface.stop()
    encoder_interface.stop()
    sensor_interface.stop()
    data_logger.stop()


if __name__ == "__main__":
    test_runtime()

    # target = Path("/media/Data/Tyche-A2")
    # convert_old_data(root_directory=target, remove_sources=True)
