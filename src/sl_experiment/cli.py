"""This module provides click-based Command-Line Interface (CLI) scripts that allow using various features from this
library through the terminal."""

import click
from pathlib import Path

from sl_experiment.experiment import SessionData

from .zaber_bindings import _CRCCalculator, discover_zaber_devices
from .experiment import (
    _HeadBar,
    _LickPort,
    _MicroControllerInterfaces,
    _BehavioralTraining,
    _VideoSystems,
    _ZaberPositions,
    _LickTrainingDescriptor,
    lick_training_logic,
)
from ataraxis_base_utilities import console, LogLevel
from ataraxis_data_structures import DataLogger

# Precalculated default valve calibration data. This is used as the 'default' field for our mesoscope-vr CLI
DEFAULT_VALVE_CALIBRATION_DATA = (
    (15000, 1.8556),
    (30000, 3.4844),
    (45000, 7.1846),
    (60000, 10.0854),
)


@click.command()
@click.option(
    "--string", "-i", prompt="Enter the string to be checksummed", help="The string to calculate the CRC checksum for."
)
def calculate_crc(string: str) -> None:
    """Calculates the CRC32-XFER checksum for the input string."""
    calculator = _CRCCalculator()
    crc_checksum = calculator.string_checksum(string)
    click.echo(f"The CRC32-XFER checksum for the input string '{string}' is: {crc_checksum}")


@click.command()
@click.option(
    "-e",
    "--errors",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display errors encountered when connecting to all evaluated serial ports.",
)
def list_devices(errors: bool) -> None:
    """Displays information about all Zaber devices available through USB ports of the host-system."""
    discover_zaber_devices(silence_errors=not errors)


@click.command()
@click.option(
    "-o",
    "--output_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="/home/cybermouse/Desktop/TestOut",
    show_default=True,
    help=(
        "The path to the directory used to save the output data. Typically, this would be the path to a "
        "temporary testing directory on the VRPC."
    ),
)
@click.option(
    "-ap",
    "--actor_port",
    type=str,
    show_default=True,
    default="/dev/ttyACM0",
    help="The USB port used by the actor MicroController.",
)
@click.option(
    "-sp",
    "--sensor_port",
    type=str,
    show_default=True,
    default="/dev/ttyACM1",
    help="The USB port used by the sensor MicroController.",
)
@click.option(
    "-ep",
    "--encoder_port",
    type=str,
    show_default=True,
    default="/dev/ttyACM2",
    help="The USB port used by the encoder MicroController.",
)
@click.option(
    "-hp",
    "--headbar_port",
    type=str,
    show_default=True,
    default="/dev/ttyUSB0",
    help="The USB port used by the HeadBar controller.",
)
@click.option(
    "-lp",
    "--lickport_port",
    type=str,
    show_default=True,
    default="/dev/ttyUSB1",
    help="The USB port used by the LickPort controller.",
)
@click.option(
    "-fc",
    "--face_camera",
    type=int,
    default=0,
    show_default=True,
    help="The index of the face camera in the list of all available Harvester-managed cameras.",
)
@click.option(
    "-fc",
    "--left_camera",
    type=int,
    default=0,
    show_default=True,
    help="The index of the left camera in the list of all available OpenCV-managed cameras.",
)
@click.option(
    "-fc",
    "--right_camera",
    type=int,
    default=2,
    show_default=True,
    help="The index of the right camera in the list of all available OpenCV-managed cameras.",
)
@click.option(
    "-cp",
    "--cti_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti",
    show_default=True,
    help="The path to the GeniCam CTI file used to connect to Harvesters-managed cameras.",
)
@click.option(
    "-s",
    "--screens_on",
    is_flag=True,
    default=False,
    help="Communicates whether the VR screens are currently ON.",
)
@click.option(
    "-vd",
    "--valve_calibration_data",
    type=(int, float),
    multiple=True,
    default=DEFAULT_VALVE_CALIBRATION_DATA,
    show_default=True,
    help=(
        "Supplies the data used by the solenoid valve module to determine how long to keep the valve open to "
        "deliver requested water volumes. Provides calibration data as pairs of numbers, for example: "
        "--valve-calibration-data 15000 1.8556."
    ),
)
def mesoscope_vr_cli(
    output_path: Path,
    actor_port: str,
    sensor_port: str,
    encoder_port: str,
    headbar_port: str,
    lickport_port: str,
    face_camera: int,
    left_camera: int,
    right_camera: int,
    cti_path: Path,
    screens_on: bool,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
):
    """Provides the Command-Line Interface (CLI) to control all custom-made Mesoscope-VR components used during
    experiment and training runtimes.

    Notably, this CLI does not directly interface with Unity or the ScanImage software. Some of the indirect interfaces
    would likely function, but this CLI is not designed to work in an experiment setting. Primarily, this CLI is used
    to calibrate zaber motors and interface with the solenoid valve between experiment and training runtimes.
    """

    # Enables the console
    if not console.enabled:
        console.enable()

    # Initializes the data logger
    logger = DataLogger(
        output_directory=output_path,
        instance_name="behavior",  # Creates behavior_log subfolder under the output directory
        sleep_timer=0,
        exist_ok=True,
        process_count=1,
        thread_count=10,
    )
    logger.start()

    # Initializes binding classes
    headbar = _HeadBar(headbar_port, output_path.joinpath("zaber_positions.yaml"))
    lickport = _LickPort(lickport_port, output_path.joinpath("zaber_positions.yaml"))
    microcontrollers = _MicroControllerInterfaces(
        data_logger=logger,
        screens_on=screens_on,
        actor_port=actor_port,
        sensor_port=sensor_port,
        encoder_port=encoder_port,
        valve_calibration_data=valve_calibration_data,
    )
    microcontrollers.start()

    cameras = _VideoSystems(
        data_logger=logger,
        output_directory=output_path,
        face_camera_index=face_camera,
        left_camera_index=left_camera,
        right_camera_index=right_camera,
        harvesters_cti_path=cti_path,
    )

    # Starts fame acquisition and display. Saving is triggered via the appropriate CLI command
    cameras.start_face_camera()
    cameras.start_body_cameras()

    message = "Mesoscope-VR CLI: Activated."
    console.echo(message=message, level=LogLevel.SUCCESS)

    message = (
        "Supported Zaber position commands: home, mount, calibrate, restore, park, export_positions. Each command "
        "moves the HeadBar and LickPort motors to the respective positions."
    )
    console.echo(message=message, level=LogLevel.INFO)

    message = (
        "Supported Camera commands: save_face, save_body. Each command starts saving frames acquired by the face "
        "camera or body cameras respectively."
    )
    console.echo(message=message, level=LogLevel.INFO)

    message = (
        "Supported MicroController commands: encoder_on, encoder_off, break_on, break_off, screen_on, screen_off, "
        "torque_on, torque_off, start_mesoscope, stop_mesoscope, frames_on, frames_off, lick_on, lick_off. Each "
        "command interfaces with its respective hardware module. This list does not include solenoid valve commands, "
        "for solenoid valve support see the message below."
    )
    console.echo(message=message, level=LogLevel.INFO)

    message = (
        "Supported Solenoid Valve commands (via MicroController): open_valve, close_valve, reference_valve, "
        "deliver_reward, calibrate_valve_15, calibrate_valve_30, calibrate_valve_45, calibrate_valve_60. it is "
        "recommended to move Zaber motors into calibration position before running any valve commands."
    )
    console.echo(message=message, level=LogLevel.INFO)

    while True:
        command = input("Use 'q' to quit. Enter command: ")
        if command == "home":
            message = f"Moving HeadBar and LickPort motors to home position."
            console.echo(message=message, level=LogLevel.INFO)
            headbar.prepare_motors(wait_until_idle=False)
            lickport.prepare_motors(wait_until_idle=True)
            headbar.wait_until_idle()

        if command == "mount":
            message = f"Moving HeadBar and LickPort motors to mounting position."
            console.echo(message=message, level=LogLevel.INFO)
            headbar.mount_position(wait_until_idle=False)
            lickport.mount_position(wait_until_idle=True)
            headbar.wait_until_idle()

        if command == "calibrate":
            message = f"Moving HeadBar and LickPort motors to calibration position."
            console.echo(message=message, level=LogLevel.INFO)
            headbar.calibrate_position(wait_until_idle=False)
            lickport.calibrate_position(wait_until_idle=True)
            headbar.wait_until_idle()

        if command == "restore":
            message = f"RRestoring HeadBar and LickPort motors to the position loaded from the .yaml file."
            console.echo(message=message, level=LogLevel.INFO)
            headbar.restore_position(wait_until_idle=False)
            lickport.restore_position(wait_until_idle=True)
            headbar.wait_until_idle()

        if command == "park":
            message = f"Moving HeadBar and LickPort motors to parking position."
            console.echo(message=message, level=LogLevel.INFO)
            headbar.park_position(wait_until_idle=False)
            lickport.park_position(wait_until_idle=True)
            headbar.wait_until_idle()

        if command == "export_positions":
            message = f"Exporting current HeadBar and LickPort motor positions as a .yaml file."
            console.echo(message=message, level=LogLevel.INFO)
            head_bar_positions = headbar.get_positions()
            lickport_positions = lickport.get_positions()
            zaber_positions = _ZaberPositions(
                headbar_z=head_bar_positions[0],
                headbar_pitch=head_bar_positions[1],
                headbar_roll=head_bar_positions[2],
                lickport_z=lickport_positions[0],
                lickport_x=lickport_positions[1],
                lickport_y=lickport_positions[2],
            )
            zaber_positions.to_yaml(file_path=output_path.joinpath("zaber_positions.yaml"))

        if command == "save_face":
            message = f"Initializing face-camera frame saving."
            console.echo(message=message, level=LogLevel.INFO)
            cameras.save_face_camera_frames()

        if command == "save_body":
            message = f"Initializing body-camera frame saving."
            console.echo(message=message, level=LogLevel.INFO)
            cameras.save_body_camera_frames()

        if command == "encoder_on":
            message = f"Initializing encoder monitoring."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.enable_encoder_monitoring()

        if command == "encoder_off":
            message = f"Stopping encoder monitoring."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.disable_encoder_monitoring()

        if command == "break_on":
            message = f"Enabling break."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.enable_break()

        if command == "break_off":
            message = f"Disabling break."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.disable_break()

        if command == "screen_on":
            message = f"Turning VR screens ON."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.enable_vr_screens()

        if command == "screen_off":
            message = f"Turning VR screens OFF."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.disable_vr_screens()

        if command == "torque_on":
            message = f"Initializing torque monitoring."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.enable_torque_monitoring()

        if command == "torque_off":
            message = f"Stopping torque monitoring."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.disable_torque_monitoring()

        if command == "start_mesoscope":
            message = f"Sending mesoscope frame acquisition START trigger."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.start_mesoscope()

        if command == "stop_mesoscope":
            message = f"Sending mesoscope frame acquisition STOP trigger."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.stop_mesoscope()

        if command == "frames_on":
            message = f"Initializing mesoscope frame acquisition timestamp monitoring."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.enable_mesoscope_frame_monitoring()

        if command == "frames_off":
            message = f"Stopping mesoscope frame acquisition timestamp monitoring."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.disable_mesoscope_frame_monitoring()

        if command == "lick_on":
            message = f"Initializing lick monitoring."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.enable_lick_monitoring()

        if command == "lick_off":
            message = f"Stopping lick monitoring."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.disable_lick_monitoring()

        if command == "open_valve":
            message = f"Opening solenoid valve."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.open_valve()

        if command == "close_valve":
            message = f"Closing solenoid valve."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.close_valve()

        if command == "deliver_reward":
            message = f"Delivering 5 uL water reward."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.deliver_reward()

        if command == "reference_valve":
            message = f"Running the reference solenoid valve calibration procedure."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.reference_valve()

        if command == "calibrate_valve_15":
            message = f"Running 15 ms solenoid valve calibration."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.calibrate_valve(pulse_duration=15)

        if command == "calibrate_valve_30":
            message = f"Running 30 ms solenoid valve calibration."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.calibrate_valve(pulse_duration=30)

        if command == "calibrate_valve_45":
            message = f"Running 45 ms solenoid valve calibration."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.calibrate_valve(pulse_duration=45)

        if command == "calibrate_valve_60":
            message = f"Running 60 ms solenoid valve calibration."
            console.echo(message=message, level=LogLevel.INFO)
            microcontrollers.calibrate_valve(pulse_duration=60)

        if command == "q":
            message = f"Terminating CLI runtime."
            console.echo(message=message, level=LogLevel.INFO)
            break

    # Shuts down zaber bindings
    message = f"Shutting down Zaber motors."
    console.echo(message=message, level=LogLevel.INFO)
    headbar.park_position(wait_until_idle=False)
    lickport.park_position(wait_until_idle=True)
    headbar.wait_until_idle()
    headbar.disconnect()
    lickport.disconnect()

    # Shuts down microcontroller interfaces
    microcontrollers.stop()

    # Shuts down cameras
    cameras.stop()

    # Stops the data logger
    logger.stop()

    # Compresses the logs
    logger.compress_logs(
        remove_sources=True, verbose=True, verify_integrity=False, memory_mapping=False, compress=False
    )


@click.command()
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project for which to run the lick training session.",
)
@click.option(
    "-a",
    "--animal",
    type=str,
    required=True,
    help="The name of the animal for which to run the lick training session.",
)
@click.option(
    "-ad",
    "--average_delay",
    type=int,
    show_default=True,
    default=12,
    help="The average number of seconds the has to pass between two consecutive reward deliveries during training.",
)
@click.option(
    "-md",
    "--maximum_deviation",
    type=int,
    show_default=True,
    default=6,
    help=(
        "The maximum number of seconds that can be used to increase or decrease the delay between two consecutive "
        "reward deliveries during training. This is used to generate delays using a pseudorandom sampling, to remove "
        "trends in reward delivery patterns.",
    ),
)
@click.option(
    "-mv",
    "--maximum_volume",
    type=float,
    show_default=True,
    default=1.0,
    help="The maximum volume of water, in milliliters, that can be delivered during training.",
)
@click.option(
    "-mt",
    "--maximum_time",
    type=int,
    show_default=True,
    default=40,
    help="The maximum time to run the training, in minutes.",
)
@click.option(
    "-ap",
    "--actor_port",
    type=str,
    show_default=True,
    default="/dev/ttyACM0",
    help="The USB port used by the actor MicroController.",
)
@click.option(
    "-sp",
    "--sensor_port",
    type=str,
    show_default=True,
    default="/dev/ttyACM1",
    help="The USB port used by the sensor MicroController.",
)
@click.option(
    "-ep",
    "--encoder_port",
    type=str,
    show_default=True,
    default="/dev/ttyACM2",
    help="The USB port used by the encoder MicroController.",
)
@click.option(
    "-hp",
    "--headbar_port",
    type=str,
    show_default=True,
    default="/dev/ttyUSB0",
    help="The USB port used by the HeadBar controller.",
)
@click.option(
    "-lp",
    "--lickport_port",
    type=str,
    show_default=True,
    default="/dev/ttyUSB1",
    help="The USB port used by the LickPort controller.",
)
@click.option(
    "-fc",
    "--face_camera",
    type=int,
    default=0,
    show_default=True,
    help="The index of the face camera in the list of all available Harvester-managed cameras.",
)
@click.option(
    "-fc",
    "--left_camera",
    type=int,
    default=0,
    show_default=True,
    help="The index of the left camera in the list of all available OpenCV-managed cameras.",
)
@click.option(
    "-fc",
    "--right_camera",
    type=int,
    default=2,
    show_default=True,
    help="The index of the right camera in the list of all available OpenCV-managed cameras.",
)
@click.option(
    "-cp",
    "--cti_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti",
    show_default=True,
    help="The path to the GeniCam CTI file used to connect to Harvesters-managed cameras.",
)
@click.option(
    "-s",
    "--screens_on",
    is_flag=True,
    default=False,
    help="Communicates whether the VR screens are currently ON.",
)
@click.option(
    "-vd",
    "--valve_calibration_data",
    type=(int, float),
    multiple=True,
    default=DEFAULT_VALVE_CALIBRATION_DATA,
    show_default=True,
    help=(
        "Supplies the data used by the solenoid valve module to determine how long to keep the valve open to "
        "deliver requested water volumes. Provides calibration data as pairs of numbers, for example: "
        "--valve-calibration-data 15000 1.8556."
    ),
)
@click.option(
    "-lr",
    "--local_root_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="/media/Data/Experiments",
    show_default=True,
    help="The path to the root directory on the local machine (VRPC) that stores experiment project folders.",
)
@click.option(
    "-sp",
    "--server_root_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="/media/cybermouse/Extra Data/server/storage",
    show_default=True,
    help="The path to the root directory on the BioHPC lab server that stores experiment project folders.",
)
@click.option(
    "-np",
    "--nas_root_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="/home/cybermouse/nas/rawdata",
    show_default=True,
    help="The path to the root directory on the NAS that stores experiment project folders.",
)
def lick_training(
    animal: str,
    project: str,
    average_delay: int,
    maximum_deviation: int,
    maximum_volume: int,
    maximum_time: int,
    actor_port: str,
    sensor_port: str,
    encoder_port: str,
    headbar_port: str,
    lickport_port: str,
    face_camera: int,
    left_camera: int,
    right_camera: int,
    cti_path: str,
    screens_on: bool,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
    local_root_path,
    server_root_path,
    nas_root_path,
):
    """Runs a single lick training session for the specified animal and project combination, using the input
    parameters.

    This CLI command allows running lick training sessions via the terminal. The only parameters that have to be
    provided at each runtime are the animal and project name. Every other parameter can be overwritten, but has been
    statically configured to work for our current Mesoscope-VR system configuration.
    """
    # Converts input paths to Pth instances.
    cti_path = Path(cti_path)
    local_root_path = Path(local_root_path)
    server_root_path = Path(server_root_path)
    nas_root_path = Path(nas_root_path)

    # Initializes the session data manager class.
    session_data = SessionData(
        animal_name=animal,
        project_name=project,
        generate_mesoscope_paths=False,  # No need for mesoscope when running lick training.
        local_root_directory=local_root_path,
        server_root_directory=server_root_path,
        nas_root_directory=nas_root_path,
    )

    # Pre-generates the SessionDescriptor class and populates it with training data
    descriptor = _LickTrainingDescriptor(
        average_reward_delay_s=average_delay,
        maximum_deviation_from_average_s=maximum_deviation,
        training_time_m=maximum_time,
    )

    # Initializes the main runtime interface class.
    runtime = _BehavioralTraining(
        session_data=session_data,
        descriptor=descriptor,
        actor_port=actor_port,
        sensor_port=sensor_port,
        encoder_port=encoder_port,
        headbar_port=headbar_port,
        lickport_port=lickport_port,
        face_camera_index=face_camera,
        left_camera_index=left_camera,
        right_camera_index=right_camera,
        harvesters_cti_path=cti_path,
        screens_on=screens_on,
        valve_calibration_data=valve_calibration_data,
    )

    # Runs the lick training session.
    lick_training_logic(
        runtime=runtime,
        average_reward_delay=average_delay,
        maximum_deviation_from_mean=maximum_deviation,
        maximum_water_volume=maximum_volume,
        maximum_training_time=maximum_time,
    )
