"""This module provides click-based Command-Line Interface (CLI) scripts that allow using various features from this
library through the terminal."""

import click
from pathlib import Path
from sl_experiment.experiment import SessionData

from .zaber_bindings import _CRCCalculator, discover_zaber_devices
from .experiment import _BehavioralTraining, _LickTrainingDescriptor, lick_training_logic, calibrate_valve_logic
from ataraxis_base_utilities import console, LogLevel

# Precalculated default valve calibration data. This is used as the 'default' field for our valve interface cli
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
        "trends in reward delivery patterns."
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
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="/media/Data/Experiments",
    show_default=True,
    help="The path to the root directory on the local machine (VRPC) that stores experiment project folders.",
)
@click.option(
    "-sp",
    "--server_root_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="/media/cbsuwsun/storage/sun_data",
    show_default=True,
    help="The path to the root directory on the BioHPC lab server that stores experiment project folders.",
)
@click.option(
    "-np",
    "--nas_root_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
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

    # Enables the console
    if not console.enabled:
        console.enable()

    message = f"Initializing lick training runtime..."
    console.echo(message=message, level=LogLevel.INFO)

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
        maximum_training_time_m=maximum_time,
        maximum_water_volume_ml=maximum_volume,
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


@click.command()
@click.option(
    "-ap",
    "--actor_port",
    type=str,
    show_default=True,
    default="/dev/ttyACM0",
    help="The USB port used by the actor MicroController.",
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
def calibrate_valve_cli(
    actor_port: str,
    headbar_port: str,
    lickport_port: str,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
):
    """Instantiates a terminal-driven interface to interact with the water delivery solenoid valve.

    This CLI command is designed to fill, empty, check, and, if necessary, recalibrate the solenoid valve used to
    deliver water to animals during training and experiment runtimes. This CLI is typically used twice per day: before
    the first runtime and after the last runtime.
    """
    # Enables the console
    if not console.enabled:
        console.enable()

    message = f"Initializing valve calibration runtime..."
    console.echo(message=message, level=LogLevel.INFO)

    # Runs the calibration runtime.
    calibrate_valve_logic(
        actor_port=actor_port,
        headbar_port=headbar_port,
        lickport_port=lickport_port,
        valve_calibration_data=valve_calibration_data,
    )
