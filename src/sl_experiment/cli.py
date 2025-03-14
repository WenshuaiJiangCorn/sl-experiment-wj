"""This module provides click-based Command-Line Interface (CLI) scripts that allow using various features from this
library through the terminal."""

from pathlib import Path
from datetime import (
    datetime,
)

import click
from ataraxis_base_utilities import LogLevel, console

from sl_experiment.experiment import SessionData

from .experiment import (
    run_train_logic,
    _BehavioralTraining,
    lick_training_logic,
    run_experiment_logic,
    calibrate_valve_logic,
    _RunTrainingDescriptor,
    _LickTrainingDescriptor,
)
from .gs_data_parser import ParseData, SurgeryData, FilteredSurgeries, _WaterSheetData, _SurgerySheetData
from .zaber_bindings import _CRCCalculator, discover_zaber_devices

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
    default=20,
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
    maximum_volume: float,
    maximum_time: int,
    actor_port: str,
    sensor_port: str,
    encoder_port: str,
    headbar_port: str,
    lickport_port: str,
    face_camera: int,
    left_camera: int,
    right_camera: int,
    cti_path: Path | str,
    screens_on: bool,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
    local_root_path: Path | str,
    server_root_path: Path | str,
    nas_root_path: Path | str,
) -> None:
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
) -> None:
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


@click.command()
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project for which to run the run training session.",
)
@click.option(
    "-a",
    "--animal",
    type=str,
    required=True,
    help="The name of the animal for which to run the run training session.",
)
@click.option(
    "-is",
    "--initial_speed",
    type=float,
    show_default=True,
    default=0.05,
    help="The initial speed, in centimeters per second, the animal must maintain to obtain water rewards.",
)
@click.option(
    "-id",
    "--initial_duration",
    type=float,
    show_default=True,
    default=0.05,
    help=(
        "The initial duration, in seconds, the animal must maintain above-threshold running speed to obtain water "
        "rewards."
    ),
)
@click.option(
    "-it",
    "--increase_threshold",
    type=float,
    show_default=True,
    default=0.1,
    help=(
        "The volume of water delivered to the animal, in milliliters, after which the speed and duration will be "
        "increased by the specified step-sizes. This is used to make the training progressively harder for the animal "
        "as a factor of how well it performs (and gets water rewards)."
    ),
)
@click.option(
    "-ss",
    "--speed_step",
    type=float,
    show_default=True,
    default=0.05,
    help=(
        "The amount, in centimeters per second, to increase the speed threshold each time the animal receives the "
        "volume of water specified by the 'increase-threshold' parameter. This determines how much harder tha training "
        "becomes at each increase point."
    ),
)
@click.option(
    "-ds",
    "--duration_step",
    type=float,
    show_default=True,
    default=0.05,
    help=(
        "The amount, in seconds, to increase the duration threshold each time the animal receives the volume of water "
        "specified by the 'increase-threshold' parameter. This determines how much harder tha training becomes at "
        "each increase point."
    ),
)
@click.option(
    "-ms",
    "--maximum_speed",
    type=float,
    show_default=True,
    default=10.0,
    help=(
        "The maximum speed, in centimeters per second, the animal must maintain to obtain water rewards. This option "
        "is used to limit how much the training logic can increase the speed threshold when the animal performs well "
        "during training."
    ),
)
@click.option(
    "-md",
    "--maximum_duration",
    type=float,
    show_default=True,
    default=10.0,
    help=(
        "The maximum duration, in seconds, the animal must maintain above-threshold running speed to obtain water "
        "rewards. This option is used to limit how much the training logic can increase the duration threshold when "
        "the animal performs well during training."
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
def run_training(
    animal: str,
    project: str,
    initial_speed: float,
    initial_duration: float,
    increase_threshold: float,
    speed_step: float,
    duration_step: float,
    maximum_speed: float,
    maximum_duration: float,
    maximum_volume: float,
    maximum_time: int,
    actor_port: str,
    sensor_port: str,
    encoder_port: str,
    headbar_port: str,
    lickport_port: str,
    face_camera: int,
    left_camera: int,
    right_camera: int,
    cti_path: Path | str,
    screens_on: bool,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
    local_root_path: Path | str,
    server_root_path: Path | str,
    nas_root_path: Path | str,
) -> None:
    """Runs a single run training session for the specified animal and project combination, using the input
    parameters.

    This CLI command allows running 'run' training sessions via the terminal. The only parameters that have to be
    provided at each runtime are the animal and project name. Every other parameter can be overwritten, but has been
    statically configured to work for our current Mesoscope-VR system configuration.
    """

    # Enables the console
    if not console.enabled:
        console.enable()

    message = f"Initializing run training runtime..."
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
        generate_mesoscope_paths=False,  # No need for mesoscope during run training.
        local_root_directory=local_root_path,
        server_root_directory=server_root_path,
        nas_root_directory=nas_root_path,
    )

    # Pre-generates the SessionDescriptor class and populates it with training data
    descriptor = _RunTrainingDescriptor(
        initial_running_speed_cm_s=initial_speed,
        initial_speed_duration_s=initial_duration,
        increase_threshold_ml=increase_threshold,
        increase_running_speed_cm_s=speed_step,
        increase_speed_duration_s=duration_step,
        maximum_running_speed_cm_s=maximum_speed,
        maximum_speed_duration_s=maximum_duration,
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

    # Runs the training session.
    run_train_logic(
        runtime=runtime,
        initial_speed_threshold=initial_speed,
        initial_duration_threshold=initial_duration,
        speed_increase_step=increase_threshold,
        duration_increase_step=speed_step,
        increase_threshold=duration_step,
        maximum_speed_threshold=maximum_speed,
        maximum_duration_threshold=maximum_duration,
        maximum_water_volume=maximum_volume,
        maximum_training_time=maximum_time,
    )


@click.command()
def run_experiment() -> None:
    """Runs the reference experiment runtime.

    This CLI is intended EXCLUSIVELY to test the Mesoscope-Vr system. Do not use this for real projects.
    """
    run_experiment_logic()


def _normalize_date(date: str) -> str:
    """
    This method converts the input date from MM/DD/YY format to M/D/YY format
    to retrieve the correct date for the daily water restriction data from the
    water log.
    """
    try:
        dt = datetime.strptime(date, "%m/%d/%y")
        return dt.strftime("%-m/%-d/%y")

    except ValueError:
        raise ValueError(f"Invalid date format: {date}. Expected 'MM/DD/YY'.")


def _force_valid_date_input() -> str:
    """
    This method continuously prompts the user for a valid date.
    """
    while True:
        date = click.prompt("Enter the date in the format MM/DD/YY", type=str)

        try:
            normalized_date = _normalize_date(date)
            return normalized_date
        except ValueError:
            click.echo("Invalid date format.")


def _valid_mouse_id(mouse_id: str, available_ids: set[int]) -> int:
    """
    This method continuously prompts the user to input a valid mouse ID from
    the available entries from the chosen log.
    """
    try:
        mouse_id_int = int(mouse_id)
        if mouse_id_int in available_ids:
            return mouse_id_int
        else:
            raise ValueError(f"Mouse ID {mouse_id_int} is not in the list of available IDs.")

    except ValueError:
        raise ValueError(f"Invalid mouse ID")


@click.command()
@click.option(
    "--log-choice",
    type=click.Choice(["1", "2"]),
    default="1",
    help="The type of log to fetch: 1 for Surgery Log, 2 for Water Log.",
)
@click.option(
    "--mouse-id",
    type=int,
    help="The ID of the mouse to fetch data for.",
)
@click.option(
    "--date",
    type=str,
    help="The date of the water log entry to fetch (format: MM/DD/YY).",
)
@click.option(
    "--list-water-mice",
    is_flag=True,
    help="List all mouse IDs available in the water log and exit.",
)
@click.option(
    "--list-surgery-mice",
    is_flag=True,
    help="List all mouse IDs available in the surgery log and exit.",
)
@click.option(
    "--id-from-tab-surgery",
    type=str,
    help="The tab name for the surgery log to fetch data based on the mouse ID.",
)
def process_gs_log(
    log_choice: str,
    mouse_id: int,
    date: str,
    list_water_mice: bool,
    list_surgery_mice: bool,
    id_from_tab_surgery: str,
) -> None:
    if list_water_mice:
        list_water_mouse_ids()
        return

    if list_surgery_mice:
        if not id_from_tab_surgery:
            click.echo("Error: --id-from-tab-surgery is required when listing surgery mice.")
            return
        list_surgery_mouse_ids(id_from_tab_surgery)
        return

    if log_choice == "1":
        if not id_from_tab_surgery:
            click.echo("Error: --id-from-tab-surgery is required for surgery log.")
            return
        process_surgery_log(id_from_tab_surgery, mouse_id)

    elif log_choice == "2":
        process_water_log(mouse_id, date, list_water_mice)

    else:
        click.echo("Invalid log choice. Enter 1 for Surgery Log or 2 for Water Log.")


def process_water_log(mouse_id: int | None, date: str | None, list_water_mice: bool) -> None:
    """
    Processes the water log for a given mouse ID and date.
    Handles input validation and forces valid inputs.
    """
    sheet_data = _WaterSheetData()
    sheet_data._fetch_data_from_tab("MouseInfo")
    tab_data = sheet_data.data.get("MouseInfo", [])
    headers = tab_data[0]
    mouse_col_index = headers.index("mouse")

    available_ids = set()

    for row in tab_data[1:]:
        if len(row) > mouse_col_index:
            cell_value = row[mouse_col_index]
            if isinstance(cell_value, str) and cell_value.strip():
                try:
                    available_ids.add(int(cell_value))
                except ValueError:
                    continue

    if list_water_mice:
        list_water_mouse_ids()
        return

    if mouse_id is None:
        list_water_mouse_ids()
        while True:
            mouse_id_input = click.prompt("Enter the mouse ID", type=str)
            try:
                mouse_id = _valid_mouse_id(mouse_id_input, available_ids)
                break
            except ValueError as e:
                click.echo(str(e))

    if date is None:
        date = click.prompt("Enter the date in the form MM/DD/YY)", type=str)

    try:
        date = _normalize_date(date)
    except ValueError:
        click.echo(f"Invalid date format: {date}.")
        date = _force_valid_date_input()

    fetch_water_log_data(mouse_id, date)


def process_surgery_log(tab_name: str, mouse_id: int | None) -> None:
    """
    This method ensures the user provides a valid tab name and mouse ID from the surgery log.
    Mouse IDs are extracted from the header row of each tab.
    """
    sheet_data = _SurgerySheetData(project_name=tab_name)
    sheet_data._parse()
    mouse_col_index = sheet_data.headers["id"]

    available_ids = set()
    for row in sheet_data.data:
        if len(row) > mouse_col_index:
            cell_value = row[mouse_col_index]
            if isinstance(cell_value, str) and cell_value.strip():
                try:
                    available_ids.add(int(cell_value))
                except ValueError:
                    continue

    if not available_ids:
        click.echo(f"No mouse IDs found in tab '{tab_name}'.")
        return

    if mouse_id is None:
        list_surgery_mouse_ids(tab_name)

        while True:
            mouse_id_input = click.prompt("Enter the mouse ID", type=str)
            try:
                mouse_id = _valid_mouse_id(mouse_id_input, available_ids)
                break
            except ValueError as e:
                click.echo(str(e))

    fetch_surgery_log_data(tab_name, mouse_id)


def list_water_mouse_ids() -> None:
    """
    Lists all mouse IDs available in the water log.
    """
    sheet_data = _WaterSheetData()
    sheet_data._fetch_data_from_tab("MouseInfo")
    tab_data = sheet_data.data.get("MouseInfo", [])
    headers = tab_data[0]
    water_key = headers.index("mouse")

    if water_key == -1:
        click.echo("'mouse' column not found in the water log.")
        return

    mouse_ids = set()
    for row in tab_data[1:]:
        if len(row) > water_key:
            cell_value = row[water_key]
            if isinstance(cell_value, str) and cell_value.strip():
                try:
                    mouse_ids.add(int(cell_value))
                except ValueError:
                    continue

    if mouse_ids:
        click.echo("Available Mouse IDs for Water Log:")
        for mouse_id in sorted(mouse_ids):
            click.echo(mouse_id)
    else:
        click.echo("No mouse IDs found in the water log.")


def list_surgery_mouse_ids(tab_name: str) -> None:
    """
    Lists all mouse IDs available in the surgery log for the specified tab.
    """
    sheet_data = _SurgerySheetData(project_name=tab_name)
    sheet_data._parse()

    mouse_col_index = sheet_data.headers.get("id", -1)
    if mouse_col_index == -1:
        click.echo(f"'id' column not found in tab '{tab_name}'. Available columns: {list(sheet_data.headers.keys())}")
        return

    mouse_ids = set()
    for row in sheet_data.data:
        if len(row) > mouse_col_index:
            cell_value = row[mouse_col_index]
            if isinstance(cell_value, str) and cell_value.strip():
                try:
                    mouse_ids.add(int(cell_value))
                except ValueError:
                    continue

    if mouse_ids:
        click.echo(f"Available Mouse IDs in tab '{tab_name}':\n" + "\n".join(map(str, sorted(mouse_ids))))
    else:
        click.echo(f"No mouse IDs found in tab '{tab_name}'.")


def fetch_water_log_data(mouse_id: int, date: str) -> None:
    """
    Fetches both the baseline information about the mouse and the water restriction
    data for a given date.
    """
    water_log = ParseData(tab_name="MouseInfo", mouse_id=mouse_id, date=date)
    mouse_instance = water_log.mouse_instances.get(str(mouse_id))
    if mouse_instance is None:
        click.echo(f"Mouse ID {mouse_id} not found in the water log.")
        return

    if mouse_instance:
        if date in mouse_instance.daily_log:
            click.echo(mouse_instance)
        else:
            click.echo(f"No water restriction data found for mouse {mouse_id} on {date}.")
    else:
        click.echo(f"No data found for mouse {mouse_id}.")


def fetch_surgery_log_data(tab_name: str, mouse_id: int) -> FilteredSurgeries:
    """
    Fetches the protocol data, implant data, injection data, brain data, and drug
    data for the given mouse ID.
    """
    sheet_data = _SurgerySheetData(project_name=tab_name)
    sheet_data._parse()

    filtered_surgeries = FilteredSurgeries(surgeries=[])
    for row in sheet_data.data:
        surgery_data = SurgeryData(headers=sheet_data.headers, row=row, tab_name=tab_name)

        if surgery_data.protocol_data.id == mouse_id:
            filtered_surgeries.surgeries.append(surgery_data)

    return filtered_surgeries
