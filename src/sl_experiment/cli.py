"""This module provides click-based Command-Line Interface (CLI) scripts that allow using various features from this
library through the terminal."""

from pathlib import Path

import click

from .experiment import (
    ExperimentState,
    run_train_logic,
    lick_training_logic,
    run_experiment_logic,
    vr_maintenance_logic,
)
from .zaber_bindings import CRCCalculator, discover_zaber_devices
from .data_preprocessing import SessionData, purge_redundant_data

# Precalculated default valve calibration data. This should be defined separately for each project, as the valve
# is replaced and recalibrated fairly frequently.
valve_calibration_data = (
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
    calculator = CRCCalculator()
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
    "-e",
    "--experimenter",
    type=str,
    required=True,
    help="The ID of the experimenter supervising the training session.",
)
@click.option(
    "-a",
    "--animal",
    type=str,
    required=True,
    help="The name of the animal undergoing the lick training session.",
)
@click.option(
    "-w",
    "--animal_weight",
    type=float,
    required=True,
    help="The weight of the animal, in grams, at the beginning of the training session.",
)
@click.option(
    "-ad",
    "--average_delay",
    type=int,
    show_default=True,
    default=12,
    help="The average number of seconds that has to pass between two consecutive reward deliveries during training.",
)
@click.option(
    "-md",
    "--maximum_deviation",
    type=int,
    show_default=True,
    default=6,
    help=(
        "The maximum number of seconds that can be used to increase or decrease the delay between two consecutive "
        "reward deliveries during training. This is used to generate reward delays using a pseudorandom sampling to "
        "remove trends in reward delivery patterns."
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
def lick_training(
    experimenter: str,
    animal: str,
    animal_weight: float,
    average_delay: int,
    maximum_deviation: int,
    maximum_volume: float,
    maximum_time: int,
) -> None:
    """Runs a reference lick training session for the specified animal, using the input parameters.

    The CLI is primarily designed to calibrate and test the Sun lab Mesoscope-VR system and to demonstrate how to
    implement lick training for custom projects. Depending on the animal id, this CLI statically uses 'TestMice' or
    'Template' project.
    """

    # To distinguish real test mice used to calibrate sun lab equipment from the 'virtual' mouse used to test the
    # hardware, they use two different projects. Real project implementations of this CLI should statically set the
    # project name for their project
    if int(animal) == 666:
        project = "Template"
    else:
        project = "TestMice"

    # Surgery and water restriction log data has to be defined separately for each project, as each may use separate
    # Google Sheet files. Here, we use the two standard sheets used in the lab to test and calibrate this library.
    surgery_id = "1aEdF4gaiQqltOcTABQxN7mf1m44NGA-BTFwZsZdnRX8"
    water_restriction_id = "12yMl60O9rlb4VPE70swRJEWkMvgsL7sgVx1qYYcij6g"

    # Initializes the session data manager class. This generates the necessary data directories on all PCs used in
    # out data acquisition, processing, and storage pipelines.
    session_data = SessionData(
        animal_id=animal,
        project_name=project,
        session_type="lick_training",
        surgery_sheet_id=surgery_id,
        water_log_sheet_id=water_restriction_id,
        credentials_path="/media/Data/Experiments/sl-surgery-log-0f651e492767.json",
        local_root_directory="/media/Data/Experiments",
        server_root_directory="/media/cbsuwsun/storage/sun_data",
        nas_root_directory="/home/cybermouse/nas/rawdata",
    )

    # Runs the lick training session.
    lick_training_logic(
        experimenter=experimenter,
        animal_weight=animal_weight,
        session_data=session_data,
        valve_calibration_data=valve_calibration_data,
        average_reward_delay=average_delay,
        maximum_deviation_from_mean=maximum_deviation,
        maximum_water_volume=maximum_volume,
        maximum_training_time=maximum_time,
    )


@click.command()
def maintain_vr() -> None:
    """Exposes a terminal interface to interact with the water delivery solenoid valve and the running wheel break.

    This CLI command is designed to fill, empty, check, and, if necessary, recalibrate the solenoid valve used to
    deliver water to animals during training and experiment runtimes. Also, it is capable of locking or unlocking the
    wheel breaks, which is helpful when cleaning the wheel (after each session) and maintaining the wrap around the
    wheel surface (weekly to monthly).

    Since valve maintenance requires accurate valve calibration data which may change frequently, it is advised to
    reimplement this CLI for each project, similar to all other 'reference' CLI commands from this library.
    """
    # Runs the calibration runtime.
    vr_maintenance_logic(valve_calibration_data=valve_calibration_data)


@click.command()
@click.option(
    "-e",
    "--experimenter",
    type=str,
    required=True,
    help="The ID of the experimenter supervising the training session.",
)
@click.option(
    "-a",
    "--animal",
    type=str,
    required=True,
    help="The name of the animal undergoing the run training session.",
)
@click.option(
    "-w",
    "--animal_weight",
    type=float,
    required=True,
    help="The weight of the animal, in grams, at the beginning of the training session.",
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
def run_training(
    experimenter: str,
    animal: str,
    animal_weight: float,
    initial_speed: float,
    initial_duration: float,
    increase_threshold: float,
    speed_step: float,
    duration_step: float,
    maximum_speed: float,
    maximum_duration: float,
    maximum_volume: float,
    maximum_time: int,
) -> None:
    """Runs a reference run training session for the specified animal, using the input parameters.

    The CLI is primarily designed to calibrate and test the Sun lab Mesoscope-VR system and to demonstrate how to
    implement run training for custom projects. Depending on the animal id, this CLI statically uses 'TestMice' or
    'Template' project.
    """

    # To distinguish real test mice used to calibrate sun lab equipment from the 'virtual' mouse used to test the
    # hardware, they use two different projects. Real project implementations of this CLI should statically set the
    # project name for their project
    if int(animal) == 666:
        project = "Template"
    else:
        project = "TestMice"

    # Surgery and water restriction log data has to be defined separately for each project, as each may use separate
    # Google Sheet files. Here, we use the two standard sheets used in the lab to test and calibrate this library.
    surgery_id = "1aEdF4gaiQqltOcTABQxN7mf1m44NGA-BTFwZsZdnRX8"
    water_restriction_id = "12yMl60O9rlb4VPE70swRJEWkMvgsL7sgVx1qYYcij6g"

    # Initializes the session data manager class.
    session_data = SessionData(
        animal_id=animal,
        project_name=project,
        session_type="run_training",
        surgery_sheet_id=surgery_id,
        water_log_sheet_id=water_restriction_id,
        credentials_path="/media/Data/Experiments/sl-surgery-log-0f651e492767.json",
        local_root_directory="/media/Data/Experiments",
        server_root_directory="/media/cbsuwsun/storage/sun_data",
        nas_root_directory="/home/cybermouse/nas/rawdata",
    )

    # Runs the training session.
    run_train_logic(
        experimenter=experimenter,
        animal_weight=animal_weight,
        session_data=session_data,
        valve_calibration_data=valve_calibration_data,
        initial_speed_threshold=initial_speed,
        initial_duration_threshold=initial_duration,
        speed_increase_step=speed_step,
        duration_increase_step=duration_step,
        increase_threshold=increase_threshold,
        maximum_speed_threshold=maximum_speed,
        maximum_duration_threshold=maximum_duration,
        maximum_water_volume=maximum_volume,
        maximum_training_time=maximum_time,
    )


@click.command()
@click.option(
    "-e",
    "--experimenter",
    type=str,
    required=True,
    help="The ID of the experimenter supervising the experiment session.",
)
@click.option(
    "-a",
    "--animal",
    type=str,
    required=True,
    help="The name of the animal undergoing the experiment session.",
)
@click.option(
    "-w",
    "--animal_weight",
    type=float,
    required=True,
    help="The weight of the animal, in grams, at the beginning of the experiment session.",
)
def run_experiment(
    experimenter: str,
    animal: str,
    animal_weight: float,
) -> None:
    """Runs a reference experiment session for the specified animal, using the input parameters.

    The CLI is primarily designed to calibrate and test the Sun lab Mesoscope-VR system and to demonstrate how to
    implement experiment runtimes for custom projects. Depending on the animal id, this CLI statically uses 'TestMice'
    or 'Template' project.
    """

    # Statically defines the cue length map for the test Unity task
    cue_length_map = {0: 30.0, 1: 30.0, 2: 30.0, 3: 30.0, 4: 30.0}  # Ivan's task: 4 cues and 4 gray regions

    # Defines the sequence of experiment 'states'
    baseline = ExperimentState(experiment_state_code=1, vr_state_code=1, state_duration_s=30.0)
    run = ExperimentState(
        experiment_state_code=2,
        vr_state_code=2,
        state_duration_s=120.0,  # 2 minutes
    )
    cooldown = ExperimentState(experiment_state_code=3, vr_state_code=1, state_duration_s=15.0)
    experiment_state_sequence = (baseline, run, cooldown)

    # To distinguish real test mice used to calibrate sun lab equipment from the 'virtual' mouse used to test the
    # hardware, they use two different projects. Real project implementations of this CLI should statically set the
    # project name for their project
    if int(animal) == 666:
        project = "Template"
    else:
        project = "TestMice"

    # Surgery and water restriction log data has to be defined separately for each project, as each may use separate
    # Google Sheet files. Here, we use the two standard sheets used in the lab to test and calibrate this library.
    surgery_id = "1aEdF4gaiQqltOcTABQxN7mf1m44NGA-BTFwZsZdnRX8"
    water_restriction_id = "12yMl60O9rlb4VPE70swRJEWkMvgsL7sgVx1qYYcij6g"

    # Initializes the session data manager class.
    session_data = SessionData(
        animal_id=animal,
        project_name=project,
        session_type="experiment",
        surgery_sheet_id=surgery_id,
        water_log_sheet_id=water_restriction_id,
        credentials_path="/media/Data/Experiments/sl-surgery-log-0f651e492767.json",
        local_root_directory="/media/Data/Experiments",
        server_root_directory="/media/cbsuwsun/storage/sun_data",
        nas_root_directory="/home/cybermouse/nas/rawdata",
    )

    # Runs the experiment session using the input parameters.
    run_experiment_logic(
        experimenter=experimenter,
        animal_weight=animal_weight,
        session_data=session_data,
        valve_calibration_data=valve_calibration_data,
        cue_length_map=cue_length_map,
        experiment_state_sequence=experiment_state_sequence,
    )


@click.command()
@click.option(
    "-s",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    prompt="Enter the paths to the target session directory:",
    help="The path to the session directory to preprocess.",
)
def preprocess_session(session_path: Path) -> None:
    """Preprocesses the target session's data.

    This command aggregates all session data on the VRPC, compresses the data to optimize it for network transmission
    and storage, and transfers the data to the NAS and the BioHPC cluster. It automatically skips already completed
    processing stages as necessary to optimize runtime performance.

    Primarily, this command is intended to retry or resume failed or interrupted preprocessing runtimes.
    Preprocessing should be carried out immediately after data acquisition to optimize the acquired data for long-term
    storage and distribute it to the NAS and the BioHPC cluster for further processing and storage.
    """
    session_path = Path(session_path)  # Ensures the path is wrapped into a Path object instance.
    session_data = SessionData.from_path(path=session_path)  # Restores SessionData from the cache .yaml file.
    session_data.preprocess_session_data()  # Runs the preprocessing logic.


@click.command()
@click.option(
    "-u",
    "--remove_ubiquitin",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to remove ubiquitin-marked mesoscope_frames directories from the ScanImagePC.",
)
@click.option(
    "-t",
    "--remove_telomere",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to remove raw_data directories from the VRPC if their counterparts on the "
        "BioHPC server contain telomere markers."
    ),
)
def purge_data(remove_ubiquitin: bool, remove_telomere: bool) -> None:
    """Depending on configuration, removes all redundant data directories from the ScanImagePC, VRPC, or both.

    This command should be used at least weekly to remove no longer necessary data from the PCs used during data
    acquisition. Unless this function is called, our preprocessing pipelines will NOT remove the data, eventually
    leading to both PCs running out of storage space.
    """
    purge_redundant_data(
        remove_ubiquitin=remove_ubiquitin,
        remove_telomere=remove_telomere,
        local_root_path=Path("/media/Data/Experiments"),
        server_root_path=Path("/media/cbsuwsun/storage/sun_data"),
        mesoscope_root_path=Path("/home/cybermouse/scanimage/mesodata"),
    )
