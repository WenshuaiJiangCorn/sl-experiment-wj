from pathlib import Path

from _typeshed import Incomplete

from .experiment import (
    ExperimentState as ExperimentState,
    run_train_logic as run_train_logic,
    lick_training_logic as lick_training_logic,
    run_experiment_logic as run_experiment_logic,
    vr_maintenance_logic as vr_maintenance_logic,
)
from .zaber_bindings import (
    CRCCalculator as CRCCalculator,
    discover_zaber_devices as discover_zaber_devices,
)
from .data_preprocessing import (
    purge_redundant_data as purge_redundant_data,
    preprocess_session_directory as preprocess_session_directory,
)

valve_calibration_data: Incomplete

def calculate_crc(string: str) -> None:
    """Calculates the CRC32-XFER checksum for the input string."""

def list_devices(errors: bool) -> None:
    """Displays information about all Zaber devices available through USB ports of the host-system."""

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

def maintain_vr() -> None:
    """Exposes a terminal interface to interact with the water delivery solenoid valve and the running wheel break.

    This CLI command is designed to fill, empty, check, and, if necessary, recalibrate the solenoid valve used to
    deliver water to animals during training and experiment runtimes. Also, it is capable of locking or unlocking the
    wheel breaks, which is helpful when cleaning the wheel (after each session) and maintaining the wrap around the
    wheel surface (weekly to monthly).

    Since valve maintenance requires accurate valve calibration data which may change frequently, it is advised to
    reimplement this CLI for each project, similar to all other 'reference' CLI commands from this library.
    """

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

def run_experiment(experimenter: str, animal: str, animal_weight: float) -> None:
    """Runs a reference experiment session for the specified animal, using the input parameters.

    The CLI is primarily designed to calibrate and test the Sun lab Mesoscope-VR system and to demonstrate how to
    implement experiment runtimes for custom projects. Depending on the animal id, this CLI statically uses 'TestMice'
    or 'Template' project.
    """

def preprocess_session(session_path: Path) -> None:
    """Preprocesses the target session's data.

    Primarily, this command is intended to retry or resume failed or interrupted preprocessing runtimes.
    Preprocessing should be carried out immediately after data acquisition to optimize the acquired data for long-term
    storage and distribute it to the NAS and the BioHPC cluster for further processing and storage.

    This command aggregates all session data on the VRPC, compresses the data to optimize it for network transmission
    and storage, and transfers the data to the NAS and the BioHPC cluster. It automatically skips already completed
    processing stages as necessary to optimize runtime performance.
    """

def purge_data(remove_ubiquitin: bool, remove_telomere: bool) -> None:
    """Depending on configuration, removes all redundant data directories from the ScanImagePC, VRPC, or both.

    This command should be used at least weekly to remove no longer necessary data from the PCs used during data
    acquisition. Unless this function is called, our preprocessing pipelines will NOT remove the data, eventually
    leading to both PCs running out of storage space.
    """
