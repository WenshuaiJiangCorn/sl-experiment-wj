from _typeshed import Incomplete

from .experiment import (
    ExperimentState as ExperimentState,
    run_train_logic as run_train_logic,
    lick_training_logic as lick_training_logic,
    run_experiment_logic as run_experiment_logic,
    vr_maintenance_logic as vr_maintenance_logic,
)
from .zaber_bindings import (
    _CRCCalculator as _CRCCalculator,
    discover_zaber_devices as discover_zaber_devices,
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
