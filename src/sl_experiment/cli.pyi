from pathlib import Path

from .mesoscope_vr import (
    CRCCalculator as CRCCalculator,
    experiment_logic as experiment_logic,
    maintenance_logic as maintenance_logic,
    run_training_logic as run_training_logic,
    lick_training_logic as lick_training_logic,
    purge_redundant_data as purge_redundant_data,
    window_checking_logic as window_checking_logic,
    discover_zaber_devices as discover_zaber_devices,
    preprocess_session_data as preprocess_session_data,
)

def calculate_crc(input_string: str) -> None:
    """Calculates the CRC32-XFER checksum for the input string."""

def list_devices(errors: bool) -> None:
    """Displays information about all Zaber devices available through USB ports of the host-system."""

def maintain_acquisition_system() -> None:
    """Exposes a terminal interface to interact with the water delivery solenoid valve and the running wheel break.

    This CLI command is primarily designed to fill, empty, check, and, if necessary, recalibrate the solenoid valve
    used to deliver water to animals during training and experiment runtimes. Also, it is capable of locking or
    unlocking the wheel breaks, which is helpful when cleaning the wheel (after each session) and maintaining the wrap
    around the wheel surface (weekly to monthly).
    """

def lick_training(
    user: str,
    animal: str,
    project: str,
    animal_weight: float,
    minimum_delay: int,
    maximum_delay: int,
    maximum_volume: float,
    maximum_time: int,
    unconsumed_rewards: int,
    restore_parameters: bool,
) -> None:
    """Runs the lick training session for the specified animal and project combination.

    Lick training is the first phase of preparing the animal to run experiment runtimes in the lab, and is usually
    carried out over the first two days of head-fixed training. Primarily, this training is designed to teach the
    animal to operate the lick-port and associate licking at the port with water delivery.
    """

def run_training(
    user: str,
    project: str,
    animal: str,
    animal_weight: float,
    initial_speed: float,
    initial_duration: float,
    increase_threshold: float,
    speed_step: float,
    duration_step: float,
    maximum_volume: float,
    maximum_time: int,
    unconsumed_rewards: int,
    maximum_idle_time: int,
    restore_parameters: bool,
) -> None:
    """Runs the run training session for the specified animal and project combination.

    Run training is the second phase of preparing the animal to run experiment runtimes in the lab, and is usually
    carried out over the five days following the lick training sessions. Primarily, this training is designed to teach
    the animal how to run the wheel treadmill while being head-fixed and associate getting water rewards with running
    on the treadmill. Over the course of training, the task requirements are adjusted to ensure the animal performs as
    many laps as possible during experiment sessions lasting ~60 minutes.
    """

def run_experiment(user: str, project: str, experiment: str, animal: str, animal_weight: float, guided: bool) -> None:
    """Runs the requested experiment session for the specified animal and project combination.

    Experiment runtimes are carried out after the lick and run training sessions Unlike training session commands, this
    command can be used to run different experiments. Each experiment runtime is configured via the user-defined
    configuration .yaml file, which should be stored inside the 'configuration' folder of the target project. The
    experiments are discovered by name, allowing a single project to have multiple different experiments. To create a
    new experiment configuration, use the 'sl-create-experiment' CLI command.
    """

def check_window(project: str, animal: str) -> None:
    """Runs the cranial window and surgery quality checking session for the specified animal and project combination.

    Before the animals are fully inducted (included) into a project, the quality of the surgical intervention
    (craniotomy and window implantation) is checked to ensure the animal will produce high-quality scientific data. As
    part of this process, various parameters of the Mesoscope-VR data acquisition system are also calibrated to best
    suit the animal. This command aggregates all steps necessary to verify and record the quality of the animal's window
    and to generate customized Mesoscope-VR parameters for the animal.
    """

def preprocess_session(session_path: Path) -> None:
    """Preprocesses the target session's data.

    This command aggregates all session data on the VRPC, compresses the data to optimize it for network transmission
    and storage, and transfers the data to the NAS and the BioHPC cluster. It automatically skips already completed
    processing stages as necessary to optimize runtime performance.

    Primarily, this command is intended to retry or resume failed or interrupted preprocessing runtimes.
    Preprocessing should be carried out immediately after data acquisition to optimize the acquired data for long-term
    storage and distribute it to the NAS and the BioHPC cluster for further processing and storage.
    """

def purge_data() -> None:
    """Removes all redundant data directories for ALL projects from the ScanImagePC and the VRPC.

    Redundant data purging is now executed automatically as part of data preprocessing. This command is primarily
    maintained as a fall-back option if automated data purging fails for any reason. Data purging should be carried out
    at least weekly to remove no longer necessary data from the PCs used during data acquisition.
    """
