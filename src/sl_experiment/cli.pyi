from pathlib import Path

from .experiment import (
    run_train_logic as run_train_logic,
    lick_training_logic as lick_training_logic,
    run_experiment_logic as run_experiment_logic,
    vr_maintenance_logic as vr_maintenance_logic,
)
from .data_classes import (
    SessionData as SessionData,
    ProjectConfiguration as ProjectConfiguration,
    replace_root_path as replace_root_path,
)
from .zaber_bindings import (
    CRCCalculator as CRCCalculator,
    discover_zaber_devices as discover_zaber_devices,
)
from .data_preprocessing import (
    purge_redundant_data as purge_redundant_data,
    preprocess_session_data as preprocess_session_data,
)

def calculate_crc(string: str) -> None:
    """Calculates the CRC32-XFER checksum for the input string."""

def list_devices(errors: bool) -> None:
    """Displays information about all Zaber devices available through USB ports of the host-system."""

def lick_training(
    user: str,
    animal: str,
    project: str,
    animal_weight: float,
    minimum_delay: int,
    maximum_delay: int,
    maximum_volume: float,
    maximum_time: int,
) -> None:
    """Runs the lick training session for the specified animal and project combination.

    Lick training is the first phase of preparing the animal to run experiment runtimes in the lab, and is usually
    carried out over the first two days of head-fixed training. Primarily, this training is designed to teach the
    animal to operate the lick-port and associate licking at the port with water delivery.
    """

def maintain_vr(project: str) -> None:
    """Exposes a terminal interface to interact with the water delivery solenoid valve and the running wheel break.

    This CLI command is primarily designed to fill, empty, check, and, if necessary, recalibrate the solenoid valve
    used to deliver water to animals during training and experiment runtimes. Also, it is capable of locking or
    unlocking the wheel breaks, which is helpful when cleaning the wheel (after each session) and maintaining the wrap
    around the wheel surface (weekly to monthly).

    The interface also contains Zaber motors (HeadBar and LickPort) bindings to facilitate testing the quality of
    implanted cranial windows before running training sessions for new animals.
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
) -> None:
    """Runs the run training session for the specified animal and project combination.

    Run training is the second phase of preparing the animal to run experiment runtimes in the lab, and is usually
    carried out over the five days following the lick training sessions. Primarily, this training is designed to teach
    the anima how to run the wheel treadmill while being head-fixed and associate getting water rewards with running
    on the treadmill. Over the course of training, the task requirements are adjusted to ensure the animal performs as
    many laps as possible during experiment sessions lasting ~60 minutes.
    """

def run_experiment(user: str, project: str, experiment: str, animal: str, animal_weight: float) -> None:
    """Runs the requested experiment session for the specified animal and project combination.

    Experiment runtimes are carried out after the lick and run training sessions. Unlike training runtimes, experiment
    runtimes use the Virtual Reality (VR) system and rely on Unity game engine to resolve the experiment task logic
    during runtime. Also, experiments use the Mesoscope to acquire the brain activity data, which is mostly handled by
    the ScanImage software.

    Unlike training CLIs, this CLI can be used to run a variety of experiments. Each experiment is configured via the
    user-written configuration .yaml file, which should be stored inside the 'configuration' folder of the target
    project. The experiments are discovered by name, allowing a single project to have multiple different experiments.
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

def purge_data(project: str, remove_ubiquitin: bool, remove_telomere: bool) -> None:
    """Depending on configuration, removes all redundant data directories for ALL projects from the ScanImagePC,
    VRPC, or both.

    This command should be used at least weekly to remove no longer necessary data from the PCs used during data
    acquisition. Unless this function is called, our preprocessing pipelines will NOT remove the data, eventually
    leading to both PCs running out of storage space. Note, despite the command taking in a project name, it removes
    redundant data for all projects stored in the same root folder as the target project.
    """

def replace_local_root_directory(path: str) -> None:
    """Replaces the current local project root directory with the specified directory.

    To ensure all projects are saved in the same location, this library statically resolves and saves the path to the
    root directory in default user directory. Since this directory is typically hidden, this CLI can be used to
    conveniently replace the local directory path, if necessary.
    """
