"""A Python library that provides tools to acquire, manage, and preprocess scientific data in the Sun (NeuroAI) lab.

See https://github.com/Sun-Lab-NBB/sl-experiment for more details.
API documentation: https://sl-experiment.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Kushaan Gupta, Natalie Yeung, Katlynn Ryu, Jasmine Si
"""

from .experiment import run_train_logic, lick_training_logic, run_experiment_logic, vr_maintenance_logic
from .data_processing import process_log_directory
from .data_preprocessing import (
    SessionData,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    RuntimeHardwareConfiguration,
    MesoscopeExperimentDescriptor,
    purge_redundant_data,
)

__all__ = [
    "lick_training_logic",
    "vr_maintenance_logic",
    "run_train_logic",
    "run_experiment_logic",
    "purge_redundant_data",
    "process_log_directory",
    "SessionData",
    "LickTrainingDescriptor",
    "RunTrainingDescriptor",
    "MesoscopeExperimentDescriptor",
    "RuntimeHardwareConfiguration",
]
