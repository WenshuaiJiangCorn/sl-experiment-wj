"""A Python library that provides the interfaces and runtime control bindings used to train animals and record
experiment data in the Sun (NeuroAI) lab at Cornell university.

See https://github.com/Sun-Lab-NBB/sl-experiment for more details.
API documentation: https://sl-experiment.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Natalie Yeung, Katlynn Ryu, Jasmine Si
"""

from .experiment import run_train_logic, lick_training_logic, run_experiment_logic, vr_maintenance_logic
from .data_preprocessing import purge_redundant_data, preprocess_session_directory
from .data_processing import process_log_directory

__all__ = [
    "lick_training_logic",
    "vr_maintenance_logic",
    "run_train_logic",
    "run_experiment_logic",
    "preprocess_session_directory",
    "purge_redundant_data",
    "process_log_directory"
]
