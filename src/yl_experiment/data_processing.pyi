from typing import Any
from pathlib import Path

import numpy as np
from numpy.typing import NDArray as NDArray
from ataraxis_data_structures import DataLogger
from ataraxis_communication_interface import ExtractedModuleData as ExtractedModuleData

from .microcontroller import (
    AMCInterface as AMCInterface,
    ModuleTypeCodes as ModuleTypeCodes,
)

def _interpolate_data(
    timestamps: NDArray[np.uint64], data: NDArray[Any], seed_timestamps: NDArray[np.uint64], is_discrete: bool
) -> NDArray[Any]: ...
def _parse_valve_data(
    extracted_module_data: ExtractedModuleData,
    output_file: Path,
    scale_coefficient: np.float64,
    nonlinearity_exponent: np.float64,
) -> None: ...
def _parse_lick_data(
    extracted_module_data: ExtractedModuleData, output_file: Path, lick_threshold: np.uint16
) -> None: ...
def process_microcontroller_log(
    data_logger: DataLogger, microcontroller: AMCInterface, output_directory: Path
) -> None: ...
