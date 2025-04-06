"""A Python library that provides tools to acquire, manage, and preprocess scientific data in the Sun (NeuroAI) lab.

See https://github.com/Sun-Lab-NBB/sl-experiment for more details.
API documentation: https://sl-experiment.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Kushaan Gupta, Natalie Yeung, Katlynn Ryu, Jasmine Si
"""

# Unlike most other libraries, all of this library's features are realized via the click-based CLI commands
# automatically exposed by installing the library into a conda environment. All explicit exports here are intended
# exclusively for Sun lab data processing libraries that reuse some classes and functions defined in this library
# to further process the acquired data.

from .data_classes import HardwareConfiguration
from .transfer_tools import transfer_directory
from .packaging_tools import calculate_directory_checksum
