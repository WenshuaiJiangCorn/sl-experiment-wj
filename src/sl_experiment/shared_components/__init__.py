"""This package stores data acquisition and preprocessing assets shared by multiple data acquisition systems."""

from .shared_tools import get_version_data
from .module_interfaces import (
    TTLInterface,
    LickInterface,
    ValveInterface,
)

__all__ = [
    "TTLInterface",
    "ValveInterface",
    "LickInterface",
    "get_version_data",
]
