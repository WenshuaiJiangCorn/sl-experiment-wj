"""A heavily refactored version of the sl-experiment library used to run experiments in the Yapici lab."""

from ataraxis_base_utilities import console

# Ensures the console is enabled whenever this library is imported.
if not console.enabled:
    console.enable()
