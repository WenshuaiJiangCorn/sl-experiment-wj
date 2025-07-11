"""A Python library that provides tools to acquire, manage, and preprocess scientific data in the Sun (NeuroAI) lab.

See https://github.com/Sun-Lab-NBB/sl-experiment for more details.
API documentation: https://sl-experiment-api-docs.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Kushaan Gupta, Natalie Yeung, Katlynn Ryu, Jasmine Si
"""

# Unlike most other libraries, all of this library's features are realized via the click-based CLI commands
# automatically exposed by installing the library into a conda environment. Therefore, it currently does not contain
# any explicit API exports.

from ataraxis_base_utilities import console


# Defines a callback function to use instead of the default sys.exit call used by ataraxis console.
def _pass_callback(__error: str | int | None = None) -> None:
    pass


# Ensures console is enabled whenever this library is imported.
if not console.enabled:
    console.set_reraise(enabled=True)
    console.set_callback(callback=_pass_callback)
    console.enable()
