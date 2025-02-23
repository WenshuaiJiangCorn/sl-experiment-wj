"""This module provides click-based Command-Line Interface (CLI) scripts that allow using various features from this
library through the terminal."""

import click
from pathlib import Path
from .zaber_bindings import _CRCCalculator, discover_zaber_devices
from .experiment import _HeadBar, _LickPort


@click.command()
@click.option(
    "--string", "-i", prompt="Enter the string to be checksummed", help="The string to calculate the CRC checksum for."
)
def calculate_crc(string: str) -> None:
    """Calculates the CRC32-XFER checksum for the input string."""
    calculator = _CRCCalculator()
    crc_checksum = calculator.string_checksum(string)
    click.echo(f"The CRC32-XFER checksum for the input string '{string}' is: {crc_checksum}")


@click.command()
@click.option(
    "-e",
    "--errors",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display errors encountered when connecting to all evaluated serial ports.",
)
def list_devices(errors: bool) -> None:
    """Displays information about all Zaber devices available through USB ports of the host-system."""
    discover_zaber_devices(silence_errors=not errors)


@click.command()
@click.option(
    "-zp",
    "--zaber_positions_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="/home/cybermouse/Desktop/TestOut/zaber_positions.yaml",
    help=(
        "The path to the zaber_positions.yaml file. If the file exists, it will be used during runtime to restore "
        "motor positions. Also, this path will be used to save the motor positions extracted during runtime."
    ),
)
@click.option(
    "-hp", "--headbar_port", type=str, default="/dev/ttyUSB0", help="The USB port used by the HeadBar controller."
)
@click.option(
    "-lp", "--lickport_port", type=str, default="/dev/ttyUSB1", help="The USB port used by the LickPort controller."
)
def mesoscope_vr_cli(zaber_positions_path: Path, headbar_port: str, lickport_port: str):
    """Provides the CLI to control all Mesoscope-VR components used during experiment and training runtimes.

    Primarily, this is used to calibrate zaber motors and interface with the solenoid valve between experiment and
    training runtimes.
    """

    # Initializes binding classes
    headbar = _HeadBar(headbar_port, zaber_positions_path)
    lickport = _LickPort(lickport_port, zaber_positions_path)

    while True:
        command = input("Enter command: ")

        if command == 'home':
            headbar.prepare_motors(wait_until_idle=False)
            lickport.prepare_motors(wait_until_idle=True)
            headbar.wait_until_idle()

        if command == 'mount':
            headbar.mount_position(wait_until_idle=False)
            lickport.mount_position(wait_until_idle=True)
            headbar.wait_until_idle()

        if command == 'calibrate':
            headbar.calibrate_position(wait_until_idle=False)
            lickport.calibrate_position(wait_until_idle=True)
            headbar.wait_until_idle()

        if command == 'restore':
            headbar.restore_position(wait_until_idle=False)
            lickport.restore_position(wait_until_idle=True)
            headbar.wait_until_idle()

        if command == 'park':
            headbar.park_position(wait_until_idle=False)
            lickport.park_position(wait_until_idle=True)
            headbar.wait_until_idle()

        # Shuts down zaber bindings
        headbar.park_position(wait_until_idle=False)
        lickport.park_position(wait_until_idle=True)
        headbar.wait_until_idle()
        headbar.disconnect()
        lickport.disconnect()
