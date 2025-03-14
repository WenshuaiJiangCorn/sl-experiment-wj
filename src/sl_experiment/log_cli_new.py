from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import (
    time as dt_time,
    datetime,
)

import click
import numpy as np
from ataraxis_base_utilities import LogLevel, console

from .water_log import ParseData, _WaterSheetData
from .gs_data_parser import SurgeryData, FilteredSurgeries, _SurgerySheetData


def _normalize_date(date: str) -> str:
    """
    This method converts the input date from MM/DD/YY format to M/D/YY format
    to retrieve the correct date for the daily water restriction data from the
    water log.
    """
    try:
        dt = datetime.strptime(date, "%m/%d/%y")
        return dt.strftime("%-m/%-d/%y")

    except ValueError:
        raise ValueError(f"Invalid date format: {date}. Expected 'MM/DD/YY'.")


def _force_valid_date_input() -> str:
    """
    This method continuously prompts the user for a valid date.
    """
    while True:
        date = click.prompt("Enter the date in the format MM/DD/YY", type=str)

        try:
            normalized_date = _normalize_date(date)
            return normalized_date
        except ValueError:
            click.echo("Invalid date format.")


def _valid_mouse_id(mouse_id: str, available_ids: set[int]) -> int:
    """
    This method continuously prompts the user to input a valid mouse ID from
    the available entries from the chosen log.
    """
    try:
        mouse_id_int = int(mouse_id)
        if mouse_id_int in available_ids:
            return mouse_id_int
        else:
            raise ValueError(f"Mouse ID {mouse_id_int} is not in the list of available IDs.")

    except ValueError:
        raise ValueError(f"Invalid mouse ID")


@click.command()
@click.option(
    "--log-choice",
    type=click.Choice(["1", "2"]),
    default="1",
    help="The type of log to fetch: 1 for Surgery Log, 2 for Water Log.",
)
@click.option(
    "--mouse-id",
    type=int,
    help="The ID of the mouse to fetch data for.",
)
@click.option(
    "--date",
    type=str,
    help="The date of the water log entry to fetch (format: MM/DD/YY).",
)
@click.option(
    "--list-water-mice",
    is_flag=True,
    help="List all mouse IDs available in the water log and exit.",
)
@click.option(
    "--list-surgery-mice",
    is_flag=True,
    help="List all mouse IDs available in the surgery log and exit.",
)
@click.option(
    "--id-from-tab-surgery",
    type=str,
    help="The tab name for the surgery log to fetch data based on the mouse ID.",
)
def process_gs_log(
    log_choice: str,
    mouse_id: int,
    date: str,
    list_water_mice: bool,
    list_surgery_mice: bool,
    id_from_tab_surgery: str,
) -> None:
    if list_water_mice:
        list_water_mouse_ids()
        return

    if list_surgery_mice:
        if not id_from_tab_surgery:
            click.echo("Error: --id-from-tab-surgery is required when listing surgery mice.")
            return
        list_surgery_mouse_ids(id_from_tab_surgery)
        return

    if log_choice == "1":
        if not id_from_tab_surgery:
            click.echo("Error: --id-from-tab-surgery is required for surgery log.")
            return
        process_surgery_log(id_from_tab_surgery, mouse_id)

    elif log_choice == "2":
        process_water_log(mouse_id, date, list_water_mice)

    else:
        click.echo("Invalid log choice. Enter 1 for Surgery Log or 2 for Water Log.")


def process_water_log(mouse_id: int | None, date: str | None, list_water_mice: bool) -> None:
    """
    Processes the water log for a given mouse ID and date.
    Handles input validation and forces valid inputs.
    """
    sheet_data = _WaterSheetData()
    sheet_data._fetch_data_from_tab("MouseInfo")
    tab_data = sheet_data.data.get("MouseInfo", [])
    headers = tab_data[0]
    mouse_col_index = headers.index("mouse")

    available_ids = set()

    for row in tab_data[1:]:
        if len(row) > mouse_col_index:
            cell_value = row[mouse_col_index]
            if isinstance(cell_value, str) and cell_value.strip():
                try:
                    available_ids.add(int(cell_value))
                except ValueError:
                    continue

    if list_water_mice:
        list_water_mouse_ids()
        return

    if mouse_id is None:
        list_water_mouse_ids()
        while True:
            mouse_id_input = click.prompt("Enter the mouse ID", type=str)
            try:
                mouse_id = _valid_mouse_id(mouse_id_input, available_ids)
                break
            except ValueError as e:
                click.echo(str(e))

    if date is None:
        date = click.prompt("Enter the date in the form MM/DD/YY)", type=str)

    try:
        date = _normalize_date(date)
    except ValueError:
        click.echo(f"Invalid date format: {date}.")
        date = _force_valid_date_input()

    fetch_water_log_data(mouse_id, date)


def process_surgery_log(tab_name: str, mouse_id: int | None) -> None:
    """
    This method ensures the user provides a valid tab name and mouse ID from the surgery log.
    Mouse IDs are extracted from the header row of each tab.
    """
    sheet_data = _SurgerySheetData(tab_name=tab_name)
    sheet_data._parse()
    mouse_col_index = sheet_data.headers["id"]

    available_ids = set()
    for row in sheet_data.data:
        if len(row) > mouse_col_index:
            cell_value = row[mouse_col_index]
            if isinstance(cell_value, str) and cell_value.strip():
                try:
                    available_ids.add(int(cell_value))
                except ValueError:
                    continue

    if not available_ids:
        click.echo(f"No mouse IDs found in tab '{tab_name}'.")
        return

    if mouse_id is None:
        list_surgery_mouse_ids(tab_name)

        while True:
            mouse_id_input = click.prompt("Enter the mouse ID", type=str)
            try:
                mouse_id = _valid_mouse_id(mouse_id_input, available_ids)
                break
            except ValueError as e:
                click.echo(str(e))

    fetch_surgery_log_data(tab_name, mouse_id)


def list_water_mouse_ids() -> None:
    """
    Lists all mouse IDs available in the water log.
    """
    sheet_data = _WaterSheetData()
    sheet_data._fetch_data_from_tab("MouseInfo")
    tab_data = sheet_data.data.get("MouseInfo", [])
    headers = tab_data[0]
    water_key = headers.index("mouse")

    if water_key == -1:
        click.echo("'mouse' column not found in the water log.")
        return

    mouse_ids = set()
    for row in tab_data[1:]:
        if len(row) > water_key:
            cell_value = row[water_key]
            if isinstance(cell_value, str) and cell_value.strip():
                try:
                    mouse_ids.add(int(cell_value))
                except ValueError:
                    continue

    if mouse_ids:
        click.echo("Available Mouse IDs for Water Log:")
        for mouse_id in sorted(mouse_ids):
            click.echo(mouse_id)
    else:
        click.echo("No mouse IDs found in the water log.")


def list_surgery_mouse_ids(tab_name: str) -> None:
    """
    Lists all mouse IDs available in the surgery log for the specified tab.
    """
    sheet_data = _SurgerySheetData(tab_name=tab_name)
    sheet_data._parse()

    mouse_col_index = sheet_data.headers.get("id", -1)
    if mouse_col_index == -1:
        click.echo(f"'id' column not found in tab '{tab_name}'. Available columns: {list(sheet_data.headers.keys())}")
        return

    mouse_ids = set()
    for row in sheet_data.data:
        if len(row) > mouse_col_index:
            cell_value = row[mouse_col_index]
            if isinstance(cell_value, str) and cell_value.strip():
                try:
                    mouse_ids.add(int(cell_value))
                except ValueError:
                    continue

    if mouse_ids:
        click.echo(f"Available Mouse IDs in tab '{tab_name}':\n" + "\n".join(map(str, sorted(mouse_ids))))
    else:
        click.echo(f"No mouse IDs found in tab '{tab_name}'.")


def fetch_water_log_data(mouse_id: int, date: str) -> None:
    """
    Fetches both the baseline information about the mouse and the water restriction
    data for a given date.
    """
    water_log = ParseData(tab_name="MouseInfo", mouse_id=mouse_id, date=date)
    mouse_instance = water_log.mouse_instances.get(str(mouse_id))
    if mouse_instance is None:
        click.echo(f"Mouse ID {mouse_id} not found in the water log.")
        return

    if mouse_instance:
        if date in mouse_instance.daily_log:
            click.echo(mouse_instance)
        else:
            click.echo(f"No water restriction data found for mouse {mouse_id} on {date}.")
    else:
        click.echo(f"No data found for mouse {mouse_id}.")


def fetch_surgery_log_data(tab_name: str, mouse_id: int) -> FilteredSurgeries:
    """
    Fetches the protocol data, implant data, injection data, brain data, and drug
    data for the given mouse ID.
    """
    sheet_data = _SurgerySheetData(tab_name=tab_name)
    sheet_data._parse()

    filtered_surgeries = FilteredSurgeries(surgeries=[])
    for row in sheet_data.data:
        surgery_data = SurgeryData(headers=sheet_data.headers, row=row, tab_name=tab_name)

        if surgery_data.protocol_data.id == mouse_id:
            filtered_surgeries.surgeries.append(surgery_data)

    return filtered_surgeries
