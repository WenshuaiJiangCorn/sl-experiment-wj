from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import (
    time as dt_time,
    datetime,
)
import click
import numpy as np
from ataraxis_base_utilities import LogLevel, console
from water_log import ParseData, _SheetData


def _normalize_date(date: str) -> str:
    try:
        dt = datetime.strptime(date, "%m/%d/%y")
        return dt.strftime("%-m/%-d/%y") 
    
    except ValueError:
        raise ValueError(f"Invalid date format: {date}. Expected 'MM/DD/YY'.")
    

def _force_valid_date_input() -> str:
    while True:
        date = click.prompt("Enter the date in the format MM/DD/YY", type=str)

        try:
            normalized_date = _normalize_date(date)
            return normalized_date
        except ValueError:
            click.echo("Invalid date format.")


def _valid_mouse_id(mouse_id: str, available_ids: set[int]) -> int:
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
    "-log",
    "--log-choice",
    type=int,
    default=None,
    help="The type of log to fetch: options include surgery and water log.",
)
@click.option(
    "-id",
    "--mouse-id",
    type=int,
    default=None,
    help="The ID of the mouse to fetch data for.",
)
@click.option(
    "-dt",
    "--date",
    type=str,
    default=None,
    help="The date of the water log entry to fetch in the format MM/DD/YY.",
)
@click.option(
    "-all",
    "--list-all-mice",
    is_flag=True,
    help="List all mouse IDs available in the water log and exit.",
)

def main(log_choice: int, mouse_id: int, date: str, list_all_mice: bool) -> None:
    if list_all_mice:
        list_water_mouse_ids()
        return

    if log_choice is None:
        log_choice = click.prompt(
            "Press 1 to view data from Surgery Log and 2 for Water Log",
            type=int,
        )
        
    if log_choice == 1:
        process_surgery_log()

    elif log_choice == 2:
        process_water_log(mouse_id, date, list_all_mice)
        
    else:
        click.echo("Invalid log choice. Enter 1 for Surgery Log or 2 for Water Log.")


def process_water_log(mouse_id: int, date: str, list_all_mice: bool) -> None:
    if list_all_mice:
        list_water_mouse_ids()
        return

    sheet_data = _SheetData()
    sheet_data._fetch_data_from_tab("MouseInfo")
    tab_data = sheet_data.data.get("MouseInfo", [])
    headers = tab_data[0] if tab_data else []
    mouse_col_index = headers.index("mouse") if "mouse" in headers else -1

    available_ids = set()
    for row in tab_data[1:]:
        if len(row) > mouse_col_index:
            try:
                available_ids.add(int(row[mouse_col_index]))
            except ValueError:
                click.echo(f"Invalid mouse ID found in the data: {row[mouse_col_index]}")

    if mouse_id is None:
        click.echo("Available mouse IDs:")
        list_water_mouse_ids()  
        while True:
            mouse_id_input = click.prompt("Please enter the mouse ID", type=str)
            try:
                mouse_id = _valid_mouse_id(mouse_id_input, available_ids)
                break
            except ValueError as e:
                click.echo(str(e))

    if date is None:
        date = click.prompt("Please enter the date (format: M/DD/YY)", type=str)

    fetch_water_log_data(mouse_id, date)


def process_surgery_log():
    pass


def list_water_mouse_ids() -> None:
    """
    Lists all mouse IDs available in the water log.
    """
    sheet_data = _SheetData()
    sheet_data._fetch_data_from_tab("MouseInfo")
    tab_data = sheet_data.data.get("MouseInfo", [])
    headers = tab_data[0] if tab_data else []
    mouse_col_index = headers.index("mouse") if "mouse" in headers else -1

    if mouse_col_index == -1:
        click.echo("'mouse' column not found in the water log.")
        return

    mouse_ids = set()
    for row in tab_data[1:]:
        if len(row) > mouse_col_index:
            try:
                mouse_id = int(row[mouse_col_index])
                mouse_ids.add(mouse_id)
            except ValueError:
                print(f"Invalid mouse ID: {row[mouse_col_index]}")

    if mouse_ids:
        click.echo("Available Mouse IDs for Water Log:")
        for mouse_id in sorted(mouse_ids):
            click.echo(mouse_id)
    else:
        click.echo("No mouse IDs found in the water log.")


def list_surgery_mouse_ids() -> None:
    pass


def fetch_water_log_data(mouse_id: int, date: str) -> None:
    if date is None:
        click.echo("No date provided. Please enter a valid date.")
        date = _force_valid_date_input()

    else:
        try:
            date = _normalize_date(date)  
        except ValueError:
            click.echo(f"Invalid date format: {date}. Please use the format 'MM/DD/YY'.")
            date = _force_valid_date_input()

    water_log = ParseData(tab_name="MouseInfo", mouse_id=mouse_id, date=date)
    mouse_instance = water_log.mouse_instances.get(mouse_id)
    
    if mouse_instance:
        if date in mouse_instance.daily_log:
            click.echo(mouse_instance)  

        else:
            click.echo(f"No water log data found for mouse {mouse_id} on {date}.")
    else:
        click.echo(f"No data found for mouse {mouse_id}.")


def fetch_surgery_log_data(self):
    pass


if __name__ == "__main__":
    main()