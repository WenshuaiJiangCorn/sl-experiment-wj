from typing import Dict, List, Type, Optional
from pathlib import Path
from datetime import (
    time as dt_time,
    datetime,
    timezone,
)
from dataclasses import field, dataclass

import numpy as np
from gs_data_parser import _convert_date_time
from ataraxis_data_structures import YamlConfig
from googleapiclient.discovery import build  # type: ignore
from google.oauth2.service_account import Credentials


class _WaterSheetData:
    """
    This class initializes key identifiers for the Google Sheet, including the spreadsheet URL,
    the cell range, and all tabs within the sheet. OAuth 2.0 scopes are used to link
    and grant access to Google APIs for data parsing.
    """

    def __init__(self):
        self.sheet_id = "1AofPA9J2jqg8lORGfc-rKn5s8YCr6EpH9ZMsUoWzcHQ"
        self.range = "A1:Z"
        self.SERVICE_ACCOUNT_FILE = "/Users/natalieyeung/Documents/GitHub/sl-mesoscope/water_log.json"
        self.SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
        self.data: Dict[str, List[List[Optional[str]]]] = {}

    def _get_tab_data(self, tab_name: str) -> List[List[str]]:
        """
        Retrieves data from the specified tab in the Google Sheet.
        """
        creds = Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)  # type: ignore
        service = build("sheets", "v4", credentials=creds)
        range_name = f"'{tab_name}'!{self.range}"
        result = service.spreadsheets().values().get(spreadsheetId=self.sheet_id, range=range_name).execute()
        return result.get("values", [])

    def _replace_empty(self, row_data: List[List[str]]) -> List[List[Optional[str]]]:
        """
        Replaces empty cells and cells containing 'n/a', '--' or '---' with None. This funcation
        also ensures that cells in the main grid are processed and that all rows  have equal length.
        """
        result: List[List[Optional[str]]] = []

        for row in row_data:
            processed_row: List[Optional[str]] = []

            for cell in row:
                if not cell.strip():
                    break

                if cell.strip().lower() in {"n/a", "--", "---", ""}:
                    processed_row.append(None)
                else:
                    processed_row.append(cell)

            result.append(processed_row)

        max_row_length = max(len(row) for row in result)
        for row in result:
            while len(row) < max_row_length:
                row.append(None)

        return result

    def _fetch_data_from_tab(self, tab_name: str) -> None:
        """
        Fetches data from the specified tab, processes it, and stores it in self.data.
        """
        tab_data = self._get_tab_data(tab_name)
        processed_data = self._replace_empty(tab_data)
        self.data[tab_name] = processed_data

    def _get_tab_names(self) -> List[str]:
        """
        Retrieves the metadata including the names of all tabs in the Google Sheet.
        """
        creds = Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)  # type: ignore
        service = build("sheets", "v4", credentials=creds)
        sheet_metadata = service.spreadsheets().get(spreadsheetId=self.sheet_id).execute()
        sheets = sheet_metadata.get("sheets", "")
        return [sheet["properties"]["title"] for sheet in sheets]

    def _write_to_sheet(self, tab_name: str, range_name: str, values: List[List[str]]) -> None:
        """
        Updates a specified cell or range in a Google Sheets document with the given values using
        the provided tab name and range.
        """

        creds = Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)  # type: ignore
        service = build("sheets", "v4", credentials=creds)
        full_range = f"'{tab_name}'!{range_name}"
        body = {"values": values}

        result = (
            service.spreadsheets()
            .values()
            .update(
                spreadsheetId=self.sheet_id,
                range=full_range,
                valueInputOption="RAW",
                body=body,
            )
            .execute()
        )


@dataclass
class DailyLog(YamlConfig):
    """
    This class stores the daily tracking data and water intake metrics for a mouse during
    the water restriction period.

    Args:
        row: A list of values representing a single row of data from the Google Sheet.
        headers: A list mapping column names (headers) to their respective indices in the row.

    Attributes:
        date_time: Stores the date and time water was provided to the mouse.
        weight: Stores the weight of the mouse.
        baseline_percent: Stores the percentage relative to the mouse's original weight.
        water_given: Stores the amount of water provided to water per day in mL.
        given_by: Stores the netID of the water provider.
    """

    date_time: np.uint64 = np.uint64(0)
    weight: float = 0.0
    baseline_percent: float = 0.0
    water_given: float = 0.0
    given_by: str | None = None

    def __init__(self, row: List[str], headers: List[str]):
        self.date_time = _convert_date_time(date=row[headers.index("date")], time=row[headers.index("time")])
        self.weight = float(row[headers.index("weight (g)")] or self.weight)
        self.baseline_percent = float(row[headers.index("baseline %")] or self.baseline_percent)
        self.water_given = float(row[headers.index("water given (mL)")] or self.water_given)
        self.given_by = row[headers.index("given by:")] or self.given_by


@dataclass
class MouseKey(YamlConfig):
    """
    This class stores a mouse's identifying details and daily water restriction data from the
    DailyLog class.

    Args:
        row: A list of values representing a single row of data from the Google Sheet.
        headers: A list mapping column names (headers) to their respective indices in the row.

    Attributes:
        mouse_id: Stores the mouseID. The mouseID corresponds to the ID on the surgery log.
        cage: Stores the cage number which the mouse is placed in.
        ear_punch: Stores whether the mouse has an ear punch in either ear.
        sex: Stores the gender of the mouse.
        baseline_weight: Stores the original weight of the mouse.
        target_weight: Stores the weight goal for the mouse during water restriction.
        daily_log: A dictionary storing instances of the DailyLog class, where each entry
                   corresponds to a specific date and contains the water log data on the day.
    """

    mouse_id: int = 0
    cage: int = 0
    ear_punch: str | None = None
    baseline_weight: float = 0.0
    target_weight: float = 0.0
    sex: str | None = None
    daily_log: Dict[str, DailyLog] = field(default_factory=dict)

    def __init__(self, row: List[str], headers: List[str]):
        self.mouse_id = int(row[headers.index("mouse")])
        self.cage = int(row[headers.index("cage #")])
        self.ear_punch = row[headers.index("ID (ear)")]
        self.sex = row[headers.index("sex")]
        self.baseline_weight = float(row[headers.index("baseline weight (g)")])
        self.target_weight = float(row[headers.index("target weight (g)")])
        self.daily_log = {}

    def __repr__(self) -> str:
        """
        Returns a string representation of the MouseKey instance including all attributes
        from the MouseKey class and data from the DailyLog class.
        """
        daily_log_str = ", ".join([f"{date}: {log}" for date, log in self.daily_log.items()])
        return (
            f"MouseKey(mouse_id={self.mouse_id}, "
            f"cage={self.cage}, "
            f"ear_punch={self.ear_punch}, "
            f"sex={self.sex}, "
            f"baseline_weight={self.baseline_weight}, "
            f"target_weight={self.target_weight}, "
            f"daily_log={{{daily_log_str}}})"
        )


class ParseData:
    """
    Parses data from a Google Sheet for a specific mouse.

    Args:
        tab_name: Stores the name of the tab containing the key identification information
                  for all mice.
        mouse_id: Stores the specific ID of the mouse to retrieve data from.
        date: Stores the specific date of the water log entry to fetch.

    Attributes:
        mouse_classes: A dictionary that maps mouse IDs to their respective MouseKey subclasses.
        mouse_instances: A dictionary that maps mouse IDs to their initialized MouseKey instances containing
                         the mouse's data and corresponding daily logs.
    """

    def __init__(self, tab_name: str, mouse_id: int, date: str):
        self.tab_name = tab_name
        self.mouse_id = mouse_id
        self.date = date
        self.sheet_data = _WaterSheetData()
        self.sheet_data._fetch_data_from_tab(tab_name)
        self.mouse_classes: Dict[str, Type[MouseKey]] = {}
        self.mouse_instances: Dict[str, MouseKey] = {}
        self._create_mouse_subclasses()
        self._link_to_tab()

    def _extract_mouse_id(self, tab_data: List[List[str]], mouse_col_index: int) -> List[int]:
        """
        Extracts unique mouse IDs from the main tab containing key identification information
        for each mice.
        """
        mouse_ids = []

        for row in tab_data[1:]:
            if len(row) > mouse_col_index:
                try:
                    mouse_id = int(row[mouse_col_index])
                    if mouse_id not in mouse_ids:
                        mouse_ids.append(mouse_id)
                except ValueError:
                    print(f"Invalid mouse ID: {row[mouse_col_index]}")
        return mouse_ids

    def _create_mouse_subclasses(self) -> None:
        """
        Creates a subclass for the specified mouse and initializes its instance with data.

        This method checks if the given mouse ID exists in the sheet. If found, it creates
        a subclass of MouseKey named Mouse_<mouse_id> and finds the corresponding row in the dataset
        and initializes it.
        """

        tab_data = self.sheet_data.data.get(self.tab_name, [])
        headers = tab_data[0]
        mouse_col_index = headers.index("mouse")

        if self.mouse_id in self._extract_mouse_id(tab_data, mouse_col_index):
            subclass_name = f"Mouse_{self.mouse_id}"
            self.mouse_classes[self.mouse_id] = type(subclass_name, (MouseKey,), {})

            for row in tab_data[1:]:
                if int(row[mouse_col_index]) == self.mouse_id:
                    self.mouse_instances[self.mouse_id] = MouseKey(row, headers)
                    break

    def _link_to_tab(self) -> None:
        """
        This method links the mouse ID to its corresponding tab and initializes DailyLog subclasses.
        If the tab corresponding to the mouseID is found, it creates a DailyLog instance for
        every date and stores it in the daily_log dictionary of the corresponding mouse instance.
        """

        all_tabs = self.sheet_data._get_tab_names()
        tab_name = str(self.mouse_id)

        if tab_name in all_tabs:
            self.sheet_data._fetch_data_from_tab(tab_name)

            linked_tab_data = self.sheet_data.data[tab_name]
            headers = linked_tab_data[1]
            data_rows = linked_tab_data[2:]

            for row in data_rows:
                if len(row) > 1:
                    date = row[headers.index("date")]
                    if date and (self.date in [None, date]):
                        daily_log = DailyLog(row, headers)
                        self.mouse_instances[self.mouse_id].daily_log[date] = daily_log


# Write Data
# sheet_data = _WaterSheetData()
# sheet_data._write_to_sheet(
#     tab_name="MouseInfo",
#     range_name="A6",
#     values=[["6"]]
# )

# Read Data
water_log = ParseData(tab_name="MouseInfo", mouse_id=1, date="2/8/25")
mouse_instance = water_log.mouse_instances.get(water_log.mouse_id)
mouse_instance.to_yaml(file_path=Path("water_log.yaml"))
