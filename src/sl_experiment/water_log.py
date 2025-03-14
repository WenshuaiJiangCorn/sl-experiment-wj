from typing import Dict, List, Type, Optional
from pathlib import Path
from datetime import (
    time as dt_time,
)
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import console
from ataraxis_data_structures import YamlConfig
from googleapiclient.discovery import build  # type: ignore
from google.oauth2.service_account import Credentials

from .gs_data_parser import _convert_date_time


class _WaterSheetData:
    """
    This class initializes key identifiers for the Google Sheet, including the spreadsheet URL,
    the cell range, and all tabs within the sheet. OAuth 2.0 scopes are used to link
    and grant access to Google APIs for data parsing.

    Args:
        tab name: Stores a list of tab names for the water log, where each tab corresponds to an individual mouse ID.
        row_data: Stores data from each row when it is processed to replace empty or irrelevant cells.
        range_name: Stores the range of the sheet to parse data from. The range is set to the entire sheet by default.
    """

    def __init__(self) -> None:
        self.sheet_id = "1AofPA9J2jqg8lORGfc-rKn5s8YCr6EpH9ZMsUoWzcHQ"
        self.range = "A1:Z"
        self.SERVICE_ACCOUNT_FILE = "/Users/natalieyeung/Documents/GitHub/sl-mesoscope/water_log.json"
        self.SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
        self.data: Dict[str, List[List[Optional[str]]]] = {}

    def _get_service(self) -> build:
        """
        Authenticates the Google Sheets API using the service account credentials and the defined API scope.
        It then builds and returns the Google Sheets API service client to enable the script to interact with
        the sheet.
        """
        creds = Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)  # type: ignore
        return build("sheets", "v4", credentials=creds)

    def _get_tab_data(self, tab_name: str) -> List[List[str]]:
        """
        Retrieves data from the specified tab in the Google Sheet.
        """
        service = self._get_service()
        range_name = f"'{tab_name}'!{self.range}"
        result = service.spreadsheets().values().get(spreadsheetId=self.sheet_id, range=range_name).execute()
        # return result.get("values", [])
        values = result.get("values", [])
        return values

    def _fetch_data_from_tab(self, tab_name: str) -> None:
        """
        Fetches data from the specified tab, processes it, and stores it in self.data.
        """
        tab_data = self._get_tab_data(tab_name)
        self.data[tab_name] = self._replace_empty(tab_data)

    def _replace_empty(self, row_data: List[List[str]]) -> List[List[Optional[str]]]:
        """
        Replaces empty cells and cells containing 'n/a', '--' or '---' with None. This funcation
        also ensures that cells in the main grid are processed and that all rows  have equal length.
        """
        result: List[List[Optional[str]]] = []

        for row in row_data:
            processed_row: List[Optional[str]] = []

            for cell in row:
                if cell.strip().lower() in {"n/a", "--", "---", ""}:
                    processed_row.append(None)
                else:
                    processed_row.append(cell)
            result.append(processed_row)

        max_row_length = max(len(row) for row in result)

        for row in result:
            row.extend([""] * (max_row_length - len(row)))

        return result

    def _get_tab_names(self) -> List[str]:
        """
        Retrieves the metadata including the names of all tabs in the Google Sheet.
        """
        service = self._get_service()
        sheet_metadata = service.spreadsheets().get(spreadsheetId=self.sheet_id).execute()
        sheets = sheet_metadata.get("sheets", "")
        return [sheet["properties"]["title"] for sheet in sheets]


class _WriteData(_WaterSheetData):
    """
    This class provides write access to the water restriction log Google Sheet to update daily water
    log records. It allows the script to modify specific attributes within the sheet, such as
    weight (g), water given (mL), the NetID of the water provider, and the time. The cell to update
    is located based on the mouse ID and date.

    Args:
        mouseID: Identifies the mouse for which data is being updated. The mouseID is
                 used to locate the corresponding tab to update.
        date: Stores the date corresponding to the row that should be updated.
        attribute: The specific column header to update.
        value(s): The new value to be written into the cell.
    """

    def _write_to_sheet(self, tab_name: str, range_name: str, values: List[List[str]]) -> None:
        """
        This method handles connection and write access to the Google Sheets API. It allows data
        to be written to multiple cells within a specified range. It also configures the formatting
        of the written data to be centered within the cell.
        """

        service = self._get_service()
        full_range = f"'{tab_name}'!{range_name}"
        body = {"values": values}
        service.spreadsheets().values().update(
            spreadsheetId=self.sheet_id, range=full_range, valueInputOption="RAW", body=body
        ).execute()

        col_letter = range_name[0].upper()
        row_number = int(range_name[1:])
        col_index = ord(col_letter) - ord("A")
        sheet_metadata = service.spreadsheets().get(spreadsheetId=self.sheet_id).execute()

        sheet_id = None
        for sheet in sheet_metadata["sheets"]:
            if sheet["properties"]["title"] == tab_name:
                sheet_id = sheet["properties"]["sheetId"]
                break

        if sheet_id is None:
            raise ValueError(f"Tab '{tab_name}' not found in the Google Sheet.")

        requests = [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_number - 1,
                        "endRowIndex": row_number,
                        "startColumnIndex": col_index,
                        "endColumnIndex": col_index + 1,
                    },
                    "cell": {"userEnteredFormat": {"horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
                    "fields": "userEnteredFormat.horizontalAlignment,userEnteredFormat.verticalAlignment",
                }
            }
        ]

        service.spreadsheets().batchUpdate(spreadsheetId=self.sheet_id, body={"requests": requests}).execute()

    def _write_to_cell(self, mouseID: int, date: str, attribute: str, value: str) -> None:
        """
        Writes a specific value to the Google Sheet based on mouse ID, date, and attribute. The
        correct row is located using the input date and the corresponding attribute in the headers
        headers list.
        """

        tab_name = str(mouseID)
        self._fetch_data_from_tab(tab_name)
        tab_data = self.data.get(tab_name, [])

        row_index = -1
        for i, row in enumerate(tab_data):
            if not row or len(row) <= 1:
                continue
            if row[1] == date:
                row_index = i
                break

        if row_index == -1:
            raise ValueError(f"No row found for date {date} in tab {tab_name}.")

        headers = tab_data[1]
        if attribute == "weight":
            value = float(value)
            col_index = headers.index("weight (g)")

        elif attribute == "given by":
            col_index = headers.index("given by:")

        elif attribute == "water given":
            col_index = headers.index("water given (mL)")

        elif attribute == "time":
            _convert_date_time(date, value)
            col_index = headers.index("time")

        else:
            raise ValueError(
                f"Invalid attribute: {attribute}. Only 'weight', 'given by:', 'water given (mL)', and 'time' can be updated."
            )

        tab_data[row_index][col_index] = value
        col_letter = chr(ord("A") + col_index)
        row_number = row_index + 1
        cell_range = f"{col_letter}{row_number}"

        self._write_to_sheet(tab_name, cell_range, [[value]])


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
# sheet_data = _WriteData()
# sheet_data._write_to_cell(mouseID=1, date="2/28/25", attribute="given by", value="ik278")
# sheet_data._write_to_cell(mouseID=1, date="2/28/25", attribute="water given", value="1")
# sheet_data._write_to_cell(mouseID=1, date="2/28/25", attribute="weight", value="28.0")
# sheet_data._write_to_cell(mouseID=1, date="2/28/25", attribute="time", value="16:00")

# Read Data
# water_log = ParseData(tab_name="MouseInfo", mouse_id=3, date="2/8/25")
# mouse_instance = water_log.mouse_instances.get(water_log.mouse_id)
# mouse_instance.to_yaml(file_path=Path("water_log.yaml"))
