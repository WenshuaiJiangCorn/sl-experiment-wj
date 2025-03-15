import re
from typing import Optional
from pathlib import Path
from datetime import (
    date as dt_date,
    time as dt_time,
    datetime,
    timezone,
)
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import console
from ataraxis_data_structures import YamlConfig
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# Stores schemas for supported date formats.
_supported_date_formats: set[str] = {"%m-%d-%y", "%m-%d-%Y", "%m/%d/%y", "%m/%d/%Y"}


def _convert_date_time(date: str, time: str) -> int:
    """Converts the input date and time strings into the UTC timestamp.

    This function is used to convert date and time strings parsed from the Google Sheet into the microseconds since
    UTC epoch onset, which is the primary time format used by all other library components.

    Args:
        date: The date string in the format "%m-%d-%y" or "%m-%d-%Y".
        time: The time string in the format "%H:%M".

    Returns:
        The number of microseconds elapsed since UTC epoch onset as an integer.

    Raises:
        ValueError: If date or time are not non-empty strings. If the date or time format does not match any of the
            supported formats.
    """

    # Ensures date and time are provided
    if not isinstance(date, str) or len(date) < 1:
        message = (
            f"Unable to convert the input date and time into a UTC timestamp when parsing Google Sheet data. Expected "
            f"non-empty string inputs for 'date' argument, but encountered {date} of type {type(date).__name__}."
        )
        console.error(message=message, error=ValueError)

    if not isinstance(time, str) or len(time) < 1:
        message = (
            f"Unable to convert the input date and time into a UTC timestamp when parsing Google Sheet data. Expected "
            f"non-empty string inputs for 'time' argument, but encountered {time} of type {type(time).__name__}."
        )
        console.error(message=message, error=ValueError)

    # Precreates date and time object placeholders.
    date_obj: dt_date = dt_date(1990, 1, 1)
    time_obj: dt_time = dt_time(0, 0)

    # Parses the time object
    try:
        time_obj = datetime.strptime(time, "%H:%M").time()
    except ValueError:
        message = (
            f"Invalid time format encountered when parsing Google Sheet data. Expected the supported time format "
            f"(%H:%M), but encountered {time}."
        )
        console.error(message=message, error=ValueError)

    # Parses the date object
    for date_format in _supported_date_formats:
        try:
            date_obj = datetime.strptime(date, date_format).date()
            break
        except ValueError:
            continue
    else:
        message = (
            f"Invalid date format encountered when parsing Google Sheet data. Expected one of the supported formats "
            f"({sorted(_supported_date_formats)}), but encountered {date}."
        )
        console.error(message=message, error=ValueError)

    # Constructs the full DT object and converts it into the UTC timestamp in microseconds.
    full_datetime = datetime.combine(date=date_obj, time=time_obj)
    full_datetime = full_datetime.replace(tzinfo=timezone.utc)

    # Gets and translates second timestamp (float) into microseconds (int). The returns it to caller
    return int(full_datetime.timestamp() * 1_000_000)


def extract_numerical(substring: str) -> Optional[float]:
    """
    Extracts a numerical value from a substring representing a coordinate.
    """
    match = re.search(r"([-+]?\d*\.?\d+)\s*(AP|ML|DV)", part)
    return float(match.group(1)) if match else None

def parse_coordinates(coordinate_string: str) -> tuple[float, float, float]:
    """
    Parses and stores the numerical part for each coordinate into the AP, ML and DV
    attributes of a Coordinates object.
    """
    coordinates = Coordinates()
    if coord_string:
        for part in coord_string.split(","):
            part = part.strip()
            if "AP" in part.upper():
                coordinates.ap = Coordinates.extract_numerical(part)
            elif "ML" in part.upper():
                coordinates.ml = Coordinates.extract_numerical(part)
            elif "DV" in part.upper():
                coordinates.dv = Coordinates.extract_numerical(part)
    return coordinates

@dataclass()
class SubjectData:
    """Stores the subject (mouse) ID information."""
    id: int
    """Stores the unique ID (name) of the subject. Assumes all mice are given a numeric ID, rather than a string name.
    """
    cage: int
    """Stores the number of the latest cage used to house the subject."""
    ear_punch: str
    """Stores the ear tag location of the subject."""
    sex: str
    """Stores the gender of the subject."""
    genotype: str
    """Stores the genotype of the subject."""
    date_of_birth_us: int
    """Stores the date of birth of the subject as the number of microseconds elapsed since UTC epoch onset."""
    weight_g: float
    """Stores the weight of the subject pre-surgery, in grams."""
    location_housed: str
    """Stores the latest location used to house the subject after the surgery."""
    status: str
    """Stores the latest status of the subject (alive / deceased)."""


@dataclass()
class ProcedureData:
    """Stores the general information about the surgical procedure."""
    surgery_start_us: int
    """Stores the date and time when the surgery was started as microseconds elapsed since UTC epoch onset."""
    surgery_end_us: int
    """Stores the date and time when the surgery has ended as microseconds elapsed since UTC epoch onset."""
    surgeon: str
    """Stores the name or ID of the surgeon. If the intervention was carrie out by multiple surgeon, all participating
    surgeon names and IDs are stored as part of the same string."""
    protocol: str
    """Stores the experiment protocol number (ID) used during the surgery."""


@dataclass
class Coordinates:
    """
    This class represents stereotaxic coordinates for implants or injections in a brain structure.
    The class also contains methods to extract numerical values from a coordinate strings and parse them
    into a Coordinates object.

    Args:
        coord_string: A string containing the stereotaxic coordinates in the format "AP: X, ML: Y, DV: Z".
                      If the cell containing the coordinates is empty or None, the method will return a
                      Coordinates object with all attributes set to None.

    Attributes:
        ap: Stores the anteroposterior distance in nm of an implant or injection within a given brain structure.
        ml: Stores the medial-lateral distance in nm of an implant or injection in a given brain structure.
        dv: Stores the dorsal-ventral distance in nm of an implant or injection in a given brain structure.
    """

    ap: float
    ml: float
    dv: float




@dataclass
class ImplantData:
    """
    This class handles all data related to implants performed during surgery. The class is designed
    to store data for only two implants as it is highly unlikely that more than two injections
    will be performed on the same mouse.

    Args:
        headers: A dictionary mapping column names (headers) to their respective indices in the row.
        row: A list of values representing a single row of data from the Google Sheet.

    Attributes:
        implant1, implant2: Stores whether an injection was performed
        implant1_region, implant2_region: Stores the structure of the brain where the injection was
                                              conducted
        implant1_coordinates, implant2_coordinates: Stores the stereotaxic coordinates of each implant. The
                                                    coordinates are pre-procssed in the Coordinates dataclass.
    """

    implant1: Optional[str] = None
    implant1_location: Optional[str] = None
    implant1_coordinates: Optional[Coordinates] = None
    implant2: Optional[str] = None
    implant2_location: Optional[str] = None
    implant2_coordinates: Optional[Coordinates] = None


@dataclass
class InjectionData:
    """
    This class handles all data related to injections performed during surgery. The class is designed
    to store data for only two injections as it is highly unlikely that more than two injections
    will be performed on the same mouse.

    Args:
        headers: A dictionary mapping column names (headers) to their respective indices in the row.
        row: A list of values representing a single row of data from the Google Sheet.

    Attributes:
        injection1, injection2: Stores whether an injection was performed
        injection1_region, injection2_region: Stores the structure of the brain where the injection was
                                              conducted
        injection1_coordinates, injection2_coordinates: Stores the stereotaxic coordinates of each injection.
                                                        The coordinates are pre-procssed in the Coordinates dataclass.
    """

    injection1: Optional[str] = None
    injection1_region: Optional[str] = None
    injection1_coordinates: Optional[Coordinates] = None
    injection2: Optional[str] = None
    injection2_region: Optional[str] = None
    injection2_coordinates: Optional[Coordinates] = None

    def __init__(self, headers: dict[str, int], row: list[Optional[str]]):
        injections = ("injection1", "injection2")

        for injection in injections:
            if f"{injection}" in headers:
                setattr(self, injection, row[headers[f"{injection}"]])

            if f"{injection}_region" in headers:
                setattr(self, f"{injection}_region", row[headers[f"{injection}_region"]])

            if f"{injection}_coordinates" in headers and row[headers[f"{injection}_coordinates"]]:
                coords_str = row[headers[f"{injection}_coordinates"]]
                if coords_str:
                    parsed_coords = Coordinates.parse_coordinates(coords_str)
                else:
                    parsed_coords = None
                setattr(self, f"{injection}_coordinates", parsed_coords)


@dataclass
class Drug:
    """
    This class contains all drug-related data for implant and injection procedures. It maps drug
    dosages (in mL) from the headers to the corresponding attributes of the class. All dosages
    are stored as floats.

    Args:
        headers: A dictionary mapping column names (headers) to their respective indices in the row.
        row: A list of values representing a single row of data from the Google Sheet.

    Attributes:
        lrs: Stores the amount of LRS administered in mL.
        ketoprofen: Stores the amount of ketoprofen administered in mL.
        buprenorphine: Stores the amount of buprenorphine administered in mL.
        dexomethazone: Stores the amount of dexomethazone administered in mL.
    """

    lrs: Optional[float] = None
    ketoprofen: Optional[float] = None
    buprenorphine: Optional[float] = None
    dexomethazone: Optional[float] = None

    def __init__(self, headers: dict[str, int], row: list[Optional[str]]):
        drug_mapping = {
            "lrs (ml)": "lrs",
            "ketoprofen (ml)": "ketoprofen",
            "buprenorphine (ml)": "buprenorphine",
            "dexomethazone (ml)": "dexomethazone",
        }

        for header_name, attr_name in drug_mapping.items():
            if header_name in headers:
                value = row[headers[header_name]]

                if value:
                    setattr(self, attr_name, float(value))
                else:
                    setattr(self, attr_name, None)


@dataclass
class BrainData:
    """
    This class handles all post-op brain-related data from an extracted Google Sheet row.
    It maps values from the "Brain Location" and "Brain status" headers and stores it in
    the brain_location and brain_status attributes.

    Args:
        headers: A dictionary mapping column names (headers) to their respective indices in the row.
        row: A list of values representing a single row of data from the Google Sheet.

    Attributes:
        brain_location: Stores the physical location of the brain post-op.
        brain_status: Stores the status of the brain post-op.
    """

    brain_location: str | None = None
    brain_status: str | None = None

    def __init__(self, headers: dict[str, int], row: list[Optional[str]]):
        self.brain_location = row[headers["brain location"]]
        self.brain_status = row[headers["brain status"]]


class SurgeryData:
    """
    This dataclass combines the processed hierarchies of data from the ProtocolData, ImplantData,
    InjectionData, Drug, and BrainData classes with additional information from the inter-
    and post-operative phases of the surgery.

    Args:
        headers: A dictionary mapping column names (headers) to their respective indices in the row.
        row: A list of values representing a single row of data from the Google Sheet.

    Attributes:
        protocol_data: Stores an instance of the ProtocolData class.
        implant_data: Stores an instance of the ImplantData class.
        injection_data: Stores an instance of the InjectionData class.
        drug_data: Stores an instance of the DrugData class.
        iso_o2_ratio: Stores the the isoflurane to oxygen ratio used during surgery.
        start: Stores the start time of the surgical procedure.
        end: Stores the end time of the surgical procedure.
        duration: Stores the duration of the surgical procedure.
        surgery_notes: Stores the observations of the mouse during surgery.
        post_op_notes: Stores the status and observations of the mouse post-surgery.
        sac_date: Stores the sacrifice date from non-surgical procedures such as perfusions.
        brain_data: Stores an instance of the BrainData class.
    """

    def __init__(self, headers: dict[str, int], row: list[Optional[str]], tab_name: str):
        self.tab_name: str = tab_name
        self.protocol_data = SubjectData(
            id=int(row[headers.get("id", -1)]),
            surgery_date_us=_convert_date_time(date=row[headers.get("date", -1)], time="00:00"),
            surgeon=row[headers.get("surgeon", -1)],
            protocol=row[headers.get("protocol", -1)],
            cage=int(row[headers.get("cage #", -1)]),
            ear_punch=row[headers.get("ear punch", -1)],
            sex=row[headers.get("sex", -1)],
            genotype=row[headers.get("genotype", -1)],
            date_of_birth_us=_convert_date_time(date=row[headers.get("dob", -1)], time="00:00"),
            weight_g=float(row[headers.get("weight (g)", -1)]),
            location_housed=row[headers.get("location housed", -1)],
            status=row[headers.get("status", -1)],
        )
        self.implant_data = ImplantData(headers=headers, row=row)
        self.injection_data = InjectionData(headers=headers, row=row)
        self.drug_data = Drug(headers=headers, row=row)
        self.iso_o2_ratio: str = row[headers["iso:o2 ratio"]]
        self.start: int = _convert_date_time(date=row[headers["date"]], time=row[headers["start"]])
        self.end: int = _convert_date_time(date=row[headers["date"]], time=row[headers["end"]])
        self.duration: np.uint64 = self.end - self.start
        self.surgery_notes: str = row[headers["surgery notes"]]
        self.post_op_notes: str = row[headers["post-op notes"]]
        self.sac_date: np.uint64 = _convert_date_time(date=row[headers["sac date"]], time=None)
        self.brain_data: BrainData = BrainData(headers=headers, row=row)


class _SurgerySheetData:
    """
    This class initializes key identifiers for the Google Sheet, including the spreadsheet URL,
    the cell range, and all tabs within the sheet. OAuth 2.0 scopes are used to link
    and grant access to Google APIs for data parsing.
    """

    def __init__(
        self,
        project_name: str,
        credentials_path: Path,
        sheet_id: str,
    ):
        self._project_name = project_name
        self._sheet_id = sheet_id

        # Generates the credentials' object to access the target Google Sheet. Since we are only reading the data from
        # the surgery log, we can use the 'readonly' access mode for added file safety.
        credentials = Credentials.from_service_account_file(
            filename=str(credentials_path), scopes=("https://www.googleapis.com/auth/spreadsheets.readonly",)
        )

        # Uses the credentials' object to build the access service for the target Google Sheet. This service is then
        # used to fetch the sheet data via HTTP request(s).
        self._service = build(serviceName="sheets", version="v4", credentials=credentials)

        # Retrieves all values stored in the first row of the target sheet tab. Each tab represents a particular
        # project. The first row contains the headers for all data columns stored in the sheet.
        headers = (
            self._service.spreadsheets()
            .values()
            .get(spreadsheetId=sheet_id, range=f"'{self._project_name}'!1:1")  # extracts the entire first row
            .execute()
        )

        # Converts headers to a list of strings and raises an error if the header list is empty
        header_values = headers.get("values", [[]])[0]
        if not header_values:
            message = (
                f"Unable to parse the surgery data for the project {project_name} Google Sheet. The first row of the "
                f"target tab appears to be empty. Instead, the first row should contain the column headers."
            )
            console.error(message, error=RuntimeError)

        # Creates a dictionary mapping header values to Google Sheet column letters
        self._headers: dict[str, str] = {}
        for i, header in enumerate(header_values):
            # Converts column index to column letter (0 -> A, 1 -> B, etc.)
            column_letter = self._convert_index_to_column_letter(i)
            self._headers[str(header)] = column_letter

        # Retrieves all animal names (IDs) from the 'ID' column. Each ID is z-filled to a triple-digit string for
        # sorting to behave predictably. This data is stored as a tuple of IDs.
        id_column = self._get_column_id("ID")
        animal_ids = (
            self._service.spreadsheets()
            .values()
            .get(
                spreadsheetId=sheet_id,
                range=f"{self._project_name}!{id_column}2:{id_column}",  # row 2 onward, row 1 stores headers
                majorDimension="COLUMNS"  # Gets data in column-major order
            )
            .execute()
        )
        id_list = animal_ids.get("values", [[]])[0]
        self._animals: tuple[str, ...] = tuple([str(animal_id).zfill(3) for animal_id in id_list])
        if len(self._animals) == 0:
            message = (
                f"Unable to parse the surgery data for the project {project_name} Google Sheet. The ID column of the "
                f"sheet contains no data, indicating that the log does not contain any animals."
            )
            console.error(message, error=RuntimeError)

    def extract_animal_data(self, animal_id: int) -> None:
        # Converts input ID to the same format as stored IDs for comparison
        formatted_id = str(animal_id).zfill(3)

        # Checks if the animal ID exists in the tuple of animal IDs generated at class initialization. If not, raises an
        # error
        if formatted_id not in self._animals:
            message = (
                f"Unable to extract the surgery data for the project {self._project_name} and animal {animal_id}. The "
                f"specified animal ID is not contained in the 'ID' column of the parsed Google Sheet."
            )
            console.error(message=message, error=ValueError)

        # Finds the index of the target animal in the ID value tuple to determine the row number to parse from the
        # sheet. The index is modified by 2 because: +1 for 0-indexing to 1-indexing, +1 to account for the header row
        animal_index = self._animals.index(formatted_id)
        row_number = animal_index + 2

        # Retrieves the entire row of data for the target animal
        row_data = (
            self._service.spreadsheets()
            .values()
            .get(
                spreadsheetId=self._sheet_id,
                range=f"'{self._project_name}'!{row_number}:{row_number}"
            )
            .execute()
        )

        # Converts the data from dictionary format into a list of strings.
        row_values = row_data.get("values")[0]

        # Replaces empty cells and value placeholders ('n/a'', '--' or '---') with None.
        row_values = self._replace_empty_values(row_values)

        # Creates a dictionary mapping headers (column names) to the animal-specific extracted values for these
        # headers.
        animal_data = {}
        for i, header in enumerate(self._headers):
            animal_data[header] = row_values[i] if i < len(row_values) else None

        print(animal_data)

    @staticmethod
    def _convert_index_to_column_letter(index):
        """Converts a 0-based column index to an Excel-style (Google Sheet) column letter (A, B, C, ... Z, AA, AB, ...).
        """
        result = ""
        while index >= 0:
            remainder = index % 26
            result = chr(65 + remainder) + result  # 65 is ASCII for 'A'
            index = index // 26 - 1
        return result

    def _get_column_id(self, column_name: str) -> str:
        """Returns the Google Sheet column ID (letter) for the given column name.

        This method assumes that the header name comes from the data extracted from the header row of the processed
        sheet. It does not contain any guards against retrieving a non-existent column.

        Args:
            column_name: The name of the column as it appears in the header row.

        Returns:
            The column ID (e.g., "A", "B", "C") corresponding to the column name.
        """
        return self._headers[column_name]

    @staticmethod
    def _replace_empty_values(row_data: list[str]) -> list[str | None]:
        """ Replaces empty cells and cells containing 'n/a', '--' or '---' inside the input row_data list with None.

        This internal method is used when retrieving animal data to filter out empty cells and values.

        Args:
            row_data: The list of cell values from a single Google Sheet row.
        """
        return [None if cell.strip().lower() in {"", "n/a", "--", "---"} else cell for cell in row_data]


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
        self.data: dict[str, list[list[Optional[str]]]] = {}

    def _get_service(self) -> build:
        """
        Authenticates the Google Sheets API using the service account credentials and the defined API scope.
        It then builds and returns the Google Sheets API service client to enable the script to interact with
        the sheet.
        """
        creds = Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)  # type: ignore
        return build("sheets", "v4", credentials=creds)

    def _get_tab_data(self, tab_name: str) -> list[list[str]]:
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

    def _replace_empty(self, row_data: list[list[str]]) -> list[list[Optional[str]]]:
        """
        Replaces empty cells and cells containing 'n/a', '--' or '---' with None. This funcation
        also ensures that cells in the main grid are processed and that all rows  have equal length.
        """
        result: list[list[Optional[str]]] = []

        for row in row_data:
            processed_row: list[Optional[str]] = []

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

    def _get_tab_names(self) -> list[str]:
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

    def _write_to_sheet(self, tab_name: str, range_name: str, values: list[list[str]]) -> None:
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

    def __init__(self, row: list[str], headers: list[str]):
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
    daily_log: dict[str, DailyLog] = field(default_factory=dict)

    def __init__(self, row: list[str], headers: list[str]):
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
        self.mouse_classes: dict[str, MouseKey] = {}
        self.mouse_instances: dict[str, MouseKey] = {}
        self._create_mouse_subclasses()
        self._link_to_tab()

    def _extract_mouse_id(self, tab_data: list[list[str]], mouse_col_index: int) -> list[int]:
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

sheet_id = "1aEdF4gaiQqltOcTABQxN7mf1m44NGA-BTFwZsZdnRX8"
project_name = "Practice mice"
credentials = Path("/home/cyberaxolotl/Downloads/sl-surgery-log-0f651e492767.json")
data = _SurgerySheetData(project_name=project_name, credentials_path=credentials, sheet_id=sheet_id)
data.extract_animal_data(animal_id=2)
