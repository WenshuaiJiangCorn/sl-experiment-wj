import re
from typing import Dict, List, Optional
from pathlib import Path
from datetime import (
    time as dt_time,
    datetime,
    timezone,
)
from dataclasses import field, dataclass

import numpy as np
from ataraxis_data_structures import YamlConfig
from googleapiclient.discovery import build  # type: ignore
from google.oauth2.service_account import Credentials


def _convert_date_time(date: Optional[str], time: Optional[str]) -> int:
    """
    Converts a date and time string to a UTC timestamp.

    Args:
        date: The date string in the format "%m-%d-%y".
        time: The time string in the format "%H:%M". If None, it is set to a default of "00:00"
    """
    if not date:
        return 0

    date_obj = datetime.strptime(date, "%m-%d-%y").date()

    if time:
        time_obj = datetime.strptime(time, "%H:%M").time()
    else:
        time_obj = dt_time(0, 0)

    full_datetime = datetime.combine(date_obj, time_obj)
    full_datetime = full_datetime.replace(tzinfo=timezone.utc)
    utc_timestamp = int(full_datetime.timestamp())

    return utc_timestamp


@dataclass
class ProtocolData:
    """
    This class stores basic information about the mouse. It maps data from each row in the Google Sheet
    to the corresponding attributes using the provided headers.

    Args:
        headers: A dictionary mapping column names (headers) to their respective indices in the row.
        row: A list of values representing a single row of data from the Google Sheet.

    Attributes:
        id: Stores the unique ID of the mouse.
        date: Stores the date of the surgery.
        surgeon: Stores the operating surgeon(s).
        protocol: Stores the protocol ID of the surgery.
        cage: Stores the cage number which the mouse is placed post-surgery.
        ear_punch: Stores whether the mouse has an ear punch.
        sex: Stores the gender of the mouse.
        genotype: Stores the genotype of the mouse.
        dob: Stores the date of birth of the mouse
        weight: Stores the weight of the mouse pre-surgery.
        location_housed: Stores the location where the mouse's cage is kept.
        status: Stores the current status of the mouse, indicating whether it is alive or deceased.
    """

    id: int = 0
    date: np.uint64 = np.uint64(0)
    surgeon: str | None = None
    protocol: str | None = None
    cage: int = 0
    ear_punch: str = ""
    sex: str | None = None
    genotype: str | None = None
    dob: np.uint64 = np.uint64(0)
    weight: float = 0.0
    location_housed: str | None = None
    status: str | None = None

    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
        self.id = int(row[headers.get("id", -1)])
        self.date = _convert_date_time(date=row[headers.get("date", -1)], time=None)
        self.surgeon = row[headers.get("surgeon", -1)]
        self.protocol = row[headers.get("protocol", -1)]
        self.cage = int(row[headers.get("cage #", -1)])
        self.ear_punch = row[headers.get("ear punch", -1)]
        self.sex = row[headers.get("sex", -1)]
        self.genotype = row[headers.get("genotype", -1)]
        self.dob = _convert_date_time(date=row[headers.get("dob", -1)], time=None)
        self.weight = float(row[headers.get("weight (g)", -1)])
        self.location_housed = row[headers.get("location housed", -1)]
        self.status = row[headers.get("status", -1)]


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
        AP: Stores the anteroposterior distance in nm of an implant or injection within a given brain structure.
        ML: Stores the medial-lateral distance in nm of an implant or injection in a given brain structure.
        DV: Stores the dorsal-ventral distance in nm of an implant or injection in a given brain structure.
    """

    AP: Optional[float] = None
    ML: Optional[float] = None
    DV: Optional[float] = None

    @staticmethod
    def extract_numerical(part: str) -> Optional[float]:
        """
        Extracts a numerical value from a substring representing a coordinate.
        """
        match = re.search(r"([-+]?\d*\.?\d+)\s*(AP|ML|DV)", part)
        return float(match.group(1)) if match else None

    @staticmethod
    def parse_coordinates(coord_string: Optional[str]) -> "Coordinates":
        """
        Parses and stores the numerical part for each coordinate into the AP, ML and DV
        attributes of a Coordinates object.
        """
        coordinates = Coordinates()
        if coord_string:
            for part in coord_string.split(","):
                part = part.strip()
                if "AP" in part.upper():
                    coordinates.AP = Coordinates.extract_numerical(part)
                elif "ML" in part.upper():
                    coordinates.ML = Coordinates.extract_numerical(part)
                elif "DV" in part.upper():
                    coordinates.DV = Coordinates.extract_numerical(part)
        return coordinates


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

    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
        implants = ("implant1", "implant2")

        for implant in implants:
            if implant in headers:
                setattr(self, implant, row[headers[implant]])

            if f"{implant}_location" in headers:
                setattr(self, f"{implant}_location", row[headers[f"{implant}_location"]])

            implant_coords_key = f"{implant}_coordinates"
            if implant_coords_key in headers and row[headers[implant_coords_key]]:
                coords_str = row[headers[implant_coords_key]]
                if coords_str:
                    parsed_coords = Coordinates.parse_coordinates(coords_str)
                else:
                    parsed_coords = None
                setattr(self, implant_coords_key, parsed_coords)


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

    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
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

    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
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

    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
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

    def __init__(self, headers: Dict[str, int], row: List[Optional[str]], tab_name: str):
        self.tab_name: str = tab_name
        self.protocol_data = ProtocolData(headers=headers, row=row)
        self.implant_data = ImplantData(headers=headers, row=row)
        self.injection_data = InjectionData(headers=headers, row=row)
        self.drug_data = Drug(headers=headers, row=row)
        self.iso_o2_ratio: str = row[headers["iso:o2 ratio"]]
        self.start: np.uint64 = _convert_date_time(date=row[headers["date"]], time=row[headers["start"]])
        self.end: np.uint64 = _convert_date_time(date=row[headers["date"]], time=row[headers["end"]])
        self.duration: np.uint64 = self.end - self.start
        self.surgery_notes: str = row[headers["surgery notes"]]
        self.post_op_notes: str = row[headers["post-op notes"]]
        self.sac_date: np.uint64 = _convert_date_time(date=row[headers["sac date"]], time=None)
        self.brain_data: BrainData = BrainData(headers=headers, row=row)


class _SheetData:
    """
    This class initializes key identifiers for the Google Sheet, including the spreadsheet URL,
    the cell range, and all tabs within the sheet. OAuth 2.0 scopes are used to link
    and grant access to Google APIs for data parsing.
    """

    def __init__(self, tab_name: str):
        self.sheet_id = "1fOM2SenU7Dcz6Y1fw_cd7g4eJRuxXdjgZUofOuMNo7k"  # Replace with actual sheet ID
        self.range = "A1:ZZ"
        self.SERVICE_ACCOUNT_FILE = (
            "/Users/natalieyeung/Documents/GitHub/sl-mesoscope/mesoscope_data.json"  # Replace with actual credentials
        )
        self.SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        self.data: List[List[Optional[str]]] = []
        self.headers: Dict[str, int] = {}
        self.tab_name = tab_name

    def _get_sheet_data(self) -> List[List[str]]:
        """
        Retrieves non-empty rows from the specified tab in the Google Sheet. This method ensures
        that only populated rows are processed to handle variations in row counts across different tabs.
        """

        creds = Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)  # type: ignore
        service = build("sheets", "v4", credentials=creds)

        range_name = f"{self.tab_name}!{self.range}"
        result = service.spreadsheets().values().get(spreadsheetId=self.sheet_id, range=range_name).execute()

        tab_data = result.get("values", [])
        return [row for row in tab_data if row]

    def _replace_empty(self, row_data: List[List[str]]) -> List[List[Optional[str]]]:
        """
        Replaces empty cells and cells containing 'n/a', '--' or '---' with None.
        """
        result: List[List[Optional[str]]] = []

        for row in row_data:
            processed_row: List[Optional[str]] = []

            for cell in row:
                if cell.strip().lower() in {"n/a", "--", "---"}:
                    processed_row.append(None)
                else:
                    processed_row.append(cell)

            result.append(processed_row)

        return result

    def _parse(self) -> None:
        """
        Processes the raw data fetched from the Google Sheet to extract and modify the headers
        to remove any invalid characters and whitespaces.
        """
        raw_data = self._get_sheet_data()
        replaced_data = self._replace_empty(raw_data)
        first_row = replaced_data[0]

        self.headers = {}

        for i, column in enumerate(first_row):
            column_str = str(column).lower()
            self.headers[column_str] = i

        self.data = replaced_data[1:]

    def _return_all(self) -> List[SurgeryData]:
        """
        Parses each row of sheet data into a SurgeryData instance and returns a list of these instances.
        """
        surgeries = []
        for row in self.data:
            surgery_data = SurgeryData(headers=self.headers, row=row, tab_name=self.tab_name)
            surgeries.append(surgery_data)
        return surgeries


@dataclass
class FilteredSurgeries(YamlConfig):
    """
    A wrapper class to store filtered surgeries for serialization using the to_yaml method.

    Attributes:
        surgeries (List[SurgeryData]): A list of SurgeryData objects representing filtered surgeries.
    """

    surgeries: List[SurgeryData] = field(default_factory=list)


def extract_mouse(tab_name: str, mouse_id: int) -> FilteredSurgeries:
    """
    Fetches data from the specified tab in the Google Sheet and filters it based on the mouse ID provided.
    """
    sheet_data = _SheetData(tab_name)
    sheet_data._parse()
    surgeries = sheet_data._return_all()

    filtered_data = []
    for surgery in surgeries:
        if surgery.protocol_data.id == mouse_id and surgery.tab_name == tab_name:
            filtered_data.append(surgery)

    return FilteredSurgeries(surgeries=filtered_data)


# Main
filtered_surgeries = extract_mouse(tab_name="Sheet1", mouse_id=2)
filtered_surgeries.to_yaml(file_path=Path("mouse_data.yaml"))
