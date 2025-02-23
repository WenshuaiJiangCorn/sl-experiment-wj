from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from datetime import datetime
from ataraxis_base_utilities import console
from ataraxis_data_structures import YamlConfig
from dataclasses import is_dataclass
import re 


@dataclass
class AttributeData:
    values: List[Optional[str]]
    

@dataclass
class ProtocolData:
    id: str
    date: str
    surgeon: str
    protocol: str
    cage: int
    ear_punch: str 
    sex: str
    genotype: str
    dob: str
    weight: float
    location_housed: str 
    status: str

    def __repr__(self):
        """
        Returns the string representation of an instance of the ProtocolData class.
        """
        return (
            f"ProtocolData(ID:{self.id}, Date:{self.date}, Surgeon:{self.surgeon}, Protocol:{self.protocol}, "
            f"Cage #:{self.cage}, Ear punch:{self.ear_punch}, Sex:{self.sex}, Genotype:{self.genotype}, DOB:{self.dob}, "
            f"Weight (g):{self.weight}, Location housed:{self.location_housed}, Status:{self.status})"
        )


@dataclass
class Coordinates:
    AP: Optional[float] = None
    ML: Optional[float] = None
    DV: Optional[float] = None

    def __repr__(self):
        return f"Coordinates(AP={self.AP}, ML={self.ML}, DV={self.DV})"

    @staticmethod
    def extract_numerical(part: str) -> Optional[float]:
        """
        Extracts a numerical value from a string like "1.8 AP" or "2 ML".
        """
        pattern = r"([-+]?\d*\.?\d+)\s*(AP|ML|DV)"
        match = re.search(pattern, part)

        if match:
            numeric_value = match.group(1)  
            return float(numeric_value)  
        else:
            return None
        

    @staticmethod
    def parse_coordinates(coord_string: Optional[str]) -> 'Coordinates':
        """
        Parses a coordinate string like "1.8 AP, 2 ML, 0 DV" into a Coordinates object.
        """
        coordinates = Coordinates()

        if coord_string:
            coord_substring = [part.strip() for part in coord_string.split(",")]

            for part in coord_substring:
                if "AP" in part.upper():
                    coordinates.AP = Coordinates.extract_numerical(part)
                elif "ML" in part.upper():
                    coordinates.ML = Coordinates.extract_numerical(part)
                elif "DV" in part.upper():
                    coordinates.DV = Coordinates.extract_numerical(part)

        return coordinates
    

@dataclass
class ImplantData:
    implant1: Optional[str] = None
    implant1_location: Optional[str] = None
    implant1_coordinates: Optional[Coordinates] = None
    implant2: Optional[str] = None
    implant2_location: Optional[str] = None
    implant2_coordinates: Optional[Coordinates] = None

    def __init__(self, headers: Dict[str, int], row: tuple[Optional[str], ...]):
    
        implants = ("implant1", "implant2")

        for implant in implants:

            if f"{implant}" in headers:
                setattr(self, implant, row[headers[f"{implant}"]])

                if f"{implant}_location" in headers:
                    setattr(self, f"{implant}_location", row[headers[f"{implant}_location"]])

                if f"{implant}_coordinates" in headers and row[headers[f"{implant}_coordinates"]]:
                    coords_str = row[headers[f"{implant}_coordinates"]]
                    setattr(
                        self, f"{implant}_coordinates", Coordinates.parse_coordinates(coords_str))
                    

    def __repr__(self):
        return (
            f"Implant1: {self.implant1}, Implant1_location: {self.implant1_location}, "
            f"Implant1_coordinates: {self.implant1_coordinates}, "
            f"Implant2: {self.implant2}, Implant2_location: {self.implant2_location}, "
            f"Implant2_coordinates: {self.implant2_coordinates}"
        )
    

@dataclass
class InjectionData:
    injection1: Optional[str] = None
    injection1_region: Optional[str] = None
    injection1_coordinates: Optional[Coordinates] = None
    injection2: Optional[str] = None
    injection2_region: Optional[str] = None
    injection2_coordinates: Optional[Coordinates] = None

    def __init__(self, headers: Dict[str, int], row: tuple[Optional[str], ...]):
    
        injections = ("injection1", "injection2")

        for injection in injections:
    
            if f"{injection}" in headers:
                setattr(self, injection, row[headers[f"{injection}"]])

                if f"{injection}_region" in headers:
                    setattr(self, f"{injection}_region", row[headers[f"{injection}_region"]])

                if f"{injection}_coordinates" in headers and row[headers[f"{injection}_coordinates"]]:
                    coords_str = row[headers[f"{injection}_coordinates"]]
                    setattr(
                        self, f"{injection}_coordinates", Coordinates.parse_coordinates(coords_str))
                    

    def __repr__(self):
        return (
            f"injection1: {self.injection1}, injection1_location: {self.injection1_region}, "
            f"injection1_coordinates: {self.injection1_coordinates}, "
            f"injection2: {self.injection2}, injection2_location: {self.injection2_region}, "
            f"injection2_coordinates: {self.injection2_coordinates}"
        )
    

@dataclass
class BrainData:
    brain_location: str
    brain_status: str

    def __init__(self, headers: dict, row_values: list):
        self.brain_location = row_values[headers["brain_location"]]
        self.brain_status = row_values[headers["brain_status"]]


    def __repr__(self):
        """
        Returns the brain location and brain status of the mouse post-surgery. 
        """
        return (
            f"BrainData(Brain location:{self.brain_location}, Brain status:{self.brain_status})"
        )


@dataclass
class Drug:
    lrs: Optional[float] = None
    ketoprofen: Optional[float] = None
    buprenorphin: Optional[float] = None
    dexomethazone: Optional[float] = None

    def __init__(self, headers: Dict[str, int], row: tuple[Optional[str], ...]):
        """
        Initializes the attributes of the Drug class for columns containing data for LRS,
        ketoprofen, buprenorphin, and dexomethazone dosages.
        """
        drug_list = ("lrs", "ketoprofen", "buprenorphin", "dexomethazone")

        for header, i in headers.items():
            updated_header = header.strip().lower().replace(" (ml)", "").replace("_(ml)", "")

            if updated_header in drug_list:
                value = row[i].strip() if row[i] else ""
                setattr(self, updated_header, float(value) if value else None)


    def __repr__(self):
        """
        Returns the representation string of an instance of the Drug class.
        """
        drug_fields = {key: getattr(self, key) for key in ["lrs", "ketoprofen", "buprenorphin", "dexomethazone"]}
        return f"Drug({drug_fields})"
    

@dataclass
class SurgeryData:
    _protocol_data: ProtocolData
    _implant_data: ImplantData
    _injection_data: InjectionData
    _drug_data: Drug
    _brain_data: BrainData
    _start: str
    _end: str
    _duration: str 
    _surgery_notes: str
    _post_op_notes: str
    
    def __repr__(self):
        """
        Returns the combined representation string the ProtocolData, ImplantData, InjectionData classes and surgery notes. 
        """
        return (
            f"({self._protocol_data}, {self._implant_data}, {self._injection_data}, "
            f"{self._drug_data}, {self._start}, {self._end}, {self._duration}, "
            f"{self._surgery_notes}, {self._post_op_notes}, {self._brain_data})"
        )


class ParseData:
    def __init__(self):
        self._sheet_data = self._SheetData()
        self._sheet_data._parse()


    class _SheetData:
        def __init__(self):
            """
            Initializes key identifiers of the Google Sheets such as the spreadsheet URL,  
            the range of the sheet, OAuth 2.0 scopes to request access Google APIs
            and parse the data.
            """
            self.sheet_id = '1fOM2SenU7Dcz6Y1fw_cd7g4eJRuxXdjgZUofOuMNo7k'  # Replace based on sheet 
            self.range = 'A1:ZZ'
            self.SERVICE_ACCOUNT_FILE = '/Users/natalieyeung/Documents/GitHub/sl-mesoscope/mesoscope_data.json'   # Replace based on sheet 
            self.SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
            self.data: List[AttributeData] = []
            self.headers: Dict[str, int] = {}


        def _get_sheet_data(self) -> List[List[str]]:
            """
            Parses data from the connected Google Sheets based on the specified range. 
            """
            creds = Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)
            service = build('sheets', 'v4', credentials=creds)
            result = service.spreadsheets().values().get(spreadsheetId=self.sheet_id, range=self.range).execute()
            return result.get('values', [])
        

        def _replace_empty(self, row_data: List[List[str]]) -> List[List[Optional[str]]]:
            """
            Replaces empty cells and cells containing 'n/a' or '--' with None.
            """
            max_columns = len(row_data[0])
            
            result = []
            for row in row_data:

                processed_row = []

                for val in row + [None] * (max_columns - len(row)):
                    if val in {'', 'n/a', 'N/A', '--', '---'}:
                        processed_row.append(None)
                    elif val and val.strip():
                        processed_row.append(val.strip())
                    else:
                        processed_row.append(None)
                result.append(processed_row)

            # print(result)
            return result
            

        def _parse(self):
            """
            Extracts headers of the processed data from its entries. This method also assigns
            indices to each header name.
            """
            raw_data = self._get_sheet_data()
            replaced_data = self._replace_empty(raw_data)
            if not replaced_data:
                return

            first_row = replaced_data[0]
            self.headers = {}

            for i, column in enumerate(first_row):
                normalized_column = column.lower().replace(" ", "_").replace(":", "_").replace("-", "_")
                self.headers[normalized_column] = i
            
            # print("Parsed Headers:", self.headers)  
            self.data = [AttributeData(values=row) for row in replaced_data[1:]]


    def _get_mice(self, ID: str, date: str, protocol: str, genotype: str) -> List[SurgeryData]:
        if not self._sheet_data.data:
            return []

        search_key = {key: value for key, value in zip(
            ["id", "date", "protocol", "genotype"], 
            [ID.lower(), date.lower(), protocol.lower(), genotype.lower()]
        )}

        column_mapping = {
            "cage_#": "cage",
            "weight_(g)": "weight"
        }

        return [
            SurgeryData(
                ProtocolData(**{
                    key: row.values[self._sheet_data.headers[key]].strip().lower()
                    for key in ["id", "date", "protocol", "genotype"]
                }, **{
                    column_mapping.get(key, key): row.values[self._sheet_data.headers[key]]
                    for key in ["surgeon", "sex", "dob", "ear_punch", "cage_#", "weight_(g)", "status", "location_housed"]
                }),
                ImplantData(self._sheet_data.headers, row.values),
                InjectionData(self._sheet_data.headers, row.values),
                Drug(self._sheet_data.headers, row.values),
                BrainData(self._sheet_data.headers, row.values),
                *[
                    f"{key}: {row.values[self._sheet_data.headers[key]]}"
                    for key in ["start", "end", "duration", "surgery_notes", "post_op_notes"]
                ]
            )
            for row in self._sheet_data.data
            if all(row.values[self._sheet_data.headers[key]].strip().lower() == search_key[key] for key in search_key)
        ]


# MAIN 
mice_data = ParseData()
results = mice_data._get_mice(ID='2', genotype="WT", date="1-24-25", protocol="2024-0019")
for result in results:
    print(result)
    # print(is_dataclass(result) and isinstance(result, SurgeryData)) 
    # print(f"Protocol: {result._protocol_data.genotype}")
    # print(f"Injection 1 region: {result._injection_data.injection1_region}")
    # print(f"Injection 1 coords: {result._injection_data.injection1_coordinates.AP}")
    # print(f"ketoprofen: {result._drug_data.ketoprofen}")
    # print(result._post_op_notes)
    # print(result._start)
  

