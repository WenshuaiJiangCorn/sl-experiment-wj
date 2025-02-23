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
    def __init__(self, ID: str, date: str, protocol: str, genotype: str, sex: str, dob: str, cage: int, weight: float, ear_punch: str):
        self._id = ID
        self._date = date
        self._protocol = protocol
        self._sex = sex
        self._genotype = genotype
        self._dob = dob
        self._cage = cage
        self._weight = weight
        self._ear_punch = ear_punch


    def __post_init__(self):
        """
        Parses the date of surgery into the form ("%-m-%-d-%y").
        """
        if self.date:
            try:
                parsed_date = datetime.strptime(self.date, "%Y-%m-%d")
                self.date = parsed_date.strftime("%-m-%-d-%y")

            except ValueError:
                pass


    def __repr__(self):
        """
        Returns the representation string of an instance of the ProtocolData class. 
        """
        return f"ProtocolData(ID:{self._id}, Date:{self._date}, Protocol:{self._protocol}, Cage #:{self._cage}, Genotype:{self._genotype}, Sex:{self._sex}, DOB:{self._dob}, Weight:{self._weight}, Ear punch:{self._ear_punch})"


@dataclass
class ImplantData:

    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
        """
        Initializes the attributes of the InjectionData class for any column headers containing
        the string "injection". Coordinates are parsed if the header contains "coordinates".
        """
        self._implant = [] 

        for header, i in headers.items():
            if "implant" in header.lower():
                value = row[i]

                if "coordinates" in header.lower():
                    value = Coordinates.parse_coordinates(row[i]) 

                attribute_name = f"_{header.lower()}"
                setattr(self, attribute_name, value)
                self._implant.append((header.lower(), value))
                

    def __repr__(self):
        return f"ImplantData({self._implant})"


    def get_injection_data(self) -> List[tuple[str, str]]:
        return self._implant


@dataclass
class InjectionData:

    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
        """
        Initializes the attributes of the InjectionData class for any column headers containing
        the string "injection". 
        """
        self._injection = [] 

        for header, i in headers.items():
            if "injection" in header.lower():
                value = row[i]

                if "coordinates" in header.lower():
                    value = Coordinates.parse_coordinates(row[i]) 

                attribute_name = f"_{header.lower()}"
                setattr(self, attribute_name, value)
                self._injection.append((header.lower(), value))
                

    def __repr__(self):
        return f"InjectionData{self._injection}"


    def get_injection_data(self) -> List[tuple[str, str]]:
        return self._injection


@dataclass
class Coordinates:

    _AP: Optional[float] = None
    _ML: Optional[float] = None
    _DV: Optional[float] = None

    def __repr__(self):
        return f"Coordinates(AP={self._AP}, ML={self._ML}, DV={self._DV})"
    
    
    @staticmethod
    def extract_numerical(part: str) -> Optional[float]:
        pattern = r"([-+]?\d*\.?\d+)\s*(AP|ML|DV)"
        match = re.search(pattern, part)

        if match:
            numeric_value = match.group(1)  
            return float(numeric_value)  
        else:
            return None
        
        
    @staticmethod
    def parse_coordinates(coord_string: Optional[str]) -> 'Coordinates':
        coordinates = Coordinates()
        
        if coord_string:
            coord_substring = [part.strip() for part in coord_string.split(",")]

            for part in coord_substring:
                if "AP" in part.upper():
                    coordinates._AP = Coordinates.extract_numerical(part)
                elif "ML" in part.upper():
                    coordinates._ML = Coordinates.extract_numerical(part)
                elif "DV" in part.upper():
                    coordinates._DV = Coordinates.extract_numerical(part)

        return coordinates
    

@dataclass
class Drug:
    # Should NOT be OPTIONAL 
    LRS: Optional[float] = None
    ketoprofen: Optional[float] = None
    buprenorphin: Optional[float] = None
    dexomethazone: Optional[float] = None


    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
        """
        Initializes the attributes of the Drug dataclass for columns containing data for the LRS,
        ketoprofen, buprenorphin and dexomethazone dosages.
        """
        drug_list = ["lrs", "ketoprofen", "buprenorphin", "dexomethazone"]

        for header, i in headers.items():
            updated_header = header.strip().lower().replace(" (ml)", "")
            if any(drug in updated_header for drug in drug_list):
                setattr(self, f"_{updated_header}", row[i])


    def __repr__(self):
        """
        Returns the representation string of an instance of the Drug class. The drug fields are 
        initialized as a dictionary. 
        """
        drug_fields = {}

        for key, value in self.__dict__.items():
            if any(drug in key for drug in ["lrs", "ketoprofen", "buprenorphin", "dexomethazone"]):
                drug_fields[f"{key.lstrip('_')}"] = value
                
        return f"Drug({drug_fields})"
    

@dataclass
class SurgeryData:
    
    _protocol_data: ProtocolData
    _implant_data: ImplantData
    _injection_data: InjectionData
    _drug_data: Drug
    _surgery_notes: str

    def __repr__(self):
        """
        Returns the combined representation string the ProtocolData, ImplantData, InjectionData classes and surgery notes. 
        """
        return f"({self._protocol_data}, {self._implant_data}, {self._injection_data}, {self._drug_data}, {self._surgery_notes})"
    

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
                normalized_column = column.lower().replace(" ", "_").replace(":", "_")
                self.headers[normalized_column] = i
            
            # print("Parsed Headers:", self.headers)  
            self.data = [AttributeData(values=row) for row in replaced_data[1:]]



    def _get_mice(self, ID: str, date: str, protocol: str, genotype: str) -> List[SurgeryData]:
        if not self._sheet_data.data:
            return []

        search_key = {key: value for key, value in zip(
            ["id", "date", "protocol", "genotype"], [ID.lower(), date.lower(), protocol.lower(), genotype.lower()]
        )}

        return [
            SurgeryData(
                ProtocolData(
                    ID=row.values[self._sheet_data.headers["id"]].strip().lower(),
                    date=row.values[self._sheet_data.headers["date"]].strip().lower(),
                    protocol=row.values[self._sheet_data.headers["protocol"]].strip().lower(),
                    genotype=row.values[self._sheet_data.headers["genotype"]].strip().lower(),
                    sex=row.values[self._sheet_data.headers["sex"]],
                    dob=row.values[self._sheet_data.headers["dob"]],
                    ear_punch=row.values[self._sheet_data.headers.get("ear_punch")],
                    cage=row.values[self._sheet_data.headers["cage_#"]],
                    weight=row.values[self._sheet_data.headers["weight_(g)"]]
                ),
                ImplantData(self._sheet_data.headers, row.values),
                InjectionData(self._sheet_data.headers, row.values),
                Drug(self._sheet_data.headers, row.values),
                row.values[self._sheet_data.headers["surgery_notes"]]
            )
            for row in self._sheet_data.data
            if all(row.values[self._sheet_data.headers[key]].strip().lower() == search_key[key] for key in search_key)
        ]


# MAIN 
mice_data = ParseData()
results = mice_data._get_mice(ID='2', genotype="WT", date="1-24-25", protocol="2024-0019")
for result in results:
    print(result)
    print(is_dataclass(result) and isinstance(result, SurgeryData)) 
    # print(f"Cage: {result._protocol_data._cage}")
    # print(f"Protocol: {result._protocol_data._genotype}")
    # print(f"Injection 1 region: {result._injection_data._injection1_region}")
    # print(f"Implant1: {result._implant_data._implant1}")
    # print(f"AP: {result._injection_data._injection1_coordinates._AP}")
    # print(result._surgery_notes)
  

