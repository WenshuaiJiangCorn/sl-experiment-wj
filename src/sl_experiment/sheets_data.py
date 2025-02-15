from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from datetime import datetime
from ataraxis_base_utilities import console


@dataclass
class AttributeData:
    values: List[Optional[str]]


@dataclass
class ProtocolData:
    def __init__(self, ID: str, date: str, surgeon: str, protocol: str):
        self._ID = ID
        self._date = date
        self._surgeon = surgeon
        self._protocol = protocol

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
        return f"ProtocolData(ID={self._ID}, surgeon={self._surgeon}, date={self._date}, protocol={self._protocol})"


@dataclass
class ImplantData:
    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
        """
        Initializes the attributes of the ImplantData class for any column headers containing the string "implant".
        """
        for header, i in headers.items():
            if "implant" in header.lower():
                setattr(self, f"_{header}", row[i])

    def __repr__(self):
        """
        Returns the representation string of an instance of the ImplantData class. The implant fields are
        initialized as a dictionary.
        """
        implant_fields = {}

        for key, value in self.__dict__.items():
            if "implant" in key.lower():
                implant_fields[key] = value

        return f"ImplantData({implant_fields})"


@dataclass
class InjectionData:
    def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):
        """
        Initializes the attributes of the InjectionData class for any column headers containing the string "implant".
        """
        for header, i in headers.items():
            if "injection" in header.lower():
                setattr(self, f"_{header}", row[i])

    def __repr__(self):
        """
        Returns the representation string of an instance of the InjectionData class. The implant fields are
        initialized as a dictionary.
        """
        injection_fields = {}

        for key, value in self.__dict__.items():
            if "injection" in key.lower():
                injection_fields[key] = value

        return f"InjectionData({injection_fields})"


# @dataclass
# class Drug:

#     LRS: Optional[float] = None
#     ketoprofen: Optional[float] = None
#     buprenorphin: Optional[float] = None
#     dexomethazone: Optional[float] = None

#     def __init__(self, headers: Dict[str, int], row: List[Optional[str]]):

#         drug_list = ["lrs", "ketoprofen", "buprenorphin", "dexomethazone"]


@dataclass
class IndividualMouseData:
    _protocol_data: ProtocolData
    _implant_data: ImplantData
    _injection_data: InjectionData

    def __getattr__(self, name: str) -> Optional[str]:
        """
        Checks if an attribute from ProtocolData, ImplantData and InjectionData classes
        and obtains the value if it exists.
        """
        if hasattr(self._protocol_data, name):
            return getattr(self._protocol_data, name)

        if hasattr(self._implant_data, name):
            return getattr(self._implant_data, name)

        if hasattr(self._injection_data, name):
            return getattr(self._injection_data, name)

        message = f"{self.__class__.__name__}' object has no attribute '{name}'"
        console.error(message=message, error=AttributeError)
        raise AttributeError("Unknown attribute")

    def __repr__(self):
        """
        Returns the combined representation string the ProtocolData and ImplatnData classes.
        """
        return f"({self._protocol_data}, {self._implant_data}, {self._injection_data})"


class MiceData:
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
            self.sheet_id = "1fOM2SenU7Dcz6Y1fw_cd7g4eJRuxXdjgZUofOuMNo7k"  # Replace based on sheet
            self.range = "A1:Z"
            self.SERVICE_ACCOUNT_FILE = (
                "/Users/natalieyeung/Documents/GitHub/sl-mesoscope/mesoscope_data.json"  # Replace based on sheet
            )
            self.range = "A1:Z"
            self.SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
            self.data: List[AttributeData] = []
            self.headers: Dict[str, int] = {}

        def _get_sheet_data(self) -> List[List[str]]:
            """
            Parses data from the connected Google Sheets based on the specified range.
            """
            creds = Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)
            service = build("sheets", "v4", credentials=creds)
            result = service.spreadsheets().values().get(spreadsheetId=self.sheet_id, range=self.range).execute()
            return result.get("values", [])

        def _replace_empty(self, row_data: List[List[str]]) -> List[List[Optional[str]]]:
            """
            Replaces empty cells and cells containing 'n/a' or '--' with None.
            """
            max_columns = len(row_data[0])

            result = []
            for row in row_data:
                processed_row = []

                for val in row + [None] * (max_columns - len(row)):
                    if val in {"", "n/a", "N/A", "--"}:
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
                col_lower = column.lower()
                self.headers[col_lower] = i
            # print("Parsed Headers:", self.headers)
            self.data = [AttributeData(values=row) for row in replaced_data[1:]]

    def _get_mice(self, ID: str, surgeon: str, date: str, protocol: str) -> List[IndividualMouseData]:
        """
        Returns a list of DataObject instances containing ProtocolData and ImplantData.

        The data for individual mice is first queried using attributes such as ID,
        Surgeon, Date, and Protocol. This allows further querying of specific attributes of
        the mouse.
        """
        if not self._sheet_data.data:
            return []

        results = []

        for row in self._sheet_data.data:
            row_id = row.values[self._sheet_data.headers.get("id")]
            row_surgeon = row.values[self._sheet_data.headers.get("surgeon")]
            row_date = row.values[self._sheet_data.headers.get("date")]
            row_protocol = row.values[self._sheet_data.headers.get("protocol")]

            row_id = row_id.strip().lower() if row_id else None
            row_surgeon = row_surgeon.strip().lower() if row_surgeon else None
            row_date = row_date.strip().lower() if row_date else None
            row_protocol = row_protocol.strip().lower() if row_protocol else None

            if (
                row_id == ID.lower()
                and row_surgeon == surgeon.lower()
                and row_date == date.lower()
                and row_protocol == protocol.lower()
            ):
                protocol_data = ProtocolData(ID=row_id, date=row_date, surgeon=row_surgeon, protocol=row_protocol)

                implant_data = ImplantData(self._sheet_data.headers, row.values)
                injection_data = InjectionData(self._sheet_data.headers, row.values)

                results.append(IndividualMouseData(protocol_data, implant_data, injection_data))

        return results


# main
# mice_data = MiceData()
# results = mice_data._get_mice(ID='2', surgeon="Chelsea", date="1-24-25", protocol="2024-0019")
# for result in results:
#     print(result)
#     print(f"ID: {result._ID}")
#     print(f"Protocol: {result._protocol}")
#     print(f"Injection 1 region: {result._injection1}")
