# sl-experiment

A Python library that provides tools to acquire, manage, and preprocess scientific data in the Sun (NeuroAI) lab.

![PyPI - Version](https://img.shields.io/pypi/v/sl-experiment)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sl-experiment)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/sl-experiment)
![PyPI - Status](https://img.shields.io/pypi/status/sl-experiment)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/sl-experiment)
___

## Detailed Description

All experimental projects in the lab are based on this interface library. Primarily, it contains the bindings for all 
subsystems used to collect behavior data and control experiment runtimes. This library is purpose-built for the hardware
used in the lab and will likely require extensive modifications to be used elsewhere.

This library leverages the codebase developed in the lab as part of the 'Ataraxis' project to interface with 
individual system components. It is publicly accessible primarily to serve as an example of how to leverage 'Ataraxis' 
libraries to implement custom projects in scientific and industrial contexts. Navigate to the ReadMe / API 
documentation of specific 'Ataraxis' libraries if you need help with a particular system component or interface.
___

## Features

- Written for and extensively tested on the Linux operating system.
- Abstracts working with all hardware components used to conduct experiments in the Sun Lab by exposing an easy-to-use
  state-machine API.
- Contains built-in pipelines to pre-process and distribute the acquired data over lab servers and NAS storage drives.
- Maximizes data throughput, integrity, and resilience to interruptions.
- GPL 3 License.
___

## Table of Contents

- [Software Dependencies](#software-dependencies)
- [Hardware Dependencies](#software-dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)
___

## Software Dependencies
- [MQTT broker](https://mosquitto.org/). The broker should be running locally on the VRPC with default IP and Port 
  configuration.
- [FFMPEG](https://www.ffmpeg.org/download.html). We recommend using the latest available release.
- [MvImpactAcquire](https://assets-2.balluff.com/mvIMPACT_Acquire/) GenTL producer. This library works with version 
  **2.9.2**, but higher versions may be fine too.
- [Zaber Launcher](https://software.zaber.com/zaber-launcher/download). We recommend the latest available release.
- 
---

## Hardware Dependencies
- [Nvidia GPU](https://www.nvidia.com/en-us/). This library uses GPU hardware acceleration to encode acquired video 
  data. Any GPU with hardware encoding chip(s) should be fine, the library was tested with RTX 4090.
- [Teensy MicroController Boards](https://www.pjrc.com/teensy/). This library is designed to run with 3 Teensy 4.1 
  MicroControllers.
___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for everyone who is not an active project developer.
Developers should see the [Developers](#Developers) section for more details on installing from source. The instructions
below assume you are ***not*** a developer.

1. Download this repository to your local machine using your preferred method, such as Git-cloning. Use one
   of the stable releases from [GitHub](https://github.com/Sun-Lab-NBB/sl-experiment/releases).
2. Unpack the downloaded zip and note the path to the binary wheel (`.whl`) file contained in the archive.
3. Run ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file, to install the 
   wheel into the active python environment.

### pip
Use the following command to install the library using pip: ```pip install sl-experiment```.
___

## Google Sheets API Integration

To connect your Google Sheets, first log into the Google Cloud Console using the Gmail account of 
the Google Sheet owner. Create a new project and navigate to APIs & Services > Library and enable the Google Sheets API 
for the project. Under IAM & Admin > Service Accounts, create a service account. This will generate a service account 
ID in the format of your-service-account@gserviceaccount.com. Share the Google Sheets you would like to access with this
service account email and provide it with Editor access. Once the service account is created, select Manage Keys from 
the Actions menu. If a key does not already exist, create a new key and download the private key in JSON format. This 
JSON file should be added to your project directory and will be used for authentication when making requests to the Google 
Sheets API. 

1. Google API Client Setup

    The script for data parsing requires the google-api-python-client and google-auth libraries to interact with the Google 
    Sheets API. Install all dependencies with the following command:
    pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib. 

2. Sheet Configuration

    To connect to your Google Sheet, the sheet ID, range, service account file and scope must be specified. The sheet ID 
    is the unique identifier found in the URL between /d/ and /edit. To parse data from the entire sheet, set the range
    to A1:Z for up to 26 columns or A1:ZZ if exceeded. The tab name is optional and is only required if the sheet contains 
    multiple tabs. If no tab name is provided, the default is set as the first tab. The scope defines the application's 
    access level to the Google Sheets API. The full list of Google Sheets API scopes can be found here 
    [https://developers.google.com/identity/protocols/oauth2/scopes].

    sheet_id = "your-google-sheet-id"
    range = "range"
    tab_name = "tab name"
    SERVICE_ACCOUNT_FILE = "path/to/your/service-account-key.json"
    SCOPES = ["Scope code"]

3. Authentication using service account credentials and Retrieving Data

    creds = Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)  
    service = build("sheets", "v4", credentials=creds)
    range_name = f"{self.tab_name}!{self.range}"
    result = service.spreadsheets().values().get(spreadsheetId=self.sheet_id, range=range_name).execute()
___

## API Documentation

See the [API documentation](https://sl-experiment-api.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.
___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library.

### Installing the library

The easiest way to ensure you have most recent development dependencies and library source files is to install the 
python environment for your OS (see below). All environments used during development are exported as .yml files and as 
spec.txt files to the [envs](envs) folder. The environment snapshots were taken on each of the three explicitly 
supported OS families: Windows 11, OSx Darwin, and GNU Linux.

**Note!** Since the OSx environment was built for the Darwin platform (Apple Silicon), it may not work on Intel-based 
Apple devices.

1. If you do not already have it installed, install [tox](https://tox.wiki/en/latest/user_guide.html) into the active
   python environment. The rest of this installation guide relies on the interaction of local tox installation with the
   configuration files included in with this library.
2. Download this repository to your local machine using your preferred method, such as git-cloning. If necessary, unpack
   and move the project directory to the appropriate location on your system.
3. ```cd``` to the root directory of the project using your command line interface of choice. Make sure it contains
   the `tox.ini` and `pyproject.toml` files.
4. Run ```tox -e import``` to automatically import the os-specific development environment included with the source 
   distribution. Alternatively, you can use ```tox -e create``` to create the environment from scratch and automatically
   install the necessary dependencies using pyproject.toml file. 
5. If either step 4 command fails, use ```tox -e provision``` to fix a partially installed environment.

**Hint:** while only the platforms mentioned above were explicitly evaluated, this project will likely work on any 
common OS, but may require additional configurations steps.

### Additional Dependencies

In addition to installing the development environment, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. These 
   versions will be installed in-addition to the main Python version installed in the development environment.
   The easiest way to get tox to work as intended is to have separate python distributions, but using 
   [pyenv](https://github.com/pyenv/pyenv) is a good alternative. This is needed for the 'test' task to work as 
   intended.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check [tox.ini file](tox.ini) for details about 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All commits to this project have to successfully complete the ```tox``` task before being pushed to GitHub. 
To minimize the runtime duration for this task, use ```tox --parallel```.

For more information, check the 'Usage' section of the 
[ataraxis-automation project](https://github.com/Sun-Lab-NBB/ataraxis-automation#Usage) documentation.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself are prone to various failures. In 
most cases, this is related to their caching behavior. Despite a considerable effort to disable caching behavior known 
to be problematic, in some cases it cannot or should not be eliminated. If you run into an unintelligible error with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a cli command is very likely to fix the issue.
___

## Versioning

We use [semantic versioning](https://semver.org/) for this project. For the versions available, see the 
[tags on this repository](https://github.com/Sun-Lab-NBB/sl-experiment/tags).

---

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Kushaan Gupta ([Inkaros](https://github.com/kushaangupta))
- Natalie Yeung
- Katlynn Ryu ([katlynn-ryu](https://github.com/KatlynnRyu))
- Jasmine Si

___

## License

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.
___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The creators of all other projects used in our development automation pipelines and source code 
  [see pyproject.toml](pyproject.toml).

---
