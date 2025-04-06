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

This library functions as the central hub for collecting and preprocessing the data in the Sun lab shared by all 
individual lab projects. To do so, it exposes the API that allows interfacing with the hardware making up the overall 
Mesoscope-VR (Virtual Reality) system used in the lab and working with the data collected via this hardware. Primarily, 
this involves specializing the general-purpose libraries, such as 
[ataraxis-video-system](https://github.com/Sun-Lab-NBB/ataraxis-video-system), 
[ataraxis-micro-controller](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller) 
and [ataraxis-data-structures](https://github.com/Sun-Lab-NBB/ataraxis-data-structures) to work within the specific 
hardware implementations used in the lab.

This library is explicitly designed to work with the specific hardware and data handling strategies used in the Sun lab,
and will likely not work in other contexts without extensive modification. It is made public to serve as the real-world 
example of how to use 'Ataraxis' libraries to acquire and preprocess scientific data.

Currently, the Mesoscope-VR system consists of three major parts: 
1. The [2P-Random-Access-Mesoscope (2P-RAM)](https://elifesciences.org/articles/14472), assembled by 
   [Thor Labs](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10646) and controlled by 
   [ScanImage](https://www.mbfbioscience.com/products/scanimage/) software. The Mesoscope control and data acquisition 
   are performed by a dedicated computer referred to as the 'ScanImagePC' or 'Mesoscope PC.' 
2. The [Unity game engine](https://unity.com/products/unity-engine) running the Virtual Reality game world used in all 
   experiments to control the task environment and resolve the task logic. The virtual environment runs on the main data
   acquisition computer referred to as the 'VRPC.'
3. The [microcontroller-powered](https://github.com/Sun-Lab-NBB/sl-micro-controllers) hardware that allows 
   bidirectionally interfacing with the Virtual Reality world and collecting non-visual animal behavior data. This 
   hardware, as well as dedicated camera hardware used to record visual behavior data, is controlled through the 'VRPC'.
___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [System Assembly](#system-assembly)
- [API Documentation](#api-documentation)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)
___

## Dependencies

### Main Dependency
- ***Linux*** operating system. While the library may also work on Windows and macOS, it has been explicitly written for
  and tested on mainline [6.11 kernel](https://kernelnewbies.org/Linux_6.11) and Ubuntu 24.10 distribution of the GNU 
  Linux operating system.

### Software Dependencies
***Note!*** This list only includes external dependencies that are required to run the library, in addition to all 
dependencies automatically installed from pip / conda as part of library installation. The dependencies below have to
be installed and configured on the **VRPC** before calling runtime commands via the command-line interface (CLI) exposed
by this library.

- [MQTT broker](https://mosquitto.org/). The broker should be running locally with the **default** IP (27.0.0.1) and 
  Port (1883) configuration.
- [FFMPEG](https://www.ffmpeg.org/download.html). As a minimum, the version of FFMPEG should support H265 and H264 
  codecs with hardware acceleration (Nvidia GPU). It is typically safe to use the latest available version.
- [MvImpactAcquire](https://assets-2.balluff.com/mvIMPACT_Acquire/) GenTL producer. This library is used with version 
  **2.9.2**, which is freely distributed. Higher GenTL producer versions will likely work too, but they require 
  purchasing a license.
- [Zaber Launcher](https://software.zaber.com/zaber-launcher/download). Use the latest available release.
- [Unity Game Engine](https://unity.com/products/unity-engine). Use the latest available release.
---

### Hardware Dependencies

**Note!** These dependencies only apply to the 'VRPC,' the main PC that will be running the data acquisition and 
preprocessing pipelines.

- [Nvidia GPU](https://www.nvidia.com/en-us/). This library uses GPU hardware acceleration to encode acquired video 
  data. Any Nvidia GPU with hardware encoding chip(s) should work as expected. The library was tested with RTX 4090.
- A CPU with at least 12, preferably 16 physical cores. This library has been tested with 
  [AMD Ryzen 7950X CPU](https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-9-7950x.html). 
  It is recommended to use CPUs with 'full' cores, instead of the modern Intel’s design of 'e' and 'p' cores 
  for predictable performance of all library components.
- A 10-Gigabit capable motherboard or Ethernet adapter, such as [X550-T2](https://shorturl.at/fLLe9). Primarily, this is
  required for the high-quality machine vision camera used to record videos of the animal’s face. We also used 10-G 
  lines for transferring the data between the PCs used in the data acquisition process and destinations used for 
  long-term data storage.
___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for everyone who is not an active project developer.

1. Download this repository to your local machine using your preferred method, such as Git-cloning. Use one
   of the stable releases from [GitHub](https://github.com/Sun-Lab-NBB/sl-experiment/releases).
2. Unpack the downloaded zip and note the path to the binary wheel (`.whl`) file contained in the archive.
3. Run ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file, to install the 
   wheel into the active python environment.

### pip
Use the following command to install the library using pip: ```pip install sl-experiment```.
___

## System Assembly

The Mesoscope-VR system consists of multiple interdependent components. We are constantly making minor changes to the 
system to optimize its performance and facilitate novel experiments and projects carried out in the lab. Treat this 
section as a general system assembly guide, but consult our publications over this section for instructions on building 
specific system implementations used for various projects.

Physical assembly and mounting of ***all*** hardware components mentioned in specific subsections below is discussed in 
the [main VR environment assembly section](#virtual-reality-environment).

### Zaber Motors
All brain activity recordings with the mesoscope require the animal to be head-fixed. To orient head-fixed animals on 
the Virtual Reality treadmill (running wheel) and promote task performance, we use two groups of motors controlled 
though Zaber motor controllers. The first group, called the **HeadBar**, is used to position of the animal’s head in 
Z, Pitch, and Roll axes. Together with the movement axes of the Mesoscope, this allows for a wide range of 
motions necessary to promote good animal running behavior and brain activity data collection. The second group of 
motors, called the **LickPort**, controls the position of the water delivery port (and sensor) in X, Y, and Z axes. This
is used to ensure all animals have comfortable access to the water delivery tube, regardless of their head position.

The current snapshot of Zaber motor configurations used in the lab, alongside motor parts list and electrical wiring 
instructions, is available 
[here](https://drive.google.com/drive/folders/1SL75KE3S2vuR9TTkxe6N4wvrYdK-Zmxn?usp=drive_link).

### Behavior Cameras
To record the animal’s behavior, we use a group of three cameras. The 'face_camera' is a high-end machine-vision camera
used to record the animal’s face with ~3-MP resolution. The 'left-camera' and 'right_camera' are 1080P security cameras 
used to record the body of the animal. Only the data recorded by the face_camera is currently used during data 
processing and analysis. We use custom [ataraxis-video-system](https://github.com/Sun-Lab-NBB/ataraxis-video-system) 
bindings to interface with and record the frames acquired by all cameras.

Specific information about the components used by the camera systems, as well as assembly instructions and face_camera
parameters are available [here]https://drive.google.com/drive/folders/1l9dLT2s1ysdA3lLpYfLT1gQlTXotq79l?usp=sharing).

### MicroControllers
To interface with all components of the Mesoscope-VR system **other** than cameras and Zaber motors, we use Teensy 4.1 
microcontrollers with specialized [ataraxis-micro-controller](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller) 
code. Currently, we use three isolated microcontroller systems: "Actor," "Sensor," and "Encoder."

For instructions on assembling and wiring the electronic components used in each microcontroller system, as well as the 
code running on each microcontroller, see the 
[microcontroller repository](https://github.com/Sun-Lab-NBB/sl-micro-controllers).

### Unity Game World


### Google Sheets API Integration

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

    To connect to your Google Sheet, the sheet ID, range, service account file, and scope must be specified. The sheet ID 
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

### Virtual Reality Environment:

___

## API Documentation

See the [API documentation](https://sl-experiment-api.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.
___

## Versioning

We use [semantic versioning](https://semver.org/) for this project. For the versions available, see the 
[tags on this repository](https://github.com/Sun-Lab-NBB/sl-experiment/tags).

---

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Kushaan Gupta ([kushaangupta](https://github.com/kushaangupta))
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
