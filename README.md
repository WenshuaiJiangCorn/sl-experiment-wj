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
   hardware, as well as dedicated camera hardware used to record visual behavior data, is controlled through the 'VRPC.'
___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [System Assembly](#system-assembly)
- [Usage](#usage)
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
section as a general system composition guide, but consult our publications over this section for instructions on 
building specific system implementations used for various projects.

Physical assembly and mounting of ***all*** hardware components mentioned in the specific subsections below is discussed
in the [main Mesoscope-VR assembly section](#mesoscope-vr-assembly).

### Zaber Motors
All brain activity recordings with the mesoscope require the animal to be head-fixed. To orient head-fixed animals on 
the Virtual Reality treadmill (running wheel) and promote task performance, we use two groups of motors controlled 
though Zaber motor controllers. The first group, the **HeadBar**, is used to position of the animal’s head in 
Z, Pitch, and Roll axes. Together with the movement axes of the Mesoscope, this allows for a wide range of 
motions necessary to promote good animal running behavior and brain activity data collection. The second group of 
motors, the **LickPort**, controls the position of the water delivery port (and sensor) in X, Y, and Z axes. This
is used to ensure all animals have comfortable access to the water delivery tube, regardless of their head position.

The current snapshot of Zaber motor configurations used in the lab, alongside motor parts list and electrical wiring 
instructions, is available 
[here](https://drive.google.com/drive/folders/1SL75KE3S2vuR9TTkxe6N4wvrYdK-Zmxn?usp=drive_link).

**Warning!** Zaber motors have to be configured correctly to work with this library. Unless you restore Zaber settings 
from the snapshots available from the link above, it is likely that the motors will behave unexpectedly. This can damage
the surrounding equipment, animals, or motors themselves and should be avoided at all costs.

To manually configure the motors to work with this library, you need to overwrite the non-volatile 'user settings' of 
each motor device (controller) with the data expected by this library. See the 
[API documentation](https://sl-experiment-api.netlify.app/) for the **ZaberSettings** class to learn more about the 
settings used by this library. See the source code from the [zaber_bindings.py](/src/sl_experiment/zaber_bindings.py)
module to learn how these settings are used during runtime.

### Behavior Cameras
To record the animal’s behavior, we use a group of three cameras. The **face_camera** is a high-end machine-vision 
camera used to record the animal’s face with approximately 3-MegaPixel resolution. The **left-camera** and 
**right_camera** are 1080P security cameras used to record the body of the animal. Only the data recorded by the 
**face_camera** is currently used during data processing and analysis. We use custom 
[ataraxis-video-system](https://github.com/Sun-Lab-NBB/ataraxis-video-system) bindings to interface with and record the 
frames acquired by all cameras.

Specific information about the components used by the camera systems, as well as the snapshot of the configuration 
parameters used by the **face_camera**, is available 
[here]https://drive.google.com/drive/folders/1l9dLT2s1ysdA3lLpYfLT1gQlTXotq79l?usp=sharing).

### MicroControllers
To interface with all components of the Mesoscope-VR system **other** than cameras and Zaber motors, we use Teensy 4.1 
microcontrollers with specialized [ataraxis-micro-controller](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller) 
code. Currently, we use three isolated microcontroller systems: **Actor**, **Sensor**, and **Encoder**.

For instructions on assembling and wiring the electronic components used in each microcontroller system, as well as the 
code running on each microcontroller, see the 
[microcontroller repository](https://github.com/Sun-Lab-NBB/sl-micro-controllers).

### Unity Game World
The task environment used in Sun lab experiments is rendered and controlled by the Unity game engine. To make Unity work
with this library, each project-specific Unity task must use the bindings and assets released as part of our 
[GIMBL-tasks repository](https://github.com/Sun-Lab-NBB/GIMBL-tasks). Follow the instructions from that repository to 
set up Unity Game engine to run Sun lab experiment tasks.

**Note** This library does not contain tools capable of initializing Unity Game engine. The desired Virtual Reality task
has to be started manually before initializing the main experiment runtime through this library.

### Google Sheets API Integration

This library is statically configured to interact with various Google Sheet files used in the Sun lab. Currently, this 
includes two logs: the **surgery log** and the **water restriction log**. Primarily, this part of the library is 
designed as a convenience feature for lab members and to back up and store all project-related data in the same place.
Generally, we expect that each project uses a unique set of log files, but shares the Goggle API credentials used to 
access and parse the sheet data.

#### Setting up Google Sheets API Access

**If you already have a service Google Sheets API account, skip to the next section.**

1. Log into the [Google Cloud Console](https://shorturl.at/qiDYc). 
2. Create a new project
3. Navigate to APIs & Services > Library and enable the Google Sheets API for the project. 
4. Under IAM & Admin > Service Accounts, create a service account. This will generate a service account ID in the format
   of `your-service-account@gserviceaccount.com`.
5. Select Manage Keys from the Actions menu and, if a key does not already exist, create a new key and download the 
   private key in JSON format. This key is then used to access the Google Sheets.

#### Adding Google Sheets Access to the Service Account
To access the **surgery log** and the **water restriction log** Google Sheets as part of this library runtime, create 
and share these log files with the email of the service account created above. The service account requires **Viewer** 
access to the **surgery log** file and **Editor** access to the **water restriction log** file.

**Note!** This feature expects that both log files are formatted according to the available Sun lab templates. 
Otherwise, the parsing algorithm will not behave as expected, leading to runtime failure.

### Mesoscope-VR Assembly:
This section is currently a placeholder. Since we are actively working on the final Mesoscope-VR design, it will be 
populated once we have a final design implementation.
___

## Data Structure and Management

This 

--- 

## Usage

All user-facing library functionality is realized through a set of Command-Line Interface (CLI) commands automatically 
exposed when the library is pip-installed into a python environment. Some of these commands take additional arguments 
that allow further configuring their runtime. Use `--help` argument when calling any of the commands described below to
see the list of supported arguments together with their descriptions and default values.

To use any of the commands described below, activate the python environment where the libray is installed, e.g., with 
`conda activate myenv` and type one of the commands described below.

***Warning!*** Most commands described below use the terminal to communicate important runtime information to the user 
or request user feedback. **Make sure you carefully read every message printed to the terminal during runtime**. 
Failure to do so may damage the equipment or harm the animal.

### sl-crc
This command takes in a string-value and returns a CRC-32 XFER checksum of the input string. This is used to generate a 
numeric checksum for each Zaber Device by check-summing its label (name). This checksum should be stored under User 
Setting 0. During runtime, it is used to ensure that each controller has been properly configured to work with this 
library, by comparing the checksum loaded from User Setting 0 to the checksum generated using the device’s label.

### sl-devices
This command is used during initial system configuration to discover the USB ports assigned to all Zaber devices. This 
is used for generating te project_configuration.yaml files that, amongst other information, communicate the USB ports 
used by various Mesoscope-VR system components during runtime.

### sl-replace-root
This command is used to replace the path to the VRPC folder where all projects are saved, which is stored in the 
non-volatile user-specific memory used by this library. When one of the main runtime commands from this library is 
used for the **first ever time**, the library asks the user to define a directory where to save all projects. All future
calls to this library will use the same path and assume the projects are stored in that directory. Since the path is 
stored in a typically hidden service directory, this command simplifies finding and replacing the path if this need 
ever arises.

### sl-maintain-vr
This command is typically used twice during each experiment or training day. First, it is used at the beginning of the 
day to prepare the Mesoscope-VR system for runtime by filling the water delivery system with water and, if necessary, 
replacing the running-wheel surface wrap. Second, it is used at the end of each day to empty the water delivery system.

This runtime is also co-opted to check the cranial windows of newly implanted animals to determine whether they should
be included in a project. To do so, the command allows changing the position of the HeadBar and LickPort manipulators 
and generating a snapshot of Mesoscope and Zaber positions, as well as the screenshot of the cranial window.

***Note!*** Since this runtime fulfills multiple functions, it uses an 'input'-based terminal interface to accept 
further commands during runtime. To prevent visual bugs, the input does not print anything to the terminal and appears 
as a blank new line. If you see a blank new line with no terminal activity, this indicates that the system is ready 
to accept one of the supported commands. All supported commands are printed to the terminal as part of the runtime 
initialization.

#### Supported vr-maintenance commands
1.  `open`. Opens the water delivery valve.
2.  `close`. Closes the water delivery valve.
3.  `close_10`. Closes the water delivery valve after a 10-second delay.
4.  `reference`. Triggers 200 valve pulses with each pulse calibrated to deliver 5 uL of water. This commands is used to
    check whether the valve calibration data matches the actual state of the valve at the beginning of each training or 
    experiment day. The reference runtime should overall dispense ~ 1 ml of water.
5.  `calibrate_15`. Runs 200 valve pulses, keeping the valve open for 15-milliseconds for each pulse. This is used to 
    generate valve calibration data.
6.  `calibarte_30`. Same as above, but uses 30-millisecond pulses.
7.  `calibrate_45`. Same as above, but uses 45-millisecond pulses.
8.  `calibrate_60`. Same as above, but uses 60-millisecond pulses.
9.  `lock`. Locks the running wheel (engages running wheel break).
10. `unlock`. Unlocks the running wheel (disengages running wheel break).
11. `maintain`. Moves the HeadBar and LickPort to the predefined VR maintenance position stored inside non-volatile
    Zaber device memory.
12. `mount`. Moves the HeadBar and LickPort to the predefined animal mounting position stored inside non-volatile
    Zaber device memory. This is used when checking the cranial windows of newly implanted animals.
13. `image`. Moves the HeadBar and LickPort to the predefined brain imaging position stored inside non-volatile
    Zaber device memory. This is used when checking the cranial windows of newly implanted animals.
14. `snapshot`. Generates a snapshot of the Zaber motor positions, Mesoscope positions, and the screenshot of the 
    cranial window. This saves the system configuration for the checked animal, so that it can be reused during future 
    training and experiment runtimes

### sl-lick-train
Runs a single lick-training session. All animals in the Sun lab undergo a two-stage training protocol before they start 
participating in project-specific experiments. The first phase of the training protocol is lick training, where the 
animals are trained to operate the lick-tube while being head-fixed. This training is carried out for 2 days and uses 
the same runtime protocol, resolved by this command.

### sl-run-train
Runs a single run-training session. The second phase of the Sun lab training protocol is run training, where the 
animals run on the wheel treadmill while being head-fixed to get water rewards. This training is carried out for the 
5 days following the lick-training and uses the same runtime protocol, resolved by this command.

### sl-experiment
Runs a single experiment session. Each project has to define one or more experiment configurations that can be executed 
via this command. Every experiment configuration may be associated with a unique Unity VR task, which has to be
activated independently of running this command.

***Critical!*** Study the [API documentation](https://sl-experiment-api.netlify.app/) of the ExperimentConfiguration and
ExperimentState classes available from the [data_classes.py](/src/sl_experiment/data_classes.py) module. Each experiment
called via this command requires a well-configured EXPERIMENT_NAME.yaml file inside the 'configuration' directory of the
project. During project initialization, an example experiment configuration file is dumped in the 'configuration' 
directory alongside the main **project_configuration.yaml** file to assist users in writing their own experiment 
configurations.

### sl-process
This command can be called to preprocess the target training or experiment session data folder. Typically, this library
calls the preprocessing pipeline as part of the runtime command, so there is no need to use this command. However, if 
the runtime or preprocessing is unexpectedly interrupted, call this command to ensure the target session is preprocessed
and transferred to the long-term storage destinations.

### sl-purge
To maximize data integrity, this library does not automatically delete redundant data from the ScanImagePC or the VRPC, 
even if the data has been safely backed up to long-term storage destinations. This command discovers all redundant data
marked for deletion by various Sun lab pipelines and deletes it from the ScanImagePC or the VRPC. 

***Critical*** This command has to be called at least weekly to prevent running out of disk space on the ScanImagePC and
VRPC.

---

## API Documentation

See the [API documentation](https://sl-experiment-api.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.
___

## Recovering from Interruptions

--

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
