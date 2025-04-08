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

This library functions as the central hub for collecting and preprocessing the data shared by all individual Sun lab 
projects. To do so, it exposes the API that allows interfacing with the hardware making up the overall Mesoscope-VR 
(Virtual Reality) system used in the lab and working with the data collected via this hardware. Primarily, this involves
specializing varius general-purpose libraries, released as part of the 'Ataraxis' science-automation project
to work within the specific hardware implementations used in the lab.

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
- [Recovering from Interruptions](#recovering-from-interruptions)
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

**Note!** These dependencies only apply to the 'VRPC,' the main PC that runs the data acquisition and 
preprocessing pipelines. Hardware dependencies for the ScanImagePC are determined by ThorLabs.

- [Nvidia GPU](https://www.nvidia.com/en-us/). This library uses GPU hardware acceleration to encode acquired video 
  data. Any Nvidia GPU with hardware encoding chip(s) should work as expected. The library was tested with RTX 4090.
- A CPU with at least 12, preferably 16, physical cores. This library has been tested with 
  [AMD Ryzen 7950X CPU](https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-9-7950x.html). 
  It is recommended to use CPUs with 'full' cores, instead of the modern Intel’s design of 'e' and 'p' cores 
  for predictable performance of all library components.
- A 10-Gigabit capable motherboard or Ethernet adapter, such as [X550-T2](https://shorturl.at/fLLe9). Primarily, this is
  required for the high-quality machine vision camera used to record videos of the animal’s face. We also use 10-Gigabit
  lines for transferring the data between the PCs used in the data acquisition process and destinations used for 
  long-term data storage (see [data management section](#data-structure-and-management)).
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
though Zaber motor controllers. The first group, the **HeadBar**, is used to position the animal’s head in 
Z, Pitch, and Roll axes. Together with the movement axes of the Mesoscope, this allows for a wide range of 
motions necessary to promote good animal running behavior and brain activity data collection. The second group of 
motors, the **LickPort**, controls the position of the water delivery port (and sensor) in X, Y, and Z axes. This
is used to ensure all animals have comfortable access to the water delivery tube, regardless of their head position.

The current snapshot of Zaber motor configurations used in the lab, alongside motor parts list and electrical wiring 
instructions, is available 
[here](https://drive.google.com/drive/folders/1SL75KE3S2vuR9TTkxe6N4wvrYdK-Zmxn?usp=drive_link).

**Warning!** Zaber motors have to be configured correctly to work with this library. To (re)configure the motors to work
with the library, apply the setting snapshots from the link above via the 
[Zaber Launcher](https://software.zaber.com/zaber-launcher/download) software. Make sure you read the instructions in 
the 'Applying Zaber Configuration' document for the correct application procedure.

**Although this is highly discouraged, you can also edit the motor settings manually**. To configure the motors
to work with this library, you need to overwrite the non-volatile User Data of each motor device (controller) with
the data expected by this library. See the [API documentation](https://sl-experiment-api-docs.netlify.app/) for the 
**ZaberSettings** class to learn more about the settings used by this library. See the source code from the 
[zaber_bindings.py](/src/sl_experiment/zaber_bindings.py) module to learn how these settings are used during runtime.

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

**Note** This library does not contain tools to initialize Unity Game engine. The desired Virtual Reality task
has to be started ***manually*** before initializing the main experiment runtime through this library. The main Unity 
repository contains more details about starting the virtual reality tasks when running experiments.

### Google Sheets API Integration

This library is statically configured to interact with various Google Sheet files used in the Sun lab. Currently, this 
includes two files: the **surgery log** and the **water restriction log**. Primarily, this part of the library is 
designed as a convenience feature for lab members and to back up and store all project-related data in the same place.

#### Setting up Google Sheets API Access

**If you already have a service Google Sheets API account, skip to the next section.** Typically, we use the same 
service account for all projects and log files.

1. Log into the [Google Cloud Console](https://shorturl.at/qiDYc). 
2. Create a new project.
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
***This section is currently a placeholder. Since we are actively working on the final Mesoscope-VR design, it will be 
populated once we have a final design implementation.***

___

## Data Structure and Management

The library defines a fixed structure for storing all acquired data which uses a 4-level directory tree hierarchy: 
**root**, **project**, **animal**, and **session**.

Currently, our pipeline uses a total of four computers when working with data. **VRPC** and **ScanImagePC** are used to 
acquire and preprocess the data. After preprocessing, the data is moved to the **BioHPC server** and the 
**Synology NAS** for long-term storage and preprocessing. All data movement is performed over 10-Gigabit local networks 
within the lab and broader Cornell infrastructure.

***Critical!*** Although this library primarily operates the VRPC, it expects the root data directories for all other 
PCs used for data acquisition or storage in the lab to be **mounted to the VRPC filesystem using the SMB3 protocol**. 
In turn, this allows the library to maintain the same data hierarchies across all storage machines.

Generally, the library tries to maintain at least two copies of data for long-term storage: one on the NAS and the other
on the BioHPC server. Moreover, until `sl-purge` (see below) command is used to clear the VRPC storage, an additional 
copy of the acquired data is also stored on the VRPC for each recorded session. While this design achieves high data 
integrity (and redundancy), we **highly encourage** all lab members to manually back up critical data to external 
SSD / HDD drives.

### Root Directory
When a training, experiment, or maintenance runtime command from this library is called for the first time, the library 
asks the user to provide the path to the root project directory on the VRPC. The data for all projects after this point 
is stored in that directory. This directory is referred to as the local **root** directory. Moreover, each
project can be configured with paths to the root directories on all other computers used for data acquisition or 
storage (see below). However, it is expected that all projects in the lab use the same root directories for all 
computers.

### Project directory
When a new --project (-p) argument value is provided to any runtime command, the library generates a new **project**
directory under the static **root** directory. The project directory uses the project name provided via the command
argument as its name. As part of this process, a **configuration** subdirectory is also created under the 
**project** directory.

***Critical!*** Inside the **configuration** subdirectory, the library automatically creates a 
**project_configuration.yaml** file. Open that file with a text editor and edit the fields in the file to specify the 
project configuration. Review the [API documentation](https://sl-experiment-api-docs.netlify.app/) for the 
**ProjectConfiguration** class to learn more about the purpose of each configuration file field.

Together with the **project_configuration.yaml**, the library also creates an example **default_experiment.yaml**
file. Each experiment that needs to be carried out as part of this project needs to have a dedicated .yaml file, named
after the experiment. For example, to run the 'default_experiment,' the library uses the configurations stored in 
the 'default_experiment.yaml' file. You can use the default_experiment.yaml as an example for writing additional 
experiment configurations. Review the [API documentation](https://sl-experiment-api-docs.netlify.app/) for the 
**ExperimentConfiguration** and **ExperimentState** classes to learn more about the purpose of each field inside the 
experiment configuration .yaml file.

### Animal directory
When a new --animal (-a) argument value is provided to any runtime command, the library generates a new **animal**
directory under the **root** and **project** directory combination. The directory uses the ID of the animal, 
provided via the command argument as its name.

Under each animal directory, two additional directories are created. First, the **persistent_data** directory, which
is used to store the information that has to stay on the VRPC when the acquired data is transferred from the VRPC to 
other destinations. Second, the **metadata** directory, which is used to store information that does not change between
sessions, such as the information about the surgical procedures performed on the animal.

### Session directory
When any training or experiment runtime command is called, a new session directory is created under the **root**, 
**project** and **animal** directory combination. The session name is derived from the current UTC timestamp, accurate 
to microseconds. Together with other runtime controls, this makes it impossible to have sessions with duplicate names 
and ensures all sessions can always be sorted chronologically.

Since the same directory tree is reused for data processing, all data acquired by this library is stored under the 
**raw_data** subdirectory, generated for each session. Overall, an example path to the acquired data can therefore 
look like this: `/media/Data/Experiments/TestMice/666/2025-11-11-05-03-234123/raw_data`. Our data processing pipelines 
generate new files and subdirectories under the **processed_data** directory using the same **root**, **project**, 
**animal**, and **session** combination, e.g.`server/sun_data/TestMice/666/2025-11-11-05-03-234123/processed_data`.

### Raw Data contents
After acquisition and preprocessing, the **raw_data** folder will contain the following files and subdirectories:
1. **zaber_positions.yaml**: Stores the snapshot of the HeadBar and LickPort motor positions taken at the end of the 
   runtime session.
2. **hardware_configuration.yaml**: Stores the snapshot of some configuration parameters used by the hardware module 
   interfaces during runtime.
3. **session_data.yaml**: Stores the paths and other data-management-related information used during runtime to save and
   preprocess the acquired data.
4. **session_descriptor.yaml**: Stores session-type-specific information, such as the training task parameters or 
   experimenter notes. For experiment runtimes, this file is co-opted to store the Mesoscope objective positions.
5. **ax_checksum.txt**: Stores an xxHash-128 checksum used to verify data integrity when it is transferred to the 
   long-term storage destination.
6. **behavior_data_log**: Stores compressed .npz log files. All non-video data acquired by the VRPC during runtime is 
   stored in these log files. This includes all messages sent or received by each microcontroller and the timestamps 
   for the frames acquired by each camera.
7. **camera_frames** Stores the behavior videos recorded by each of the cameras.
8. **mesoscope_frames**: Stores all Mesoscope-acquired data (frames, motion estimation files, etc.). This directory 
   will be empty for training sessions, as they do not acquire Mesoscope data.

### ScanImagePC

The ScanImagePC uses a modified directory structure. First, under its **root** directory, there has to be a 
**mesoscope_frames** directory, where ***All*** ScanImage data is saved during each session runtime. During 
preprocessing, the library automatically empties this directory, allowing the same directory to be (re)used by all 
experiment sessions.

Under the same **root** directory, the library also creates a **persistent_data** directory. That directory follows the 
same hierarchy (**project** and **animal**) as the VRPC. Like the VRPC’s **persistent_data** directory, it is used to 
keep the data that should not be removed from the ScanImagePC even after all data acquired for a particular session is 
moved over for long-term storage.

**Note!** For each runtime that uses the mesoscope, the library requires the user to generate a screenshot of the 
cranial window and the dot-alignment window. The easiest way to make this work is to reconfigure the default Windows 
screenshot directory (ScanImagePC uses Windows OS) to be the root ScanImagePC data directory. This way, hitting 
'Windows+PrtSc' will automatically generate the .png screenshot under the ScanImagePc root directory, which is the 
expected location used by this library.

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
numeric checksum for each Zaber Device by check-summing its label (name). This checksum should be stored under user 
Setting 0. During runtime, it is used to ensure that each controller has been properly configured to work with this 
library by comparing the checksum loaded from User Setting 0 to the checksum generated using the device’s label.

### sl-devices
This command is used during initial system configuration to discover the USB ports assigned to all Zaber devices. This 
is used when updating the project_configuration.yaml files that, amongst other information, communicate the USB ports 
used by various Mesoscope-VR system components during runtime.

### sl-replace-root
This command is used to replace the path to the **root** directory on the VRPC (where all projects are saved), which is 
stored in a user-specific default directory. When one of the main runtime commands from this library is used for the 
**first ever time**, the library asks the user to define a directory where to save all projects. All future
calls to this library use the same path and assume the projects are stored in that directory. Since the path is 
stored in a typically hidden service directory, this command simplifies finding and replacing the path if this need 
ever arises.

### sl-maintain-vr
This command is typically used twice during each experiment or training day. First, it is used at the beginning of the 
day to prepare the Mesoscope-VR system for runtime by filling the water delivery system and, if necessary, replacing 
the running-wheel surface wrap. Second, it is used at the end of each day to empty the water delivery system.

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
4.  `reference`. Triggers 200 valve pulses with each pulse calibrated to deliver 5 uL of water. This command is used to
    check whether the valve calibration data stored in the project_configuration.yaml of the project specified when 
    calling the runtime command is accurate. This is done at the beginning of each training or experiment day. The 
    reference runtime should overall dispense ~ 1 ml of water.
5.  `calibrate_15`. Runs 200 valve pulses, keeping the valve open for 15-milliseconds for each pulse. This is used to 
    generate valve calibration data.
6.  `calibarte_30`. Same as above, but uses 30-millisecond pulses.
7.  `calibrate_45`. Same as above, but uses 45-millisecond pulses.
8.  `calibrate_60`. Same as above, but uses 60-millisecond pulses.
9.  `lock`. Locks the running wheel (engages running-wheel break).
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
animals are trained to operate the lick-tube while being head-fixed. This training is carried out for 2 days.

### sl-run-train
Runs a single run-training session. The second phase of the Sun lab training protocol is run training, where the 
animals run on the wheel treadmill while being head-fixed to get water rewards. This training is carried out for the 
5 days following the lick-training.

### sl-experiment
Runs a single experiment session. Each project has to define one or more experiment configurations that can be executed 
via this command. Every experiment configuration may be associated with a unique Unity VR task, which has to be
activated independently of running this command. See the [project directory notes](#project-directory) to learn about 
experiment configuration files which are used by this command.

**Critical!** Since this library does not have a way of starting Unity game engine or ScanImage software, both have to 
be initialized **manually** before running the sl-experiment command. See the main 
[Unity repository](https://github.com/Sun-Lab-NBB/GIMBL-tasks) for details on starting experiment task runtimes. To 
prepare the ScanImage software for runtime, enable 'External Triggers' and configure the system to take **start** and 
**stop** triggers from the ports wired to the Actor microcontroller as described in our 
[microcontroller repository](https://github.com/Sun-Lab-NBB/sl-micro-controllers). Then, hit 'Loop' to 'arm' the system
to start frame acquisition when it receives the 'start' TTL trigger from this library.

### sl-process
This command can be called to preprocess the target training or experiment session data folder. Typically, this library
calls the preprocessing pipeline as part of the runtime command, so there is no need to use this command separately. 
However, if the runtime or preprocessing is unexpectedly interrupted, call this command to ensure the target session is 
preprocessed and transferred to the long-term storage destinations.

### sl-purge
To maximize data integrity, this library does not automatically delete redundant data from the ScanImagePC or the VRPC, 
even if the data has been safely backed up to long-term storage destinations. This command discovers all redundant data
marked for deletion by various Sun lab pipelines and deletes it from the ScanImagePC or the VRPC. 

***Critical!*** This command has to be called at least weekly to prevent running out of disk space on the ScanImagePC 
and VRPC.

---

## API Documentation

See the [API documentation](https://sl-experiment-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.
___

## Recovering from Interruptions
While it is not typical for the data acquisition or preprocessing pipelines to fail during runtime, it is not 
impossible. The library can recover or gracefully terminate the runtime for most code-generated errors, so this is 
usually not a concern. However, if a major interruption (i.e., power outage) occurs or the ScanImagePC encounters an 
interruption, manual intervention is typically required before the VRPC can run new data acquisition or preprocessing 
runtimes.

### Data acquisition interruption

***Critical!*** If you encounter an interruption during data acquisition (training or experiment runtime), it is 
impossible to resume the interrupted session. Moreover, since this library acts independently of the ScanImage software
managing the Mesoscope, you will need to manually shut down the other acquisition process. If VRPC is interrupted, 
terminate Mesoscope data acquisition via the ScanImage software. If the Mesoscope is interrupted, use 'ESC+Q' to 
terminate the VRPC data acquisition.

If VRPC is interrupted during data acquisition, follow this instruction:
1. If the session involved mesoscope imaging, shut down the Mesoscope acquisition process and make sure all required 
   files (frame stacks, motion estimator data, cranial window screenshot) have been generated adn saved to the 
   **mesoscope_frames** folder.
2. Remove the animal from the Mesoscope-VR system.
3. Use Zaber Launcher to **manually move the HeadBarRoll axis to have a positive angle** (> 0 degrees). This is 
   critical! If this is not done, the motor will not be able to home during the next session and will instead collide 
   with the movement guard, at best damaging the motor and, at worst, the mesoscope or the animal.
4. Go into the 'Device Settings' tab of the Zaber Launcher, click on each Device tab (NOT motor!) and navigate to its 
   User Data section. Then **flip Setting 1 from 0 to 1**. Without this, the library will refuse to operate the Zaber 
   Motors.
5. If the session involved mesoscope imaging, **rename the mesoscope_frames folder to prepend the session name, using an
   underscore to separate the folder name from the session name**. For example, from mesoscope_frames → 
   2025-11-11-05-03-234123_mesoscope_frames. Critical! if this is not done, the library may **delete** any leftover 
   mesoscope files during the next runtime and will not be able to properly preprocess the frames for the interrupted
   session during the next step.
6. Call the `sl-process` command and provide it with the path to the session directory of the interrupted session. This
   will preprocess and transfer all collected data to the long-term storage destinations. This way, you can preserve 
   any data acquired before the interruption and prepare the system for running the next session.

***Note!*** If the interruption occurs on the ScanImagePC (Mesoscope) and you use the 'ESC+Q' combination, there is 
no need to do any of the steps above. Using ESC+Q executes a 'graceful' VRPC interruption process which automatically
executes the correct shutdown sequence and data preprocessing.

### Data preprocessing interruption
To recover from an error encountered during preprocessing, call the `sl-process` command and provide it with the path 
to the session directory of the interrupted session. The preprocessing pipeline should automatically resume an 
interrupted runtime.

---

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
