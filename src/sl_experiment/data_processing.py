"""This module contains the functions used by the general data processing pipeline which runs on the BioHPC server.
Since some processing methods rely on custom code exposed by each AMC hardware module interface, it makes more sense to
implement these methods as part of this library. These methods are not used during data acquisition or preprocessing!
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import polars as pl
from numpy.typing import NDArray
from numpy.lib.npyio import NpzFile
from ataraxis_video_system import extract_logged_video_system_data
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_communication_interface import extract_logged_hardware_module_data
from typing import Any


def _interpolate_data(
    timestamps: NDArray[np.uint64],
    data: NDArray[np.integer[Any] | np.floating[Any]],
    seed_timestamps: NDArray[np.uint64],
    is_discrete: bool,
) -> NDArray[np.signedinteger[Any] | np.unsignedinteger[Any] | np.floating[Any]]:
    """Interpolates data values for the provided seed timestamps.

    Primarily, this service function is used to align dispensed water values and auditory tone states during ValveModule
    data parsing.

    Notes:
        This function expects seed_timestamps and timestamps arrays to be monotonically increasing.

        Discrete interpolated data will be returned as an array with the same datatype as the input data. Continuous
        interpolated data will always use float_64 datatype.

    Args:
        timestamps: The one-dimensional numpy array that stores the timestamps for source datapoints.
        data: The two-dimensional numpy array that stores the source datapoints.
        seed_timestamps: The one-dimensional numpy array that stores the timestamps for which to interpolate the data
            values.
        is_discrete: A boolean flag that determines whether the data is discrete or continuous.

    Returns:
        A numpy NDArray with the same dimension as the seed_timestamps array that stores the interpolated data values.
    """
    # Discrete data
    if is_discrete:
        # Preallocates the output array
        interpolated_data = np.empty(seed_timestamps.shape, dtype=data.dtype)

        # Handles boundary conditions in bulk using boolean masks. All seed timestamps below the minimum source
        # timestamp are statically set to data[0], and all seed timestamps above the maximum source timestamp are set
        # to data[-1].
        below_min = seed_timestamps < timestamps[0]
        above_max = seed_timestamps > timestamps[-1]
        within_bounds = ~(below_min | above_max)  # The portion of the seed that is within the source timestamp boundary

        # Assigns out-of-bounds values in-bulk
        interpolated_data[below_min] = data[0]
        interpolated_data[above_max] = data[-1]

        # Processes within-boundary timestamps by finding the last known certain value to the left of each seed
        # timestamp and setting each seed timestamp to that value.
        if np.any(within_bounds):
            indices = np.searchsorted(timestamps, seed_timestamps[within_bounds], side="right") - 1
            interpolated_data[within_bounds] = data[indices]

        return interpolated_data

    # Continuous data. Note, due to interpolation, continuous data is always returned using float_64 datatype.
    else:
        return np.interp(seed_timestamps, timestamps, data)  # type: ignore


def parse_encoder_data(log_path: Path, output_directory: Path, cm_per_pulse: np.float64) -> None:
    """Extracts and saves the data acquired by the module during runtime as a .feather file.

    Args:
        log_path: The path to the .npz archive that stores the data logged by the module during runtime.
        output_directory: The path to the directory where to save the parsed data as a .feather file.
        cm_per_pulse: The conversion factor to translate raw encoder pulses into distance in centimeters.
    """

    # Extracts data from the log file
    log_data = extract_logged_hardware_module_data(log_path=log_path, module_type=2, module_id=1)

    # Here, we only look for event-codes 51 (CCW displacement) and event-codes 52 (CW displacement).

    # Gets the data, defaulting to an empty list if the data is missing
    ccw_data = log_data.get(np.uint8(51), [])
    cw_data = log_data.get(np.uint8(52), [])

    # The way EncoderModule is implemented guarantees there is at least one CW code message with the displacement
    # of 0 that is received by the PC. In the worst case scenario, there will be no CCW codes and the parsing will
    # not work. To avoid that issue, we generate an artificial zero-code CCW value at the same timestamp + 1
    # microsecond as the original CW zero-code value. This does not affect the accuracy of our data, just makes the
    # code work for edge-cases.
    if not ccw_data:
        first_timestamp = cw_data[0]["timestamp"]
        ccw_data = [{"timestamp": first_timestamp + 1, "data": 0}]
    elif not cw_data:
        first_timestamp = ccw_data[0]["timestamp"]
        cw_data = [{"timestamp": first_timestamp + 1, "data": 0}]

    # Precreates the output arrays, based on the number of recorded CW and CCW displacements.
    total_length = len(ccw_data) + len(cw_data)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    displacements = np.empty(total_length, dtype=np.float64)

    # Processes CCW rotations (Code 51). CCW rotation is interpreted as positive displacement
    n_ccw = len(ccw_data)
    timestamps[:n_ccw] = [value["timestamp"] for value in ccw_data]  # Extracts timestamps for each value
    # The values are initially using the uint32 type. This converts them to float64 during the initial assignment
    displacements[:n_ccw] = [np.float64(value["data"]) for value in ccw_data]

    # Processes CW rotations (Code 52). CW rotation is interpreted as negative displacement
    timestamps[n_ccw:] = [value["timestamp"] for value in cw_data]  # CW data just fills remaining space after CCW.
    displacements[n_ccw:] = [-np.float64(value["data"]) for value in cw_data]

    # Sorts both arrays based on timestamps.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    displacements = displacements[sort_indices]

    # Converts individual displacement vectors into aggregated absolute position of the mouse. The position is also
    # translated from encoder pulse counts into centimeters. The position is referenced to the start of the
    # experimental trial (beginning of the VR track) as 0-value. Positive positions mean moving forward along the
    # track, negative positions mean moving backward along the track.
    positions: NDArray[np.float64] = np.round(np.cumsum(displacements * cm_per_pulse), decimals=8)

    # Replaces -0.0 values with 0.0. This is a convenience conversion to improve the visual appearance of numbers
    # to users
    positions = np.where(np.isclose(positions, -0.0) & (np.signbit(positions)), 0.0, positions)

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": timestamps,
            "traveled_distance_cm": positions,
        }
    )

    # Saves the DataFrame to the output directory as a Feather file with lz4 compression
    module_dataframe.write_ipc(file=output_directory.joinpath("encoder_data.feather"), compression="lz4")


def parse_ttl_data(log_path: Path, output_directory: Path) -> None:
    """Extracts and saves the data acquired by the break module during runtime as a .feather file.

    Args:
        log_path: The path to the .npz archive that stores the data logged by the module during runtime.
        output_directory: The path to the directory where to save the parsed data as a .feather file.
    """

    # Extracts data from the log file
    log_data = extract_logged_hardware_module_data(log_path=log_path, module_type=1, module_id=1)

    # Here, we only look for event-codes 52 (InputON) and event-codes 53 (InputOFF).

    # Gets the data for both message types. The way the module is written guarantees that the PC receives code 53
    # at least once. No such guarantee is made for code 52, however. We still default to empty lists for both
    # to make this code a bit friendlier to future changes.
    on_data = log_data.get(np.uint8(52), [])
    off_data = log_data.get(np.uint8(53), [])

    # Since this code ultimately looks for rising edges, it will not find any unless there is at least one ON and
    # one OFF message. Therefore, if any of the codes is actually missing, does NOT return any data. It is expected
    # that this will be interpreted as having no mesoscope frame ttl data during analysis.
    if len(on_data) == 0 or len(off_data) == 0:
        return

    # Determines the total length of the output array using the length of ON and OFF data arrays.
    total_length = len(on_data) + len(off_data)

    # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype and the trigger
    # values are boolean. We use uint8 as it has the same memory footprint as a boolean and allows us to use integer
    # types across the entire dataset.
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    triggers: NDArray[np.uint8] = np.empty(total_length, dtype=np.uint8)

    # Extracts ON (Code 52) trigger codes. Statically assigns the value '1' to denote ON signals.
    n_on = len(on_data)
    timestamps[:n_on] = [value["timestamp"] for value in on_data]
    triggers[:n_on] = np.uint8(1)  # All code 52 signals are ON (High)

    # Extracts OFF (Code 53) trigger codes.
    timestamps[n_on:] = [value["timestamp"] for value in off_data]
    triggers[n_on:] = np.uint8(0)  # All code 53 signals are OFF (Low)

    # Sorts both arrays based on the timestamps, so that the data is in the chronological order.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    triggers = triggers[sort_indices]

    # If the last value is not 0, adds a zero-value to the end of the data sequence, one microsecond
    # after the last readout. This is to properly mark the end of the monitoring sequence.
    if triggers[-1] != 0:
        timestamps = np.append(timestamps, timestamps[-1] + 1)
        triggers = np.append(triggers, 0)

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": timestamps,
            "mesoscope_scanning_state": triggers,
        }
    )

    # Saves the DataFrame to the output directory as a Feather file with lz4 compression
    module_dataframe.write_ipc(file=output_directory.joinpath("frame_data.feather"), compression="lz4")


def parse_break_data(
    log_path: Path,
    output_directory: Path,
    maximum_break_strength: np.float64,
    minimum_break_strength: np.float64,
) -> None:
    """Extracts and saves the data acquired by the break module during runtime as a .feather file.

    Args:
        log_path: The path to the .npz archive that stores the data logged by the module during runtime.
        output_directory: The path to the directory where to save the parsed data as a .feather file.
        maximum_break_strength: The maximum torque of the break in Newton centimeters.
        minimum_break_strength: The minimum torque of the break in Newton centimeters.

    Notes:
        This method assumes that the break was used in the absolute force mode. It does not extract variable
        breaking power data.
    """

    # Extracts data from the log file
    log_data = extract_logged_hardware_module_data(log_path=log_path, module_type=3, module_id=1)

    # Here, we only look for event-codes 52 (Engaged) and event-codes 53 (Disengaged) as no experiment requires
    # variable breaking power. If we ever use variable breaking power, this section would need to be expanded to
    # allow parsing code 54 events.

    # Gets the data, defaulting to an empty list if the data is missing
    engaged_data = log_data.get(np.uint8(52), [])
    disengaged_data = log_data.get(np.uint8(53), [])

    # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype. Although trigger
    # values are boolean, we translate them into the actual torque applied by the break in Newton centimeters and
    # store them as float 64 values.
    total_length = len(engaged_data) + len(disengaged_data)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    torques: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

    # Processes Engaged (code 52) triggers. When the motor is engaged, it applies the maximum possible torque to
    # the break.
    n_engaged = len(engaged_data)
    timestamps[:n_engaged] = [value["timestamp"] for value in engaged_data]  # Extracts timestamps for each value
    # Since engaged strength means that the torque is delivering maximum force, uses the maximum force in N cm as
    # the torque value for each 'engaged' state.
    torques[:n_engaged] = [maximum_break_strength for _ in engaged_data]  # Already in rounded float 64

    # Processes Disengaged (code 53) triggers. Contrary to naive expectation, the torque of a disengaged break is
    # NOT zero. Instead, it is at least the same as the minimum break strength, likely larger due to all mechanical
    # couplings in the system.
    timestamps[n_engaged:] = [value["timestamp"] for value in disengaged_data]
    torques[n_engaged:] = [minimum_break_strength for _ in disengaged_data]  # Already in rounded float 64

    # Sorts both arrays based on timestamps.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    torques = torques[sort_indices]

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": timestamps,
            "break_torque_N_cm": torques,
        }
    )

    # Saves the DataFrame to the output directory as a Feather file with lz4 compression
    module_dataframe.write_ipc(file=output_directory.joinpath("break_data.feather"), compression="lz4")


def parse_valve_data(
    log_path: Path, output_directory: Path, scale_coefficient: np.float64, nonlinearity_exponent: np.float64
) -> None:
    """Extracts and saves the data acquired by the break module during runtime as a .feather file.

    Notes:
        Unlike other processing methods, this method generates a .feather dataset with 3 columns: time, dispensed
        water volume, and the state of the tone buzzer.

    Args:
        log_path: The path to the .npz archive that stores the data logged by the module during runtime.
        output_directory: The path to the directory where to save the parsed data as a .feather file.
        scale_coefficient: Stores the scale coefficient used in the fitted power law equation that translates valve
            pulses into dispensed water volumes.
        nonlinearity_exponent: Stores the nonlinearity exponent used in the fitted power law equation that
            translates valve pulses into dispensed water volumes.
    """
    # Extracts data from the log file
    log_data = extract_logged_hardware_module_data(log_path=log_path, module_type=5, module_id=1)

    # Here, we primarily look for event-codes 52 (Valve Open) and event-codes 53 (Valve Closed).
    # We also look for codes 55 (ToneON) and 56 (ToneOFF) however and these codes are parsed similar to the
    # ttl state codes.

    # The way this module is implemented guarantees there is at least one code 53 message, but there may be no code
    # 52 messages.
    open_data = log_data.get(np.uint8(52), [])
    closed_data = log_data[np.uint8(53)]

    # If there were no valve open events, no water was dispensed. In this case, uses the first code 53 timestamp
    # to report zero-volume reward and ends the runtime early. If the valve was never opened, there were no
    # tones, so this shorts both tone-parsing and valve-parsing
    if not open_data:
        module_dataframe = pl.DataFrame(
            {
                "time_us": np.array([closed_data[0]["timestamp"]], dtype=np.uint64),
                "dispensed_water_volume_uL": np.array([0], dtype=np.float64),
                "tone_state": np.array([0], dtype=np.uint8),
            }
        )
        module_dataframe.write_ipc(file=output_directory.joinpath("valve_data.feather"), compression="lz4")
        return

    # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype. Although valve
    # trigger values are boolean, we translate them into the total volume of water, in microliters, dispensed to the
    # animal at each time-point and store that value as a float64.
    total_length = len(open_data) + len(closed_data)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    volume: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

    # The water is dispensed gradually while the valve stays open. Therefore, the full reward volume is dispensed
    # when the valve goes from open to closed. Based on calibration data, we have a conversion factor to translate
    # the time the valve remains open into the fluid volume dispensed to the animal, which we use to convert each
    # Open/Close cycle duration into the dispensed volume.

    # Extracts Open (Code 52) trigger codes. Statically assigns the value '1' to denote Open signals.
    n_on = len(open_data)
    timestamps[:n_on] = [value["timestamp"] for value in open_data]
    volume[:n_on] = np.uint8(1)  # All code 52 signals are Open (High)

    # Extracts Closed (Code 53) trigger codes.
    timestamps[n_on:] = [value["timestamp"] for value in closed_data]
    volume[n_on:] = np.uint8(0)  # All code 53 signals are Closed (Low)

    # Sorts both arrays based on timestamps.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    volume = volume[sort_indices]

    # Find falling and rising edges. Falling edges are valve closing events, rising edges are valve opening events.
    rising_edges = np.where((volume[:-1] == 0) & (volume[1:] == 1))[0] + 1
    falling_edges = np.where((volume[:-1] == 1) & (volume[1:] == 0))[0] + 1

    # Samples the timestamp array to only include timestamps for the falling edges. That is, when the valve has
    # finished delivering water
    reward_timestamps = timestamps[falling_edges]

    # Calculates pulse durations in microseconds for each open-close cycle. Since the original timestamp array
    # contains alternating HIGH / LOW edges, each falling edge has to match to a rising edge.
    pulse_durations: NDArray[np.float64] = (timestamps[falling_edges] - timestamps[rising_edges]).astype(np.float64)

    # Converts the time the Valve stayed open into the dispensed water volume, in microliters.
    # noinspection PyTypeChecker
    volumes: NDArray[np.float64] = np.round(
        np.cumsum(scale_coefficient * np.power(pulse_durations, nonlinearity_exponent)),
        decimals=8,
    )

    # The processing logic above removes the initial water volume of 0. This re-adds the initial volume using the
    # first timestamp of the module data. That timestamp communicates the initial valve state, which should be 0.
    reward_timestamps = np.insert(reward_timestamps, 0, timestamps[0])
    volumes = np.insert(volumes, 0, 0.0)

    # Now carries out similar processing for the Tone signals
    # Same logic as with code 52 applies to code 55
    tone_on_data = log_data.get(np.uint8(55), [])
    tone_off_data = log_data.get(np.uint8(56), [])  # The empty default is to appease mypy

    tone_length = len(tone_on_data) + len(tone_off_data)
    tone_timestamps: NDArray[np.uint64] = np.empty(tone_length, dtype=np.uint64)
    tone_states: NDArray[np.uint8] = np.empty(tone_length, dtype=np.uint8)

    # Extracts ON (Code 55) Tone codes. Statically assigns the value '1' to denote On signals.
    tone_on_n = len(tone_on_data)
    tone_timestamps[:tone_on_n] = [value["timestamp"] for value in tone_on_data]
    tone_states[:tone_on_n] = np.uint8(1)  # All code 55 signals are On (High)

    # Extracts Closed (Code 53) trigger codes.
    tone_timestamps[tone_on_n:] = [value["timestamp"] for value in tone_off_data]
    tone_states[tone_on_n:] = np.uint8(0)  # All code 56 signals are Off (Low)

    # Sorts both arrays based on timestamps.
    sort_indices = np.argsort(tone_timestamps)
    tone_timestamps = tone_timestamps[sort_indices]
    tone_states = tone_states[sort_indices]

    # If the last value is not 0, adds a zero-value to the end of the data sequence, one microsecond
    # after the last readout. This is to properly mark the end of the monitoring sequence.
    if tone_states[-1] != 0:
        tone_timestamps = np.append(tone_timestamps, tone_timestamps[-1] + 1)
        tone_states = np.append(tone_states, 0)

    # Constructs a shared array that includes all reward and tone timestamps. This will be used to interpolate tone
    # and timestamp values. Sorts the generated array to arrange all timestamps in monotonically ascending order
    shared_stamps = np.concatenate([tone_timestamps, reward_timestamps])
    sort_indices = np.argsort(shared_stamps)
    shared_stamps = shared_stamps[sort_indices]

    # Interpolates the reward volumes for each tone state and tone states for each reward volume.
    out_reward = _interpolate_data(
        timestamps=reward_timestamps, data=volumes, seed_timestamps=shared_stamps, is_discrete=True
    )
    out_tones = _interpolate_data(
        timestamps=tone_timestamps, data=tone_states, seed_timestamps=shared_stamps, is_discrete=True
    )

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": shared_stamps,
            "dispensed_water_volume_uL": out_reward,
            "tone_state": out_tones,
        }
    )

    # Saves the DataFrame to the output directory as a Feather file with lz4 compression
    module_dataframe.write_ipc(file=output_directory.joinpath("valve_data.feather"), compression="lz4")


def parse_lick_data(log_path: Path, output_directory: Path, lick_threshold: np.uint16) -> None:
    """Extracts and saves the data acquired by the break module during runtime as a .feather file.

    Args:
        log_path: The path to the .npz archive that stores the data logged by the module during runtime.
        output_directory: The path to the directory where to save the parsed data as a .feather file.
        lick_threshold: The voltage threshold for detecting the interaction with the sensor as a lick.

    Notes:
        The extraction automatically filters out non-lick events by applying the class lick-threshold value. The
        time-difference between consecutive ON and OFF event edges corresponds to the time, in microseconds, the
        tongue maintained contact with the lick tube. This may include both the time the tongue physically
        touched the tube and the time there was a conductive fluid bridge between the tongue and the lick tube.

        In addition to filtering out non-lick events, the code also converts multiple consecutive above-threshold or
        below-threshold readouts into LOW and HIGH epochs. Each HIGH epoch denotes the duration, for that particular
        lick, that the tongue maintained contact with the sensor. Each LOW epoch denotes the duration between licks
        that the tongue was not making contact with the sensor.
    """

    # Extracts data from the log file
    log_data = extract_logged_hardware_module_data(log_path=log_path, module_type=4, module_id=1)

    # LickModule only sends messages with code 51 (Voltage level changed). Therefore, this extraction pipeline has
    # to apply the threshold filter, similar to how the real-time processing method.

    # Unlike the other parsing methods, this one will always work as expected since it only deals with one code and
    # that code is guaranteed to be received for each runtime.

    # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype. Lick sensor
    # voltage levels come in as uint16, but we later replace them with binary uint8 1 and 0 values.
    voltage_data = log_data[np.uint8(51)]
    total_length = len(voltage_data)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    voltages: NDArray[np.uint16] = np.empty(total_length, dtype=np.uint16)

    # Extract timestamps and voltage levels
    timestamps[:] = [value["timestamp"] for value in voltage_data]
    voltages[:] = [value["data"] for value in voltage_data]

    # Converts voltage levels to binary lick states based on the class threshold. Note, the threshold is inclusive.
    licks = np.where(voltages >= lick_threshold, np.uint8(1), np.uint8(0))

    # Sorts all arrays by timestamp. This is technically not needed as the extracted values are already sorted by
    # timestamp, but this is still done for additional safety.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    licks = licks[sort_indices]

    # Finds indices where lick state changes (either 0->1 or 1->0)
    state_changes = np.where(licks[:-1] != licks[1:])[0] + 1

    # Extracts the state values and corresponding timestamps for each change
    state_stamps = timestamps[state_changes]
    states = licks[state_changes]

    # The transformation above removes the initial lick state (0). Re-adds the initial timestamp and state to
    # the output array
    timestamps = np.insert(state_stamps, 0, timestamps[0])
    states = np.insert(states, 0, licks[0])

    # If the last value is not 0, adds a zero-value to the end of the data sequence, one microsecond
    # after the last readout. This is to properly mark the end of the monitoring sequence.
    if states[-1] != 0:
        timestamps = np.append(timestamps, timestamps[-1] + 1)
        states = np.append(states, 0)

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": timestamps,
            "lick_state": states,
        }
    )

    # Saves the DataFrame to the output directory as a Feather file with lz4 compression
    module_dataframe.write_ipc(file=output_directory.joinpath("lick_data.feather"), compression="lz4")


def parse_torque_data(log_path: Path, output_directory: Path, torque_per_adc_unit: np.float64) -> None:
    """Extracts and saves the data acquired by the break module during runtime as a .feather file.

    Args:
        log_path: The path to the .npz archive that stores the data logged by the module during runtime.
        output_directory: The path to the directory where to save the parsed data as a .feather file.
        torque_per_adc_unit: The conversion actor used to translate ADC units recorded by the torque sensor into
            the torque in Newton centimeter, applied by the animal to the wheel.

    Notes:
        Despite this method trying to translate the detected torque into Newton centimeters, it may not be accurate.
        Partially, the accuracy of the translation depends on the calibration of the interface class, which is very
        hard with our current setup. The accuracy also depends on the used hardware, and currently our hardware is
        not very well suited for working with millivolt differential voltage levels used by the sensor to report
        torque. Therefore, currently, it is best to treat the torque data extracted from this module as a very rough
        estimate of how active the animal is at a given point in time.
    """

    # Extracts data from the log file
    log_data = extract_logged_hardware_module_data(log_path=log_path, module_type=6, module_id=1)

    # Here, we only look for event-codes 51 (CCW Torque) and event-codes 52 (CW Torque). CCW torque is interpreted
    # as torque in the positive direction, and CW torque is interpreted as torque in the negative direction.

    # Gets the data, defaulting to an empty list if the data is missing
    ccw_data = log_data.get(np.uint8(51), [])
    cw_data = log_data.get(np.uint8(52), [])

    # The way TorqueModule is implemented guarantees there is at least one CW code message with the displacement
    # of 0 that is received by the PC. In the worst case scenario, there will be no CCW codes and the parsing will
    # not work. To avoid that issue, we generate an artificial zero-code CCW value at the same timestamp + 1
    # microsecond as the original CW zero-code value. This does not affect the accuracy of our data, just makes the
    # code work for edge-cases.
    if not ccw_data:
        first_timestamp = cw_data[0]["timestamp"]
        ccw_data = [{"timestamp": first_timestamp + 1, "data": 0}]
    elif not cw_data:
        first_timestamp = ccw_data[0]["timestamp"]
        cw_data = [{"timestamp": first_timestamp + 1, "data": 0}]

    # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype. Although torque
    # values are uint16, we translate them into the actual torque applied by the animal in Newton centimeters and
    # store them as float 64 values.
    total_length = len(ccw_data) + len(cw_data)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    torques: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

    # Processes CCW torques (Code 51). CCW torque is interpreted as positive torque
    n_ccw = len(ccw_data)
    timestamps[:n_ccw] = [value["timestamp"] for value in ccw_data]  # Extracts timestamps for each value
    # The values are initially using the uint16 type. This converts them to float64 and translates from raw ADC
    # units into Newton centimeters.
    torques[:n_ccw] = [np.round(np.float64(value["data"]) * torque_per_adc_unit, decimals=8) for value in ccw_data]

    # Processes CW torques (Code 52). CW torque is interpreted as negative torque
    timestamps[n_ccw:] = [value["timestamp"] for value in cw_data]  # CW data just fills remaining space after CCW.
    torques[n_ccw:] = [np.round(-np.float64(value["data"]) * torque_per_adc_unit, decimals=8) for value in cw_data]

    # Sorts both arrays based on timestamps.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    torques = torques[sort_indices]

    # If the last value is not 0, adds a zero-value to the end of the data sequence, one microsecond
    # after the last readout. This is to properly mark the end of the monitoring sequence.
    if torques[-1] != 0:
        timestamps = np.append(timestamps, timestamps[-1] + 1)
        torques = np.append(torques, 0)

    # Replaces -0.0 values with 0.0. This is a convenience conversion to improve the visual appearance of numbers
    # to users
    torques = np.where(np.isclose(torques, -0.0) & (np.signbit(torques)), 0.0, torques)

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": timestamps,
            "mouse_torque_N_cm": torques,
        }
    )

    # Saves the DataFrame to the output directory as a Feather file with lz4 compression
    module_dataframe.write_ipc(file=output_directory.joinpath("torque_data.feather"), compression="lz4")


def parse_screen_data(log_path: Path, output_directory: Path, initially_on: bool) -> None:
    """Extracts and saves the data acquired by the break module during runtime as a .feather file.

    Args:
        log_path: The path to the .npz archive that stores the data logged by the module during runtime.
        output_directory: The path to the directory where to save the parsed data as a .feather file.
        initially_on: Communicates the initial state of the screen at module interface initialization. This is used
            to determine the state of the screens after each processed screen toggle signal.

    Notes:
        This extraction method works similar to the TTLModule method. This is intentional, as ScreenInterface is
        essentially a group of 3 TTLModules.
    """

    # Extracts data from the log file
    log_data = extract_logged_hardware_module_data(log_path=log_path, module_type=7, module_id=1)

    # Here, we only look for event-codes 52 (pulse ON) and event-codes 53 (pulse OFF).

    # The way the module is implemented guarantees there is at least one code 53 message. However, if screen state
    # is never toggled, there may be no code 52 messages.
    on_data = log_data.get(np.uint8(52), [])
    off_data = log_data[np.uint8(53)]

    # If there were no ON pulses, screens never changed state. In this case, shorts to returning the data for the
    # initial screen state using the initial Off timestamp. Otherwise, parses the data
    if not on_data:
        module_dataframe = pl.DataFrame(
            {
                "time_us": np.array([off_data[0]["timestamp"]], dtype=np.uint64),
                "screen_state": np.array([initially_on], dtype=np.uint8),
            }
        )
        module_dataframe.write_ipc(file=output_directory.joinpath("screen_data.feather"), compression="lz4")
        return

    # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype and the trigger
    # values are boolean. We use uint8 as it has the same memory footprint as a boolean and allows us to use integer
    # types across the entire dataset.
    total_length = len(on_data) + len(off_data)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    triggers: NDArray[np.uint8] = np.empty(total_length, dtype=np.uint8)

    # Extracts ON (Code 52) trigger codes. Statically assigns the value '1' to denote ON signals.
    n_on = len(on_data)
    timestamps[:n_on] = [value["timestamp"] for value in on_data]
    triggers[:n_on] = np.uint8(1)  # All code 52 signals are ON (High)

    # Extracts OFF (Code 53) trigger codes.
    timestamps[n_on:] = [value["timestamp"] for value in off_data]
    triggers[n_on:] = np.uint8(0)  # All code 53 signals are OFF (Low)

    # Sorts both arrays based on the timestamps, so that the data is in the chronological order.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    triggers = triggers[sort_indices]

    # Finds rising edges (where the signal goes from 0 to 1). Then uses the indices for such events to extract the
    # timestamps associated with each rising edge, before returning them to the caller.
    rising_edges = np.where((triggers[:-1] == 0) & (triggers[1:] == 1))[0] + 1
    screen_timestamps = timestamps[rising_edges]

    # Adds the initial state of the screen using the first recorded timestamp. The module is configured to send the
    # initial state of the relay (Off) during Setup, so the first recorded timestamp will always be 0 and correspond
    # to the initial state of the screen.
    screen_timestamps = np.concatenate(([timestamps[0]], screen_timestamps))

    # Builds an array of screen states. Starts with the initial screen state and then flips the state for each
    # consecutive timestamp matching a rising edge of the toggle pulse.
    screen_states = np.zeros(len(screen_timestamps), dtype=np.uint8)
    screen_states[0] = initially_on
    for i in range(1, len(screen_states)):
        screen_states[i] = 1 - screen_states[i - 1]  # Flips between 0 and 1

    # Creates a Polars DataFrame with the processed data
    module_dataframe = pl.DataFrame(
        {
            "time_us": screen_timestamps,
            "screen_state": screen_states,
        }
    )

    # Saves the DataFrame to the output directory as a Feather file with lz4 compression
    module_dataframe.write_ipc(file=output_directory.joinpath("screen_data.feather"), compression="lz4")


def _process_camera_timestamps(log_path: Path, output_path: Path) -> None:
    """Reads the log .npz archive specified by the log_path and extracts the camera frame timestamps
    as a Polars Series saved to the output_path as a Feather file.

    Args:
        log_path: Path to the .npz log archive to be parsed.
        output_path: Path to save the output Polars Series as a Feather file.
    """
    # Extracts timestamp data from log archive
    timestamp_data = extract_logged_video_system_data(log_path)

    # Converts extracted data to Polars series.
    timestamps_series = pl.Series(name="frame_time_us", values=timestamp_data)

    # Saves extracted data using Feather format and 'lz4' compression. Lz4 allows optimizing processing time and
    # file size. These extracted files are temporary and will be removed during later processing steps.
    timestamps_series.to_frame().write_ipc(file=output_path, compression="lz4")


def _process_experiment_data(log_path: Path, output_directory: Path, cue_map: dict[int, float]) -> None:
    """Extracts the VR states, Experiment states, and the Virtual Reality cue sequence from the log generated
    by the MesoscopeExperiment instance during runtime and saves the extracted data as Polars DataFrame .feather files.

    This extraction method functions similar to camera log extraction and hardware module log extraction methods. The
    key difference is the wall cue sequence extraction, which does not have a timestamp column. Instead, it has a
    distance colum which stores the distance the animal has to run, in centimeters, to reach each cue in the sequence.
    It is expected that during data processing, the distance data will be used to align the cues to the distance ran by
    the animal during the experiment.
    """
    # Loads the archive into RAM
    archive: NpzFile = np.load(file=log_path)

    # Precreates the variables used to store extracted data
    vr_states = []
    vr_timestamps = []
    experiment_states = []
    experiment_timestamps = []
    cue_sequence: NDArray[np.uint8] = np.zeros(shape=0, dtype=np.uint8)

    # Locates the logging onset timestamp. The onset is used to convert the timestamps for logged data into absolute
    # UTC timestamps. Originally, all timestamps other than onset are stored as elapsed time in microseconds
    # relative to the onset timestamp.
    timestamp_offset = 0
    onset_us = np.uint64(0)
    timestamp: np.uint64
    for number, item in enumerate(archive.files):
        message: NDArray[np.uint8] = archive[item]  # Extracts message payload from the compressed .npy file

        # Recovers the uint64 timestamp value from each message. The timestamp occupies 8 bytes of each logged
        # message starting at index 1. If timestamp value is 0, the message contains the onset timestamp value
        # stored as 8-byte payload. Index 0 stores the source ID (uint8 value)
        if np.uint64(message[1:9].view(np.uint64)[0]) == 0:
            # Extracts the byte-serialized UTC timestamp stored as microseconds since epoch onset.
            onset_us = np.uint64(message[9:].view("<i8")[0].copy())

            # Breaks the loop onc the onset is found. Generally, the onset is expected to be found very early into
            # the loop
            timestamp_offset = number  # Records the item number at which the onset value was found.
            break

    # Once the onset has been discovered, loops over all remaining messages and extracts data stored in these
    # messages.
    for item in archive.files[timestamp_offset + 1 :]:
        message = archive[item]

        # Extracts the elapsed microseconds since timestamp and uses it to calculate the global timestamp for the
        # message, in microseconds since epoch onset.
        elapsed_microseconds = np.uint64(message[1:9].view(np.uint64)[0].copy())
        timestamp = onset_us + elapsed_microseconds

        payload = message[9:]  # Extracts the payload from the message

        # If the message is longer than 500 bytes, it is a sequence of wall cues. It is very unlikely that we
        # will log any other data with this length, so it is a safe heuristic to use.
        if len(payload) > 500:
            cue_sequence = payload.view(np.uint8).copy()  # Keeps the original numpy uint8 format

        # If the message has a length of 2 bytes and the first element is 1, the message communicates the VR state
        # code.
        elif len(payload) == 2 and payload[0] == 1:
            vr_state = np.uint8(payload[1])  # Extracts the VR state code from the second byte of the message.
            vr_states.append(vr_state)
            vr_timestamps.append(timestamp)

        # Otherwise, if the starting code is 2, the message communicates the experiment state code.
        elif len(payload) == 2 and payload[0] == 2:
            # Extracts the experiment state code from the second byte of the message.
            experiment_state = np.uint8(payload[1])
            experiment_states.append(experiment_state)
            experiment_timestamps.append(timestamp)

    # Closes the archive to free up memory
    archive.close()

    # Uses the cue_map dictionary to compute the length of each cue in the sequence. Then computes the cumulative
    # distance the animal needs to travel to reach each cue in the sequence. The first cue is associated with distance
    # of 0 (the animal starts at this cue), the distance to each following cue is the sum of all previous cue lengths.
    distance_sequence = np.zeros(len(cue_sequence), dtype=np.float64)
    distance_sequence[1:] = np.cumsum([cue_map[int(code)] for code in cue_sequence[:-1]], dtype=np.float64)

    # Converts extracted data into Polar Feather files:
    vr_dataframe = pl.DataFrame(
        {
            "time_us": vr_timestamps,
            "vr_state": vr_states,
        }
    )
    exp_dataframe = pl.DataFrame(
        {
            "time_us": experiment_timestamps,
            "experiment_state": experiment_states,
        }
    )
    cue_dataframe = pl.DataFrame(
        {
            "vr_cue": cue_sequence,
            "traveled_distance_cm": distance_sequence,
        }
    )

    # Saves the DataFrames to Feather file with lz4 compression
    vr_dataframe.write_ipc(output_directory.joinpath("vr_data.feather"), compression="lz4")
    exp_dataframe.write_ipc(output_directory.joinpath("experiment_data.feather"), compression="lz4")
    cue_dataframe.write_ipc(output_directory.joinpath("cue_data.feather"), compression="lz4")


def process_log_directory(data_directory: Path, verbose: bool = False) -> None:
    """Reads the compressed .npz log files stored in the input directory and extracts all camera frame timestamps and
    relevant behavior data stored in log files.

    This function is intended to run on the BioHPC server as part of the data processing pipeline. It is optimized to
    process all log files in parallel and extract the data stored inside the files into behavior_data directory and
    camera_frames directory.

    Notes:
        This function makes certain assumptions about the layout of the raw-data directory to work as expected.

    Args:
        data_directory: The Path to the target session raw_data directory to be processed.
        verbose: Determines whether this function should run in the verbose mode.
    """
    # Resolves the paths to the specific directories used during processing
    log_directory = data_directory.joinpath("behavior_data_log")  # Should exist inside the raw data directory
    camera_frame_directory = data_directory.joinpath("camera_frames")  # Should exist inside the raw data directory
    behavior_data_directory = data_directory.joinpath("behavior_data")
    ensure_directory_exists(behavior_data_directory)  # Generates the directory

    # Should exist inside the raw data directory
    hardware_configuration_path = data_directory.joinpath("hardware_configuration.yaml")

    # Finds all .npz log files inside the input log file directory. Assumes there are no uncompressed log files.
    compressed_files: list[Path] = [file for file in log_directory.glob("*.npz")]

    # Loads the input HardwareConfiguration file to read the hardware parameters necessary to parse the data
    hardware_configuration: RuntimeHardwareConfiguration = RuntimeHardwareConfiguration.from_yaml(  # type: ignore
        file_path=hardware_configuration_path,
    )

    # Otherwise, iterates over all compressed log files and processes them in-parallel
    with ProcessPoolExecutor() as executor:
        futures = set()
        for file in compressed_files:
            # MesoscopeExperiment log file
            if file.stem == "1_log" and hardware_configuration.cue_map is not None:
                futures.add(
                    executor.submit(
                        _process_experiment_data,
                        file,
                        behavior_data_directory,
                        hardware_configuration.cue_map,
                    )
                )

            # Face Camera timestamps
            if file.stem == "51_log":
                futures.add(
                    executor.submit(
                        _process_camera_timestamps,
                        file,
                        camera_frame_directory.joinpath("face_camera_timestamps.feather"),
                    )
                )

            # Left Camera timestamps
            if file.stem == "62_log":
                futures.add(
                    executor.submit(
                        _process_camera_timestamps,
                        file,
                        camera_frame_directory.joinpath("left_camera_timestamps.feather"),
                    )
                )

            # Right Camera timestamps
            if file.stem == "73_log":
                futures.add(
                    executor.submit(
                        _process_camera_timestamps,
                        file,
                        camera_frame_directory.joinpath("right_camera_timestamps.feather"),
                    )
                )

            # Actor AMC module data
            if file.stem == "101_log":
                # Break
                if (
                    hardware_configuration.minimum_break_strength is not None
                    and hardware_configuration.maximum_break_strength is not None
                ):
                    futures.add(
                        executor.submit(
                            parse_break_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.minimum_break_strength,
                            hardware_configuration.maximum_break_strength,
                        )
                    )

                # Valve
                if (
                    hardware_configuration.nonlinearity_exponent is not None
                    and hardware_configuration.scale_coefficient is not None
                ):
                    futures.add(
                        executor.submit(
                            parse_valve_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.scale_coefficient,
                            hardware_configuration.nonlinearity_exponent,
                        )
                    )

                # Screens
                if hardware_configuration.initially_on is not None:
                    futures.add(
                        executor.submit(
                            parse_screen_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.initially_on,
                        )
                    )

            # Sensor AMC module data
            if file.stem == "152_log":
                # Lick Sensor
                if hardware_configuration.lick_threshold is not None:
                    futures.add(
                        executor.submit(
                            parse_lick_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.lick_threshold,
                        )
                    )

                # Torque Sensor
                if hardware_configuration.torque_per_adc_unit is not None:
                    futures.add(
                        executor.submit(
                            parse_torque_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.torque_per_adc_unit,
                        )
                    )

                # Mesoscope Frame TTL module
                if hardware_configuration.has_ttl:
                    futures.add(executor.submit(parse_ttl_data, file, behavior_data_directory))

            # Encoder AMC module data
            if file.stem == "203_log":
                # Encoder
                if hardware_configuration.cm_per_pulse is not None:
                    futures.add(
                        executor.submit(
                            parse_encoder_data,
                            file,
                            behavior_data_directory,
                            hardware_configuration.cm_per_pulse,
                        )
                    )

        # Displays a progress bar to track the parsing status if the function is called in the verbose mode.
        if verbose:
            with tqdm(
                total=len(futures),
                desc=f"Parsing log sources",
                unit="source",
            ) as pbar:
                for future in as_completed(futures):
                    # Propagates any exceptions from the transfers
                    future.result()
                    pbar.update(1)
        else:
            for future in as_completed(futures):
                # Propagates any exceptions from the transfers
                future.result()
