from pathlib import Path

import numpy as np
import keyboard
import polars as pl
import tempfile

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger
from ataraxis_time import PrecisionTimer

from microcontroller import AMCInterface
from visualizers import BehaviorVisualizer

from ataraxis_video_system import (VideoSystem, 
                                   VideoEncoders, 
                                   CameraInterfaces, 
                                   EncoderSpeedPresets, 
                                   extract_logged_camera_timestamps)


class VideoSystems:
    """Container class for managing 3 VideoSystem instances.
    Arags:
        data_logger (DataLogger): DataLogger instance for logging video timestamps.
        output_directory (Path): Directory where video frames and logs will be saved.
    """

    def __init__(self, data_logger: DataLogger, output_directory: Path):
        self._cameras_started = False
        self._data_logger = data_logger

        self._left_camera = VideoSystem(
            system_id=np.uint8(101),
            data_logger=data_logger,
            output_directory=output_directory,
            camera_interface=CameraInterfaces.OPENCV,  # OpenCV interface for webcameras
            camera_index=0,  # Uses the default system webcam
            display_frame_rate=15,  # Displays the acquired data at a rate of 15 frames per second
            frame_width=640,
            frame_height=360,
            color=False,  # Acquires images in MONOCHROME mode
            video_encoder=VideoEncoders.H264,  # Uses H264 CPU video encoder.
            encoder_speed_preset=EncoderSpeedPresets.FASTER,
            quantization_parameter=25,  # Increments the default qp parameter to reflect using the H264 encoder.
        )

        self._top_camera = VideoSystem(
            system_id=np.uint8(102),
            data_logger=data_logger,
            output_directory=output_directory,
            camera_interface=CameraInterfaces.OPENCV,  # OpenCV interface for webcameras
            camera_index=1,  # Uses the default system webcam
            display_frame_rate=15,
            frame_width=1280,
            frame_height=720,
            frame_rate=30,  # Uses 30 FPS for acquisition
            color=False,  # Acquires images in MONOCHROME mode
            video_encoder=VideoEncoders.H264,  # Uses H264 CPU video encoder.
            encoder_speed_preset=EncoderSpeedPresets.SLOW,
            quantization_parameter=25,  # Increments the default qp parameter to reflect using the H264 encoder.
        )

        self._right_camera = VideoSystem(
            system_id=np.uint8(103),
            data_logger=data_logger,
            output_directory=output_directory,
            camera_interface=CameraInterfaces.OPENCV,  # OpenCV interface for webcameras
            camera_index=2,  # Uses the default system webcam
            display_frame_rate=15,  # Displays the acquired data at a rate of 30 frames per second
            frame_width=640,
            frame_height=360,
            color=False,  # Acquires images in MONOCHROME mode
            video_encoder=VideoEncoders.H264,  # Uses H264 CPU video encoder.
            encoder_speed_preset=EncoderSpeedPresets.FASTER,
            quantization_parameter=25,  # Increments the default qp parameter to reflect using the H264 encoder.
        )

    def start(self) -> None:
        """Starts all VideoSystem instances."""
        if self._cameras_started:
            console.echo("VideoSystems: Cameras have already been started.", level=LogLevel.WARNING)
            return

        console.echo("VideoSystems: Starting cameras...", level=LogLevel.INFO)
        self._top_camera.start()
        self._left_camera.start()
        self._right_camera.start()
        console.echo("VideoSystems: All cameras started.", level=LogLevel.SUCCESS)

        console.echo("Start saving the acquired frames to disk...", level=LogLevel.INFO)
        self._top_camera.start_frame_saving()
        self._left_camera.start_frame_saving()
        self._right_camera.start_frame_saving()

        self._cameras_started = True

    def stop(self) -> None:
        """Stops all VideoSystem instances."""
        console.echo("Stop saving the acquired frames to disk...", level=LogLevel.INFO)
        self._top_camera.stop_frame_saving()
        self._left_camera.stop_frame_saving()
        self._right_camera.stop_frame_saving()

        console.echo("VideoSystems: Camera frame saving stopped. Terminating the cameras...", level=LogLevel.INFO)
        self._top_camera.stop()
        self._left_camera.stop()
        self._right_camera.stop()
        self._cameras_started = False
        console.echo("VideoSystems: All cameras terminated.", level=LogLevel.SUCCESS)

    def _save_time_stamps(self, log_path: Path, output_file: Path):
        """Extracts and save time stamps of each frame, computes the frame rates of
        the interfaced cameras based on logged timestamp data.

        Args:
            log_path (Path): The path to the assembled log archive (.npz file) containing the logged data.

        Notes:
            This method save the extracted timestamps to disk and returns the computed frame rates.

        Returns:
             fps (np.float64): The computed frame rate based on the extracted timestamps.
        """
        timestamps = extract_logged_camera_timestamps(log_path=log_path)

        # Saves the extracted timestamps to a .feather file
        timestamp_array = np.array(timestamps, dtype=np.uint64)
        timestamp_dataframe = pl.DataFrame({"time_us": timestamp_array})
        timestamp_dataframe.write_ipc(file=output_file)

        # Computes and prints the frame rate of the camera based on the extracted frame timestamp data.
        time_diffs = np.diff(timestamp_array)
        fps = 1 / (np.mean(time_diffs) / 1e6)

        return fps

    def extract_video_time_stamps(self, output_directory: Path) -> None:
        """Extracts and save time stamps of each frame for all cameras, computes the frame rates of
        the interfaced cameras based on logged timestamp data.
        """
        console.echo("Extracting frame acquisition timestamps from the assembled log archive...")
        fps_top = self._save_time_stamps(
            log_path=self._data_logger.output_directory.joinpath("102_log.npz"),
            output_file=output_directory / "top_camera_timestamps.feather",
        )

        fps_left = self._save_time_stamps(
            log_path=self._data_logger.output_directory.joinpath("101_log.npz"),
            output_file=output_directory / "left_camera_timestamps.feather",
        )

        fps_right = self._save_time_stamps(
            log_path=self._data_logger.output_directory.joinpath("103_log.npz"),
            output_file=output_directory / "right_camera_timestamps.feather",
        )

        console.echo(
            message=(
                f"According to the extracted timestamps, the interfaced cameras had acquisition frame rates of:\n "
                f"Top camera has {fps_top:.2f} frames / second\n"
                f"Left camera has {fps_left:.2f} frames / second\n"
                f"Right camera has {fps_right:.2f} frames / second\n"
                f"Time stamps saved."
            ),
            level=LogLevel.SUCCESS,
        )


class LinearTrackFunctions:
    """Manages toggle, calibration and training logic of YLab linear track experiments.

        Arags:
            data_logger (DataLogger): DataLogger instance for logging experiment data.
    """

    def __init__(self, data_logger: DataLogger | None = None):
        if not console.enabled:
            console.enable()

        if data_logger is None:
            with tempfile.TemporaryDirectory(delete=False) as temp_dir_path:
                output_dir = Path(temp_dir_path).joinpath("test_output")
            self.data_logger = DataLogger(output_directory=output_dir, instance_name="temp_logger")
        else:
            self.data_logger = data_logger

        self.mc = AMCInterface(data_logger=self.data_logger)
        self.vs = VideoSystems(data_logger=self.data_logger, output_directory=self.data_logger.output_directory)
        self.visualizer = BehaviorVisualizer()

        console.echo(self.mc._controller._port)
        
        
    def _check_side(self, valve_side: str):
        """Check and return the valve object based on the specified side.

        Args:
            valve_side (str): The side of the valve ("left" or "right").
            
        Returns:
            Valve object corresponding to the specified side.
        """

        if valve_side == "left":
            return self.mc.left_valve
        elif valve_side == "right":
            return self.mc.right_valve
        else:
            console.echo("Invalid valve side specified.", level=LogLevel.ERROR)
            raise ValueError("Invalid valve side specified.")


    def _start(self):
        """Starts the microcontroller and connects to SharedMemoryArray. Must be called before any operation."""
        self.data_logger.start()
        self.mc.start()
        self.mc.connect_to_smh()


    def _stop(self):
        """Stops the microcontroller and disconnects from SharedMemoryArray. Must be called after any operation."""
        self.mc.disconnect_to_smh()
        self.mc.stop()
        self.data_logger.stop()


    def open_valve(self, valve_side: str, duration: int = 1) -> None:
        """Open a specific valve for a specific duration.

        Args:
            valve_side (str): The side of the valve to toggle ("left" or "right").
            duration (int): The desired state of the valve (True for open, False for closed).
        """
        valve = self._check_side(valve_side)

        try:
            self._start()

            timer = PrecisionTimer("s")
            console.echo(f"Open {valve_side} valve for {duration} seconds.", level=LogLevel.SUCCESS)
            valve.toggle(state=True)
            timer.delay(duration, block=True)

        finally:
            valve.toggle(state=False)
            self._stop()
            console.echo("Valve: closed.", level=LogLevel.SUCCESS)


    def calibrate_valve(self, valve_side, calibration_pulse_duration) -> None:
        """Calibrates the valve by sending a pulse of specified duration."""

        valve = self._check_side(valve_side)

        try:
            self._start()

            console.echo("Calibration starts")
            valve.calibrate(calibration_pulse_duration)

        finally:
            valve.toggle(state=False)
            self._stop()
            console.echo("Calibration: ended.", level=LogLevel.SUCCESS)


    def delivery_test(self, valve_side) -> None:
        """Delivers a specified volume (default 15uL) of fluid 40 times (same amount as second day testing) 
           through the specified valve to test dispensing."""

        valve = self._check_side(valve_side)
        timer = PrecisionTimer("s")

        try:
            self._start()
            console.echo("Delivery test starts")
            for n in range(40):
                valve.dispense_volume(volume=np.float64(15))
                if n//10 % 1:
                    console.echo(f"{n + 1} deliveries")
                timer.delay(3)

        finally:
            self._stop()
            console.echo("Delivery test: ended.", level=LogLevel.SUCCESS)

    
    def first_day_training(self) -> None:
        """Executes the first day training protocol for the linear track experiment.
           Only use right valve and camera, deliver water manually by pressing "r" key.
        """

        delivery_num = 0

        try:
            self._start()
            self.vs._right_camera.start() # Start only the right camera
            self.visualizer.open()  # Open the visualizer window
            console.echo("First day training started, press 'r' to deliver water, press 'q' to quit.")

            while not keyboard.is_pressed("q"):
                self.visualizer.update()

                if keyboard.is_pressed("r"):
                    # Deliver 10uL per manual trigger. It is not accurate since 
                    # calibration data is not from testing chamber, the command of delivering
                    # 15uL water actually delivers 10uL. 
                    self.mc.right_valve.dispense_volume(volume=np.float64(15)) 
                    delivery_num += 1
                    self.visualizer.add_right_valve_event()

                    console.echo(f"Water delivered manually, total deliveries: {delivery_num}")

                timer = PrecisionTimer("ms")
                timer.delay(delay=30, block=False)  # 10ms delay to prevent CPU overuse
        
        finally:
            total_volume = delivery_num * 11 # Convert to actual delivered volume in uL
            self.vs._right_camera.stop() # Stop only the right camera
            self.visualizer.close()
            self._stop()
            console.echo(f"First day training: ended. Total dispensed volume: {total_volume:.2f} uL", level=LogLevel.SUCCESS)
            
            
    def second_day_training(self) -> None:
        """Executes the second day training protocol for the linear track experiment.
           Deliver water every minute for 30 minutes.
        """

        cycle_num = 0

        try:
            self._start()
            self.vs._right_camera.start()  # Start right camera
            console.echo("Second day training started")

            
            timer = PrecisionTimer("s")

            while cycle_num < 40:
                # 45s delay
                timer.delay(delay=45, block=False)

                # Deliver 10uL per manual trigger. It is not accurate since 
                # calibration data is not from testing chamber, the command of delivering
                # 15uL water actually delivers 10uL. 
                self.mc.right_valve.dispense_volume(volume=np.float64(15))
                console.echo(f"Cycle {cycle_num + 1}: Delivered water through the right valve.")

                cycle_num += 1

        finally:
            total_volume = cycle_num * 11 # Convert to actual delivered volume in uL
            self.vs._right_camera.stop()  # Stop right camera
            self.visualizer.close()
            self._stop()
            console.echo(f"Second day training: ended. Total dispensed volume: {total_volume:.2f} uL", level=LogLevel.SUCCESS)