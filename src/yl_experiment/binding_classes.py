import time
from pathlib import Path

import numpy as np
import keyboard
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger
from ataraxis_video_system import VideoSystem, VideoEncoders, CameraInterfaces, extract_logged_camera_timestamps


class VideoSystems:
    """Container class for managing multiple VideoSystem instances.
    Arags:
        data_logger (DataLogger): DataLogger instance for logging video timestamps.
        output_directory (Path): Directory where video frames and logs will be saved."""
    
    def __init__(self, data_logger: DataLogger, output_directory: Path):
        self._cameras_started = False
        self._data_logger = data_logger

        self._top_camera = VideoSystem(
            system_id=np.uint8(101),
            data_logger=data_logger,
            output_directory=output_directory,
            camera_interface=CameraInterfaces.OPENCV,  # OpenCV interface for webcameras
            camera_index=0,  # Uses the default system webcam
            display_frame_rate=15,
            frame_width=1280,
            frame_height=720,
            frame_rate=30, # Uses 30 FPS for acquisition
            color=False,  # Acquires images in MONOCHROME mode
            video_encoder=VideoEncoders.H264,  # Uses H264 CPU video encoder.
            quantization_parameter=25,  # Increments the default qp parameter to reflect using the H264 encoder.
        )

        self._left_camera = VideoSystem(
            system_id=np.uint8(102),
            data_logger=data_logger,
            output_directory=output_directory,
            camera_interface=CameraInterfaces.OPENCV,  # OpenCV interface for webcameras
            camera_index=1,  # Uses the default system webcam
            display_frame_rate=15, # Displays the acquired data at a rate of 15 frames per second
            frame_width=800,
            frame_height=600,
            color=False,  # Acquires images in MONOCHROME mode
            video_encoder=VideoEncoders.H264,  # Uses H264 CPU video encoder.
            quantization_parameter=25,  # Increments the default qp parameter to reflect using the H264 encoder.
        )

        self._right_camera = VideoSystem(
            system_id=np.uint8(103),
            data_logger=data_logger,
            output_directory=output_directory,
            camera_interface=CameraInterfaces.OPENCV,  # OpenCV interface for webcameras
            camera_index=2,  # Uses the default system webcam
            display_frame_rate=15,  # Displays the acquired data at a rate of 30 frames per second
            frame_width=800,
            frame_height=600,
            color=False,  # Acquires images in MONOCHROME mode
            video_encoder=VideoEncoders.H264,  # Uses H264 CPU video encoder.
            quantization_parameter=25,  # Increments the default qp parameter to reflect using the H264 encoder.
        )

    def start(self) -> None:
        """Starts all VideoSystem instances."""

        if self._cameras_started:
            console.echo(f"VideoSystems: Cameras have already been started.", level=LogLevel.WARNING)
            return

        console.echo(f"VideoSystems: Starting cameras...", level=LogLevel.INFO)
        self._top_camera.start()
        self._left_camera.start()
        self._right_camera.start()
        console.echo(f"VideoSystems: All cameras started.", level=LogLevel.SUCCESS)

        console.echo(f"Start saving the acquired frames to disk...", level=LogLevel.INFO)
        self._top_camera.start_frame_saving()
        self._left_camera.start_frame_saving()
        self._right_camera.start_frame_saving()
        
        self._cameras_started = True


    def stop(self) -> None:
        """Stops all VideoSystem instances."""

        console.echo(f"Stop saving the acquired frames to disk...", level=LogLevel.INFO)
        self._top_camera.stop_frame_saving()
        self._left_camera.stop_frame_saving()
        self._right_camera.stop_frame_saving()
        
        console.echo(f"VideoSystems: Camera frame saving stopped. Terminating the cameras...", level=LogLevel.INFO)
        self._top_camera.stop()
        self._left_camera.stop()
        self._right_camera.stop()
        self._cameras_started = False
        console.echo(f"VideoSystems: All cameras terminated.", level=LogLevel.SUCCESS)

    
    def show_frame_rates(self) -> None:
        """Extracts and computes the frame rates of the interfaced cameras based on logged timestamp data."""

        console.echo(f"Extracting frame acquisition timestamps from the assembled log archive...")
        timestamps1 = extract_logged_camera_timestamps(log_path=self._data_logger.output_directory.joinpath(f"101_log.npz"))
        timestamps2 = extract_logged_camera_timestamps(log_path=self._data_logger.output_directory.joinpath(f"102_log.npz"))
        timestamps3 = extract_logged_camera_timestamps(log_path=self._data_logger.output_directory.joinpath(f"103_log.npz"))

        # Computes and prints the frame rate of the camera based on the extracted frame timestamp data.
        timestamp_array1 = np.array(timestamps1, dtype=np.uint64)
        time_diffs1 = np.diff(timestamp_array1)
        fps1 = 1 / (np.mean(time_diffs1) / 1e6)

        timestamp_array2 = np.array(timestamps2, dtype=np.uint64)
        time_diffs2 = np.diff(timestamp_array2)
        fps2 = 1 / (np.mean(time_diffs2) / 1e6)

        timestamp_array3 = np.array(timestamps3, dtype=np.uint64)
        time_diffs3 = np.diff(timestamp_array3)
        fps3 = 1 / (np.mean(time_diffs3) / 1e6)

        console.echo(
            message=(
                f"According to the extracted timestamps, the interfaced cameras had acquisition frame rates of: "
                f"Top camera {fps1:.2f} frames / second"
                f"Left camera {fps2:.2f} frames / second"
                f"Right camera {fps3:.2f} frames / second"
            ),
            level=LogLevel.SUCCESS,
        )

