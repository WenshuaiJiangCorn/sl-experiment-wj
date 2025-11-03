import time
from pathlib import Path

import numpy as np
import keyboard
import polars as pl

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
        timestamps = extract_logged_camera_timestamps(
            log_path=log_path
            )

        # Saves the extracted timestamps to a .feather file
        timestamp_array = np.array(timestamps, dtype=np.uint64)
        timestamp_dataframe = pl.DataFrame({"time_us": timestamp_array})
        timestamp_dataframe.write_ipc(
            file=output_file
        )

        # Computes and prints the frame rate of the camera based on the extracted frame timestamp data.
        time_diffs = np.diff(timestamp_array)
        fps = 1 / (np.mean(time_diffs) / 1e6)

        return fps
        

    def extract_video_time_stamps(self, output_directory: Path) -> None:
        """Extracts and save time stamps of each frame for all cameras, computes the frame rates of 
           the interfaced cameras based on logged timestamp data."""

        console.echo(f"Extracting frame acquisition timestamps from the assembled log archive...")
        fps_top = self._save_time_stamps(
            log_path=self._data_logger.output_directory.joinpath(f"101_log.npz"),
            output_file=output_directory / "top_camera_timestamps.feather"
            )
        
        fps_left = self._save_time_stamps(
            log_path=self._data_logger.output_directory.joinpath(f"102_log.npz"),
            output_file=output_directory / "left_camera_timestamps.feather"
            )
        
        fps_right = self._save_time_stamps(
            log_path=self._data_logger.output_directory.joinpath(f"103_log.npz"),
            output_file=output_directory / "right_camera_timestamps.feather"
            )

        console.echo(
            message=(
                f"According to the extracted timestamps, the interfaced cameras had acquisition frame rates of: "
                f"Top camera {fps_top:.2f} frames / second"
                f"Left camera {fps_left:.2f} frames / second"
                f"Right camera {fps_right:.2f} frames / second"
                f"Time stamps saved."
            ),
            level=LogLevel.SUCCESS,
        )
