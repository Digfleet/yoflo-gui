import argparse
import logging
import os
import threading
import time
import sys
import cv2
import torch
import hid  # For PTZ camera HID
import msvcrt  # For Windows-specific PTZ control with arrow keys
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download, hf_hub_download
from colorama import Fore, Style, init
import tkinter as tk
from tkinter import filedialog, simpledialog, Toplevel
import numpy as np

# ---------------------------------------------------------------------------
# The following classes and functions are adapted from the CLI version,
# integrated into a single GUI script. This merges the full set of features:
#   - Logging setup
#   - PTZ tracking & PTZ camera control
#   - ModelManager for local/remote (Hugging Face) model loading
#   - RecordingManager for object-detection-based or expression-based recording
#   - ImageUtils for bounding-box plotting & screenshot saving
#   - AlertLogger for logging alerts to a file
#   - PTZController for HID-based PTZ operations
#   - PTZTracker for autonomous camera tracking
#   - Full YO_FLO GUI class that merges all features from the older GUI
#     version with the new functionalities.
# ---------------------------------------------------------------------------

init(autoreset=True)

def setup_logging(log_to_file, log_file_path="alerts.log"):
    """
    Sets up the logging configuration for the entire application.
    If log_to_file is True, messages will also be written to a specified file.

    :param log_to_file: Boolean indicating whether to also log to a file.
    :param log_file_path: The path where the log file will be written.
    """
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(log_file_path))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)


class PTZController:
    """
    Class to control PTZ camera movements via HID commands.
    It locates a suitable HID device for the PTZ camera based on given vendor and product IDs.
    """
    def __init__(self, vendor_id=0x046D, product_id=0x085F, usage_page=65280, usage=1):
        """
        Initializes the PTZController by attempting to open a HID device matching the given parameters.

        :param vendor_id: The USB vendor ID of the PTZ device.
        :param product_id: The USB product ID of the PTZ device.
        :param usage_page: The HID usage page number.
        :param usage: The HID usage number.
        """
        self.device = None
        try:
            ptz_path = None
            for d in hid.enumerate(vendor_id, product_id):
                if d['usage_page'] == usage_page and d['usage'] == usage:
                    ptz_path = d['path']
                    break
            if ptz_path:
                self.device = hid.device()
                self.device.open_path(ptz_path)
                print("PTZ HID interface opened successfully.")
            else:
                print("No suitable PTZ HID interface found. PTZ commands may not work.")
        except IOError as e:
            print(f"Error opening PTZ device: {e}")
        except Exception as e:
            print(f"Unexpected error during PTZ device initialization: {e}")

    def send_command(self, report_id, value):
        """
        Sends a command to the PTZ device via HID write.

        :param report_id: The report ID for the PTZ control.
        :param value: The value that represents the specific command (e.g., pan left/right, tilt up/down).
        """
        if not self.device:
            print("PTZ Device not initialized.")
            return
        command = [report_id & 0xFF, value] + [0x00] * 30
        try:
            self.device.write(command)
            print(f"Command sent: report_id={report_id}, value={value}")
            time.sleep(0.2)
        except IOError as e:
            print(f"Error sending PTZ command: {e}")
        except Exception as e:
            print(f"Unexpected error sending PTZ command: {e}")

    def pan_right(self):
        """Pans the camera to the right."""
        self.send_command(0x0B, 0x02)

    def pan_left(self):
        """Pans the camera to the left."""
        self.send_command(0x0B, 0x03)

    def tilt_up(self):
        """Tilts the camera upward."""
        self.send_command(0x0B, 0x00)

    def tilt_down(self):
        """Tilts the camera downward."""
        self.send_command(0x0B, 0x01)

    def zoom_in(self):
        """Zooms the camera in."""
        self.send_command(0x0B, 0x04)

    def zoom_out(self):
        """Zooms the camera out."""
        self.send_command(0x0B, 0x05)

    def close(self):
        """
        Closes the HID device handle, if open, to release system resources.
        """
        if self.device:
            try:
                self.device.close()
                print("PTZ device closed successfully.")
            except Exception as e:
                print(f"Error closing PTZ device: {e}")


class PTZTracker:
    """
    Autonomous PTZ tracking class. Keeps a specified object centered and at a desired size.

    This class adjusts camera pan, tilt, and zoom automatically to keep
    the detected object within a certain bounding box ratio, or "zoom level."
    """
    def __init__(
        self,
        camera,
        desired_ratio=0.20,
        zoom_tolerance=0.4,
        pan_tilt_tolerance=25,
        pan_tilt_interval=0.75,
        zoom_interval=0.5,
        smoothing_factor=0.2,
        max_consecutive_errors=5,
    ):
        """
        Initializes the PTZTracker with various parameters controlling behavior.

        :param camera: A camera object that supports PTZ commands.
        :param desired_ratio: Desired fraction of the frame the object should occupy.
        :param zoom_tolerance: The tolerance around the desired_ratio before zooming in/out.
        :param pan_tilt_tolerance: Pixel difference from center before panning/tilting.
        :param pan_tilt_interval: Minimum time (in seconds) between pan/tilt commands.
        :param zoom_interval: Minimum time (in seconds) between zoom commands.
        :param smoothing_factor: Weight for exponential smoothing of bounding box size.
        :param max_consecutive_errors: Maximum camera command errors before deactivation.
        """
        if not (0 < smoothing_factor < 1):
            raise ValueError("smoothing_factor must be between 0 and 1.")
        if desired_ratio <= 0 or desired_ratio >= 1:
            raise ValueError("desired_ratio should be between 0 and 1.")
        if zoom_tolerance < 0:
            raise ValueError("zoom_tolerance must be >= 0.")
        if pan_tilt_tolerance < 0:
            raise ValueError("pan_tilt_tolerance must be >= 0.")
        if pan_tilt_interval <= 0 or zoom_interval <= 0:
            raise ValueError("Intervals must be positive.")
        if max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be at least 1.")

        self.camera = camera
        self.desired_ratio = desired_ratio
        self.zoom_tolerance = zoom_tolerance
        self.pan_tilt_tolerance = pan_tilt_tolerance
        self.pan_tilt_interval = pan_tilt_interval
        self.zoom_interval = zoom_interval
        self.smoothing_factor = smoothing_factor
        self.max_consecutive_errors = max_consecutive_errors

        self.last_pan_tilt_adjust = 0.0
        self.last_zoom_adjust = 0.0
        self.smoothed_width = None
        self.smoothed_height = None
        self.active = False
        self.consecutive_errors = 0

    def activate(self, active=True):
        """
        Activate or deactivate PTZ tracking. When deactivated, tracking resets smoothing and error counters.
        """
        self.active = active
        if not active:
            self.smoothed_width = None
            self.smoothed_height = None
            self.consecutive_errors = 0

    def adjust_camera(self, bbox, frame_width, frame_height):
        """
        Adjusts camera pan, tilt, and zoom to keep the object bounding box centered and sized per desired_ratio.

        :param bbox: A tuple (x1, y1, x2, y2) representing the object bounding box coordinates.
        :param frame_width: The width of the current frame in pixels.
        :param frame_height: The height of the current frame in pixels.
        """
        if not self.active:
            return

        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            print("Invalid bbox coordinates; skipping camera adjustment.")
            return

        bbox_width = (x2 - x1)
        bbox_height = (y2 - y1)

        if self.smoothed_width is None:
            self.smoothed_width = bbox_width
            self.smoothed_height = bbox_height
        else:
            self.smoothed_width = (
                self.smoothing_factor * bbox_width
                + (1 - self.smoothing_factor) * self.smoothed_width
            )
            self.smoothed_height = (
                self.smoothing_factor * bbox_height
                + (1 - self.smoothing_factor) * self.smoothed_height
            )

        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

        desired_width = frame_width * self.desired_ratio
        desired_height = frame_height * self.desired_ratio

        min_width = desired_width * (1 - self.zoom_tolerance)
        max_width = desired_width * (1 + self.zoom_tolerance)
        min_height = desired_height * (1 - self.zoom_tolerance)
        max_height = desired_height * (1 + self.zoom_tolerance)

        current_time = time.time()

        # Handle Pan/Tilt
        if (current_time - self.last_pan_tilt_adjust) >= self.pan_tilt_interval:
            dx = bbox_center_x - frame_center_x
            dy = bbox_center_y - frame_center_y

            pan_tilt_moved = False
            if abs(dx) > self.pan_tilt_tolerance:
                pan_tilt_moved = (
                    self._safe_camera_command("pan_left" if dx < 0 else "pan_right")
                    or pan_tilt_moved
                )
            if abs(dy) > self.pan_tilt_tolerance:
                pan_tilt_moved = (
                    self._safe_camera_command("tilt_up" if dy < 0 else "tilt_down")
                    or pan_tilt_moved
                )

            if pan_tilt_moved:
                self.last_pan_tilt_adjust = current_time

        # Handle Zoom
        if (current_time - self.last_zoom_adjust) >= self.zoom_interval:
            width_too_small = self.smoothed_width < min_width
            height_too_small = self.smoothed_height < min_height
            width_too_large = self.smoothed_width > max_width
            height_too_large = self.smoothed_height > max_height

            zoom_moved = False
            if width_too_small or height_too_small:
                zoom_moved = self._safe_camera_command("zoom_in")
            elif width_too_large or height_too_large:
                zoom_moved = self._safe_camera_command("zoom_out")

            if zoom_moved:
                self.last_zoom_adjust = current_time

        if self.consecutive_errors >= self.max_consecutive_errors:
            print("Too many consecutive camera errors, deactivating PTZ tracking.")
            self.activate(False)

    def _safe_camera_command(self, command):
        """
        Safely invokes a camera command, handling exceptions and counting errors.
        """
        if not hasattr(self.camera, command):
            print(f"Camera does not support command '{command}'.")
            return False
        try:
            method = getattr(self.camera, command)
            method()
            self.consecutive_errors = 0
            return True
        except Exception as e:
            self.consecutive_errors += 1
            print(f"Error executing camera command '{command}': {e}")
            return False


class ModelManager:
    """
    Class responsible for loading and managing a Hugging Face Transformer model and processor,
    with optional quantization settings.
    """
    def __init__(self, device, quantization=None):
        """
        Initialize the ModelManager with a torch device and an optional quantization setting.

        :param device: Torch device, e.g., 'cuda' or 'cpu'.
        :param quantization: A string (e.g., "4bit") indicating which quantization scheme to apply.
        """
        self.device = device
        self.model = None
        self.processor = None
        self.quantization = quantization

    def _get_quant_config(self):
        if self.quantization == "4bit":
            logging.info("Using 4-bit quantization.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return None

    def load_local_model(self, model_path):
        """
        Loads a local model from the specified directory path. Optionally applies quantization.

        :param model_path: Filesystem path to the local pre-trained model directory.
        :return: Boolean indicating whether the model was successfully loaded.
        """
        if not os.path.exists(model_path):
            logging.error(f"Model path {os.path.abspath(model_path)} does not exist.")
            return False
        if not os.path.isdir(model_path):
            logging.error(f"Model path {os.path.abspath(model_path)} is not a directory.")
            return False

        try:
            logging.info(f"Attempting to load model from {os.path.abspath(model_path)}")
            quant_config = self._get_quant_config()

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quant_config,
            ).eval()

            if not self.quantization:
                self.model.to(self.device)
                if torch.cuda.is_available():
                    self.model = self.model.half()
                    logging.info("Using FP16 precision for the model.")

            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            logging.info(f"Model loaded successfully from {os.path.abspath(model_path)}")
            return True
        except (OSError, ValueError, ModuleNotFoundError) as e:
            logging.error(f"Error initializing model: {e}")
        except Exception as e:
            logging.error(f"Unexpected error initializing model: {e}")
        return False

    def download_and_load_model(self, repo_id="microsoft/Florence-2-base-ft"):
        """
        Downloads a model from the Hugging Face Hub using its repository ID, then loads it locally.

        :param repo_id: The Hugging Face model repository ID to download from.
        :return: Boolean indicating whether the model was successfully downloaded and loaded.
        """
        try:
            local_model_dir = "model"
            snapshot_download(repo_id=repo_id, local_dir=local_model_dir)
            if not os.path.exists(local_model_dir):
                logging.error(
                    f"Model download failed, directory {os.path.abspath(local_model_dir)} does not exist."
                )
                return False
            if not os.path.isdir(local_model_dir):
                logging.error(
                    f"Model download failed, path {os.path.abspath(local_model_dir)} is not a directory."
                )
                return False
            logging.info(
                f"Model downloaded and initialized at {os.path.abspath(local_model_dir)}"
            )
            return self.load_local_model(local_model_dir)
        except OSError as e:
            logging.error(f"OS error during model download: {e}")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
        return False


class RecordingManager:
    """
    Class that manages video recording. Can record continuously or by detection/inference triggers.
    """
    def __init__(self, record_mode=None):
        """
        Initializes the recording manager with a specified mode.

        :param record_mode: The mode for starting/stopping recording:
            None - no recording,
            "od" - based on object detections,
            "infy"/"infn" - based on inference results (yes/no).
        """
        self.record_mode = record_mode
        self.recording = False
        self.video_writer = None
        self.video_out_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        self.last_detection_time = time.time()

    def start_recording(self, frame):
        """
        Starts video recording given an initial frame (to set up dimensions, codec, etc.).
        """
        if not self.recording and self.record_mode:
            height, width, _ = frame.shape
            self.video_writer = cv2.VideoWriter(
                self.video_out_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                20.0,
                (width, height),
            )
            self.recording = True
            logging.info(f"Started recording video: {self.video_out_path}")

    def stop_recording(self):
        """
        Stops video recording and releases the VideoWriter resource.
        """
        if self.recording:
            self.video_writer.release()
            self.recording = False
            logging.info(f"Stopped recording video: {self.video_out_path}")

    def write_frame(self, frame):
        """
        Writes a single frame to the open video file if currently recording.
        """
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

    def handle_recording_by_detection(self, detections, frame):
        """
        Starts or stops recording based on whether object detections are present.
        """
        if not self.record_mode:
            return
        current_time = time.time()
        if detections:
            self.start_recording(frame)
            self.last_detection_time = current_time
        else:
            if (current_time - self.last_detection_time) > 1:
                self.stop_recording()
                logging.info("Recording stopped due to no detection for 1+ second.")

    def handle_recording_by_inference(self, inference_result, frame):
        """
        Starts or stops recording based on inference (yes/no) results.
        """
        if self.record_mode == "infy" and inference_result == "yes":
            self.start_recording(frame)
        elif self.record_mode == "infy" and inference_result == "no":
            self.stop_recording()
        elif self.record_mode == "infn" and inference_result == "no":
            self.start_recording(frame)
        elif self.record_mode == "infn" and inference_result == "yes":
            self.stop_recording()


class ImageUtils:
    """
    Utility class for image-related operations such as drawing bounding boxes and saving screenshots.
    """
    @staticmethod
    def plot_bbox(image, detections):
        """
        Draws bounding boxes and labels on an image using OpenCV.
        """
        try:
            for bbox, label in detections:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            return image
        except cv2.error as e:
            logging.error(f"OpenCV error plotting bounding boxes: {e}")
        except Exception as e:
            logging.error(f"Error plotting bounding boxes: {e}")
        return image

    @staticmethod
    def save_screenshot(frame):
        """
        Saves a screenshot of the current frame with a timestamped filename.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            logging.info(f"Screenshot saved: {filename}")
            print(f"[{timestamp}] Screenshot saved: {filename}")
        except cv2.error as e:
            logging.error(f"OpenCV error saving screenshot: {e}")
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error saving screenshot: {e}")
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error saving screenshot: {e}")


class AlertLogger:
    """
    A simple class to log alerts both to a dedicated file (alerts.log) and to the console.
    """
    @staticmethod
    def log_alert(message):
        """
        Appends an alert message to a log file with a timestamp, and also prints to console.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open("alerts.log", "a") as log_file:
                log_file.write(f"{timestamp} - {message}\n")
            logging.info(f"{timestamp} - {message}")
            print(f"[{timestamp}] Log entry written: {message}")
        except IOError as e:
            logging.error(f"IO error logging alert: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] IO error logging alert: {e}")
        except Exception as e:
            logging.error(f"Error logging alert: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Error logging alert: {e}")


def ptz_control_thread(ptz_camera):
    """
    A simple thread function for interactive PTZ control using arrow keys and +/- zoom on Windows.
    Press 'q' to quit PTZ mode.
    """
    print("PTZ control started. Use arrow keys to pan/tilt, +/- to zoom, q to quit.")
    while True:
        ch = msvcrt.getch()
        if ch == b'\xe0':
            arrow = msvcrt.getch()
            if arrow == b'H':
                ptz_camera.tilt_up()
            elif arrow == b'P':
                ptz_camera.tilt_down()
            elif arrow == b'K':
                ptz_camera.pan_left()
            elif arrow == b'M':
                ptz_camera.pan_right()
        elif ch == b'+':
            ptz_camera.zoom_in()
        elif ch == b'-':
            ptz_camera.zoom_out()
        elif ch == b'q':
            print("Quitting PTZ control.")
            break
    ptz_camera.close()

# ---------------------------------------------------------------------------
# Below is the updated GUI class, merging new functionalities from the CLI:
# PTZ integration, quantization, logging, recording, etc.
# ---------------------------------------------------------------------------

class YO_FLO:
    def __init__(self):
        # Original GUI attributes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.headless_mode = False
        self.processor = None
        self.inference_phrases_result_labels = []
        self.scaler = torch.cuda.amp.GradScaler()
        self.inference_start_time = None
        self.inference_count = 0
        self.inference_rate_label = None
        self.class_names = []
        self.detections = []
        self.beep_active = False
        self.screenshot_active = False
        self.screenshot_on_yes_active = False
        self.screenshot_on_no_active = False
        self.target_detected = False
        self.last_beep_time = 0
        self.stop_webcam_flag = threading.Event()
        self.model_path = None
        self.phrase = None
        self.debug = False
        self.caption_label = None
        self.object_detection_active = False
        self.expression_comprehension_active = False
        self.visual_grounding_active = False
        self.visual_grounding_phrase = None
        self.webcam_threads = []
        self.webcam_indices = [0]
        self.inference_title = None
        self.inference_phrases = []
        self.inference_result_label = None
        self.inference_tree_active = False

        # NEW FIELDS from CLI integration:
        # logging & record toggles
        self.log_to_file_active = False
        self.record = None  # e.g., "od", "infy", "infn"
        self.quantization = None  # e.g., "4bit"
        # PTZ
        self.ptz_camera = None
        self.ptz_tracker = None
        self.track_object_name = None

        # The manager classes we integrate from the CLI
        self.recording_manager = None
        self.model_manager = None

        # TK root
        self.root = tk.Tk()
        self.root.withdraw()

    # -----------------------------------------------------------------------
    # Manage model: integrate new approach with ModelManager
    # -----------------------------------------------------------------------
    def init_model_manager(self, quantization_mode=None):
        """Initialize the ModelManager (from CLI) with an optional quantization mode."""
        self.quantization = quantization_mode
        self.model_manager = ModelManager(self.device, self.quantization)

    def load_local_model(self, model_path):
        """Load local model using the ModelManager."""
        if not self.model_manager:
            self.init_model_manager()
        ok = self.model_manager.load_local_model(model_path)
        if ok:
            # If load is successful, assign references
            self.model = self.model_manager.model
            self.processor = self.model_manager.processor
            self.model_path = model_path
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Model loaded successfully from {model_path}{Style.RESET_ALL}"
            )
        else:
            print(
                f"{Fore.RED}{Style.BRIGHT}Failed to load model from {model_path}{Style.RESET_ALL}"
            )

    def download_model(self, repo_id="microsoft/Florence-2-base-ft"):
        """Download model from HF using ModelManager, then load."""
        if not self.model_manager:
            self.init_model_manager()
        ok = self.model_manager.download_and_load_model(repo_id)
        if ok:
            self.model = self.model_manager.model
            self.processor = self.model_manager.processor
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Model downloaded and initialized successfully!{Style.RESET_ALL}"
            )
        else:
            print(
                f"{Fore.RED}{Style.BRIGHT}Failed to download/initialize model.{Style.RESET_ALL}"
            )

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    def toggle_file_logging(self):
        """Toggle whether to log to file using the setup_logging function."""
        self.log_to_file_active = not self.log_to_file_active
        setup_logging(self.log_to_file_active)
        print(
            f"{Fore.GREEN}{Style.BRIGHT}File logging is now {'enabled' if self.log_to_file_active else 'disabled'}{Style.RESET_ALL}"
        )

    # -----------------------------------------------------------------------
    # PTZ Camera & Tracker
    # -----------------------------------------------------------------------
    def init_ptz_camera(self):
        if not self.ptz_camera:
            self.ptz_camera = PTZController()

    def set_ptz_target_class(self):
        """
        Prompt user for the object class name to track, store in self.track_object_name.
        """
        try:
            target_class = simpledialog.askstring(
                "PTZ Target Class", 
                "Enter the object class name to track (e.g., 'person'):"
            )
            if target_class:
                self.track_object_name = target_class.strip().lower()
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}PTZ tracking target class set to: {self.track_object_name}{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Fore.YELLOW}{Style.BRIGHT}PTZ target class input cancelled.{Style.RESET_ALL}"
                )
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error setting PTZ target class: {e}{Style.RESET_ALL}")

    def start_autonomous_ptz_tracking(self):
        """
        Initializes PTZTracker with the PTZ camera if not already.
        Activates autonomous PTZ tracking for whatever object class is in self.track_object_name.
        """
        self.init_ptz_camera()
        if not self.ptz_tracker:
            self.ptz_tracker = PTZTracker(self.ptz_camera)
        self.ptz_tracker.activate(True)
        if self.track_object_name:
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Autonomous PTZ tracking activated for object: {self.track_object_name}{Style.RESET_ALL}"
            )
        else:
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Autonomous PTZ tracking activated (no target class set yet).{Style.RESET_ALL}"
            )

    def stop_autonomous_ptz_tracking(self):
        if self.ptz_tracker:
            self.ptz_tracker.activate(False)
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Autonomous PTZ tracking deactivated.{Style.RESET_ALL}"
            )

    def open_manual_ptz_control(self):
        """
        Opens a thread that listens for arrow keys / +/- to drive PTZ in real time.
        """
        self.init_ptz_camera()
        thread = threading.Thread(target=ptz_control_thread, args=(self.ptz_camera,))
        thread.start()

    # -----------------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------------
    def set_record_mode(self, mode):
        """
        mode can be one of:
         - None (no recording)
         - "od" (start/stop on object detection)
         - "infy" (start on yes, stop on no)
         - "infn" (start on no, stop on yes)
        """
        self.record = mode
        self.recording_manager = RecordingManager(self.record)
        print(
            f"{Fore.GREEN}{Style.BRIGHT}Recording mode set to {mode if mode else 'None'}{Style.RESET_ALL}"
        )

    # -----------------------------------------------------------------------
    # Additional utility from CLI: picking a tracked object
    # -----------------------------------------------------------------------
    def _pick_tracked_object(self, detections):
        """
        Pick the bounding box for the object we want to track (largest bounding box of matching label).
        """
        if not self.track_object_name:
            return None
        candidate_detections = [
            (bbox, label)
            for bbox, label in detections
            if label.lower() == self.track_object_name.lower()
        ]
        if not candidate_detections:
            return None

        def bbox_area(bb):
            return (bb[2] - bb[0]) * (bb[3] - bb[1])

        largest_bbox = max(candidate_detections, key=lambda x: bbox_area(x[0]))[0]
        return largest_bbox

    # -----------------------------------------------------------------------
    # The rest of the GUI code is the older version, now adapted to integrate
    # the new classes above. Below are modifications or insertions to make
    # them all work seamlessly.
    # -----------------------------------------------------------------------

    def update_inference_rate(self):
        if self.inference_start_time is None:
            self.inference_start_time = time.time()
        else:
            elapsed_time = time.time() - self.inference_start_time
            if elapsed_time > 0:
                inferences_per_second = self.inference_count / elapsed_time
                if self.inference_rate_label:
                    self.inference_rate_label.config(
                        text=f"Inferences/sec: {inferences_per_second:.2f}", fg="green"
                    )

    def toggle_headless(self):
        try:
            self.headless_mode = not self.headless_mode
            status = "enabled" if self.headless_mode else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Headless mode is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling headless mode: {e}{Style.RESET_ALL}"
            )

    def prepare_inputs(self, task_prompt, image, phrase=None):
        """
        Same logic as before, just references self.model_manager if needed.
        """
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(
            self.device
        )
        if phrase:
            inputs["input_ids"] = torch.cat(
                [
                    inputs["input_ids"],
                    self.processor.tokenizer(phrase, return_tensors="pt")
                    .input_ids[:, 1:]
                    .to(self.device),
                ],
                dim=1,
            )
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.half()
        return inputs

    def run_model(self, inputs):
        with torch.amp.autocast("cuda"):
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=1,
            )
        return generated_ids

    def process_object_detection_outputs(self, generated_ids, image_size):
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task="<OD>", image_size=image_size
        )
        return parsed_answer

    def process_expression_comprehension_outputs(self, generated_ids):
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return generated_text

    def run_object_detection(self, image):
        try:
            if not self.model or not self.processor:
                raise ValueError("Model or processor is not initialized.")
            task_prompt = "<OD>"
            if self.debug:
                print(f"Running object detection with task prompt: {task_prompt}")
            inputs = self.prepare_inputs(task_prompt, image)
            generated_ids = self.run_model(inputs)
            if self.debug:
                print(f"Generated IDs: {generated_ids}")
            parsed_answer = self.process_object_detection_outputs(
                generated_ids, image.size
            )
            if self.debug:
                print(f"Parsed answer: {parsed_answer}")
            detections = []
            if parsed_answer and "<OD>" in parsed_answer:
                for bbox, label in zip(
                    parsed_answer["<OD>"]["bboxes"], parsed_answer["<OD>"]["labels"]
                ):
                    if not self.class_names or label.lower() in self.class_names:
                        detections.append((bbox, label))
            return detections
        except AttributeError as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Model or processor not initialized properly: {e}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error running object detection: {e}{Style.RESET_ALL}"
            )

    def run_expression_comprehension(self, image, phrase):
        try:
            task_prompt = "<CAPTION_TO_EXPRESSION_COMPREHENSION>"
            if self.debug:
                print(
                    f"Running expression comprehension with task prompt: {task_prompt} and phrase: {phrase}"
                )
            inputs = self.prepare_inputs(task_prompt, image, phrase)
            generated_ids = self.run_model(inputs)
            if self.debug:
                print(f"Generated IDs: {generated_ids}")
            generated_text = self.process_expression_comprehension_outputs(
                generated_ids
            )
            if self.debug:
                print(f"Generated text: {generated_text}")
            return generated_text
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error running expression comprehension: {e}{Style.RESET_ALL}"
            )

    def evaluate_inference_tree(self, image):
        try:
            if not self.inference_phrases:
                print(
                    f"{Fore.RED}{Style.BRIGHT}No inference phrases set.{Style.RESET_ALL}"
                )
                return "FAIL", []
            results = []
            phrase_results = []
            for phrase in self.inference_phrases:
                result = self.run_expression_comprehension(image, phrase)
                if result:
                    if "yes" in result.lower():
                        results.append(True)
                        phrase_results.append(True)
                    else:
                        results.append(False)
                        phrase_results.append(False)
            overall_result = "PASS" if all(results) else "FAIL"
            return overall_result, phrase_results
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error evaluating inference tree: {e}{Style.RESET_ALL}"
            )
            return "FAIL", []

    def run_visual_grounding(self, image, phrase):
        try:
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            inputs = self.prepare_inputs(task_prompt, image, phrase)
            generated_ids = self.run_model(inputs)
            if self.debug:
                print(f"Generated IDs: {generated_ids}")
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            if self.debug:
                print(f"Generated text: {generated_text}")
            parsed_answer = self.processor.post_process_generation(
                generated_text, task=task_prompt, image_size=image.size
            )
            if self.debug:
                print(f"Parsed answer: {parsed_answer}")
            if task_prompt in parsed_answer and parsed_answer[task_prompt]["bboxes"]:
                return parsed_answer[task_prompt]["bboxes"][0]
            else:
                return None
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error running visual grounding: {e}{Style.RESET_ALL}"
            )

    def plot_bbox(self, image):
        try:
            if not self.detections:
                return image
            return ImageUtils.plot_bbox(image, self.detections)
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error plotting bounding boxes: {e}{Style.RESET_ALL}"
            )
            return image

    def plot_visual_grounding_bbox(self, image, bbox, phrase):
        try:
            if bbox:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    image,
                    phrase,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
            return image
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error plotting visual grounding bounding box: {e}{Style.RESET_ALL}"
            )
            return image

    # Model path selection from the GUI
    def select_model_path(self):
        try:
            root = tk.Tk()
            root.withdraw()
            model_path = filedialog.askdirectory()
            if model_path:
                self.load_local_model(model_path)
            else:
                print(
                    f"{Fore.YELLOW}{Style.BRIGHT}Model path selection cancelled.{Style.RESET_ALL}"
                )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error selecting model path: {e}{Style.RESET_ALL}"
            )

    # Overridden to call the integrated ModelManager download
    def download_model_gui(self):
        try:
            self.download_model("microsoft/Florence-2-base-ft")
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error downloading model: {e}{Style.RESET_ALL}"
            )

    def set_class_names(self):
        try:
            class_names = simpledialog.askstring(
                "Set Class Names",
                "Enter the class names you want to detect, separated by commas (e.g., 'cat, dog'):",
            )
            if class_names:
                self.class_names = [name.strip().lower() for name in class_names.split(",")]
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Set to detect: {', '.join(self.class_names)}{Style.RESET_ALL}"
                )
            else:
                self.class_names = []
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Showing all detections{Style.RESET_ALL}"
                )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error setting class names: {e}{Style.RESET_ALL}"
            )

    def set_phrase(self):
        try:
            phrase = simpledialog.askstring(
                "Set Phrase",
                "Enter the yes or no question you want answered (e.g., 'Is the person smiling?', 'Is the cat laying down?'):",
            )
            self.phrase = phrase if phrase else None
            if self.phrase:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Set to comprehend: {self.phrase}{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}No phrase set for comprehension{Style.RESET_ALL}"
                )
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error setting phrase: {e}{Style.RESET_ALL}")

    def set_visual_grounding_phrase(self):
        try:
            phrase = simpledialog.askstring(
                "Set Visual Grounding Phrase", "Enter the phrase for visual grounding:"
            )
            self.visual_grounding_phrase = phrase if phrase else None
            if self.visual_grounding_phrase:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Set visual grounding phrase: {self.visual_grounding_phrase}{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}No phrase set for visual grounding{Style.RESET_ALL}"
                )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error setting visual grounding phrase: {e}{Style.RESET_ALL}"
            )

    def set_inference_tree(self):
        try:
            self.inference_title = simpledialog.askstring(
                "Inference Title", "Enter the title for the inference tree:"
            )
            self.inference_phrases = []
            for i in range(3):
                phrase = simpledialog.askstring(
                    "Set Inference Phrase",
                    f"Enter inference phrase {i+1} (e.g., 'Is it cloudy?', 'Is it wet?'):",
                )
                if phrase:
                    self.inference_phrases.append(phrase)
                else:
                    print(
                        f"{Fore.YELLOW}{Style.BRIGHT}Cancelled setting inference phrase {i+1}.{Style.RESET_ALL}"
                    )
                    return
            if self.inference_title and self.inference_phrases:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}Inference tree set with title: {self.inference_title}{Style.RESET_ALL}"
                )
                for phrase in self.inference_phrases:
                    print(
                        f"{Fore.GREEN}{Style.BRIGHT}Inference phrase: {phrase}{Style.RESET_ALL}"
                    )
            else:
                print(
                    f"{Fore.YELLOW}{Style.BRIGHT}Inference tree setting cancelled.{Style.RESET_ALL}"
                )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error setting inference tree: {e}{Style.RESET_ALL}"
            )

    def toggle_beep(self):
        try:
            self.beep_active = not self.beep_active
            status = "active" if self.beep_active else "inactive"
            print(f"{Fore.GREEN}{Style.BRIGHT}Beep is now {status}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error toggling beep: {e}{Style.RESET_ALL}")

    def toggle_screenshot(self):
        try:
            self.screenshot_active = not self.screenshot_active
            status = "active" if self.screenshot_active else "inactive"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Screenshot on detection is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling screenshot: {e}{Style.RESET_ALL}"
            )

    def toggle_screenshot_on_yes(self):
        try:
            self.screenshot_on_yes_active = not self.screenshot_on_yes_active
            status = "active" if self.screenshot_on_yes_active else "inactive"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Screenshot on Yes Inference is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling Screenshot on Yes Inference: {e}{Style.RESET_ALL}"
            )

    def toggle_screenshot_on_no(self):
        try:
            self.screenshot_on_no_active = not self.screenshot_on_no_active
            status = "active" if self.screenshot_on_no_active else "inactive"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Screenshot on No Inference is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling Screenshot on No Inference: {e}{Style.RESET_ALL}"
            )

    def toggle_debug(self):
        try:
            self.debug = not self.debug
            status = "enabled" if self.debug else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Debug mode is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling debug mode: {e}{Style.RESET_ALL}"
            )

    def toggle_object_detection(self):
        try:
            self.object_detection_active = not self.object_detection_active
            if not self.object_detection_active:
                self.detections.clear()
                self.class_names = []
                self.update_display()
            status = "enabled" if self.object_detection_active else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Object detection is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling object detection: {e}{Style.RESET_ALL}"
            )

    def toggle_expression_comprehension(self):
        try:
            self.expression_comprehension_active = (
                not self.expression_comprehension_active
            )
            status = "enabled" if self.expression_comprehension_active else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Expression comprehension is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling expression comprehension: {e}{Style.RESET_ALL}"
            )

    def toggle_visual_grounding(self):
        try:
            self.visual_grounding_active = not self.visual_grounding_active
            status = "enabled" if self.visual_grounding_active else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Visual grounding is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling visual grounding: {e}{Style.RESET_ALL}"
            )

    def toggle_inference_tree(self):
        try:
            self.inference_tree_active = not self.inference_tree_active
            status = "enabled" if self.inference_tree_active else "disabled"
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Inference tree evaluation is now {status}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error toggling inference tree: {e}{Style.RESET_ALL}"
            )

    def update_caption_window(self, caption):
        if self.caption_label:
            if caption.lower() == "yes":
                self.caption_label.config(
                    text=caption, fg="green", bg="black", font=("Helvetica", 14, "bold")
                )
                if self.screenshot_on_yes_active and hasattr(self, "latest_image"):
                    frame_bgr = cv2.cvtColor(np.array(self.latest_image), cv2.COLOR_RGB2BGR)
                    ImageUtils.save_screenshot(frame_bgr)
            elif caption.lower() == "no":
                self.caption_label.config(
                    text=caption, fg="red", bg="black", font=("Helvetica", 14, "bold")
                )
                if self.screenshot_on_no_active and hasattr(self, "latest_image"):
                    frame_bgr = cv2.cvtColor(np.array(self.latest_image), cv2.COLOR_RGB2BGR)
                    ImageUtils.save_screenshot(frame_bgr)
            else:
                self.caption_label.config(
                    text=caption, fg="white", bg="black", font=("Helvetica", 14, "bold")
                )

    def update_inference_result_window(self, result, phrase_results):
        if self.inference_result_label:
            if result.lower() == "pass":
                self.inference_result_label.config(
                    text=result, fg="green", bg="black", font=("Helvetica", 14, "bold")
                )
            else:
                self.inference_result_label.config(
                    text=result, fg="red", bg="black", font=("Helvetica", 14, "bold")
                )
        for idx, phrase_result in enumerate(phrase_results):
            label = self.inference_phrases_result_labels[idx]
            if phrase_result:
                label.config(
                    text=f"Inference {idx+1}: PASS",
                    fg="green",
                    bg="black",
                    font=("Helvetica", 14, "bold"),
                )
            else:
                label.config(
                    text=f"Inference {idx+1}: FAIL",
                    fg="red",
                    bg="black",
                    font=("Helvetica", 14, "bold"),
                )

    def beep_sound(self):
        try:
            if os.name == "nt":
                os.system("echo \a")
            else:
                print("\a")
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error playing beep sound: {e}{Style.RESET_ALL}"
            )

    def start_webcam_detection(self):
        if self.webcam_threads:
            print(
                f"{Fore.RED}{Style.BRIGHT}Webcam detection is already running.{Style.RESET_ALL}"
            )
            return
        self.stop_webcam_flag.clear()
        for index in self.webcam_indices:
            thread = threading.Thread(
                target=self._webcam_detection_thread, args=(index,)
            )
            thread.start()
            self.webcam_threads.append(thread)

    def _webcam_detection_thread(self, index):
        try:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                print(
                    f"{Fore.RED}{Style.BRIGHT}Error: Could not open webcam {index}.{Style.RESET_ALL}"
                )
                return
            while not self.stop_webcam_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(
                        f"{Fore.RED}{Style.BRIGHT}Error: Failed to capture image from webcam {index}.{Style.RESET_ALL}"
                    )
                    break
                try:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image)
                    self.latest_image = image_pil

                    # Expression Comprehension
                    if self.expression_comprehension_active and self.phrase:
                        results = self.run_expression_comprehension(image_pil, self.phrase)
                        if results:
                            caption = "Yes" if "yes" in results.lower() else "No"
                            self.update_caption_window(caption)
                            if self.headless_mode:
                                print(f"Expression comprehension result: {caption}")
                            self.inference_count += 1
                            self.update_inference_rate()
                            # If recording mode is 'infy' or 'infn', handle it
                            if self.recording_manager:
                                self.recording_manager.handle_recording_by_inference(caption.lower(), frame)

                    # Object Detection
                    if self.object_detection_active:
                        self.detections = self.run_object_detection(image_pil)
                        if self.headless_mode:
                            print(
                                f"Object Detection results from webcam {index}: {self.detections}"
                            )
                        self.inference_count += 1
                        self.update_inference_rate()
                        # If recording mode is 'od', handle it
                        if self.recording_manager:
                            self.recording_manager.handle_recording_by_detection(self.detections, frame)

                        # PTZ tracking
                        if self.ptz_tracker and self.ptz_tracker.active:
                            primary_bbox = self._pick_tracked_object(self.detections)
                            if primary_bbox is not None:
                                h, w, _ = frame.shape
                                self.ptz_tracker.adjust_camera(primary_bbox, w, h)

                    # Visual Grounding
                    if self.visual_grounding_active and self.visual_grounding_phrase:
                        bbox = self.run_visual_grounding(
                            image_pil, self.visual_grounding_phrase
                        )
                        if bbox:
                            if not self.headless_mode:
                                frame = self.plot_visual_grounding_bbox(
                                    frame, bbox, self.visual_grounding_phrase
                                )
                            else:
                                print(
                                    f"Visual Grounding result from webcam {index}: {bbox}"
                                )
                            self.inference_count += 1
                            self.update_inference_rate()

                    # Inference Tree
                    if self.inference_tree_active and self.inference_title and self.inference_phrases:
                        inference_result, phrase_results = self.evaluate_inference_tree(
                            image_pil
                        )
                        self.update_inference_result_window(
                            inference_result, phrase_results
                        )
                        if self.headless_mode:
                            print(
                                f"Inference Tree result from webcam {index}: {inference_result}, Details: {phrase_results}"
                            )
                        self.inference_count += 1
                        self.update_inference_rate()

                    # If currently recording, write the frame out
                    if self.recording_manager and self.recording_manager.recording:
                        self.recording_manager.write_frame(frame)

                    # Show result frames if not headless
                    if not self.headless_mode:
                        # Plot any OD bounding boxes
                        bbox_image = self.plot_bbox(frame.copy())
                        cv2.imshow(f"Object Detection Webcam {index}", bbox_image)

                        current_time = time.time()
                        # beep
                        if self.beep_active and self.target_detected and current_time - self.last_beep_time > 1:
                            threading.Thread(target=self.beep_sound).start()
                            self.last_beep_time = current_time
                        # screenshot on detection
                        if self.screenshot_active and self.target_detected:
                            ImageUtils.save_screenshot(bbox_image)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    else:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                except Exception as e:
                    print(
                        f"{Fore.RED}{Style.BRIGHT}Error during frame processing in webcam {index}: {e}{Style.RESET_ALL}"
                    )
            cap.release()
            if not self.headless_mode:
                cv2.destroyWindow(f"Object Detection Webcam {index}")
        except cv2.error as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}OpenCV error in webcam detection thread {index}: {e}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}Error in webcam detection thread {index}: {e}{Style.RESET_ALL}"
            )

    def stop_webcam_detection(self):
        if not self.webcam_threads:
            print(
                f"{Fore.RED}{Style.BRIGHT}Webcam detection is not running.{Style.RESET_ALL}"
            )
            return
        self.object_detection_active = False
        self.expression_comprehension_active = False
        self.visual_grounding_active = False
        self.inference_tree_active = False
        self.update_display()
        self.stop_webcam_flag.set()
        for thread in self.webcam_threads:
            thread.join()
        self.webcam_threads = []
        print(
            f"{Fore.GREEN}{Style.BRIGHT}Webcam detection stopped successfully.{Style.RESET_ALL}"
        )

    def update_display(self):
        if not self.object_detection_active:
            empty_frame = np.zeros((480, 640, 3), np.uint8)
            cv2.imshow("Object Detection", empty_frame)
            cv2.waitKey(1)

    # -----------------------------------------------------------------------
    # GUI Menu
    # -----------------------------------------------------------------------
    def main_menu(self):
        self.root.deiconify()
        self.root.title("YO-FLO Menu")

        def on_closing():
            self.stop_webcam_detection()
            # Close PTZ camera
            if self.ptz_camera:
                self.ptz_camera.close()
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)

        try:
            model_frame = tk.LabelFrame(self.root, text="Model Management")
            model_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                model_frame, text="Select Model Path", command=self.select_model_path
            ).pack(fill="x")
            tk.Button(
                model_frame,
                text="Download Model from HuggingFace",
                command=self.download_model_gui,
            ).pack(fill="x")
            tk.Button(
                model_frame, text="Toggle File Logging", command=self.toggle_file_logging
            ).pack(fill="x")

            detection_frame = tk.LabelFrame(self.root, text="Detection Settings")
            detection_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                detection_frame,
                text="Set Classes for Object Detection",
                command=self.set_class_names,
            ).pack(fill="x")
            tk.Button(
                detection_frame,
                text="Set Phrase for Yes/No Inference",
                command=self.set_phrase,
            ).pack(fill="x")
            tk.Button(
                detection_frame,
                text="Set Grounding Phrase",
                command=self.set_visual_grounding_phrase,
            ).pack(fill="x")
            tk.Button(
                detection_frame, text="Set Inference Tree", command=self.set_inference_tree
            ).pack(fill="x")

            feature_frame = tk.LabelFrame(self.root, text="Feature Toggles")
            feature_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                feature_frame,
                text="Object Detection",
                command=self.toggle_object_detection,
            ).pack(fill="x")
            tk.Button(
                feature_frame,
                text="Yes/No Inference",
                command=self.toggle_expression_comprehension,
            ).pack(fill="x")
            tk.Button(
                feature_frame,
                text="Visual Grounding",
                command=self.toggle_visual_grounding,
            ).pack(fill="x")
            tk.Button(
                feature_frame,
                text="Inference Tree",
                command=self.toggle_inference_tree,
            ).pack(fill="x")
            tk.Button(feature_frame, text="Headless Mode", command=self.toggle_headless).pack(
                fill="x"
            )

            trigger_frame = tk.LabelFrame(self.root, text="Triggers")
            trigger_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(trigger_frame, text="Beep on Detection", command=self.toggle_beep).pack(
                fill="x"
            )
            tk.Button(
                trigger_frame,
                text="Screenshot on Detection",
                command=self.toggle_screenshot,
            ).pack(fill="x")
            tk.Button(
                trigger_frame,
                text="Screenshot on Yes",
                command=self.toggle_screenshot_on_yes,
            ).pack(fill="x")
            tk.Button(
                trigger_frame,
                text="Screenshot on No",
                command=self.toggle_screenshot_on_no,
            ).pack(fill="x")

            ptz_frame = tk.LabelFrame(self.root, text="PTZ Control")
            ptz_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                ptz_frame,
                text="Open Manual PTZ Control",
                command=self.open_manual_ptz_control,
            ).pack(fill="x")
            tk.Button(
                ptz_frame,
                text="Set PTZ Target Class",
                command=self.set_ptz_target_class,
            ).pack(fill="x")
            tk.Button(
                ptz_frame,
                text="Start Autonomous Tracking",
                command=self.start_autonomous_ptz_tracking,
            ).pack(fill="x")
            tk.Button(
                ptz_frame,
                text="Stop Autonomous Tracking",
                command=self.stop_autonomous_ptz_tracking,
            ).pack(fill="x")

            recording_frame = tk.LabelFrame(self.root, text="Recording Control")
            recording_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                recording_frame,
                text='No Recording',
                command=lambda: self.set_record_mode(None),
            ).pack(fill="x")
            tk.Button(
                recording_frame,
                text='Record on OD',
                command=lambda: self.set_record_mode("od"),
            ).pack(fill="x")
            tk.Button(
                recording_frame,
                text='Record on "Yes" (infy)',
                command=lambda: self.set_record_mode("infy"),
            ).pack(fill="x")
            tk.Button(
                recording_frame,
                text='Record on "No" (infn)',
                command=lambda: self.set_record_mode("infn"),
            ).pack(fill="x")

            webcam_frame = tk.LabelFrame(self.root, text="Webcam Control")
            webcam_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(
                webcam_frame,
                text="Start Webcam Detection",
                command=self.start_webcam_detection,
            ).pack(fill="x")
            tk.Button(
                webcam_frame,
                text="Stop Webcam Detection",
                command=self.stop_webcam_detection,
            ).pack(fill="x")

            debug_frame = tk.LabelFrame(self.root, text="Debug")
            debug_frame.pack(fill="x", padx=10, pady=5)
            tk.Button(debug_frame, text="Toggle Debug Mode", command=self.toggle_debug).pack(
                fill="x", padx=10, pady=5
            )

            # Inference Rate
            inference_rate_frame = tk.LabelFrame(self.root, text="Inference Rate")
            inference_rate_frame.pack(fill="x", padx=10, pady=5)
            self.inference_rate_label = tk.Label(
                inference_rate_frame,
                text="Inferences/sec: N/A",
                fg="white",
                bg="black",
                font=("Helvetica", 14, "bold"),
            )
            self.inference_rate_label.pack(fill="x")

            # Binary Inference
            binary_inference_frame = tk.LabelFrame(self.root, text="Binary Inference")
            binary_inference_frame.pack(fill="x", padx=10, pady=5)
            self.caption_label = tk.Label(
                binary_inference_frame,
                text="Binary Inference: N/A",
                fg="white",
                bg="black",
                font=("Helvetica", 14, "bold"),
            )
            self.caption_label.pack(fill="x")

            # Inference Tree
            inference_tree_frame = tk.LabelFrame(self.root, text="Inference Tree")
            inference_tree_frame.pack(fill="x", padx=10, pady=5)
            self.inference_result_label = tk.Label(
                inference_tree_frame,
                text="Inference Tree: N/A",
                fg="white",
                bg="black",
                font=("Helvetica", 14, "bold"),
            )
            self.inference_result_label.pack(fill="x")

            for i in range(3):
                label = tk.Label(
                    inference_tree_frame,
                    text=f"Inference {i+1}: N/A",
                    fg="white",
                    bg="black",
                    font=("Helvetica", 14, "bold"),
                )
                label.pack(fill="x")
                self.inference_phrases_result_labels.append(label)

        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Error creating menu: {e}{Style.RESET_ALL}")
        self.root.mainloop()


if __name__ == "__main__":
    try:
        setup_logging(log_to_file=False)
        yo_flo = YO_FLO()
        yo_flo.init_model_manager(quantization_mode=None)  # or "4bit" if desired
        print(
            f"{Fore.BLUE}{Style.BRIGHT}Discover YO-FLO: A proof-of-concept merging advanced vision-language features with a Tkinter GUI.{Style.RESET_ALL}"
        )
        yo_flo.main_menu()
    except Exception as e:
        print(
            f"{Fore.RED}{Style.BRIGHT}Error initializing YO-FLO: {e}{Style.RESET_ALL}"
        )
