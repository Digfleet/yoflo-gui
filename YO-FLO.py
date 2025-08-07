import argparse
import logging
import os
import threading
import time
import sys
import cv2
import torch
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download, hf_hub_download
from colorama import Fore, Style, init
import tkinter as tk
from tkinter import filedialog, simpledialog, Toplevel
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty, Full
import re
from pathlib import Path
import gc

# Conditional imports for PTZ functionality
PTZ_AVAILABLE = False
try:
    import hid  # For PTZ camera HID
    PTZ_HID_AVAILABLE = True
except ImportError:
    PTZ_HID_AVAILABLE = False
    print(f"{Fore.YELLOW}HID module not available. PTZ camera control will be disabled.{Style.RESET_ALL}")

try:
    import msvcrt  # For Windows-specific PTZ control with arrow keys
    PTZ_MSVCRT_AVAILABLE = True
except ImportError:
    PTZ_MSVCRT_AVAILABLE = False
    print(f"{Fore.YELLOW}msvcrt module not available. Manual PTZ control with arrow keys will be disabled.{Style.RESET_ALL}")

# Set overall PTZ availability based on critical components
PTZ_AVAILABLE = PTZ_HID_AVAILABLE

init(autoreset=True)

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class AppConfig:
    """Central configuration for the YO-FLO application"""
    # Model settings
    DEFAULT_MODEL: str = "microsoft/Florence-2-base-ft"
    MODEL_CACHE_DIR: str = "model"
    QUANTIZATION_OPTIONS: List[str] = field(default_factory=lambda: ["none", "4bit"])
    
    # PTZ Camera settings
    PTZ_VENDOR_ID: int = 0x046D
    PTZ_PRODUCT_ID: int = 0x085F
    PTZ_USAGE_PAGE: int = 65280
    PTZ_USAGE: int = 1
    PTZ_COMMAND_DELAY: float = 0.2
    
    # PTZ Tracking settings
    PTZ_DESIRED_RATIO: float = 0.20
    PTZ_ZOOM_TOLERANCE: float = 0.4
    PTZ_PAN_TILT_TOLERANCE: int = 25
    PTZ_PAN_TILT_INTERVAL: float = 0.75
    PTZ_ZOOM_INTERVAL: float = 0.5
    PTZ_SMOOTHING_FACTOR: float = 0.2
    PTZ_MAX_ERRORS: int = 5
    
    # Recording settings
    RECORDING_FPS: float = 20.0
    RECORDING_CODEC: str = "XVID"
    RECORDING_TIMEOUT: float = 1.0  # Stop recording after 1s of no detection
    
    # Frame processing settings
    MAX_FPS: int = 30
    MIN_PROCESS_INTERVAL: float = 1.0 / 30  # Max 30 FPS
    FRAME_QUEUE_SIZE: int = 10
    PROCESSING_THREADS: int = 4
    
    # Security settings
    ALLOWED_MODEL_EXTENSIONS: List[str] = field(default_factory=lambda: [".bin", ".json", ".safetensors", ".pt"])
    MAX_CLASS_NAME_LENGTH: int = 50
    MAX_PHRASE_LENGTH: int = 200
    ALLOWED_CLASS_NAME_CHARS: str = r'^[a-zA-Z0-9\s\-_,]+$'
    
    # Logging settings
    LOG_FILE: str = "alerts.log"
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
    
    # GUI settings
    WINDOW_TITLE: str = "YO-FLO Vision System"
    DEFAULT_WEBCAM_INDICES: List[int] = field(default_factory=lambda: [0])
    
    # Memory management
    CUDA_MEMORY_FRACTION: float = 0.8
    CLEANUP_INTERVAL: float = 60.0  # Run cleanup every 60 seconds

# Global config instance
config = AppConfig()

# ============================================================================
# SECURITY UTILITIES
# ============================================================================

class SecurityValidator:
    """Validates and sanitizes user inputs for security"""
    
    @staticmethod
    def validate_path(path: str, allowed_extensions: List[str] = None) -> Optional[Path]:
        """
        Validates a file/directory path for security issues
        
        :param path: Path to validate
        :param allowed_extensions: List of allowed file extensions
        :return: Validated Path object or None if invalid
        """
        try:
            # Convert to Path object for safe handling
            safe_path = Path(path).resolve()
            
            # Check if path exists
            if not safe_path.exists():
                logging.warning(f"Path does not exist: {safe_path}")
                return None
            
            # Prevent directory traversal
            if ".." in str(path):
                logging.error(f"Potential directory traversal attempt: {path}")
                return None
            
            # Check file extensions if provided
            if allowed_extensions and safe_path.is_file():
                if safe_path.suffix.lower() not in allowed_extensions:
                    logging.error(f"Invalid file extension: {safe_path.suffix}")
                    return None
            
            return safe_path
            
        except Exception as e:
            logging.error(f"Path validation error: {e}")
            return None
    
    @staticmethod
    def sanitize_class_names(input_string: str) -> Optional[List[str]]:
        """
        Sanitizes class name input from user
        
        :param input_string: Raw input string
        :return: List of sanitized class names or None if invalid
        """
        if not input_string or len(input_string) > config.MAX_CLASS_NAME_LENGTH * 10:
            return None
        
        # Check for allowed characters
        if not re.match(config.ALLOWED_CLASS_NAME_CHARS, input_string):
            logging.warning(f"Invalid characters in class names: {input_string}")
            return None
        
        # Split and clean individual class names
        class_names = []
        for name in input_string.split(','):
            name = name.strip().lower()
            if name and len(name) <= config.MAX_CLASS_NAME_LENGTH:
                class_names.append(name)
        
        return class_names if class_names else None
    
    @staticmethod
    def sanitize_phrase(phrase: str) -> Optional[str]:
        """
        Sanitizes phrase input from user
        
        :param phrase: Raw phrase input
        :return: Sanitized phrase or None if invalid
        """
        if not phrase or len(phrase) > config.MAX_PHRASE_LENGTH:
            return None
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>\"\'\\]', '', phrase.strip())
        
        return sanitized if sanitized else None

# ============================================================================
# IMPROVED FRAME PROCESSOR WITH THREADING
# ============================================================================

class FrameProcessor:
    """Handles frame processing with proper threading and queue management"""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize the frame processor
        
        :param max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers or config.PROCESSING_THREADS
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.processing_queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.result_queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.active_futures: List[Future] = []
        self.shutdown_flag = threading.Event()
        self.last_process_time = time.time()
        self.frame_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Performance statistics
        self.frames_processed = 0
        self.frames_dropped = 0
        self.processing_times = []
        
    def should_process_frame(self) -> bool:
        """Check if enough time has passed to process next frame (FPS limiting)"""
        current_time = time.time()
        time_elapsed = current_time - self.last_process_time
        
        if time_elapsed >= config.MIN_PROCESS_INTERVAL:
            self.last_process_time = current_time
            return True
        return False
    
    def add_frame(self, frame: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a frame to the processing queue
        
        :param frame: Frame to process
        :param metadata: Optional metadata for the frame
        :return: True if frame was added, False if queue is full
        """
        if self.shutdown_flag.is_set():
            return False
        
        if not self.should_process_frame():
            with self.stats_lock:
                self.frames_dropped += 1
            return False
        
        try:
            self.processing_queue.put_nowait({
                'frame': frame,
                'metadata': metadata or {},
                'timestamp': time.time()
            })
            return True
        except Full:
            with self.stats_lock:
                self.frames_dropped += 1
            logging.debug("Frame queue is full, dropping frame")
            return False
    
    def process_frame(self, frame_data: Dict[str, Any], 
                     processing_func: callable) -> Optional[Any]:
        """
        Process a single frame with memory management
        
        :param frame_data: Frame data dictionary
        :param processing_func: Function to process the frame
        :return: Processing result or None
        """
        frame = frame_data['frame']
        start_time = time.time()
        result = None
        
        try:
            # Process frame
            result = processing_func(frame, frame_data['metadata'])
            
            # Update statistics
            with self.stats_lock:
                self.frames_processed += 1
                self.processing_times.append(time.time() - start_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None
            
        finally:
            # Memory cleanup
            del frame
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def submit_frame_batch(self, frames: List[np.ndarray], 
                          processing_func: callable) -> List[Future]:
        """
        Submit a batch of frames for processing
        
        :param frames: List of frames to process
        :param processing_func: Function to process each frame
        :return: List of futures for the submitted tasks
        """
        futures = []
        for frame in frames:
            if self.add_frame(frame):
                future = self.executor.submit(
                    self.process_frame,
                    {'frame': frame, 'metadata': {}, 'timestamp': time.time()},
                    processing_func
                )
                futures.append(future)
                self.active_futures.append(future)
        
        # Clean up completed futures
        self.active_futures = [f for f in self.active_futures if not f.done()]
        
        return futures
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self.stats_lock:
            avg_time = np.mean(self.processing_times) if self.processing_times else 0
            return {
                'frames_processed': self.frames_processed,
                'frames_dropped': self.frames_dropped,
                'average_processing_time': avg_time,
                'queue_size': self.processing_queue.qsize(),
                'active_tasks': len(self.active_futures)
            }
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the frame processor
        
        :param wait: Whether to wait for pending tasks to complete
        """
        self.shutdown_flag.set()
        
        # Clear queues
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except Empty:
                break
        
        # Cancel active futures if not waiting
        if not wait:
            for future in self.active_futures:
                future.cancel()
        
        self.executor.shutdown(wait=wait)
        logging.info("Frame processor shutdown complete")

# ============================================================================
# IMPROVED LOGGING SETUP
# ============================================================================

def setup_logging(log_to_file: bool = False, log_level: int = logging.INFO):
    """
    Sets up the logging configuration for the entire application.
    
    :param log_to_file: Boolean indicating whether to also log to a file.
    :param log_level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    handlers = [logging.StreamHandler()]
    
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Add timestamp to log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"yoflo_{timestamp}.log"
        
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        handlers=handlers
    )
    
    # Set specific loggers to warning to reduce noise
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

# ============================================================================
# IMPROVED PTZ CONTROLLER
# ============================================================================

class PTZController:
    """
    Class to control PTZ camera movements via HID commands with improved error handling.
    """
    
    def __init__(self, vendor_id: int = None, product_id: int = None,
                 usage_page: int = None, usage: int = None):
        """
        Initializes the PTZController with configuration-based defaults
        """
        self.vendor_id = vendor_id or config.PTZ_VENDOR_ID
        self.product_id = product_id or config.PTZ_PRODUCT_ID
        self.usage_page = usage_page or config.PTZ_USAGE_PAGE
        self.usage = usage or config.PTZ_USAGE
        self.device = None
        self.device_lock = threading.Lock()
        self.command_count = 0
        self.error_count = 0
        
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize the HID device connection"""
        if not PTZ_HID_AVAILABLE:
            logging.warning("PTZ control unavailable - HID module not loaded.")
            return
        
        try:
            ptz_path = None
            for d in hid.enumerate(self.vendor_id, self.product_id):
                if d['usage_page'] == self.usage_page and d['usage'] == self.usage:
                    ptz_path = d['path']
                    break
            
            if ptz_path:
                self.device = hid.device()
                self.device.open_path(ptz_path)
                logging.info("PTZ HID interface opened successfully.")
            else:
                logging.warning("No suitable PTZ HID interface found.")
                
        except IOError as e:
            logging.error(f"Error opening PTZ device: {e}")
            self.error_count += 1
        except Exception as e:
            logging.error(f"Unexpected error during PTZ device initialization: {e}")
            self.error_count += 1
    
    def send_command(self, report_id: int, value: int) -> bool:
        """
        Sends a command to the PTZ device via HID write with thread safety.
        
        :param report_id: The report ID for the PTZ control
        :param value: The value representing the command
        :return: True if command was sent successfully
        """
        if not PTZ_HID_AVAILABLE or not self.device:
            logging.debug("PTZ Device not initialized or not available.")
            return False
        
        with self.device_lock:
            command = [report_id & 0xFF, value] + [0x00] * 30
            
            try:
                self.device.write(command)
                self.command_count += 1
                logging.debug(f"PTZ command sent: report_id={report_id}, value={value}")
                time.sleep(config.PTZ_COMMAND_DELAY)
                return True
                
            except IOError as e:
                logging.error(f"Error sending PTZ command: {e}")
                self.error_count += 1
                
                # Try to reconnect if too many errors
                if self.error_count > 5:
                    self._reconnect()
                return False
                
            except Exception as e:
                logging.error(f"Unexpected error sending PTZ command: {e}")
                self.error_count += 1
                return False
    
    def _reconnect(self):
        """Attempt to reconnect to the PTZ device"""
        logging.info("Attempting to reconnect to PTZ device...")
        self.close()
        time.sleep(1)
        self._initialize_device()
    
    def pan_right(self) -> bool:
        """Pans the camera to the right."""
        return self.send_command(0x0B, 0x02)
    
    def pan_left(self) -> bool:
        """Pans the camera to the left."""
        return self.send_command(0x0B, 0x03)
    
    def tilt_up(self) -> bool:
        """Tilts the camera upward."""
        return self.send_command(0x0B, 0x00)
    
    def tilt_down(self) -> bool:
        """Tilts the camera downward."""
        return self.send_command(0x0B, 0x01)
    
    def zoom_in(self) -> bool:
        """Zooms the camera in."""
        return self.send_command(0x0B, 0x04)
    
    def zoom_out(self) -> bool:
        """Zooms the camera out."""
        return self.send_command(0x0B, 0x05)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get PTZ controller statistics"""
        return {
            'commands_sent': self.command_count,
            'errors': self.error_count
        }
    
    def close(self):
        """Closes the HID device handle safely."""
        if not PTZ_HID_AVAILABLE:
            return
        
        with self.device_lock:
            if self.device:
                try:
                    self.device.close()
                    logging.info("PTZ device closed successfully.")
                except Exception as e:
                    logging.error(f"Error closing PTZ device: {e}")
                finally:
                    self.device = None

# ============================================================================
# PTZ TRACKER
# ============================================================================

class PTZTracker:
    """
    Autonomous PTZ tracking class with improved error handling
    """
    
    def __init__(self, camera: Optional[PTZController], 
                 desired_ratio: float = None,
                 zoom_tolerance: float = None,
                 pan_tilt_tolerance: int = None,
                 pan_tilt_interval: float = None,
                 zoom_interval: float = None,
                 smoothing_factor: float = None,
                 max_consecutive_errors: int = None):
        """
        Initializes the PTZTracker with configuration defaults
        """
        # Use config defaults if not specified
        self.desired_ratio = desired_ratio or config.PTZ_DESIRED_RATIO
        self.zoom_tolerance = zoom_tolerance or config.PTZ_ZOOM_TOLERANCE
        self.pan_tilt_tolerance = pan_tilt_tolerance or config.PTZ_PAN_TILT_TOLERANCE
        self.pan_tilt_interval = pan_tilt_interval or config.PTZ_PAN_TILT_INTERVAL
        self.zoom_interval = zoom_interval or config.PTZ_ZOOM_INTERVAL
        self.smoothing_factor = smoothing_factor or config.PTZ_SMOOTHING_FACTOR
        self.max_consecutive_errors = max_consecutive_errors or config.PTZ_MAX_ERRORS
        
        # Check if camera is available
        if not camera or not PTZ_AVAILABLE:
            self.active = False
            self.camera = None
            logging.info("PTZ Tracker initialized but inactive - PTZ functionality not available.")
            return
        
        # Validate parameters
        self._validate_parameters()
        
        self.camera = camera
        self.last_pan_tilt_adjust = 0.0
        self.last_zoom_adjust = 0.0
        self.smoothed_width = None
        self.smoothed_height = None
        self.active = False
        self.consecutive_errors = 0
        self.tracking_lock = threading.Lock()
    
    def _validate_parameters(self):
        """Validate tracker parameters"""
        if not (0 < self.smoothing_factor < 1):
            raise ValueError("smoothing_factor must be between 0 and 1.")
        if self.desired_ratio <= 0 or self.desired_ratio >= 1:
            raise ValueError("desired_ratio should be between 0 and 1.")
        if self.zoom_tolerance < 0:
            raise ValueError("zoom_tolerance must be >= 0.")
        if self.pan_tilt_tolerance < 0:
            raise ValueError("pan_tilt_tolerance must be >= 0.")
        if self.pan_tilt_interval <= 0 or self.zoom_interval <= 0:
            raise ValueError("Intervals must be positive.")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be at least 1.")
    
    def activate(self, active: bool = True):
        """Activate or deactivate PTZ tracking"""
        if not PTZ_AVAILABLE or not self.camera:
            logging.warning("Cannot activate PTZ tracking - PTZ functionality not available.")
            self.active = False
            return
        
        with self.tracking_lock:
            self.active = active
            if not active:
                self.smoothed_width = None
                self.smoothed_height = None
                self.consecutive_errors = 0
            
            status = "activated" if active else "deactivated"
            logging.info(f"PTZ tracking {status}")
    
    def adjust_camera(self, bbox: Tuple[float, float, float, float], 
                     frame_width: int, frame_height: int):
        """
        Adjusts camera to keep object centered and properly sized
        
        :param bbox: Bounding box (x1, y1, x2, y2)
        :param frame_width: Frame width in pixels
        :param frame_height: Frame height in pixels
        """
        if not self.active or not PTZ_AVAILABLE or not self.camera:
            return
        
        with self.tracking_lock:
            x1, y1, x2, y2 = bbox
            
            # Validate bounding box
            if x1 >= x2 or y1 >= y2:
                logging.debug("Invalid bbox coordinates; skipping camera adjustment.")
                return
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Initialize or update smoothed dimensions
            if self.smoothed_width is None:
                self.smoothed_width = bbox_width
                self.smoothed_height = bbox_height
            else:
                self.smoothed_width = (
                    self.smoothing_factor * bbox_width +
                    (1 - self.smoothing_factor) * self.smoothed_width
                )
                self.smoothed_height = (
                    self.smoothing_factor * bbox_height +
                    (1 - self.smoothing_factor) * self.smoothed_height
                )
            
            # Calculate centers
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            frame_center_x = frame_width / 2
            frame_center_y = frame_height / 2
            
            # Calculate desired dimensions
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
                    command = "pan_left" if dx < 0 else "pan_right"
                    pan_tilt_moved = self._safe_camera_command(command) or pan_tilt_moved
                
                if abs(dy) > self.pan_tilt_tolerance:
                    command = "tilt_up" if dy < 0 else "tilt_down"
                    pan_tilt_moved = self._safe_camera_command(command) or pan_tilt_moved
                
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
            
            # Check for too many errors
            if self.consecutive_errors >= self.max_consecutive_errors:
                logging.error("Too many consecutive camera errors, deactivating PTZ tracking.")
                self.activate(False)
    
    def _safe_camera_command(self, command: str) -> bool:
        """
        Safely execute a camera command
        
        :param command: Command name to execute
        :return: True if command succeeded
        """
        if not PTZ_AVAILABLE or not self.camera:
            self.consecutive_errors += 1
            return False
        
        if not hasattr(self.camera, command):
            logging.error(f"Camera does not support command '{command}'.")
            return False
        
        try:
            method = getattr(self.camera, command)
            success = method()
            
            if success:
                self.consecutive_errors = 0
            else:
                self.consecutive_errors += 1
            
            return success
            
        except Exception as e:
            self.consecutive_errors += 1
            logging.error(f"Error executing camera command '{command}': {e}")
            return False

# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    Enhanced model manager with better memory management
    """
    
    def __init__(self, device: torch.device, quantization: Optional[str] = None):
        """
        Initialize the ModelManager
        
        :param device: Torch device (cuda/cpu)
        :param quantization: Quantization mode (None, "4bit")
        """
        self.device = device
        self.model = None
        self.processor = None
        self.quantization = quantization
        self.model_lock = threading.Lock()
    
    def _get_quant_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration"""
        if self.quantization == "4bit":
            logging.info("Using 4-bit quantization.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return None
    
    def load_local_model(self, model_path: str) -> bool:
        """
        Load a local model with proper error handling
        
        :param model_path: Path to model directory
        :return: True if successful
        """
        with self.model_lock:
            if not os.path.exists(model_path):
                logging.error(f"Model path {os.path.abspath(model_path)} does not exist.")
                return False
            
            if not os.path.isdir(model_path):
                logging.error(f"Model path {os.path.abspath(model_path)} is not a directory.")
                return False
            
            try:
                logging.info(f"Loading model from {os.path.abspath(model_path)}")
                quant_config = self._get_quant_config()
                
                # Clear existing model
                if self.model:
                    del self.model
                    torch.cuda.empty_cache()
                
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
    
    def download_and_load_model(self, repo_id: str = "microsoft/Florence-2-base-ft") -> bool:
        """
        Download and load model from Hugging Face
        
        :param repo_id: HuggingFace repository ID
        :return: True if successful
        """
        try:
            local_model_dir = config.MODEL_CACHE_DIR
            
            # Create directory if it doesn't exist
            Path(local_model_dir).mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Downloading model from {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=local_model_dir)
            
            if not os.path.exists(local_model_dir):
                logging.error(f"Model download failed, directory {local_model_dir} does not exist.")
                return False
            
            logging.info(f"Model downloaded to {os.path.abspath(local_model_dir)}")
            return self.load_local_model(local_model_dir)
            
        except OSError as e:
            logging.error(f"OS error during model download: {e}")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
        
        return False

# ============================================================================
# RECORDING MANAGER
# ============================================================================

class RecordingManager:
    """
    Enhanced recording manager with better resource management
    """
    
    def __init__(self, record_mode: Optional[str] = None):
        """
        Initialize the recording manager
        
        :param record_mode: Recording mode (None, "od", "infy", "infn")
        """
        self.record_mode = record_mode
        self.recording = False
        self.video_writer = None
        self.video_out_path = None
        self.last_detection_time = time.time()
        self.writer_lock = threading.Lock()
        self.frame_count = 0
        self.start_time = None
    
    def start_recording(self, frame: np.ndarray) -> bool:
        """
        Start video recording
        
        :param frame: Initial frame
        :return: True if successful
        """
        with self.writer_lock:
            if self.recording or not self.record_mode:
                return False
            
            try:
                height, width = frame.shape[:2]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Create recordings directory
                record_dir = Path("recordings")
                record_dir.mkdir(exist_ok=True)
                
                self.video_out_path = str(record_dir / f"recording_{timestamp}.avi")
                
                fourcc = cv2.VideoWriter_fourcc(*config.RECORDING_CODEC)
                self.video_writer = cv2.VideoWriter(
                    self.video_out_path,
                    fourcc,
                    config.RECORDING_FPS,
                    (width, height)
                )
                
                if self.video_writer.isOpened():
                    self.recording = True
                    self.start_time = time.time()
                    self.frame_count = 0
                    logging.info(f"Started recording: {self.video_out_path}")
                    return True
                else:
                    logging.error("Failed to open video writer")
                    return False
                    
            except Exception as e:
                logging.error(f"Error starting recording: {e}")
                return False
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop video recording
        
        :return: Path to recorded video
        """
        with self.writer_lock:
            if not self.recording:
                return None
            
            try:
                if self.video_writer:
                    self.video_writer.release()
                
                self.recording = False
                duration = time.time() - self.start_time if self.start_time else 0
                
                logging.info(
                    f"Stopped recording: {self.video_out_path} "
                    f"(Duration: {duration:.2f}s, Frames: {self.frame_count})"
                )
                
                path = self.video_out_path
                self.video_out_path = None
                self.video_writer = None
                self.frame_count = 0
                self.start_time = None
                
                return path
                
            except Exception as e:
                logging.error(f"Error stopping recording: {e}")
                return None
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the video
        
        :param frame: Frame to write
        :return: True if successful
        """
        with self.writer_lock:
            if not self.recording or not self.video_writer:
                return False
            
            try:
                self.video_writer.write(frame)
                self.frame_count += 1
                return True
            except Exception as e:
                logging.error(f"Error writing frame: {e}")
                return False
    
    def handle_recording_by_detection(self, detections: List, frame: np.ndarray):
        """Handle recording based on object detection"""
        if not self.record_mode or self.record_mode != "od":
            return
        
        current_time = time.time()
        
        if detections:
            if not self.recording:
                self.start_recording(frame)
            self.last_detection_time = current_time
            self.write_frame(frame)
        else:
            if self.recording and (current_time - self.last_detection_time) > config.RECORDING_TIMEOUT:
                self.stop_recording()
    
    def handle_recording_by_inference(self, inference_result: str, frame: np.ndarray):
        """Handle recording based on inference results"""
        if not self.record_mode or self.record_mode not in ["infy", "infn"]:
            return
        
        should_record = False
        
        if self.record_mode == "infy" and inference_result.lower() == "yes":
            should_record = True
        elif self.record_mode == "infn" and inference_result.lower() == "no":
            should_record = True
        
        if should_record:
            if not self.recording:
                self.start_recording(frame)
            self.write_frame(frame)
        else:
            if self.recording:
                self.stop_recording()
    
    def cleanup(self):
        """Clean up resources"""
        if self.recording:
            self.stop_recording()

# ============================================================================
# IMAGE UTILITIES
# ============================================================================

class ImageUtils:
    """Utility class for image operations"""
    
    @staticmethod
    def plot_bbox(image: np.ndarray, detections: List[Tuple[List[float], str]]) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        :param image: Input image
        :param detections: List of (bbox, label) tuples
        :return: Image with bounding boxes
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
    def save_screenshot(frame: np.ndarray) -> Optional[str]:
        """
        Save a screenshot with timestamp
        
        :param frame: Frame to save
        :return: Path to saved screenshot
        """
        try:
            # Create screenshots directory
            screenshot_dir = Path("screenshots")
            screenshot_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = str(screenshot_dir / f"screenshot_{timestamp}.png")
            
            cv2.imwrite(filename, frame)
            logging.info(f"Screenshot saved: {filename}")
            return filename
            
        except cv2.error as e:
            logging.error(f"OpenCV error saving screenshot: {e}")
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")
        
        return None

# ============================================================================
# ALERT LOGGER
# ============================================================================

class AlertLogger:
    """Enhanced alert logging with thread safety"""
    
    def __init__(self, log_file: str = None):
        """
        Initialize alert logger
        
        :param log_file: Path to log file
        """
        self.log_file = log_file or config.LOG_FILE
        self.log_lock = threading.Lock()
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.log_path = log_dir / self.log_file
    
    def log_alert(self, message: str):
        """
        Log an alert message
        
        :param message: Alert message
        """
        with self.log_lock:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                log_entry = f"{timestamp} - {message}\n"
                
                with open(self.log_path, "a") as log_file:
                    log_file.write(log_entry)
                
                logging.info(f"Alert logged: {message}")
                
            except IOError as e:
                logging.error(f"IO error logging alert: {e}")
            except Exception as e:
                logging.error(f"Error logging alert: {e}")

# ============================================================================
# PTZ CONTROL THREAD
# ============================================================================

def ptz_control_thread(ptz_camera: PTZController):
    """
    Thread for manual PTZ control using keyboard
    
    :param ptz_camera: PTZ camera controller
    """
    if not PTZ_MSVCRT_AVAILABLE:
        print("Cannot start PTZ control thread - msvcrt module not available.")
        return
    
    if not PTZ_HID_AVAILABLE or not ptz_camera:
        print("Cannot start PTZ control thread - PTZ camera not available.")
        return
    
    print("PTZ control started. Use arrow keys to pan/tilt, +/- to zoom, q to quit.")
    
    while True:
        try:
            ch = msvcrt.getch()
            
            if ch == b'\xe0':  # Arrow key prefix
                arrow = msvcrt.getch()
                if arrow == b'H':  # Up arrow
                    ptz_camera.tilt_up()
                elif arrow == b'P':  # Down arrow
                    ptz_camera.tilt_down()
                elif arrow == b'K':  # Left arrow
                    ptz_camera.pan_left()
                elif arrow == b'M':  # Right arrow
                    ptz_camera.pan_right()
            elif ch == b'+':
                ptz_camera.zoom_in()
            elif ch == b'-':
                ptz_camera.zoom_out()
            elif ch == b'q':
                print("Quitting PTZ control.")
                break
                
        except Exception as e:
            logging.error(f"Error in PTZ control thread: {e}")
            break

# ============================================================================
# MAIN YO-FLO APPLICATION CLASS
# ============================================================================

class YO_FLO:
    def __init__(self):
        """Initialize YO-FLO with all attributes properly initialized"""
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and processor
        self.model = None
        self.processor = None
        self.model_path = None
        self.model_manager = None
        self.quantization = None
        
        # GUI elements
        self.root = tk.Tk()
        self.root.withdraw()
        self.caption_label = None
        self.inference_rate_label = None
        self.inference_result_label = None
        self.inference_phrases_result_labels = []
        
        # Detection settings
        self.class_names = []
        self.detections = []
        self.phrase = None
        self.visual_grounding_phrase = None
        self.inference_title = None
        self.inference_phrases = []
        
        # Feature flags
        self.headless_mode = False
        self.object_detection_active = False
        self.expression_comprehension_active = False
        self.visual_grounding_active = False
        self.inference_tree_active = False
        self.beep_active = False
        self.screenshot_active = False
        self.screenshot_on_yes_active = False
        self.screenshot_on_no_active = False
        self.debug = False
        self.log_to_file_active = False
        
        # Tracking and timing
        self.target_detected = False
        self.last_beep_time = 0
        self.inference_start_time = None
        self.inference_count = 0
        self.last_process_time = time.time()
        
        # Image handling
        self.latest_image = None
        self.frame_lock = threading.Lock()
        
        # Recording
        self.record = None
        self.recording_manager = None
        
        # PTZ
        self.ptz_camera = None
        self.ptz_tracker = None
        self.track_object_name = None
        
        # Threading
        self.webcam_threads = []
        self.webcam_indices = config.DEFAULT_WEBCAM_INDICES
        self.stop_webcam_flag = threading.Event()
        self.frame_processor = FrameProcessor()
        
        # Performance
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Cleanup
        self.cleanup_thread = None
        self.cleanup_flag = threading.Event()
        
        # Security validator
        self.validator = SecurityValidator()
        
        # Alert logger
        self.alert_logger = AlertLogger()
        
        # Start periodic cleanup
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start a thread for periodic memory cleanup"""
        def cleanup_worker():
            while not self.cleanup_flag.is_set():
                time.sleep(config.CLEANUP_INTERVAL)
                self._periodic_cleanup()
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _periodic_cleanup(self):
        """Perform periodic memory cleanup"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.debug("Periodic memory cleanup completed")
        except Exception as e:
            logging.error(f"Error during periodic cleanup: {e}")
    
    # -------------------------------------------------------------------------
    # Model Management
    # -------------------------------------------------------------------------
    
    def init_model_manager(self, quantization_mode: Optional[str] = None):
        """Initialize the ModelManager with proper validation"""
        if quantization_mode and quantization_mode not in config.QUANTIZATION_OPTIONS:
            logging.warning(f"Invalid quantization mode: {quantization_mode}")
            quantization_mode = None
        
        self.quantization = quantization_mode
        self.model_manager = ModelManager(self.device, self.quantization)
    
    def load_local_model(self, model_path: str):
        """Load local model with path validation"""
        safe_path = self.validator.validate_path(model_path)
        if not safe_path:
            print(f"{Fore.RED}Invalid or unsafe model path: {model_path}{Style.RESET_ALL}")
            return
        
        if not safe_path.is_dir():
            print(f"{Fore.RED}Model path must be a directory: {safe_path}{Style.RESET_ALL}")
            return
        
        if not self.model_manager:
            self.init_model_manager()
        
        ok = self.model_manager.load_local_model(str(safe_path))
        if ok:
            self.model = self.model_manager.model
            self.processor = self.model_manager.processor
            self.model_path = str(safe_path)
            print(f"{Fore.GREEN}Model loaded successfully from {safe_path}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Failed to load model from {safe_path}{Style.RESET_ALL}")
    
    def download_model(self, repo_id: str = "microsoft/Florence-2-base-ft"):
        """Download and load model from Hugging Face"""
        if not self.model_manager:
            self.init_model_manager()
        
        ok = self.model_manager.download_and_load_model(repo_id)
        if ok:
            self.model = self.model_manager.model
            self.processor = self.model_manager.processor
            print(f"{Fore.GREEN}Model downloaded and initialized successfully!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Failed to download/initialize model.{Style.RESET_ALL}")
    
    # -------------------------------------------------------------------------
    # Model Inference Methods
    # -------------------------------------------------------------------------
    
    def prepare_inputs(self, task_prompt: str, image: Image.Image, phrase: Optional[str] = None):
        """Prepare inputs for model inference"""
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
        
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
    
    def run_model(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run model inference"""
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
    
    def process_object_detection_outputs(self, generated_ids: torch.Tensor, 
                                        image_size: Tuple[int, int]) -> Dict:
        """Process object detection outputs"""
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task="<OD>", image_size=image_size
        )
        return parsed_answer
    
    def process_expression_comprehension_outputs(self, generated_ids: torch.Tensor) -> str:
        """Process expression comprehension outputs"""
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return generated_text
    
    def run_object_detection(self, image: Image.Image) -> List[Tuple[List[float], str]]:
        """Run object detection on an image"""
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
            
            parsed_answer = self.process_object_detection_outputs(generated_ids, image.size)
            
            if self.debug:
                print(f"Parsed answer: {parsed_answer}")
            
            detections = []
            if parsed_answer and "<OD>" in parsed_answer:
                for bbox, label in zip(
                    parsed_answer["<OD>"]["bboxes"], 
                    parsed_answer["<OD>"]["labels"]
                ):
                    if not self.class_names or label.lower() in self.class_names:
                        detections.append((bbox, label))
            
            return detections
            
        except AttributeError as e:
            logging.error(f"Model or processor not initialized properly: {e}")
        except Exception as e:
            logging.error(f"Error running object detection: {e}")
        
        return []
    
    def run_expression_comprehension(self, image: Image.Image, phrase: str) -> Optional[str]:
        """Run expression comprehension on an image"""
        try:
            task_prompt = "<CAPTION_TO_EXPRESSION_COMPREHENSION>"
            
            if self.debug:
                print(f"Running expression comprehension with phrase: {phrase}")
            
            inputs = self.prepare_inputs(task_prompt, image, phrase)
            generated_ids = self.run_model(inputs)
            
            if self.debug:
                print(f"Generated IDs: {generated_ids}")
            
            generated_text = self.process_expression_comprehension_outputs(generated_ids)
            
            if self.debug:
                print(f"Generated text: {generated_text}")
            
            return generated_text
            
        except Exception as e:
            logging.error(f"Error running expression comprehension: {e}")
            return None
    
    def run_visual_grounding(self, image: Image.Image, phrase: str) -> Optional[List[float]]:
        """Run visual grounding on an image"""
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
            
            return None
            
        except Exception as e:
            logging.error(f"Error running visual grounding: {e}")
            return None
    
    def evaluate_inference_tree(self, image: Image.Image) -> Tuple[str, List[bool]]:
        """Evaluate inference tree on an image"""
        try:
            if not self.inference_phrases:
                logging.error("No inference phrases set.")
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
            logging.error(f"Error evaluating inference tree: {e}")
            return "FAIL", []
    
    # -------------------------------------------------------------------------
    # GUI Methods
    # -------------------------------------------------------------------------
    
    def select_model_path(self):
        """Select model path with security validation"""
        try:
            model_path = filedialog.askdirectory(
                title="Select Model Directory",
                initialdir=os.getcwd()
            )
            
            if model_path:
                self.load_local_model(model_path)
            else:
                print(f"{Fore.YELLOW}Model path selection cancelled.{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error selecting model path: {e}{Style.RESET_ALL}")
    
    def download_model_gui(self):
        """Download model from GUI"""
        try:
            self.download_model(config.DEFAULT_MODEL)
        except Exception as e:
            print(f"{Fore.RED}Error downloading model: {e}{Style.RESET_ALL}")
    
    def set_class_names(self):
        """Set class names with input sanitization"""
        try:
            class_names_input = simpledialog.askstring(
                "Set Class Names",
                "Enter class names separated by commas (e.g., 'cat, dog'):"
            )
            
            if class_names_input:
                sanitized_names = self.validator.sanitize_class_names(class_names_input)
                
                if sanitized_names:
                    self.class_names = sanitized_names
                    print(f"{Fore.GREEN}Set to detect: {', '.join(self.class_names)}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Invalid class names input{Style.RESET_ALL}")
            else:
                self.class_names = []
                print(f"{Fore.GREEN}Showing all detections{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error setting class names: {e}{Style.RESET_ALL}")
    
    def set_phrase(self):
        """Set phrase with input sanitization"""
        try:
            phrase_input = simpledialog.askstring(
                "Set Phrase",
                "Enter a yes/no question (e.g., 'Is the person smiling?'):"
            )
            
            if phrase_input:
                sanitized_phrase = self.validator.sanitize_phrase(phrase_input)
                
                if sanitized_phrase:
                    self.phrase = sanitized_phrase
                    print(f"{Fore.GREEN}Set to comprehend: {self.phrase}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Invalid phrase input{Style.RESET_ALL}")
            else:
                self.phrase = None
                print(f"{Fore.GREEN}No phrase set for comprehension{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error setting phrase: {e}{Style.RESET_ALL}")
    
    def set_visual_grounding_phrase(self):
        """Set visual grounding phrase"""
        try:
            phrase_input = simpledialog.askstring(
                "Set Visual Grounding Phrase",
                "Enter the phrase for visual grounding:"
            )
            
            if phrase_input:
                sanitized_phrase = self.validator.sanitize_phrase(phrase_input)
                
                if sanitized_phrase:
                    self.visual_grounding_phrase = sanitized_phrase
                    print(f"{Fore.GREEN}Set visual grounding phrase: {self.visual_grounding_phrase}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Invalid phrase input{Style.RESET_ALL}")
            else:
                self.visual_grounding_phrase = None
                print(f"{Fore.GREEN}No phrase set for visual grounding{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error setting visual grounding phrase: {e}{Style.RESET_ALL}")
    
    def set_inference_tree(self):
        """Set up inference tree"""
        try:
            self.inference_title = simpledialog.askstring(
                "Inference Title",
                "Enter the title for the inference tree:"
            )
            
            self.inference_phrases = []
            for i in range(3):
                phrase = simpledialog.askstring(
                    "Set Inference Phrase",
                    f"Enter inference phrase {i+1} (e.g., 'Is it cloudy?'):"
                )
                
                if phrase:
                    sanitized = self.validator.sanitize_phrase(phrase)
                    if sanitized:
                        self.inference_phrases.append(sanitized)
                    else:
                        print(f"{Fore.RED}Invalid phrase {i+1}{Style.RESET_ALL}")
                        return
                else:
                    print(f"{Fore.YELLOW}Cancelled setting inference phrase {i+1}.{Style.RESET_ALL}")
                    return
            
            if self.inference_title and self.inference_phrases:
                print(f"{Fore.GREEN}Inference tree set with title: {self.inference_title}{Style.RESET_ALL}")
                for phrase in self.inference_phrases:
                    print(f"{Fore.GREEN}Inference phrase: {phrase}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Inference tree setting cancelled.{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error setting inference tree: {e}{Style.RESET_ALL}")
    
    # -------------------------------------------------------------------------
    # Feature Toggles
    # -------------------------------------------------------------------------
    
    def toggle_file_logging(self):
        """Toggle file logging"""
        self.log_to_file_active = not self.log_to_file_active
        setup_logging(self.log_to_file_active)
        status = "enabled" if self.log_to_file_active else "disabled"
        print(f"{Fore.GREEN}File logging is now {status}{Style.RESET_ALL}")
    
    def toggle_headless(self):
        """Toggle headless mode"""
        try:
            self.headless_mode = not self.headless_mode
            status = "enabled" if self.headless_mode else "disabled"
            print(f"{Fore.GREEN}Headless mode is now {status}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error toggling headless mode: {e}{Style.RESET_ALL}")
    
    def toggle_object_detection(self):
        """Toggle object detection"""
        self.object_detection_active = not self.object_detection_active
        if not self.object_detection_active:
            self.detections.clear()
            self.class_names = []
        status = "enabled" if self.object_detection_active else "disabled"
        print(f"{Fore.GREEN}Object detection is now {status}{Style.RESET_ALL}")
    
    def toggle_expression_comprehension(self):
        """Toggle expression comprehension"""
        self.expression_comprehension_active = not self.expression_comprehension_active
        status = "enabled" if self.expression_comprehension_active else "disabled"
        print(f"{Fore.GREEN}Expression comprehension is now {status}{Style.RESET_ALL}")
    
    def toggle_visual_grounding(self):
        """Toggle visual grounding"""
        self.visual_grounding_active = not self.visual_grounding_active
        status = "enabled" if self.visual_grounding_active else "disabled"
        print(f"{Fore.GREEN}Visual grounding is now {status}{Style.RESET_ALL}")
    
    def toggle_inference_tree(self):
        """Toggle inference tree"""
        self.inference_tree_active = not self.inference_tree_active
        status = "enabled" if self.inference_tree_active else "disabled"
        print(f"{Fore.GREEN}Inference tree evaluation is now {status}{Style.RESET_ALL}")
    
    def toggle_beep(self):
        """Toggle beep on detection"""
        self.beep_active = not self.beep_active
        status = "active" if self.beep_active else "inactive"
        print(f"{Fore.GREEN}Beep is now {status}{Style.RESET_ALL}")
    
    def toggle_screenshot(self):
        """Toggle screenshot on detection"""
        self.screenshot_active = not self.screenshot_active
        status = "active" if self.screenshot_active else "inactive"
        print(f"{Fore.GREEN}Screenshot on detection is now {status}{Style.RESET_ALL}")
    
    def toggle_screenshot_on_yes(self):
        """Toggle screenshot on yes inference"""
        self.screenshot_on_yes_active = not self.screenshot_on_yes_active
        status = "active" if self.screenshot_on_yes_active else "inactive"
        print(f"{Fore.GREEN}Screenshot on Yes Inference is now {status}{Style.RESET_ALL}")
    
    def toggle_screenshot_on_no(self):
        """Toggle screenshot on no inference"""
        self.screenshot_on_no_active = not self.screenshot_on_no_active
        status = "active" if self.screenshot_on_no_active else "inactive"
        print(f"{Fore.GREEN}Screenshot on No Inference is now {status}{Style.RESET_ALL}")
    
    def toggle_debug(self):
        """Toggle debug mode"""
        self.debug = not self.debug
        status = "enabled" if self.debug else "disabled"
        print(f"{Fore.GREEN}Debug mode is now {status}{Style.RESET_ALL}")
    
    # -------------------------------------------------------------------------
    # PTZ Control Methods
    # -------------------------------------------------------------------------
    
    def init_ptz_camera(self):
        """Initialize PTZ camera"""
        if not PTZ_AVAILABLE:
            print(f"{Fore.YELLOW}PTZ camera functionality not available.{Style.RESET_ALL}")
            return
        
        if not self.ptz_camera:
            self.ptz_camera = PTZController()
    
    def set_ptz_target_class(self):
        """Set PTZ target class"""
        if not PTZ_AVAILABLE:
            print(f"{Fore.YELLOW}PTZ camera functionality not available.{Style.RESET_ALL}")
            return
        
        try:
            target_class = simpledialog.askstring(
                "PTZ Target Class",
                "Enter the object class name to track (e.g., 'person'):"
            )
            
            if target_class:
                sanitized = self.validator.sanitize_class_names(target_class)
                if sanitized:
                    self.track_object_name = sanitized[0]
                    print(f"{Fore.GREEN}PTZ tracking target: {self.track_object_name}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Invalid target class{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}PTZ target class input cancelled.{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error setting PTZ target class: {e}{Style.RESET_ALL}")
    
    def start_autonomous_ptz_tracking(self):
        """Start autonomous PTZ tracking"""
        if not PTZ_AVAILABLE:
            print(f"{Fore.YELLOW}PTZ camera functionality not available.{Style.RESET_ALL}")
            return
        
        self.init_ptz_camera()
        if not self.ptz_tracker:
            self.ptz_tracker = PTZTracker(self.ptz_camera)
        
        self.ptz_tracker.activate(True)
        
        if self.track_object_name:
            print(f"{Fore.GREEN}Autonomous PTZ tracking activated for: {self.track_object_name}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Autonomous PTZ tracking activated (no target set).{Style.RESET_ALL}")
    
    def stop_autonomous_ptz_tracking(self):
        """Stop autonomous PTZ tracking"""
        if not PTZ_AVAILABLE:
            print(f"{Fore.YELLOW}PTZ camera functionality not available.{Style.RESET_ALL}")
            return
        
        if self.ptz_tracker:
            self.ptz_tracker.activate(False)
            print(f"{Fore.GREEN}Autonomous PTZ tracking deactivated.{Style.RESET_ALL}")
    
    def open_manual_ptz_control(self):
        """Open manual PTZ control"""
        if not PTZ_AVAILABLE or not PTZ_MSVCRT_AVAILABLE:
            print(f"{Fore.YELLOW}PTZ manual control not available.{Style.RESET_ALL}")
            return
        
        self.init_ptz_camera()
        if not self.ptz_camera:
            print(f"{Fore.YELLOW}PTZ camera could not be initialized.{Style.RESET_ALL}")
            return
        
        thread = threading.Thread(target=ptz_control_thread, args=(self.ptz_camera,), daemon=True)
        thread.start()
    
    # -------------------------------------------------------------------------
    # Recording Control
    # -------------------------------------------------------------------------
    
    def set_record_mode(self, mode: Optional[str]):
        """Set recording mode"""
        self.record = mode
        self.recording_manager = RecordingManager(self.record)
        mode_str = mode if mode else "None"
        print(f"{Fore.GREEN}Recording mode set to {mode_str}{Style.RESET_ALL}")
    
    # -------------------------------------------------------------------------
    # Frame Processing
    # -------------------------------------------------------------------------
    
    def should_process_frame(self) -> bool:
        """Check if enough time has passed for next frame"""
        current_time = time.time()
        if (current_time - self.last_process_time) >= config.MIN_PROCESS_INTERVAL:
            self.last_process_time = current_time
            return True
        return False
    
    def _pick_tracked_object(self, detections: List[Tuple[List[float], str]]) -> Optional[List[float]]:
        """Pick the largest bounding box of the tracked object"""
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
    
    def plot_bbox(self, image: np.ndarray) -> np.ndarray:
        """Plot bounding boxes on image"""
        try:
            if not self.detections:
                return image
            return ImageUtils.plot_bbox(image, self.detections)
        except Exception as e:
            logging.error(f"Error plotting bounding boxes: {e}")
            return image
    
    def plot_visual_grounding_bbox(self, image: np.ndarray, bbox: List[float], phrase: str) -> np.ndarray:
        """Plot visual grounding bounding box"""
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
            logging.error(f"Error plotting visual grounding bbox: {e}")
            return image
    
    def beep_sound(self):
        """Play beep sound"""
        try:
            if os.name == "nt":
                os.system("echo \a")
            else:
                print("\a")
        except Exception as e:
            logging.error(f"Error playing beep sound: {e}")
    
    def update_inference_rate(self):
        """Update inference rate display"""
        if self.inference_start_time is None:
            self.inference_start_time = time.time()
        else:
            elapsed_time = time.time() - self.inference_start_time
            if elapsed_time > 0:
                inferences_per_second = self.inference_count / elapsed_time
                if self.inference_rate_label:
                    self.inference_rate_label.config(
                        text=f"Inferences/sec: {inferences_per_second:.2f}",
                        fg="green"
                    )
    
    def update_caption_window(self, caption: str):
        """Update caption window"""
        if self.caption_label:
            if caption.lower() == "yes":
                self.caption_label.config(
                    text=caption,
                    fg="green",
                    bg="black",
                    font=("Helvetica", 14, "bold")
                )
                if self.screenshot_on_yes_active:
                    with self.frame_lock:
                        if self.latest_image:
                            frame_bgr = cv2.cvtColor(np.array(self.latest_image), cv2.COLOR_RGB2BGR)
                            ImageUtils.save_screenshot(frame_bgr)
            elif caption.lower() == "no":
                self.caption_label.config(
                    text=caption,
                    fg="red",
                    bg="black",
                    font=("Helvetica", 14, "bold")
                )
                if self.screenshot_on_no_active:
                    with self.frame_lock:
                        if self.latest_image:
                            frame_bgr = cv2.cvtColor(np.array(self.latest_image), cv2.COLOR_RGB2BGR)
                            ImageUtils.save_screenshot(frame_bgr)
            else:
                self.caption_label.config(
                    text=caption,
                    fg="white",
                    bg="black",
                    font=("Helvetica", 14, "bold")
                )
    
    def update_inference_result_window(self, result: str, phrase_results: List[bool]):
        """Update inference result window"""
        if self.inference_result_label:
            if result.lower() == "pass":
                self.inference_result_label.config(
                    text=result,
                    fg="green",
                    bg="black",
                    font=("Helvetica", 14, "bold")
                )
            else:
                self.inference_result_label.config(
                    text=result,
                    fg="red",
                    bg="black",
                    font=("Helvetica", 14, "bold")
                )
        
        for idx, phrase_result in enumerate(phrase_results):
            if idx < len(self.inference_phrases_result_labels):
                label = self.inference_phrases_result_labels[idx]
                if phrase_result:
                    label.config(
                        text=f"Inference {idx+1}: PASS",
                        fg="green",
                        bg="black",
                        font=("Helvetica", 14, "bold")
                    )
                else:
                    label.config(
                        text=f"Inference {idx+1}: FAIL",
                        fg="red",
                        bg="black",
                        font=("Helvetica", 14, "bold")
                    )
    
    # -------------------------------------------------------------------------
    # Webcam Detection
    # -------------------------------------------------------------------------
    
    def start_webcam_detection(self):
        """Start webcam detection"""
        if self.webcam_threads:
            print(f"{Fore.RED}Webcam detection is already running.{Style.RESET_ALL}")
            return
        
        self.stop_webcam_flag.clear()
        
        for index in self.webcam_indices:
            thread = threading.Thread(
                target=self._webcam_detection_thread,
                args=(index,),
                daemon=True
            )
            thread.start()
            self.webcam_threads.append(thread)
        
        print(f"{Fore.GREEN}Started webcam detection{Style.RESET_ALL}")
    
    def stop_webcam_detection(self):
        """Stop webcam detection"""
        if not self.webcam_threads:
            print(f"{Fore.RED}Webcam detection is not running.{Style.RESET_ALL}")
            return
        
        # Deactivate all features
        self.object_detection_active = False
        self.expression_comprehension_active = False
        self.visual_grounding_active = False
        self.inference_tree_active = False
        
        # Signal threads to stop
        self.stop_webcam_flag.set()
        
        # Wait for threads with timeout
        for thread in self.webcam_threads:
            thread.join(timeout=2.0)
        
        self.webcam_threads.clear()
        
        print(f"{Fore.GREEN}Webcam detection stopped successfully.{Style.RESET_ALL}")
    
    def _webcam_detection_thread(self, index: int):
        """Webcam detection thread with enhanced processing"""
        cap = None
        try:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                print(f"{Fore.RED}Error: Could not open webcam {index}.{Style.RESET_ALL}")
                return
            
            while not self.stop_webcam_flag.is_set():
                # Frame rate limiting
                if not self.should_process_frame():
                    time.sleep(0.001)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    print(f"{Fore.RED}Failed to capture from webcam {index}.{Style.RESET_ALL}")
                    break
                
                try:
                    # Thread-safe image storage
                    with self.frame_lock:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image_pil = Image.fromarray(image)
                        self.latest_image = image_pil
                    
                    # Process frame
                    self._process_single_frame(image_pil, frame, index)
                    
                    # Display if not headless
                    if not self.headless_mode:
                        self._display_frame(frame, index)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                except Exception as e:
                    logging.error(f"Error processing frame from webcam {index}: {e}")
                    
        except Exception as e:
            logging.error(f"Error in webcam thread {index}: {e}")
        finally:
            if cap:
                cap.release()
            if not self.headless_mode:
                cv2.destroyWindow(f"Object Detection Webcam {index}")
    
    def _process_single_frame(self, image_pil: Image.Image, frame: np.ndarray, index: int):
        """Process a single frame with all active features"""
        
        # Expression Comprehension
        if self.expression_comprehension_active and self.phrase:
            results = self.run_expression_comprehension(image_pil, self.phrase)
            if results:
                caption = "Yes" if "yes" in results.lower() else "No"
                self.update_caption_window(caption)
                if self.headless_mode:
                    print(f"Expression result: {caption}")
                self.inference_count += 1
                self.update_inference_rate()
                
                if self.recording_manager:
                    self.recording_manager.handle_recording_by_inference(caption.lower(), frame)
        
        # Object Detection
        if self.object_detection_active:
            self.detections = self.run_object_detection(image_pil)
            if self.headless_mode:
                print(f"Detections from webcam {index}: {self.detections}")
            self.inference_count += 1
            self.update_inference_rate()
            
            # Update target detected flag
            self.target_detected = bool(self.detections)
            
            if self.recording_manager:
                self.recording_manager.handle_recording_by_detection(self.detections, frame)
            
            # PTZ tracking
            if PTZ_AVAILABLE and self.ptz_tracker and self.ptz_tracker.active:
                primary_bbox = self._pick_tracked_object(self.detections)
                if primary_bbox is not None:
                    h, w = frame.shape[:2]
                    self.ptz_tracker.adjust_camera(primary_bbox, w, h)
        
        # Visual Grounding
        if self.visual_grounding_active and self.visual_grounding_phrase:
            bbox = self.run_visual_grounding(image_pil, self.visual_grounding_phrase)
            if bbox:
                if not self.headless_mode:
                    self.plot_visual_grounding_bbox(frame, bbox, self.visual_grounding_phrase)
                else:
                    print(f"Visual grounding result: {bbox}")
                self.inference_count += 1
                self.update_inference_rate()
        
        # Inference Tree
        if self.inference_tree_active and self.inference_title and self.inference_phrases:
            result, phrase_results = self.evaluate_inference_tree(image_pil)
            self.update_inference_result_window(result, phrase_results)
            if self.headless_mode:
                print(f"Inference tree result: {result}, Details: {phrase_results}")
            self.inference_count += 1
            self.update_inference_rate()
        
        # Recording
        if self.recording_manager and self.recording_manager.recording:
            self.recording_manager.write_frame(frame)
    
    def _display_frame(self, frame: np.ndarray, index: int):
        """Display frame with overlays"""
        try:
            bbox_image = self.plot_bbox(frame.copy())
            cv2.imshow(f"Object Detection Webcam {index}", bbox_image)
            
            current_time = time.time()
            
            # Beep on detection
            if self.beep_active and self.target_detected:
                if current_time - self.last_beep_time > 1:
                    threading.Thread(target=self.beep_sound, daemon=True).start()
                    self.last_beep_time = current_time
            
            # Screenshot on detection
            if self.screenshot_active and self.target_detected:
                ImageUtils.save_screenshot(bbox_image)
                
        except Exception as e:
            logging.error(f"Error displaying frame: {e}")
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    def cleanup(self):
        """Comprehensive cleanup method"""
        try:
            logging.info("Starting YO-FLO cleanup...")
            
            # Stop all threads
            self.stop_webcam_detection()
            self.cleanup_flag.set()
            
            # Clean up frame processor
            if self.frame_processor:
                self.frame_processor.shutdown(wait=False)
            
            # Clean up recording manager
            if self.recording_manager:
                self.recording_manager.cleanup()
            
            # Close PTZ camera
            if self.ptz_camera:
                self.ptz_camera.close()
            
            # Clear model from memory
            if self.model:
                del self.model
                self.model = None
            
            if self.processor:
                del self.processor
                self.processor = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Destroy all OpenCV windows
            cv2.destroyAllWindows()
            
            logging.info("YO-FLO cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
    
    # -------------------------------------------------------------------------
    # Main Menu
    # -------------------------------------------------------------------------
    
    def main_menu(self):
        """Create and display the main GUI menu"""
        self.root.deiconify()
        self.root.title(config.WINDOW_TITLE)
        
        def on_closing():
            """Handle window closing"""
            self.cleanup()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        
        try:
            # Model Management Frame
            model_frame = tk.LabelFrame(self.root, text="Model Management")
            model_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Button(
                model_frame,
                text="Select Model Path",
                command=self.select_model_path
            ).pack(fill="x")
            
            tk.Button(
                model_frame,
                text="Download Model from HuggingFace",
                command=self.download_model_gui
            ).pack(fill="x")
            
            tk.Button(
                model_frame,
                text="Toggle File Logging",
                command=self.toggle_file_logging
            ).pack(fill="x")
            
            # Detection Settings Frame
            detection_frame = tk.LabelFrame(self.root, text="Detection Settings")
            detection_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Button(
                detection_frame,
                text="Set Classes for Object Detection",
                command=self.set_class_names
            ).pack(fill="x")
            
            tk.Button(
                detection_frame,
                text="Set Phrase for Yes/No Inference",
                command=self.set_phrase
            ).pack(fill="x")
            
            tk.Button(
                detection_frame,
                text="Set Grounding Phrase",
                command=self.set_visual_grounding_phrase
            ).pack(fill="x")
            
            tk.Button(
                detection_frame,
                text="Set Inference Tree",
                command=self.set_inference_tree
            ).pack(fill="x")
            
            # Feature Toggles Frame
            feature_frame = tk.LabelFrame(self.root, text="Feature Toggles")
            feature_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Button(
                feature_frame,
                text="Object Detection",
                command=self.toggle_object_detection
            ).pack(fill="x")
            
            tk.Button(
                feature_frame,
                text="Yes/No Inference",
                command=self.toggle_expression_comprehension
            ).pack(fill="x")
            
            tk.Button(
                feature_frame,
                text="Visual Grounding",
                command=self.toggle_visual_grounding
            ).pack(fill="x")
            
            tk.Button(
                feature_frame,
                text="Inference Tree",
                command=self.toggle_inference_tree
            ).pack(fill="x")
            
            tk.Button(
                feature_frame,
                text="Headless Mode",
                command=self.toggle_headless
            ).pack(fill="x")
            
            # Triggers Frame
            trigger_frame = tk.LabelFrame(self.root, text="Triggers")
            trigger_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Button(
                trigger_frame,
                text="Beep on Detection",
                command=self.toggle_beep
            ).pack(fill="x")
            
            tk.Button(
                trigger_frame,
                text="Screenshot on Detection",
                command=self.toggle_screenshot
            ).pack(fill="x")
            
            tk.Button(
                trigger_frame,
                text="Screenshot on Yes",
                command=self.toggle_screenshot_on_yes
            ).pack(fill="x")
            
            tk.Button(
                trigger_frame,
                text="Screenshot on No",
                command=self.toggle_screenshot_on_no
            ).pack(fill="x")
            
            # PTZ Control Frame
            if PTZ_AVAILABLE:
                ptz_frame = tk.LabelFrame(self.root, text="PTZ Control")
                ptz_frame.pack(fill="x", padx=10, pady=5)
                
                tk.Button(
                    ptz_frame,
                    text="Open Manual PTZ Control",
                    command=self.open_manual_ptz_control
                ).pack(fill="x")
                
                tk.Button(
                    ptz_frame,
                    text="Set PTZ Target Class",
                    command=self.set_ptz_target_class
                ).pack(fill="x")
                
                tk.Button(
                    ptz_frame,
                    text="Start Autonomous Tracking",
                    command=self.start_autonomous_ptz_tracking
                ).pack(fill="x")
                
                tk.Button(
                    ptz_frame,
                    text="Stop Autonomous Tracking",
                    command=self.stop_autonomous_ptz_tracking
                ).pack(fill="x")
            else:
                ptz_frame = tk.LabelFrame(self.root, text="PTZ Control (Unavailable)")
                ptz_frame.pack(fill="x", padx=10, pady=5)
                
                tk.Label(
                    ptz_frame,
                    text="PTZ functionality not available - missing required modules",
                    fg="red"
                ).pack(fill="x")
            
            # Recording Frame
            recording_frame = tk.LabelFrame(self.root, text="Recording Control")
            recording_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Button(
                recording_frame,
                text="No Recording",
                command=lambda: self.set_record_mode(None)
            ).pack(fill="x")
            
            tk.Button(
                recording_frame,
                text="Record on Detection",
                command=lambda: self.set_record_mode("od")
            ).pack(fill="x")
            
            tk.Button(
                recording_frame,
                text='Record on "Yes"',
                command=lambda: self.set_record_mode("infy")
            ).pack(fill="x")
            
            tk.Button(
                recording_frame,
                text='Record on "No"',
                command=lambda: self.set_record_mode("infn")
            ).pack(fill="x")
            
            # Webcam Control Frame
            webcam_frame = tk.LabelFrame(self.root, text="Webcam Control")
            webcam_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Button(
                webcam_frame,
                text="Start Webcam Detection",
                command=self.start_webcam_detection
            ).pack(fill="x")
            
            tk.Button(
                webcam_frame,
                text="Stop Webcam Detection",
                command=self.stop_webcam_detection
            ).pack(fill="x")
            
            # Debug Frame
            debug_frame = tk.LabelFrame(self.root, text="Debug")
            debug_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Button(
                debug_frame,
                text="Toggle Debug Mode",
                command=self.toggle_debug
            ).pack(fill="x")
            
            # Inference Rate Frame
            inference_rate_frame = tk.LabelFrame(self.root, text="Inference Rate")
            inference_rate_frame.pack(fill="x", padx=10, pady=5)
            
            self.inference_rate_label = tk.Label(
                inference_rate_frame,
                text="Inferences/sec: N/A",
                fg="white",
                bg="black",
                font=("Helvetica", 14, "bold")
            )
            self.inference_rate_label.pack(fill="x")
            
            # Binary Inference Frame
            binary_inference_frame = tk.LabelFrame(self.root, text="Binary Inference")
            binary_inference_frame.pack(fill="x", padx=10, pady=5)
            
            self.caption_label = tk.Label(
                binary_inference_frame,
                text="Binary Inference: N/A",
                fg="white",
                bg="black",
                font=("Helvetica", 14, "bold")
            )
            self.caption_label.pack(fill="x")
            
            # Inference Tree Frame
            inference_tree_frame = tk.LabelFrame(self.root, text="Inference Tree")
            inference_tree_frame.pack(fill="x", padx=10, pady=5)
            
            self.inference_result_label = tk.Label(
                inference_tree_frame,
                text="Inference Tree: N/A",
                fg="white",
                bg="black",
                font=("Helvetica", 14, "bold")
            )
            self.inference_result_label.pack(fill="x")
            
            for i in range(3):
                label = tk.Label(
                    inference_tree_frame,
                    text=f"Inference {i+1}: N/A",
                    fg="white",
                    bg="black",
                    font=("Helvetica", 14, "bold")
                )
                label.pack(fill="x")
                self.inference_phrases_result_labels.append(label)
            
            # Statistics update
            def update_stats():
                if self.frame_processor:
                    stats = self.frame_processor.get_statistics()
                    # You could add a stats label here to display this info
                self.root.after(1000, update_stats)
            
            update_stats()
            
        except Exception as e:
            print(f"{Fore.RED}Error creating menu: {e}{Style.RESET_ALL}")
        
        self.root.mainloop()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with proper error handling and cleanup"""
    app = None
    
    try:
        # Setup logging
        setup_logging(log_to_file=False, log_level=logging.INFO)
        
        # Create application instance
        app = YO_FLO()
        app.init_model_manager(quantization_mode=None)
        
        print(f"{Fore.BLUE}{Style.BRIGHT}YO-FLO Vision System v2.0{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Enhanced with security, threading, and memory management{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Created with comprehensive improvements for production use{Style.RESET_ALL}")
        
        # Run the GUI
        app.main_menu()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
        
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        
    finally:
        # Ensure cleanup
        if app:
            app.cleanup()
        logging.info("Application shutdown complete")

if __name__ == "__main__":
    main()
