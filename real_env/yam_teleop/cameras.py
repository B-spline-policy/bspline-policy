# Author: Jimmy Wu and Haoyu Xiong
# Date: March 2026

import json
import os
import threading
import time

import cv2 as cv
import numpy as np

try:
    import depthai as dai

    _DEPTHAI_AVAILABLE = True
except ImportError:
    dai = None
    _DEPTHAI_AVAILABLE = False
    print("DepthAI not found, OAKCamera unavailable")

from constants import RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT

class OAKCamera:
    """Luxonis OAK via DepthAI.

    Manual controls default to None: auto exposure, auto white balance, and autofocus.
    Pass numeric values (or a JSON config) to override.

    For several OAKs on one host, pass ``device_id`` (DepthAI MXID / device ID, IP, or USB
    path) so each instance opens the correct unit; see ``depthai.Device.getAllAvailableDevices()``.
    """

    DEFAULT_CONFIG_PATH = "oak_camera_config.json"

    def __init__(
        self,
        socket=None,
        device_id=None,
        output_raw=False,
        output_video=True,
        frame_width=RAW_IMAGE_WIDTH,
        frame_height=RAW_IMAGE_HEIGHT,
        fps=30,
        resize_mode=None,
        exposure_time_us=None,
        iso=None,
        white_balance_k=None,
        manual_focus=None,
        config_path=None,
    ):
        if not _DEPTHAI_AVAILABLE:
            raise ImportError("depthai is required for OAKCamera; pip install depthai")
        if socket is None:
            socket = dai.CameraBoardSocket.CAM_A
        if resize_mode is None:
            resize_mode = dai.ImgResizeMode.CROP
        self.socket = socket
        self.device_id = str(device_id).strip() if device_id else None
        self.output_raw = output_raw
        self.output_video = output_video
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.resize_mode = resize_mode

        if config_path is not None and os.path.exists(config_path):
            config = self.load_config(config_path)
            self.exposure_time_us = config.get("exposure_time_us", exposure_time_us)
            self.iso = config.get("iso", iso)
            self.white_balance_k = config.get("white_balance_k", white_balance_k)
            self.manual_focus = config.get("manual_focus", manual_focus)
            print(f"OAKCamera: Loaded config from {config_path}")
        else:
            self.exposure_time_us = exposure_time_us
            self.iso = iso
            self.white_balance_k = white_balance_k
            self.manual_focus = manual_focus

        self.image = None
        self.last_read_time = time.time()
        self._image_lock = threading.Lock()
        self.raw_image = None
        self.raw_queue = None
        self.video_queue = None
        self.running = False

        if self.device_id:
            self._bound_device = dai.Device(dai.DeviceInfo(self.device_id))
            self.pipeline = dai.Pipeline(self._bound_device)
        else:
            self._bound_device = None
            self.pipeline = dai.Pipeline()

        self._init_pipeline()
        threading.Thread(target=self.camera_worker, daemon=True).start()
        time.sleep(2)  # Allow USB bus to settle before next device is opened

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_config(config_path, exposure_time_us, iso, white_balance_k, manual_focus):
        config = {
            "exposure_time_us": exposure_time_us,
            "iso": iso,
            "white_balance_k": white_balance_k,
            "manual_focus": manual_focus,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"OAKCamera: Saved config to {config_path}")

    def _init_pipeline(self):
        cam = self.pipeline.create(dai.node.Camera).build(self.socket)

        if self.exposure_time_us is not None and self.iso is not None:
            cam.initialControl.setManualExposure(self.exposure_time_us, self.iso)
        if self.white_balance_k is not None:
            cam.initialControl.setManualWhiteBalance(self.white_balance_k)
        if self.manual_focus is not None:
            cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            cam.initialControl.setManualFocus(self.manual_focus)

        if self.output_raw:
            self.raw_queue = cam.raw.createOutputQueue()

        if self.output_video:
            output = cam.requestOutput(
                size=(self.frame_width, self.frame_height),
                type=dai.ImgFrame.Type.BGR888p,
                resizeMode=self.resize_mode,
                fps=self.fps,
            )
            self.video_queue = output.createOutputQueue()

    def _unpack_raw10(self, raw_data, width, height, stride=None):
        if stride is None:
            stride = width * 10 // 8
        expected_size = stride * height

        if len(raw_data) < expected_size:
            raise ValueError(
                f"Data too small: {len(raw_data)} bytes, expected {expected_size}"
            )

        packed_data = np.frombuffer(raw_data, dtype=np.uint8)
        result = np.zeros((height, width), dtype=np.uint16)

        for row in range(height):
            row_start = row * stride
            row_data = packed_data[row_start : row_start + stride]
            num_groups = (width + 3) // 4
            row_bytes = num_groups * 5

            if len(row_data) < row_bytes:
                break

            row_packed = row_data[:row_bytes].reshape(-1, 5)
            row_unpacked = np.zeros((row_packed.shape[0], 4), dtype=np.uint16)

            row_unpacked[:, 0] = row_packed[:, 0].astype(np.uint16) << 2
            row_unpacked[:, 1] = row_packed[:, 1].astype(np.uint16) << 2
            row_unpacked[:, 2] = row_packed[:, 2].astype(np.uint16) << 2
            row_unpacked[:, 3] = row_packed[:, 3].astype(np.uint16) << 2

            row_unpacked[:, 0] |= row_packed[:, 4] & 0b00000011
            row_unpacked[:, 1] |= (row_packed[:, 4] & 0b00001100) >> 2
            row_unpacked[:, 2] |= (row_packed[:, 4] & 0b00110000) >> 4
            row_unpacked[:, 3] |= (row_packed[:, 4] & 0b11000000) >> 6

            row_flat = row_unpacked.flatten()
            result[row, :width] = row_flat[:width]

        return (result * 64).astype(np.uint16)

    def camera_worker(self):
        self.pipeline.start()
        self.running = True

        while self.running:
            if self.output_video and self.video_queue is not None:
                video_frame = self.video_queue.tryGet()
                if video_frame is not None:
                    assert isinstance(video_frame, dai.ImgFrame)
                    bgr_image = video_frame.getCvFrame()
                    capture_time = time.time()
                    if bgr_image is not None:
                        with self._image_lock:
                            self.image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
                            self.last_read_time = capture_time

            if self.output_raw and self.raw_queue is not None:
                raw_frame = self.raw_queue.tryGet()
                if raw_frame is not None:
                    assert isinstance(raw_frame, dai.ImgFrame)
                    data_raw = raw_frame.getData()
                    self.raw_image = self._unpack_raw10(
                        data_raw,
                        raw_frame.getWidth(),
                        raw_frame.getHeight(),
                        raw_frame.getStride(),
                    )

            time.sleep(0.001)

    def get_image(self):
        with self._image_lock:
            return self.image

    def get_raw_image(self):
        return self.raw_image

    def close(self):
        self.running = False
        time.sleep(0.1)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None


def oak_preview_tick(cameras_by_name):
    """Refresh OpenCV windows for each ``OAKCamera`` (RGB→BGR). Call from the main thread only.

    Returns ``waitKey(1) & 0xFF`` (``ord("s")`` triggers ``oak_save_snapshots`` in callers).
    """
    for name, camera in cameras_by_name.items():
        img = camera.get_image()
        if img is None:
            continue
        cv.imshow(name, cv.cvtColor(img, cv.COLOR_RGB2BGR))
    return cv.waitKey(1) & 0xFF


def oak_preview_combined(cameras_by_name, window_name='cameras'):
    """Show all cameras tiled side-by-side in a single window. Call from the main thread only.

    Each panel is labelled with the camera name. Panels with no frame yet are shown as a
    black placeholder matching the size of the first available frame.

    Returns ``waitKey(1) & 0xFF`` (``ord("s")`` triggers ``oak_save_snapshots`` in callers).
    """
    panels = []
    ref_h, ref_w = None, None

    for name, camera in cameras_by_name.items():
        img = camera.get_image()
        if img is not None:
            bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            if ref_h is None:
                ref_h, ref_w = bgr.shape[:2]
        else:
            if ref_h is not None:
                bgr = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
            else:
                bgr = None  # no frame from any camera yet

        if bgr is not None:
            cv.putText(bgr, name, (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
            panels.append(bgr)

    if panels:
        combined = np.concatenate(panels, axis=1)
        cv.imshow(window_name, combined)

    return cv.waitKey(1) & 0xFF


def oak_save_snapshots(cameras_by_name, ts=None):
    """Write ``oak-<name>-<ts>.jpg`` for each camera with a non-``None`` frame (same as ``__main__`` here)."""
    if ts is None:
        ts = int(10 * time.time()) % 100000000
    for name, camera in cameras_by_name.items():
        img = camera.get_image()
        if img is None:
            continue
        path = f"oak-{name}-{ts}.jpg"
        cv.imwrite(path, cv.cvtColor(img, cv.COLOR_RGB2BGR))
        print(f"Saved image to {path}")


if __name__ == "__main__":
    if not _DEPTHAI_AVAILABLE:
        raise SystemExit("Install depthai to run this module: pip install depthai")

    from constants import WRIST_CAMERA_SERIAL

    # Single OAK wrist camera. Set WRIST_CAMERA_SERIAL in constants.py to pick a
    # specific unit; leave it as 'TODO'/None to open the first OAK found.
    serial = str(WRIST_CAMERA_SERIAL).strip() if WRIST_CAMERA_SERIAL else ''
    device_id = serial if serial and serial != 'TODO' else None

    cameras = {
        "wrist_camera": OAKCamera(
            socket=dai.CameraBoardSocket.CAM_A,
            device_id=device_id,
            output_raw=False,
            output_video=True,
        )
    }
    try:
        while True:
            key = oak_preview_tick(cameras)
            if key == ord("s"):
                oak_save_snapshots(cameras)
    finally:
        for camera in cameras.values():
            camera.close()
        cv.destroyAllWindows()
