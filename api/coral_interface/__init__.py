import os
import logging
from PIL import Image
from collections import deque
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
import time

# Multiple devices?
import tflite_runtime.interpreter as tflite
from pydantic import BaseModel


class Handler(object):
    def __init__(self):
        self.is_initialized = False
        self.logger = logging.getLogger(__name__)
        devices = CoralWrapper.list_devices()
        self.logger.info(f"looking for devices, found {devices}")

        if len(devices) == 0:
            self.logger.warning("No devices found, using Emulator")
            self.device = DeviceEmulator()
        else:
            self.logger.info("Found hardware, starting up..")
            self.device = CoralWrapper()

    def initialize(self, model_name: str, path_to_model_file: str, labels_file: str):
        # Check whether dirs exist
        self._check_whether_file_exists(path_to_model_file)
        self._check_whether_file_exists(labels_file)
        # Initialize model
        self.device.initialize_model(
            model_name=model_name,
            path_to_model_file=path_to_model_file,
            labels_file=labels_file,
        )
        self.is_initialized = True

    def inference(self, UUID: str, image, n_labels: int = 10) -> list[dict]:
        assert (
            self.is_initialized is True
        ), "Device is not initialized, please initialize first"

        # image_bytes = base64.urlsafe_b64decode(base64_image)  # im_bytes is a binary image
        # image_file = BytesIO(image_bytes)  # convert image to file-like object
        resized_image = (
            Image.open(image).convert("RGB").resize(self.device.size, Image.AFFINE)
        )

        results = self.device.inference(resized_image=resized_image, n_labels=n_labels)
        results.UUID = UUID
        return results

    def list_devices(self):
        return self.device.list_devices()

    def _check_whether_file_exists(self, path: str):
        if not os.path.isfile(path):
            Exception(FileExistsError)
        else:
            self.logger.error(f"could not find: {path}")


class CoralWrapper(object):
    def __init__(self):
        self.model_name = None
        self.path_to_model_file = None
        self.interpreters = None
        self.labels_file = None
        self.size = None

    def initialize_model(
        self, model_name: str, path_to_model_file: str, labels_file: str
    ):
        self.model_name = model_name
        self.interpreters = deque()
        for nr, tpu in enumerate(self.list_devices()):
            interpreter = edgetpu.make_interpreter(
                model_path_or_content=path_to_model_file, device=nr
            )
            interpreter.allocate_tensors()
            self.interpreters.append({"name": tpu, "interpreter": interpreter})
        self.size = common.input_size(self.interpreters[0]["interpreter"])
        self.labels_file = labels_file

    @staticmethod
    def list_devices() -> list[dict]:
        devices = edgetpu.list_edge_tpus()
        return devices

    def inference(self, resized_image, n_labels: int = 10):
        # assure that the device has been initialized
        if self.interpreters is None:
            Exception(
                "Please make sure that you first initialize the device before trying to use it"
            )

        start_time_inference = time.time()
        # Run an inference
        common.set_input(self.interpreters[0]["interpreter"], resized_image)
        self.interpreters[0]["interpreter"].invoke()
        end_time_inference = time.time()
        execution_time = end_time_inference - start_time_inference

        # Get results
        classes = classify.get_classes(
            self.interpreters[0]["interpreter"], top_k=n_labels
        )

        # Get labels to identify what the individual labels mean
        labels = dataset.read_label_file(self.labels_file)

        # Unpack labels
        predictions = [
            SinglePrediction(label=labels.get(c.id, c.id), score=float(c.score))
            for c in classes
        ]
        results = PredictionSet(
            predictions=predictions,
            exec_time_coral_seconds=execution_time,
            power_consumption_coral_per_inference_mWh=2.5 * execution_time / 6000,
            device=self.interpreters[0]["name"],
        )
        # rotate device
        self.interpreters.rotate(1)

        return results


class DeviceEmulator(object):
    # This is a digital-twin of the device, emulating the expected behaviour of Google Coral. This is mainly
    # Done for testing purposes

    def __init__(self):
        self.model_name = None
        self.path_to_model_file = None
        self.interpreters = None
        self.labels_file = None
        self.size = None

    def initialize_model(
        self, model_name: str, path_to_model_file: str, labels_file: str
    ):
        self.model_name = model_name
        self.interpreters = deque(
            [
                {
                    "name": "emulator",
                    "interpreter": tflite.Interpreter(model_path=path_to_model_file),
                }
            ]
        )
        self.size = self.interpreters[0].get_input_details()[0]["shape"][1:3]
        self.labels_file = labels_file

    @staticmethod
    def list_devices() -> list[dict]:
        return [{"type": "emulator", "path": "python_lib"}]

    def inference(self, resized_image, n_labels: int = 10):
        # assure that the device has been initialized
        if self.interpreters is None:
            Exception(
                "Please make sure that you first initialize the device before trying to use it"
            )

        # rotate device
        self.interpreters.rotate(1)
        results = PredictionSet(
            predictions=[SinglePrediction(label="this is an emulator", score=0.0)],
            exec_time_coral_seconds=0,
            power_consumption_coral_per_inference_mWh=0,
            device="Emulator",
        )
        return results


class SinglePrediction(BaseModel):
    label: str
    score: float


class PredictionSet(BaseModel):
    predictions: list[SinglePrediction]
    exec_time_coral_seconds: float
    power_consumption_coral_per_inference_mWh: float
    power_consumption_system_per_inference_mWh: float = 0.0
    device: dict = None
    UUID: str = None
