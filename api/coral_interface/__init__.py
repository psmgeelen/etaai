from __future__ import annotations
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
        self.model_name = None
        self.model = None
        self.labels = None

    def initialize(
        self,
        model_name: str,
        path_or_bytes_model: str | bytes,
        path_or_bytes_labels: str | bytes,
        callback=False,
    ) -> dict:
        # Check whether dirs exist
        if isinstance(path_or_bytes_model, str):
            self._check_whether_file_exists(path_or_bytes_model)
        if isinstance(path_or_bytes_labels, str):
            self._check_whether_file_exists(path_or_bytes_labels)

        results = InitResults()
        # Initialize model
        try:
            self.logger.warning(f"initializing model: {model_name}")
            self.device.initialize_model(
                model_name=model_name,
                path_to_model_file=path_or_bytes_model,
                labels_file=path_or_bytes_labels,
            )
            self.is_initialized = True
            self.logger.warning(f"successfully loaded model: {model_name}")
            results.success = True
            results.description = f"successfully loaded model: {model_name}"

            # Successfully loaded model
            # Overwrite properties of the handler
            self.model_name = model_name
            self.model = path_or_bytes_model
            self.labels = path_or_bytes_labels
        except Exception:
            self.logger.error("failed to initialize model")
            self.device.initialize_model(
                model_name=self.model_name,
                path_to_model_file=self.model,
                labels_file=self.labels,
            )
            results.success = False
            results.description = f"failed to load model: {model_name}"
        if callback:
            return results.dict()

    def inference(self, UUID: str, image: bytes, n_labels: int = 10) -> list[dict]:
        assert (
            self.is_initialized is True
        ), "Device is not initialized, please initialize first"

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
        self, model_name: str, path_to_model_file: str | bytes, labels_file: str | bytes
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
        # Yield interpreter so that rotation happens before inference
        self.interpreters.rotate(1)
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
        self, model_name: str, path_to_model_file: str | bytes, labels_file: str | bytes
    ):
        self.model_name = model_name

        if isinstance(path_to_model_file, bytes):
            interpreter = tflite.Interpreter(model_content=path_to_model_file)
        else:
            interpreter = tflite.Interpreter(model_path=path_to_model_file)
        self.interpreters = deque(
            [
                {
                    "name": "emulator",
                    "interpreter": interpreter,
                }
            ]
        )
        self.size = self.interpreters[0]["interpreter"].get_input_details()[0]["shape"][
            1:3
        ]
        self.labels_file = labels_file

    @staticmethod
    def list_devices() -> list[dict]:
        yield [{"type": "emulator", "path": "python_lib"}]

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
            device={"device": "Emulator"},
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


class InitResults(BaseModel):
    success: bool = None
    description: str = None
