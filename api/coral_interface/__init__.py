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
    """The handler object handles 2 specific cases of operation.
    1. The device(s) is/are detected, therefore creating a
    queue of interpreters, respective to the devices based on
    the CoralWrapper object. The CoralWrapper handles an
    individual device.
    2. There are no devices detected, therefore loading the
    DeviceEmulator in a similar fashion as when it were
    loading CoralWrappers.

    Furthermore, the Handler aims to be the interface for the
    FastAPI implementation. One of the key aspects is to
    pass through any requests to the devices that are available.
    Another aspect is rotating the devices, ensuring a
    naive load-balancing approach. The handler also manages
    state and includes:

      * Whether the devices have been initialized,
      * What data was used for last initialization
      * Reporting the overall process with a logger

    """

    def __init__(self):
        self.is_initialized = False
        self.logger = logging.getLogger(__name__)
        devices = CoralWrapper.list_devices()
        self.logger.info(f"looking for devices, found {devices}")

        if len(devices.devices) == 0:
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
        """The initialize method initializes the Handler, and therefore the
        associated devices. If no device is available then the handler will
        initialize a DeviceEmulator class. The Method iterates over all
        available devices, meaning that all devices will be initialized with
        the same model. If initialization failed, the old model would be
        restored on the device.

        :param model_name: This is an arbitrary to track the which model has
            been loaded.
        :param path_or_bytes_model: The model can be loaded by pointing to
            path or providing an object that contain the bytes.
        :param path_or_bytes_labels: The labels can be loaded by pointing
            to path or providing an object that contain the bytes.
        :param callback: A boolean indicator whether a callback about
            initialization should be returned.
        :return: This is a callback that returns whether initialization
            was a success or not. Note that this callback will only be
            provided when the callback parameter is set to `True`
        """
        # Check whether dirs exist
        if isinstance(path_or_bytes_model, str):
            self._check_whether_file_exists(path_or_bytes_model)
        if isinstance(path_or_bytes_labels, str):
            self._check_whether_file_exists(path_or_bytes_labels)

        results = InitializationResults()
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

    def inference(
        self, UUID: str, image: bytes | str, n_labels: int = 10
    ) -> list[dict]:
        """This is a wrapper function to handle the image inference handled by
        multiple devices. The function resizes the image and parses it to the
        CoralWrapper or DeviceEmulator class to handle the inference.

        :param UUID: An arbitrary UUID that can be set by the user to
            track their image.
        :param image: Image in bytes that needs to be inferred.
        :param n_labels: Amount of Labels that the users wants returned
        :return: A list of Predictions
        """
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
    """The CoralWrapper class wraps the pycoral library into something more
    stateful. Parameters include:

    * Whether the devices have been initialized,
    * What data was used for last initialization
    * What model is being used
    * What labels are associated with the outputs
    * The size for the input, as to facilitate the image transformation.
    """

    def __init__(self):
        self.model_name = None
        self.path_to_model_file = None
        self.interpreters = None
        self.labels_file = None
        self.size = None

    def initialize_model(
        self, model_name: str, path_to_model_file: str | bytes, labels_file: str | bytes
    ):
        """The initialize method initializes the devices. The Method iterates
        over all available devices, meaning that all devices will be
        initialized with the same model. A queue object will be stored as an
        attribute the CoralWrapper class that contains all the interpreters
        that are being initialized here.

        :param model_name: This is an arbitrary to track the which model
            has been loaded.
        :param path_to_model_file: The model can be loaded by pointing
            to path or providing an object that contain the bytes.
        :param labels_file: The labels can be loaded by pointing to path
            or providing an object that contain the bytes.
        :return: None
        """
        self.model_name = model_name
        self.interpreters = deque()
        for nr, tpu in enumerate(self.list_devices().devices):
            interpreter = edgetpu.make_interpreter(
                model_path_or_content=path_to_model_file, device=nr
            )
            interpreter.allocate_tensors()
            self.interpreters.append({"name": tpu, "interpreter": interpreter})
        self.size = common.input_size(self.interpreters[0]["interpreter"])
        self.labels_file = labels_file

    @staticmethod
    def list_devices() -> Devices:
        """
        This method calls on the `list_edge_tpus` method of the pycoral library
        and returns a list of dictionaries for each detected device.

        :return: A list with devices
        """
        devices = list()
        for item in edgetpu.list_edge_tpus():
            devices.append(Device(type=item["type"], path=item["path"]))
        return Devices(devices=devices)

    def inference(self, resized_image: Image.Image, n_labels: int = 10) -> list[dict]:
        """This method processes the image and infers what is in the image by
        using a TPU. Moreover, this method handles the naive load-balancing,
        as it rotates the queue of interpreters right before every invocation.

        :param resized_image: Image Object that is being inferred
        :param n_labels: Amount of labels that have to be returned later
            on
        :return: Predictions and other specifications
        """
        # assure that the device has been initialized
        if self.interpreters is None:
            Exception(
                "Please make sure that you first "
                "initialize the device before trying to use it"
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
    """The DeviceEmulator is a digital-twin of the device, emulating the
    expected behaviour of Google Coral. This is mainly done for testing
    purposes and wraps the pycoral library into something more stateful.

    Parameters include:
    * Whether the devices have been initialized,
    * What data was used for the last initialization
    * What model is being used
    * What labels are associated with the outputs
    * The size for the input, as to facilitate the image transformation.

    Unfortunately the setup with the Google Coral requires a specific
    version of the pycoral and tensorflow-lite library, creating a
    delta in functionality when it comes to creating interpreters. As
    it stand now, it is not possible to truly load a model without a
    Google Coral.
    """

    def __init__(self):
        self.model_name = None
        self.path_to_model_file = None
        self.interpreters = None
        self.labels_file = None
        self.size = None

    def initialize_model(
        self, model_name: str, path_to_model_file: str | bytes, labels_file: str | bytes
    ):
        """The initialize method initializes the DeviceEmulator. A que object
        will be stored as an attribute, similarly as within the CoralWrapper,
        that contains one interpreter. Unfortunately the custom version of
        tensorflow-lite doesn't allow for invoking the interpreter, as this
        would require a Google Coral.

        :param model_name: This is an arbitrary to track the which model
            has been loaded.
        :param path_to_model_file: The model can be loaded by pointing
            to path or providing an object that contain the bytes.
        :param labels_file: The labels can be loaded by pointing to path
            or providing an object that contain the bytes.
        :return: None
        """
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
    def list_devices() -> Devices:
        """
        This method emulates the call of `list_edge_tpus` of the pycoral
        library and return a list containing a
        single emulator device.
        :return: A list with a single emulator device.
        """
        return Devices(Device(type="emulator", path="python_lib"))

    def inference(self, resized_image: Image.Image, n_labels: int = 10) -> list[dict]:
        """This method processes the image and emulate inference. Moreover,
        this method handles the naive load-balancing, as it rotates the queue of
        interpreters right before every invocation.

        :param resized_image: Image Object that is being inferred
        :param n_labels: Amount of labels that have to be returned later
            on
        :return: Predictions and other specifications
        """
        # assure that the device has been initialized
        if self.interpreters is None:
            Exception(
                "Please make sure that you first initialize "
                "the device before trying to use it"
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


class InitializationResults(BaseModel):
    success: bool = None
    description: str = None


class Device(BaseModel):
    type: str
    path: str


class Devices(BaseModel):
    devices: list[Device]
