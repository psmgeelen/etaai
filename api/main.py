import io
import os
import pathlib
import time
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi_health import health
from fastapi.openapi.utils import get_openapi
from coral_interface import Handler, PredictionSet, InitializationResults, Devices
import logging

# Setup Logger
logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s", level=logging.WARNING
)
logger = logging.getLogger("my-logger")
stream_handler = handler = logging.StreamHandler()
logger.addHandler(stream_handler)

script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(
    script_dir, "test/artefacts/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
)
label_file = os.path.join(script_dir, "test/artefacts/coco_labels.txt")
image_file = os.path.join(script_dir, "test/artefacts/parrot.jpg")

handler = Handler()
handler.initialize(
    path_or_bytes_model=model_file, path_or_bytes_labels=label_file, model_name="yolo"
)

app = FastAPI()


def my_schema():
    DOCS_TITLE = "Eta API"
    DOCS_VERSION = "0.1"
    openapi_schema = get_openapi(
        title=DOCS_TITLE,
        version=DOCS_VERSION,
        routes=app.routes,
    )
    openapi_schema["info"] = {
        "title": DOCS_TITLE,
        "version": DOCS_VERSION,
        "description": (
            "η.ai showcases low-precision AI's speed and power efficiency, offering a"
            " free API for professionals. It explores analogue AI for even greater"
            " gains in efficiency. The source-code to this platform is provided in this"
            " repo."
        ),
        "contact": {
            "name": "A project of geelen.io",
            "url": "https://github.com/psmgeelen/etaai",
        },
        "license": {"name": "UNLICENSE", "url": "https://unlicense.org/"},
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = my_schema


@app.get(
    "/ping",
    summary="Check whether there is a connection at all",
    description="You ping, API should Pong",
    response_description="A string saying Pong",
)
def ping():
    return "pong"


@app.get(
    "/list_devices",
    summary="Get a list of all the available devices",
    description=(
        "This request returns a list of devices. If no hardware is found, it will"
        " return the definition of the DeviceEmulator class"
    ),
    response_description="A dictionary with a list of devices",
    response_model=Devices,
)
def list_devices():
    return handler.list_devices()


@app.post(
    "/load_model",
    summary="Load a new model into all devices that are available",
    description=(
        "This endpoint can serve a new tensorflow-lite model to the TPU. Please note"
        " that it will also require the labels file. If the model fails to load, it"
        " will automatically perform a rollback."
    ),
    response_description=(
        "Postive or Negative feedback in regards to loading of the model"
    ),
    response_model=InitializationResults,
)
def load_model(
    modelname: str = Form(...),
    modelfile: UploadFile = File(...),
    labelsfile: UploadFile = File(...),
):
    model = io.BytesIO(modelfile.file.read()).getvalue()
    labels = io.BytesIO(labelsfile.file.read()).getvalue()
    callback = handler.initialize(
        model_name=modelname,
        path_or_bytes_model=model,
        path_or_bytes_labels=labels,
        callback=True,
    )
    return callback


@app.put(
    "/inference",
    summary="Detect what objects are in the picture",
    description=(
        "This endpoint enables you to send an image and get back predictions about the"
        " object within that picture."
    ),
    response_description=(
        "Predictions on the image that has been put onto the endpoint. Note that I will"
        " also share some additional statistics to prove efficiency of this project."
    ),
    response_model=PredictionSet,
)
def inference(
    UUID: str = Form(...),
    image: UploadFile = File(...),
    nlabels: int = Form(...),
):
    try:
        downloaded_image: bytes = image.file.read()
        results = handler.inference(
            UUID=UUID, image=io.BytesIO(downloaded_image), n_labels=nlabels
        )

        return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise e
        return {"error": e}


# Health
def _healthcheck_list_devices():
    devices = handler.list_devices()
    if len(devices.devices) != 0:
        return devices
    else:
        return "error: cant list devices"


def _healthcheck_inference_e2e():
    script_dir = pathlib.Path(__file__).parent.absolute()
    model_file = os.path.join(
        script_dir, "test/artefacts/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
    )
    label_file = os.path.join(script_dir, "test/artefacts/coco_labels.txt")
    image_file = os.path.join(script_dir, "test/artefacts/parrot.jpg")

    handler.initialize(
        path_or_bytes_model=model_file,
        path_or_bytes_labels=label_file,
        model_name="yolo",
    )
    results = handler.inference(UUID="tracking_id", image=image_file, n_labels=3)
    if results is not None:
        return "succesfully infered test image"
    else:
        return "error: cant infer test image"


def _healthcheck_ping():
    hostname = "google.com"  # example
    response = os.system("ping -c 1 " + hostname)
    print(response)

    # and then check the response...
    if response == 0:
        return str(response)
    else:
        return False


app.add_api_route(
    "/health",
    health([_healthcheck_list_devices, _healthcheck_inference_e2e, _healthcheck_ping]),
    summary="Check the health of the service",
    description=(
        "The healthcheck not only checks whether the service is up, but it will also check"
        " for internet connectivity, whether the hardware is callable and it does an"
        " end-to-end test. The healthcheck therefore can become blocking by nature. Use"
        " with caution!"
    ),
    response_description=(
        "The response is only focused around the status. 200 is OK, anything else and"
        " there is trouble."
    ),
)
