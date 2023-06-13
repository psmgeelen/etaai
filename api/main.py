import io
import os
import pathlib
import time
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi_health import health
from coral_interface import Handler
import logging

# Setup Logger
logging.basicConfig(format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger("my-logger")
stream_handler = handler = logging.StreamHandler()
logger.addHandler(stream_handler)

logger.error(
    "Something happened",
    extra = {"tags": {"service": "my-service"}},
)


script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(
    script_dir, "test/artefacts/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
)
label_file = os.path.join(script_dir, "test/artefacts/coco_labels.txt")
image_file = os.path.join(script_dir, "test/artefacts/parrot.jpg")

handler = Handler()
handler.initialize(
    path_to_model_file="test/artefacts/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite",
    labels_file="test/artefacts/coco_labels.txt",
    model_name="yolo"
)

app = FastAPI()


# TODO, should we make this async?
# TODO, look into documentation (sphinx and swagger)
# TODO, look into building containers
# TODO, look into CI
# TODO, look into remotely initializing the device
@app.get("/ping")
def ping():
    return "pong"


@app.get("/list_devices")
def list_devices():
    return handler.list_devices()


@app.put("/initialize_device")
def list_devices():
    return handler.list_devices()


@app.put("/inference")
def inference(
    UUID: str = Form(...), image: UploadFile = File(...), nlabels: int = Form(...)
):
    start_time_call = time.time()
    try:
        downloaded_image = image.file.read()
        results = handler.inference(
            UUID=UUID, image=io.BytesIO(downloaded_image), n_labels=nlabels
        )
        end_time_call = time.time()
        execution_time = end_time_call - start_time_call
        results.power_consumption_system_per_inference_mWh = 50 * execution_time / 6000
        return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise e
        return {"error": e}


# Health
def _healthcheck_list_devices():
    devices = handler.list_devices()
    if len(devices) != 0:
        return devices
    else:
        return "error: cant list devices"


def _healthcheck_inference():
    image_file = os.path.join(script_dir, "test/artefacts/parrot.jpg")
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
    health([_healthcheck_list_devices, _healthcheck_inference, _healthcheck_ping]),
)
