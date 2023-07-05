from fastapi.testclient import TestClient
import os
import pathlib
from main import app

client = TestClient(app)


def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.content.decode() == '"pong"'


def test_list_devices():
    response = client.get("/list_devices")
    assert response.status_code == 200
    assert isinstance(response.json()["devices"], list)
    assert isinstance(response.json(), dict)


def test_inference():
    script_dir = pathlib.Path(__file__).parent.absolute()
    image_path = os.path.join(script_dir, "test/artefacts/parrot.jpg")
    file = {"image": (image_path, open(image_path, "rb"))}
    response = client.put("/inference", files=file, data={"UUID": "1", "nlabels": "3"})
    results = response.json()
    assert response.status_code == 200, response.status_code
    assert (
        "predictions" in results
    ), "Key missing from output, check Pydantic data-object"
    assert "device" in results, "Key missing from output, check Pydantic data-object"
    assert (
        "exec_time_coral_seconds" in results
    ), "Key missing from output, check Pydantic data-object"
    assert (
        "power_consumption_coral_per_inference_mWh" in results
    ), "Key missing from output, check Pydantic data-object"


def test_loading_new_model():
    script_dir = pathlib.Path(__file__).parent.absolute()
    modelfile = os.path.join(
        script_dir, "test/artefacts/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
    )
    labelsfile = os.path.join(
        script_dir, "test/artefacts/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
    )
    files = {
        "modelfile": (modelfile, open(modelfile, "rb")),
        "labelsfile": (labelsfile, open(labelsfile, "rb")),
    }
    model_name = "testing"
    response = client.post("/load_model", files=files, data={"modelname": model_name})
    results = response.json()
    assert response.status_code == 200, response.status_code
    assert "success" in results, "Key missing from output, check Pydantic data-object"
    assert (
        "description" in results
    ), "Key missing from output, check Pydantic data-object"
    assert results["success"] is True, "Failed to load a new model"
