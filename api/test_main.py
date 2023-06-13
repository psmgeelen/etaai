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
    assert isinstance(response.json(), list)


def test_inference():
    script_dir = pathlib.Path(__file__).parent.absolute()
    image_path = os.path.join(script_dir, "test/artefacts/parrot.jpg")
    file = {"image": (image_path, open(image_path, "rb"))}
    response = client.put("/inference", files=file, data={"UUID": "1", "nlabels": "3"})
    results = response.json()
    assert response.status_code == 200
    assert "predictions" in results
    assert "device" in results
    assert "exec_time_coral_seconds" in results
    assert "power_consumption_coral_per_inference_mWh" in results
