import sys
import os

sys.path.append("../..")

from pprint import pprint
import pathlib
from api.api.coral_interface import Handler

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(
    script_dir, "artefacts/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
)
label_file = os.path.join(script_dir, "artefacts/coco_labels.txt")
image_file = os.path.join(script_dir, "artefacts/parrot.jpg")

handler = Handler()
handler.initialize(
    path_or_bytes_model=model_file, path_or_bytes_labels=label_file, model_name="yolo"
)


results = handler.inference(UUID="tracking_id", image=image_file, n_labels=3)

pprint(results.dict())
