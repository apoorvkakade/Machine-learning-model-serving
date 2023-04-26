from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
from PIL import Image
import numpy as np
import settings
import helper_utilities
import uuid
from fastapi import FastAPI, File, UploadFile
import time
import json
import io
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from typing_extensions import Annotated

# initialize our Flask API application 
app = FastAPI()

print("* Loading model...")
model = ResNet50(weights="imagenet")
print("* Model loaded")

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	# return the processed image
	return image


@app.route("/")
def homepage():
	return "Welcome to the DISML project"
@app.post("/predict")
def predict(image: Annotated[bytes, File()]):
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image,(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
    image = image.copy(order="C")
    preds = model.predict(image)
    results = decode_predictions(preds)
    for resultSet in results:
        for (imagenetID, label, prob) in resultSet:
            r = {"label": label, "probability": float(prob)}
            output.append(r)
        data["predictions"] = output
        data["success"] = True
    return data

if __name__ == "__main__":
	print("* Starting web service...")
	app.run(host="0.0.0.0", port=8000)