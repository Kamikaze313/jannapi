from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from pydantic import BaseModel
import uvicorn
import pandas as pd


app = FastAPI()
class PINRequest(BaseModel):
    pin: int




class PINResponse(BaseModel):
    place: str
    district: str
    hospital_type: str





pin_data = pd.read_excel("C:/Users/NAJID/Documents/lllll/O.xlsx")
np.set_printoptions(suppress=True)
model = load_model("C:/Users/NAJID/Documents/model/keras_model.h5", compile=False)
class_names = open("C:/Users/NAJID/Documents/model/labels.txt", "r").readlines()



@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(file.file).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return {'class_name': class_name, 'confidence_score': float(confidence_score)}




@app.post("/pin")
async def get_pin_details(pin_request: PINRequest):
    pin = pin_request.pin
    filtered_data = pin_data[pin_data["PIN"] == pin]
    if not filtered_data.empty:
        place = filtered_data["Place"].iloc[0]
        district = filtered_data["District"].iloc[0]
        hospital_type = filtered_data["Hospital Type"].iloc[0]
        return PINResponse(place=place, district=district, hospital_type=hospital_type)
    else:
        return {"error": "Invalid PIN"}



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)