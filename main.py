from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
import uvicorn

# -------------------- MODEL --------------------

base_model = VGG19(include_top=False, input_shape=(240, 240, 3))

x = base_model.output
flat = Flatten()(x)

class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)

class_2 = Dense(1152, activation='relu')(drop_out)

output = Dense(2, activation='softmax')(class_2)

model_03 = Model(base_model.inputs, output)

model_03.load_weights("vgg_unfrozen.h5")

print("Model Loaded Successfully")

# -------------------- FASTAPI --------------------

app = FastAPI()

# Static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="templates")

# Upload folder
UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -------------------- FUNCTIONS --------------------

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img_path):

    image = cv2.imread(img_path)

    image = Image.fromarray(image, 'RGB')

    image = image.resize((240, 240))

    image = np.array(image)

    input_img = np.expand_dims(image, axis=0)

    result = model_03.predict(input_img)

    result01 = np.argmax(result, axis=1)

    return result01[0]

# -------------------- ROUTES --------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    value = getResult(file_path)

    result = get_className(value)

    return {
        "result": result
    }

# -------------------- RUN SERVER --------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)