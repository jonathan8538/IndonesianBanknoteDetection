from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import base64
import re
from io import BytesIO

model = YOLO('runs/detect/train/weights/best.pt')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str 


@app.post("/detect")
async def detect(data: ImageData):
    img_str = re.sub("^data:image/.+;base64,", "", data.image)
    img_bytes = base64.b64decode(img_str)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results = model(img)[0]
    annotated_img = results.plot() 

    _, buffer = cv2.imencode('.jpg', annotated_img)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
