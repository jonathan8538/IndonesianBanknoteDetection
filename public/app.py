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

model = YOLO('./best.pt')
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

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detections.append({
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "label": label,
                "confidence": confidence
            })

    return { "detections": detections }
