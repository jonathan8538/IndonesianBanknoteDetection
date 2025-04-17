# main.py
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

# Load the model
model = YOLO('runs/detect/train/weights/best.pt')  # Replace with your model path

# FastAPI app
app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend origin for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class ImageData(BaseModel):
    image: str  # Base64-encoded image

# Inference endpoint
@app.post("/detect")
async def detect(data: ImageData):
    # Clean Base64 header
    img_str = re.sub("^data:image/.+;base64,", "", data.image)
    img_bytes = base64.b64decode(img_str)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model(img)[0]
    annotated_img = results.plot()  # Draw boxes and labels

    # Convert back to JPEG for response
    _, buffer = cv2.imencode('.jpg', annotated_img)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
