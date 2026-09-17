import io

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app import model_loader
from app.schemas import XrayPrediction

router = APIRouter(prefix="/predict", tags=["computer-vision"])


@router.post("/xray", response_model=XrayPrediction)
async def predict_xray(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("L")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded file as an image")

    tensor = model_loader.xray_transform(image).unsqueeze(0)

    with torch.no_grad():
        logit = model_loader.cv_model(tensor)
        pneumonia_prob = torch.sigmoid(logit).item()

    label = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"
    confidence = pneumonia_prob if label == "PNEUMONIA" else 1 - pneumonia_prob

    note = None
    if not model_loader.cv_model_is_trained:
        note = (
            "This model's classification head is untrained (placeholder). "
            "Run 03_xray_cv.ipynb against the full Kaggle dataset and drop the "
            "resulting xray_cnn.pt into models/ to get real predictions -- until "
            "then, treat this label/confidence as meaningless, not as a real result."
        )

    return XrayPrediction(
        label=label,
        confidence=round(confidence, 4),
        model_trained=model_loader.cv_model_is_trained,
        note=note,
    )
