import torch

from fastapi import APIRouter

from app import model_loader
from app.schemas import LungCancerFeatures, LungCancerPrediction

router = APIRouter(prefix="/predict", tags=["deep-learning"])


@router.post("/lung-cancer-dl", response_model=LungCancerPrediction)
def predict_lung_cancer_dl(features: LungCancerFeatures):
    x = features.to_ordered_array(model_loader.lung_cancer_feature_columns)

    # Unlike the Random Forest endpoint, this one MUST scale inputs the same
    # way training did -- the saved scaler travels with the model precisely
    # so this step can't be forgotten or reimplemented slightly differently.
    x_scaled = model_loader.mlp_scaler.transform([x])
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        logit = model_loader.mlp_model(x_tensor)
        probability = torch.sigmoid(logit).item()

    label = "YES" if probability >= 0.5 else "NO"
    return LungCancerPrediction(
        probability=round(probability, 4),
        label=label,
        model="mlp",
    )
