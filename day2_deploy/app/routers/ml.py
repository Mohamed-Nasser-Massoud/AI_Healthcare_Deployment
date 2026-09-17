import pandas as pd
from fastapi import APIRouter

from app import model_loader
from app.schemas import LungCancerFeatures, LungCancerPrediction

router = APIRouter(prefix="/predict", tags=["classical-ml"])


@router.post("/lung-cancer-ml", response_model=LungCancerPrediction)
def predict_lung_cancer_ml(features: LungCancerFeatures):
    x = features.to_ordered_array(model_loader.lung_cancer_feature_columns)
    # Wrapped in a DataFrame with the training-time column names -- the RF
    # was fit on a DataFrame (notebook 01), so a bare list at inference time
    # works but throws a "missing feature names" warning on every call.
    x_df = pd.DataFrame([x], columns=model_loader.lung_cancer_feature_columns)
    probability = model_loader.rf_model.predict_proba(x_df)[0][1]
    label = "YES" if probability >= 0.5 else "NO"
    return LungCancerPrediction(
        probability=round(float(probability), 4),
        label=label,
        model="random_forest",
    )
