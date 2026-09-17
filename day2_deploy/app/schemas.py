"""Pydantic models for every request/response the API handles.

Keeping these in one file (rather than scattered across routers) means the
auto-generated /docs page shows a single, consistent schema list, and it's
one place to check when a field name changes.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LungCancerFeatures(BaseModel):
    """Input for both the classical-ML and DL lung cancer endpoints.

    Deliberately uses friendly types (Literal, bool) instead of making the
    caller know the training data's internal 1/2 encoding -- that encoding
    is an implementation detail of how the CSV happened to be formatted,
    not something an API consumer should need to know about.
    """

    gender: Literal["M", "F"]
    age: int = Field(ge=1, le=120)
    smoking: bool
    yellow_fingers: bool
    anxiety: bool
    peer_pressure: bool
    chronic_disease: bool
    fatigue: bool
    allergy: bool
    wheezing: bool
    alcohol_consuming: bool
    coughing: bool
    shortness_of_breath: bool
    swallowing_difficulty: bool
    chest_pain: bool

    def to_ordered_array(self, feature_columns: List[str]) -> list:
        """Build the numeric feature vector in the exact column order the
        model was trained on (loaded from lung_cancer_feature_columns.pkl,
        not hardcoded here -- two separate hardcoded orderings is exactly
        how this kind of bug creeps in).
        """
        values_by_name = {
            "GENDER": 1 if self.gender == "M" else 0,
            "AGE": self.age,
            "SMOKING": int(self.smoking),
            "YELLOW_FINGERS": int(self.yellow_fingers),
            "ANXIETY": int(self.anxiety),
            "PEER_PRESSURE": int(self.peer_pressure),
            "CHRONIC_DISEASE": int(self.chronic_disease),
            "FATIGUE": int(self.fatigue),
            "ALLERGY": int(self.allergy),
            "WHEEZING": int(self.wheezing),
            "ALCOHOL_CONSUMING": int(self.alcohol_consuming),
            "COUGHING": int(self.coughing),
            "SHORTNESS_OF_BREATH": int(self.shortness_of_breath),
            "SWALLOWING_DIFFICULTY": int(self.swallowing_difficulty),
            "CHEST_PAIN": int(self.chest_pain),
        }
        return [values_by_name[col] for col in feature_columns]

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "M", "age": 65, "smoking": True, "yellow_fingers": True,
                "anxiety": False, "peer_pressure": False, "chronic_disease": True,
                "fatigue": True, "allergy": False, "wheezing": True,
                "alcohol_consuming": True, "coughing": True, "shortness_of_breath": True,
                "swallowing_difficulty": False, "chest_pain": True,
            }
        }


class LungCancerPrediction(BaseModel):
    probability: float = Field(description="Predicted probability of the YES (cancer) class")
    label: Literal["YES", "NO"]
    model: str


class XrayPrediction(BaseModel):
    label: Literal["NORMAL", "PNEUMONIA"]
    confidence: float
    model_trained: bool = Field(description="False if the CV model is still an untrained placeholder head")
    note: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=3, ge=1, le=10, description="Number of source documents to retrieve")


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
