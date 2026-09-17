# RespiraAI backend — Day 2 Morning

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"   # never hardcode this in the repo
```

Folder layout expected (matches the Day 1 deliverable):

```
respiratory-ai-platform/
├── data/              # from Day 1
├── notebooks/         # from Day 1
├── models/            # lung_cancer_rf.pkl, lung_cancer_mlp.pt, lung_cancer_scaler.pkl,
│                       lung_cancer_feature_columns.pkl (from notebooks 01/02),
│                       xray_cnn.pt (from notebook 03, once you've trained on the full dataset)
├── app/                # this backend
└── requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Then open **http://localhost:8000/** — the frontend (three tabs: lung cancer risk, chest X-ray, ask a question), served by the same FastAPI app as a static file so there's no separate frontend server or CORS setup needed.

For the raw API instead, **http://localhost:8000/docs** is the auto-generated interactive explorer.

## Test with curl

```bash
# Health check -- tells you honestly which models are actually ready
curl http://localhost:8000/health

# Classical ML lung cancer risk
curl -X POST http://localhost:8000/predict/lung-cancer-ml \
  -H "Content-Type: application/json" \
  -d '{"gender":"M","age":65,"smoking":true,"yellow_fingers":true,"anxiety":false,
       "peer_pressure":false,"chronic_disease":true,"fatigue":true,"allergy":false,
       "wheezing":true,"alcohol_consuming":true,"coughing":true,
       "shortness_of_breath":true,"swallowing_difficulty":false,"chest_pain":true}'

# Same input, DL model
curl -X POST http://localhost:8000/predict/lung-cancer-dl \
  -H "Content-Type: application/json" \
  -d '{"gender":"M","age":65,"smoking":true,"yellow_fingers":true,"anxiety":false,
       "peer_pressure":false,"chronic_disease":true,"fatigue":true,"allergy":false,
       "wheezing":true,"alcohol_consuming":true,"coughing":true,
       "shortness_of_breath":true,"swallowing_difficulty":false,"chest_pain":true}'

# Chest X-ray (multipart image upload)
curl -X POST http://localhost:8000/predict/xray \
  -F "file=@../data/chest_xray_sample/NORMAL/IM-0353-0001.jpeg"

# RAG Q&A
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between asthma and COPD?"}'
```

## What's real right now vs. what needs more setup

| Endpoint | Status |
|---|---|
| `/predict/lung-cancer-ml` | Fully real — trained Random Forest from notebook 01 |
| `/predict/lung-cancer-dl` | Fully real — trained MLP + scaler from notebook 02 |
| `/predict/xray` | **Runs, but honestly reports `model_trained: false`** until you train on the full dataset (notebook 03) and drop `xray_cnn.pt` into `models/` |
| `/ask` | Fully real once `GROQ_API_KEY` is set with network access — returns a `502` with a clear message if the LLM call fails, rather than crashing |

`/health` reports the true state of all four at a glance — check it first if anything looks off.

## What was actually verified before you got this

Every endpoint above was tested end-to-end with FastAPI's `TestClient` against the real trained
artifacts (not mocks): the ML/DL endpoints against real high-risk and low-risk feature
combinations, the X-ray endpoint against a real sample image plus a deliberately invalid file
(confirms the 400 error path), and `/ask` against a real question (confirms the 502 error path
works correctly when the LLM provider is unreachable — the same path that would fire on a real
Groq outage in production, not just in a sandboxed environment). Missing-field requests were also
checked to confirm they return a `422` validation error rather than a `500`.
