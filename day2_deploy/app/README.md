# RespiraAI backend — Day 2

## Setup (local)

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"   # never hardcode this in the repo
```

Two-step install on purpose: plain `pip install torch` from PyPI grabs the CUDA build by default on
Linux, pulling ~500MB+ of nvidia-cublas/cudnn/etc. that a CPU-only deploy never uses. Installing
torch/torchvision from the CPU-only index first, then everything else from `requirements.txt`,
keeps the environment (and the deployed instance) lean.

Folder layout expected (matches the Day 1 deliverable):

```
respiratory-ai-platform/
├── data/              # from Day 1
├── notebooks/         # from Day 1
├── models/            # lung_cancer_rf.pkl, lung_cancer_mlp.pt, lung_cancer_scaler.pkl,
│                       lung_cancer_feature_columns.pkl (from notebooks 01/02),
│                       xray_cnn.pt (from notebook 03, once you've trained on the full dataset)
├── app/                # this backend
├── frontend/           # the three-tab UI
├── render.yaml         # deploy config (see below)
├── Procfile
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

## Deploy (no Docker)

This deploys as a native Python web service — no Dockerfile, no container registry, just a
build command and a start command that Render runs directly.

1. Push this repo to GitHub (needs to include `app/`, `frontend/`, `models/`, `render.yaml`,
   `requirements.txt` — `data/` and `notebooks/` aren't needed at runtime, only for retraining).
2. On [render.com](https://render.com): **New +** → **Blueprint** → point it at the repo. Render
   reads `render.yaml` automatically and sets up the build/start commands, so there's nothing to
   configure by hand.
3. In the service's **Environment** tab, set `GROQ_API_KEY` to your real key — `render.yaml`
   deliberately leaves this one blank (`sync: false`) so it never ends up committed to the repo.
4. Deploy. First build takes a few minutes (torch is the slow part even as the CPU-only wheel);
   later deploys are faster since Render caches the build.
5. Render gives you a `https://respirai.onrender.com`-style URL — `/` is the frontend, `/docs` is
   the API explorer, exactly like running it locally.

**Free-tier caveats worth knowing before a live demo:** Render's free web services spin down after
a period of inactivity and take 30-60 seconds to wake back up on the next request — hit `/health`
a minute before you actually need the app responsive, rather than discovering the cold-start delay
mid-demo. 512MB RAM on the free instance is enough for these four models, but wouldn't be if
`xray_cnn.pt` grew into something much larger than MobileNetV2's ~14MB.

**Not using Render?** `Procfile` (`web: uvicorn app.main:app --host 0.0.0.0 --port $PORT`) works
as-is on other buildpack-style platforms (Railway, for instance) — you'd just need to replicate the
CPU-only torch install as a custom build step there too, since `Procfile` alone doesn't cover it.

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
