# AI_Healthcare_Deployment

AI in healthcare capstone with four components:

- Classical ML and PyTorch MLP lung-cancer risk prediction
- MobileNetV2 chest X-ray pneumonia classification
- TF-IDF retrieval-augmented respiratory-condition Q&A
- FastAPI backend with a plain HTML frontend

## Local deployment

The production-ready service is in `day2_deploy/`.

```powershell
cd day2_deploy
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
$env:GROQ_API_KEY = "your-key"
uvicorn app.main:app --reload
```

Open `http://localhost:8000/` or `http://localhost:8000/docs`.

## Render deployment

Deploy `day2_deploy/` as the Render service directory, or create a Blueprint from its `render.yaml`. Add `GROQ_API_KEY` in Render's Environment settings. Never commit the key.

The full X-ray training archive and extracted dataset are intentionally excluded from Git because they are too large for a normal GitHub repository. The trained deployment artifact `day2_deploy/models/xray_cnn.pt` is included.

## Notebooks

The four notebooks document training and indexing:

1. `01_lung_cancer_ml.ipynb`
2. `02_lung_cancer_dl.ipynb`
3. `03_xray_cv.ipynb`
4. `04_rag_pipeline.ipynb`
