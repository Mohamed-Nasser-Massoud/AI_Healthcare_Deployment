"""Loads every model/index exactly once, at app startup -- called from
main.py's startup hook, never from inside a request handler. Loading a
Random Forest or a CNN's weights from disk on every request would work,
but it's the single most common way people accidentally make a "fast"
model feel slow in production.

Module-level globals here are a deliberate, simple choice for a weekend
build. In a larger app you'd wrap this in a class and use FastAPI's
dependency injection instead -- but for four models behind four routes,
that indirection buys nothing yet.
"""
import json
import os

import joblib
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
RAG_STORE_DIR = os.path.join(BASE_DIR, "rag_store")

# ---- populated by load_all() ----
rf_model = None
lung_cancer_feature_columns = None

mlp_model = None
mlp_scaler = None

cv_model = None
cv_model_is_trained = False
xray_transform = None

tfidf_vectorizer = None
doc_vectors = None
doc_names = None
doc_texts = None

llm_client = None
LLM_MODEL_NAME = "openai/gpt-oss-20b"

_status = {}  # what actually loaded, for /health


class LungCancerMLP(nn.Module):
    """Must match the architecture trained in 02_lung_cancer_dl.ipynb exactly --
    state_dict loading is positional/by-shape, so a mismatched architecture
    either errors loudly (good) or, worse, silently loads garbage into the
    wrong layers if shapes happen to coincide. Keeping this definition and
    the notebook's definition in sync by hand is a real weekness of this
    weekend-scope build; a slightly more mature version would import this
    class from one shared module instead of duplicating it.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def _load_lung_cancer_ml():
    global rf_model, lung_cancer_feature_columns
    rf_path = os.path.join(MODELS_DIR, "lung_cancer_rf.pkl")
    cols_path = os.path.join(MODELS_DIR, "lung_cancer_feature_columns.pkl")
    rf_model = joblib.load(rf_path)
    lung_cancer_feature_columns = joblib.load(cols_path)
    _status["lung_cancer_ml"] = True


def _load_lung_cancer_dl():
    global mlp_model, mlp_scaler
    scaler_path = os.path.join(MODELS_DIR, "lung_cancer_scaler.pkl")
    weights_path = os.path.join(MODELS_DIR, "lung_cancer_mlp.pt")
    mlp_scaler = joblib.load(scaler_path)

    in_dim = len(lung_cancer_feature_columns)
    model = LungCancerMLP(in_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    mlp_model = model
    _status["lung_cancer_dl"] = True


def _load_cv_model():
    """Tries pretrained ImageNet weights (needs internet at startup -- fine
    in a normal deploy, not fine in a locked-down sandbox); falls back to
    random init if that fails so the app still starts. Then tries to load
    real fine-tuned weights (xray_cnn.pt) on top; if that file doesn't
    exist yet (you haven't run notebook 03 against the full dataset), the
    model stays flagged as untrained rather than the app pretending it's
    ready.
    """
    global cv_model, cv_model_is_trained, xray_transform

    xray_transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        weights = tv_models.MobileNet_V2_Weights.DEFAULT
        net = tv_models.mobilenet_v2(weights=weights)
    except Exception as e:
        print(f"[model_loader] Could not fetch pretrained ImageNet weights ({e}); "
              f"using random init instead. Predictions will be meaningless either way "
              f"until xray_cnn.pt (fine-tuned weights) is loaded below.")
        net = tv_models.mobilenet_v2(weights=None)

    net.classifier[1] = nn.Linear(net.last_channel, 1)

    weights_path = os.path.join(MODELS_DIR, "xray_cnn.pt")
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        cv_model_is_trained = True
    else:
        cv_model_is_trained = False
        print(f"[model_loader] {weights_path} not found -- CV endpoint will run with an "
              f"UNTRAINED head. Run 03_xray_cv.ipynb against the full dataset and drop "
              f"xray_cnn.pt into models/ to get real predictions.")

    net.eval()
    cv_model = net
    _status["xray_cv"] = cv_model_is_trained  # True only if genuinely trained


def _load_rag_index():
    global tfidf_vectorizer, doc_vectors, doc_names, doc_texts, llm_client

    tfidf_vectorizer = joblib.load(os.path.join(RAG_STORE_DIR, "tfidf_vectorizer.pkl"))
    doc_vectors = joblib.load(os.path.join(RAG_STORE_DIR, "doc_vectors.pkl"))
    with open(os.path.join(RAG_STORE_DIR, "doc_metadata.json")) as f:
        meta = json.load(f)
    doc_names = meta["names"]
    doc_texts = meta["texts"]

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("[model_loader] GROQ_API_KEY is not set -- /ask will return a 502 until it is. "
              "Set it as an environment variable (never hardcode it in the image/repo).")
    llm_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key or "unset")
    _status["rag"] = bool(groq_api_key)


def load_all():
    _load_lung_cancer_ml()
    _load_lung_cancer_dl()
    _load_cv_model()
    _load_rag_index()
    print(f"[model_loader] startup complete: {_status}")


def health_status() -> dict:
    return {"status": "ok", "models_loaded": _status}


# ---------------------------------------------------------------------------
# RAG helpers -- same retrieve()/ask_rag() logic as 04_rag_pipeline.ipynb,
# operating on the index loaded above instead of rebuilding it every call.
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = 3):
    query_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors)[0]
    top_idx = scores.argsort()[::-1][:k]
    return [(doc_names[i], doc_texts[i], float(scores[i])) for i in top_idx]


def ask_rag(question: str, k: int = 3) -> dict:
    retrieved = retrieve(question, k=k)
    context_block = "\n\n".join(f"[Source: {name}]\n{text}" for name, text, _ in retrieved)

    prompt = f"""Answer the question using ONLY the information in the sources below.
If the sources don't contain enough information to answer, say so honestly rather than guessing.
Keep the answer concise (3-5 sentences) and do not invent medical claims not present in the sources.

Sources:
{context_block}

Question: {question}

Answer:"""

    response = llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
        reasoning_effort="low",
    )
    answer = response.choices[0].message.content
    return {"answer": answer, "sources": [name for name, _, _ in retrieved]}
