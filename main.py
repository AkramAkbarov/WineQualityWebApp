from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from model import Winequality

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pth"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Model sinifleri 0-5: yaradici "serab evi" adlari (sadalanan keyfiyyet pilləsi)
WINE_TIER_NAMES: tuple[str, ...] = (
    "Villa Xəzər — kölgə arxası",
    "Adağ qırmızısı — stolun dostu",
    "Mərkəzi serenada — balans axını",
    "Qızıl koridor — axşam tədbiri",
    "Velvet axşam — ehtiyat qarışım",
    "Ulduz bağı '18 — kolleksiya",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Winequality | None = None
scaler = None
load_error: str | None = None


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., ge=3.0, le=16.0)
    volatile_acidity: float = Field(..., ge=0.0, le=2.0)
    citric_acid: float = Field(..., ge=0.0, le=1.5)
    residual_sugar: float = Field(..., ge=0.0, le=70.0)
    chlorides: float = Field(..., ge=0.0, le=1.0)
    free_sulfur_dioxide: float = Field(..., ge=0.0, le=350.0)
    total_sulfur_dioxide: float = Field(..., ge=0.0, le=450.0)
    density: float = Field(..., ge=0.98, le=1.04)
    pH: float = Field(..., ge=2.5, le=4.5)
    sulphates: float = Field(..., ge=0.0, le=2.5)
    alcohol: float = Field(..., ge=5.0, le=16.0)

    @model_validator(mode="after")
    def check_sulfur(self):
        if self.free_sulfur_dioxide > self.total_sulfur_dioxide:
            raise ValueError(
                "free_sulfur_dioxide total_sulfur_dioxide-dən böyük ola bilməz"
            )
        return self

    def to_tensor(self) -> torch.Tensor:
        values = [getattr(self, name) for name in FEATURE_NAMES]
        x = np.array(values, dtype=np.float32).reshape(1, -1)
        if scaler is not None:
            x = scaler.transform(x)
        return torch.tensor(x, dtype=torch.float32, device=device)


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    tier_names: list[str]
    confidence: float
    probabilities: dict[str, float]


def _load_checkpoint() -> dict:
    try:
        checkpoint = torch.load(
            MODEL_PATH, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        # Pure state_dict (OrderedDict of layer weights)
        if any(k.startswith("linear_layer_stack") for k in checkpoint):
            return checkpoint
    raise ValueError(
        "model.pth formatı tanınmadı. torch.save(model.state_dict(), 'model.pth') istifadə edin."
    )


def load_artifacts() -> Winequality:
    global scaler

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"model.pth tapılmadı: {MODEL_PATH}. Faylı layihə qovluğuna yerləşdirin."
        )

    net = Winequality()
    state = _load_checkpoint()
    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()

    scaler = None
    if SCALER_PATH.is_file():
        try:
            import joblib

            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded: {SCALER_PATH}")
        except Exception as exc:
            raise RuntimeError(f"scaler.pkl oxunmadı: {exc}") from exc
    else:
        print("WARNING: scaler.pkl not found; using raw feature values.")

    return net


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, load_error
    try:
        model = load_artifacts()
        load_error = None
        print("Model loaded successfully.")
    except Exception as exc:
        model = None
        load_error = str(exc)
        print(f"Model load failed: {load_error}")
    yield


app = FastAPI(title="Wine Quality API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    path = BASE_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="index.html tapılmadı")
    return FileResponse(path)


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "device": str(device),
        "model_path": str(MODEL_PATH),
        "error": load_error,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: WineFeatures):
    if model is None:
        detail = load_error or "Model yüklənməyib. Serveri yenidən başladın."
        raise HTTPException(status_code=503, detail=detail)

    with torch.inference_mode():
        logits = model(features.to_tensor())
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(torch.argmax(probs).item())
        confidence = float(probs[pred_class].item())

    return PredictionResponse(
        predicted_class=pred_class,
        predicted_label=WINE_TIER_NAMES[pred_class],
        tier_names=list(WINE_TIER_NAMES),
        confidence=round(confidence, 4),
        probabilities={
            f"class_{i}": round(float(probs[i].item()), 4) for i in range(6)
        },
    )


@app.get("/style.css")
async def css():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")


@app.get("/script.js")
async def js():
    return FileResponse(BASE_DIR / "script.js", media_type="application/javascript")
