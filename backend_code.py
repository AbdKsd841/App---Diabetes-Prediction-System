"""
app.py — Fully implemented FastAPI backend with MongoDB (Motor) integration

Features
- POST /predict — validate patient data, call external ML model API, format response (risk scores + SHAP), store to MongoDB
- POST /predictions/{prediction_id}/comments — attach/append doctor comments to an existing prediction
- GET /patients/{patient_id}/predictions — fetch historical predictions for a patient (paginated)
- GET /health — health check

How to run (see also comments at bottom):
1) Create a virtualenv and install dependencies from the list below (requirements snippet).
2) Set environment variables: MONGO_URI, MONGO_DB, MODEL_ENDPOINT, MODEL_API_KEY (and optionally PORT).
3) Start with: uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}

Dependencies (pin or adjust as needed):
- fastapi
- uvicorn[standard]
- pydantic>=2
- motor
- httpx
- python-dotenv (optional, if you want to load a .env)

"""
from __future__ import annotations

import os
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import Body, FastAPI, HTTPException, Path, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# ----------------------------------------------------------------------------
# Utility: Pydantic-compatible ObjectId type for responses
# ----------------------------------------------------------------------------
class PydanticObjectId(str):
    """Serialize MongoDB ObjectId to string in responses."""

    @classmethod
    def validate(cls, v: Any) -> "PydanticObjectId":
        if isinstance(v, ObjectId):
            return cls(str(v))
        # accept string that is a valid 24-hex ObjectId
        if isinstance(v, str):
            try:
                ObjectId(v)
                return cls(v)
            except Exception as e:  # invalid format
                raise ValueError("Invalid ObjectId string") from e
        raise ValueError("ObjectId must be bson.ObjectId or valid hex string")


# ----------------------------------------------------------------------------
# Configuration via environment variables
# ----------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "risk_service")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT", "http://localhost:9000/predict")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "changeme-dev-key")

# Optional: allow all during development; restrict in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# ----------------------------------------------------------------------------
# FastAPI app setup
# ----------------------------------------------------------------------------
app = FastAPI(title="Diabetic Risk Prediction", version="1.0.0")

# CORS (configure properly in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# MongoDB client & database handle (Motor is async)
# ----------------------------------------------------------------------------
mongo_client: Optional[AsyncIOMotorClient] = None
mongo_db: Optional[AsyncIOMotorDatabase] = None


@app.on_event("startup")
async def startup() -> None:
    """Create Mongo client, DB, and indexes on startup."""
    global mongo_client, mongo_db
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB]

    # Create indexes to speed up patient history queries and time-ordered scans
    await mongo_db.predictions.create_index("patient_id")
    await mongo_db.predictions.create_index("created_at")


@app.on_event("shutdown")
async def shutdown() -> None:
    global mongo_client
    if mongo_client:
        mongo_client.close()


# ----------------------------------------------------------------------------
# Data models (Pydantic v2 style)
# ----------------------------------------------------------------------------
class VitalSigns(BaseModel):
    systolic_bp: Optional[int] = Field(None, ge=50, le=260, description="Systolic blood pressure (mmHg)")
    diastolic_bp: Optional[int] = Field(None, ge=30, le=200, description="Diastolic blood pressure (mmHg)")
    heart_rate: Optional[int] = Field(None, ge=20, le=240)


class PatientData(BaseModel):
    """Incoming patient features for prediction.

    NOTE: Add/remove fields as your model requires. Use `extras` for anything
    not explicitly modeled here to maintain forward-compatibility.
    """

    patient_id: str = Field(..., description="Stable ID used to group history")
    age: int = Field(..., ge=0, le=120)
    sex: Literal["male", "female", "other"]
    bmi: Optional[float] = Field(None, ge=5, le=90)
    hba1c: Optional[float] = Field(None, ge=0, le=20, description="Hemoglobin A1c (%)")
    vitals: Optional[VitalSigns] = None

    # Flexible extra features
    extras: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional features")

    @field_validator("extras")
    @classmethod
    def limit_extras(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        # Prevent obviously unsafe/nonsensical payloads (e.g., nested giant blobs)
        if len(v) > 200:
            raise ValueError("Too many extra features (max 200)")
        return v


class ModelRequest(BaseModel):
    """Payload forwarded to the external ML model service.

    You may re-map/rename fields here to match your model contract.
    """

    features: Dict[str, Any]


class RiskScores(BaseModel):
    """Per-complication risk scores in [0, 1]."""

    # Example complications; adjust for your use case.
    retinopathy: Optional[float] = Field(None, ge=0.0, le=1.0)
    nephropathy: Optional[float] = Field(None, ge=0.0, le=1.0)
    neuropathy: Optional[float] = Field(None, ge=0.0, le=1.0)
    cardiovascular: Optional[float] = Field(None, ge=0.0, le=1.0)


class SHAPExplanation(BaseModel):
    """SHAP values: mapping complication -> feature -> contribution.

    Example:
    {
      "retinopathy": {"hba1c": 0.12, "age": 0.03},
      "nephropathy": {"bmi": -0.05}
    }
    """

    values: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class PredictionPayload(BaseModel):
    """Request body for /predict that includes patient data and optional doctor comments."""

    patient: PatientData
    doctor_comment: Optional[str] = Field(None, max_length=5000)


class PredictionResponse(BaseModel):
    prediction_id: PydanticObjectId
    patient_id: str
    created_at: datetime
    risk_scores: RiskScores
    shap: SHAPExplanation
    doctor_comment: Optional[str] = None


class CommentCreate(BaseModel):
    doctor_id: Optional[str] = Field(None, description="Identifier for who wrote the comment")
    comment: str = Field(..., max_length=5000)


class CommentOut(BaseModel):
    author: Optional[str] = None
    comment: str
    created_at: datetime


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
async def call_model_api(patient: PatientData) -> Dict[str, Any]:
    """Call the external ML model endpoint and return its JSON.

    This function maps our internal PatientData into the expected model input.
    Adjust the mapping to match your model's API contract.
    """

    # Build a flat feature dict. You may change this to match the model contract.
    features: Dict[str, Any] = {
        "patient_id": patient.patient_id,
        "age": patient.age,
        "sex": patient.sex,
        "bmi": patient.bmi,
        "hba1c": patient.hba1c,
    }

    if patient.vitals is not None:
        features.update({
            "systolic_bp": patient.vitals.systolic_bp,
            "diastolic_bp": patient.vitals.diastolic_bp,
            "heart_rate": patient.vitals.heart_rate,
        })

    # Merge extras; extras override none-values if present
    for k, v in patient.extras.items():
        features[k] = v

    payload = ModelRequest(features=features).model_dump()

    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"}

    # Use a short, sane timeout; tune per your infra
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        try:
            resp = await client.post(MODEL_ENDPOINT, json=payload, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Model service unreachable: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Model service error: {resp.status_code} {resp.text}")

    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Model service returned non-JSON response")

    # Expected contract (example):
    # {
    #   "risk_scores": {"retinopathy": 0.42, "nephropathy": 0.18, ...},
    #   "shap": {"retinopathy": {"age": 0.02, ...}, "nephropathy": {...}}
    # }

    if "risk_scores" not in data or "shap" not in data:
        raise HTTPException(status_code=502, detail="Model response missing required fields (risk_scores, shap)")

    return data


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health", tags=["system"]) 
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"], status_code=status.HTTP_201_CREATED)
async def predict(payload: PredictionPayload = Body(...)) -> PredictionResponse:
    """Validate patient data, call the model, store the result + optional comment, and return the response."""

    if mongo_db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    # 1) Validate & call external model
    model_result = await call_model_api(payload.patient)

    # 2) Shape into our response models (validates ranges/types)
    risk_scores = RiskScores(**model_result.get("risk_scores", {}))
    shap = SHAPExplanation(values=model_result.get("shap", {}))

    # 3) Persist to MongoDB
    doc = {
        "patient_id": payload.patient.patient_id,
        "input_features": payload.patient.model_dump(),  # store validated inputs
        "risk_scores": risk_scores.model_dump(),
        "shap": shap.model_dump(),
        "doctor_comment": payload.doctor_comment,
        "created_at": now_utc(),
        "comments": [],  # threaded comments appended later
    }

    insert_result = await mongo_db.predictions.insert_one(doc)

    # 4) Build API response
    return PredictionResponse(
        prediction_id=PydanticObjectId.validate(insert_result.inserted_id),
        patient_id=payload.patient.patient_id,
        created_at=doc["created_at"],
        risk_scores=risk_scores,
        shap=shap,
        doctor_comment=payload.doctor_comment,
    )


@app.post("/predictions/{prediction_id}/comments", response_model=CommentOut, tags=["predictions"])
async def add_comment(
    prediction_id: str = Path(..., description="Mongo ObjectId of the prediction document"),
    body: CommentCreate = Body(...),
) -> CommentOut:
    """Attach a doctor (or reviewer) comment to an existing prediction document."""

    if mongo_db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        oid = ObjectId(prediction_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid prediction_id")

    comment_doc = {
        "author": body.doctor_id,
        "comment": body.comment,
        "created_at": now_utc(),
    }

    res = await mongo_db.predictions.update_one(
        {"_id": oid},
        {"$push": {"comments": comment_doc}}
    )

    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return CommentOut(**comment_doc)


@app.get("/patients/{patient_id}/predictions", tags=["predictions"])
async def patient_history(
    patient_id: str = Path(..., description="Patient identifier used at prediction time"),
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=200, description="Page size"),
) -> Dict[str, Any]:
    """Fetch historical predictions for the given patient, most recent first."""

    if mongo_db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    cursor = (
        mongo_db.predictions
        .find({"patient_id": patient_id})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )

    results: List[Dict[str, Any]] = []
    async for doc in cursor:
        results.append({
            "prediction_id": str(doc.get("_id")),
            "created_at": doc.get("created_at"),
            "risk_scores": doc.get("risk_scores"),
            "doctor_comment": doc.get("doctor_comment"),
            "comments": doc.get("comments", []),
        })

    return {
        "patient_id": patient_id,
        "count": len(results),
        "items": results,
        "skip": skip,
        "limit": limit,
    }


# ----------------------------------------------------------------------------
# Example: local dev bootstrap (optional)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow `python app.py` to run a dev server without uvicorn CLI.
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

"""
Quick-start notes (for reference):

requirements.txt (example)
--------------------------
fastapi
uvicorn[standard]
pydantic>=2
motor
httpx
python-dotenv

.env (example)
--------------
MONGO_URI=mongodb://localhost:27017
MONGO_DB=risk_service
MODEL_ENDPOINT=http://localhost:9000/predict
MODEL_API_KEY=dev-key-123

Run
---
python -m venv .venv && source .venv/bin/activate  # (On Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

Sample requests
---------------
POST http://localhost:8000/predict
{
  "patient": {
    "patient_id": "P-001",
    "age": 57,
    "sex": "male",
    "bmi": 29.4,
    "hba1c": 7.8,
    "vitals": {"systolic_bp": 132, "diastolic_bp": 84, "heart_rate": 72},
    "extras": {"smoker": false, "years_with_diabetes": 8}
  },
  "doctor_comment": "Initial consult; verify renal function."
}

POST http://localhost:8000/predictions/<prediction_id>/comments
{
  "doctor_id": "dr-alex",
  "comment": "Start ACE inhibitor if BP persists >130/80."
}

GET http://localhost:8000/patients/P-001/predictions?limit=10

Notes
-----
- This service *forwards* features to an external model API and expects back a JSON like:
  {
    "risk_scores": {"retinopathy": 0.42, "nephropathy": 0.18},
    "shap": {
      "retinopathy": {"age": 0.02, "hba1c": 0.12},
      "nephropathy": {"bmi": -0.05}
    }
  }
  Adapt the mapping in `call_model_api` as necessary.
- Mongo collection used: `predictions`. Indexes on `patient_id` and `created_at` are created at startup.
- All timestamps are stored in UTC.
"""
