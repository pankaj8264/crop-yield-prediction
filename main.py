# ==============================================================
# File Name : main.py
# Project   : Crop Yield Prediction System
# Description:
# Main FastAPI application. All API endpoints are here.
#
# COMPLETE FLOW (matches diagram exactly):
#
# ── AUTH ────────────────────────────────────────────
#   POST /register/farmer       → anyone can register as farmer
#   POST /register/supervisor   → only supervisor can create supervisor
#   POST /login                 → get JWT token
#
# ── FARMER DASHBOARD (role = farmer) ────────────────
#   GET  /farmer/profile            → My Profile
#   PUT  /farmer/profile/update     → Update Farm Details
#   GET  /farmer/recommendations    → Crop Recommendations
#   POST /predict                   → Predict Crop Yield
#   GET  /predictions/history       → My Predictions
#   POST /reports                   → Create Report
#   GET  /reports                   → My Reports
#
# ── SUPERVISOR DASHBOARD (role = supervisor) ─────────
#   GET  /dashboard/stats                  → Dashboard & Overview
#   GET  /supervisor/farmers               → Manage Farmers (view)
#   DELETE /supervisor/farmers/{id}        → Delete Farmer
#   GET  /predictions/history/all          → View All Predictions
#   GET  /supervisor/analytics             → Analytics & Statistics
#   GET  /supervisor/logs                  → System Logs
#   GET  /users                            → All Users
#   GET  /reports                          → View All Reports
#   DELETE /reports/{id}                   → Delete Report
#   POST /register/supervisor              → Manage Supervisors
#
# ── HEALTH ──────────────────────────────────────────
#   GET  /                          → Health Check
# ==============================================================

import joblib
import pandas as pd
import numpy as np
from typing import List
from sqlalchemy import func

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sklearn.preprocessing import LabelEncoder
from fastapi import UploadFile, File
import os
import shutil
from database import Base, engine, get_db
from models import User, Role, Prediction, PredictionHistory, Report, SystemLog
from schemas import (
    FarmerRegister,
    FarmerProfileUpdate,
    SupervisorCreate,
    UserResponse,
    UserList,
    Token,
    PredictionRequest,
    PredictionResponse,
    PredictionHistoryResponse,
    ReportCreate,
    ReportResponse,
    DashboardStats,
    AnalyticsResponse,
    SystemLogResponse,
    Message,
)
from auth import (
    hash_password,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_current_supervisor,
    get_current_farmer,
)

# ==============================================================
# Create All Tables in PostgreSQL
# Safe to run every time — skips existing tables
# ==============================================================
Base.metadata.create_all(bind=engine)

# ==============================================================
# Load ML Model and Scaler
# These files must be in the same folder as main.py:
#   - crop_yield_model.pkl  (trained XGBoost model)
#   - scaler.pkl            (StandardScaler from notebook)
#
# If files not found → falls back to simple formula
# ==============================================================
try:
    ml_model = joblib.load("crop_yield_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    print("✅ ML Model and Scaler loaded successfully")
except FileNotFoundError:
    ml_model = None
    scaler   = None
    print("⚠️  Model files not found — using fallback formula")

# ==============================================================
# FastAPI App
# ==============================================================
app = FastAPI(
    title       = "Crop Yield Prediction System",
    description = "API matching the Flow & Access Structure diagram",
    version     = "1.0.0",
)

# ==============================================================
# CORS — allows frontend (React/Flutter) to call this API
# In production: replace "*" with your frontend URL
# ==============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ==============================================================
# STARTUP — Seed Roles
# Creates "farmer" and "supervisor" in roles table
# Only runs once — skips if already exists
# ==============================================================
@app.on_event("startup")
def seed_roles():
    db = next(get_db())
    try:
        for role_name in ("farmer", "supervisor"):
            if not db.query(Role).filter(Role.role_name == role_name).first():
                db.add(Role(role_name=role_name))
                print(f"✅ Role seeded: {role_name}")
        db.commit()
    finally:
        db.close()


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def get_role_or_500(db: Session, role_name: str) -> Role:
    """Get role by name. Raises 500 if not found (misconfiguration)."""
    role = db.query(Role).filter(Role.role_name == role_name).first()
    if not role:
        raise HTTPException(status_code=500, detail=f"Role '{role_name}' not in database")
    return role


def check_unique_user(db: Session, username: str, email: str):
    """Raises 400 if username or email already registered."""
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")


def log_action(db: Session, user_id: int, action: str, details: str):
    """
    Saves a system log entry.
    Called after every important action.

    action examples: "LOGIN", "PREDICTION", "DELETE_FARMER"
    """
    db.add(SystemLog(user_id=user_id, action=action, details=details))
    # Note: caller must db.commit() after this


# ==============================================================
# ── HEALTH CHECK ────────────────────────────────────────────
# ==============================================================

@app.get("/", tags=["Health"])
def root():
    """Check if API is running. No auth required."""
    return {
        "status" : "ok",
        "message": "Crop Yield Prediction API is running ✅"
    }


# ==============================================================
# ── AUTH ROUTES ─────────────────────────────────────────────
# FLOW: User → Register → Login → Get Token → Use Token
# ==============================================================

@app.post("/register/farmer", response_model=UserResponse, tags=["Auth"])
def register_farmer(payload: FarmerRegister, db: Session = Depends(get_db)):
    """
    FLOW (diagram step 1): New User Registers → Role = FARMER
    Anyone can register. No auth token needed.

    Steps:
      1. Check username and email are unique
      2. Get farmer role from DB
      3. Hash password
      4. Create user record
      5. Log the registration
    """
    # Step 1: Check unique
    check_unique_user(db, payload.username, payload.email)

    # Step 2: Get farmer role
    role = get_role_or_500(db, "farmer")

    # Step 3 & 4: Create user
    user = User(
        full_name       = payload.full_name,
        username        = payload.username,
        email           = payload.email,
        phone           = payload.phone,
        hashed_password = hash_password(payload.password),
        state           = payload.state,
        district        = payload.district,
        village         = payload.village,
        land_area       = payload.land_area,
        role_id         = role.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Step 5: Log it
    log_action(db, user.id, "REGISTER", f"New farmer registered: {user.username}")
    db.commit()

    return user


@app.post("/register/supervisor", response_model=UserResponse, tags=["Auth"])
def register_supervisor(
    payload            : SupervisorCreate,
    db                 : Session = Depends(get_db),
    current_supervisor : User    = Depends(get_current_supervisor),  # ← only supervisor can call this
):
    """
    FLOW (diagram): Supervisor Creation box
    ONLY an existing supervisor can create another supervisor.
    Farmer trying this → 403 Forbidden (from get_current_supervisor)

    Steps:
      1. Caller verified as supervisor (via dependency)
      2. Check unique username/email
      3. Create supervisor user
      4. Log the action
    """
    check_unique_user(db, payload.username, payload.email)
    role = get_role_or_500(db, "supervisor")

    user = User(
        full_name       = payload.full_name,
        username        = payload.username,
        email           = payload.email,
        phone           = payload.phone,
        hashed_password = hash_password(payload.password),
        state           = payload.state,
        district        = payload.district,
        village         = payload.village,
        land_area       = payload.land_area,
        role_id         = role.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    log_action(
        db, current_supervisor.id,
        "CREATE_SUPERVISOR",
        f"{current_supervisor.username} created supervisor: {user.username}"
    )
    db.commit()

    return user


@app.post("/login", response_model=Token, tags=["Auth"])
def login(
    form_data : OAuth2PasswordRequestForm = Depends(),
    db        : Session = Depends(get_db)
):
    """
    FLOW (diagram step 2): Login with Username + Password
    Returns JWT token on success.

    ⚠️ IMPORTANT: Send as form-data (x-www-form-urlencoded)
    Fields: username, password

    Steps:
      1. Verify username + password
      2. Check account is active
      3. Create JWT token with user_id, username, role
      4. Log the login
      5. Return token

    Token contains:
      { user_id, username, role }

    After login, send token as:
      Authorization: Bearer <token>
    """
    # Step 1: Check credentials
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    # Step 2: Check active
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive")

    # Step 3: Create token with role info
    token = create_access_token({
        "user_id" : user.id,
        "username": user.username,
        "role"    : user.role.role_name,   # "farmer" or "supervisor"
    })

    # Step 4: Log login
    log_action(db, user.id, "LOGIN", f"{user.username} logged in as {user.role.role_name}")
    db.commit()

    return Token(access_token=token, token_type="bearer")


# ==============================================================
# ── FARMER ROUTES ───────────────────────────────────────────
# FLOW: Role Check → ROLE = FARMER → Farmer Dashboard
# All routes here require farmer token.
# ==============================================================

@app.get("/farmer/profile", tags=["Farmer"])
def get_farmer_profile(current_farmer: User = Depends(get_current_farmer)):
    """
    FLOW: Farmer Dashboard → My Profile → View Profile
    Returns all profile details of the logged-in farmer.

    Token required: farmer role
    """
    return {
        "id"        : current_farmer.id,
        "full_name" : current_farmer.full_name,
        "username"  : current_farmer.username,
        "email"     : current_farmer.email,
        "phone"     : current_farmer.phone,
        "state"     : current_farmer.state,
        "district"  : current_farmer.district,
        "village"   : current_farmer.village,
        "land_area" : current_farmer.land_area,
        "role"      : current_farmer.role.role_name,
        "is_active" : current_farmer.is_active,
        "joined"    : current_farmer.created_at,
    }


@app.put("/farmer/profile/update", response_model=Message, tags=["Farmer"])
def update_farmer_profile(
    payload        : FarmerProfileUpdate,
    db             : Session = Depends(get_db),
    current_farmer : User    = Depends(get_current_farmer),
):
    """
    FLOW: Farmer Dashboard → Update Farm Details
    Updates only the fields provided in the request body.

    Token required: farmer role

    Example — update only phone:
      {"phone": "9999999999"}
    """
    # Update only fields that were sent
    if payload.full_name is not None : current_farmer.full_name = payload.full_name
    if payload.phone     is not None : current_farmer.phone     = payload.phone
    if payload.state     is not None : current_farmer.state     = payload.state
    if payload.district  is not None : current_farmer.district  = payload.district
    if payload.village   is not None : current_farmer.village   = payload.village
    if payload.land_area is not None : current_farmer.land_area = payload.land_area

    db.commit()
    db.refresh(current_farmer)

    log_action(db, current_farmer.id, "PROFILE_UPDATE", f"{current_farmer.username} updated profile")
    db.commit()

    return Message(message="Profile updated successfully ✅")


@app.get("/farmer/recommendations", tags=["Farmer"])
def get_recommendations(current_farmer: User = Depends(get_current_farmer)):
    """
    FLOW: Farmer Dashboard → Crop Recommendations
    Returns crop suggestions based on season, rainfall, temperature.

    Token required: farmer role
    """
    return {
        "farmer"          : current_farmer.full_name,
        "state"           : current_farmer.state,
        "recommendations" : [
            {
                "crop"           : "Rice",
                "best_season"    : "Kharif",
                "min_rainfall_mm": 100,
                "temperature_c"  : "25-35°C",
                "tip"            : "Requires flooded fields. Best in humid conditions."
            },
            {
                "crop"           : "Wheat",
                "best_season"    : "Rabi",
                "min_rainfall_mm": 30,
                "temperature_c"  : "10-25°C",
                "tip"            : "Cool dry weather. Irrigate at crown root stage."
            },
            {
                "crop"           : "Maize",
                "best_season"    : "Kharif",
                "min_rainfall_mm": 60,
                "temperature_c"  : "20-30°C",
                "tip"            : "Well-drained fertile soil. Avoid waterlogging."
            },
            {
                "crop"           : "Cotton",
                "best_season"    : "Kharif",
                "min_rainfall_mm": 50,
                "temperature_c"  : "25-35°C",
                "tip"            : "Deep black soil. Pest management is critical."
            },
            {
                "crop"           : "Sugarcane",
                "best_season"    : "Whole Year",
                "min_rainfall_mm": 150,
                "temperature_c"  : "20-35°C",
                "tip"            : "Long season (12-18 months). Needs rich soil + irrigation."
            },
        ]
    }


# ==============================================================
# ── PREDICTION ROUTES ────────────────────────────────────────
# FLOW: Farmer Dashboard → Predict Crop Yield → POST /predict
#       → ML Model → Save to DB → Return result
# ==============================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def create_prediction(
    payload        : PredictionRequest,
    db             : Session = Depends(get_db),
    current_farmer : User    = Depends(get_current_farmer),
):
    """
    FLOW: Farmer Dashboard → Predict Crop Yield
    Sends input to ML model, saves result, returns prediction.

    Token required: farmer role

    Steps:
      1. Run input through ML model
      2. Generate recommendation text
      3. Save to predictions table
      4. Save to prediction_history table
      5. Log the action
      6. Return result
    """
    print(f">>> Prediction requested by: {current_farmer.username}")

    # Step 1: ML prediction
    predicted_yield = run_ml_model(payload)

    # Step 2: Recommendation
    recommendation = get_recommendation_text(payload, predicted_yield)

    # Step 3: Save latest prediction
    prediction = Prediction(
        user_id         = current_farmer.id,
        crop            = payload.crop,
        season          = payload.season,
        state           = payload.state,
        area            = payload.area,
        rainfall        = payload.rainfall,
        fertilizer      = payload.fertilizer,
        pesticide       = payload.pesticide,
        temperature     = payload.temperature,
        predicted_yield = predicted_yield,
        recommendation  = recommendation,
    )
    db.add(prediction)

    # Step 4: Save to history (permanent record)
    history = PredictionHistory(
        user_id           = current_farmer.id,
        crop              = payload.crop,
        season            = payload.season,
        state             = payload.state,
        area              = payload.area,
        rainfall          = payload.rainfall,
        fertilizer        = payload.fertilizer,
        pesticide         = payload.pesticide,
        temperature       = payload.temperature,
        predicted_yield   = predicted_yield,
        recommendation    = recommendation,
        prediction_status = "Completed",
    )
    db.add(history)
    db.commit()

    # Step 5: Log
    log_action(
        db, current_farmer.id, "PREDICTION",
        f"{current_farmer.username} predicted {payload.crop} in {payload.state}: {predicted_yield} t/ha"
    )
    db.commit()

    return PredictionResponse(predicted_yield=predicted_yield, recommendation=recommendation)


def run_ml_model(payload: PredictionRequest) -> float:
    """
    Runs the XGBoost ML model to predict crop yield.

    Matches EXACTLY what the notebook does:
      1. Build dataframe with all 16 features
      2. Label encode Crop, Season, State
      3. Create bins (Rainfall, Area, Temp, Fertilizer)
      4. Scale using saved StandardScaler
      5. Predict with XGBoost model

    Falls back to simple formula if model not loaded.
    """
    if ml_model is None or scaler is None:
        # Simple fallback formula
        base = payload.area * 1.5
        adj  = (payload.rainfall * 0.01) + (payload.fertilizer * 0.02) - (payload.pesticide * 0.01)
        return round(max(base + adj, 0), 2)

    # ── Build input dataframe (matches training columns) ──
    data = pd.DataFrame([{
        "Crop"            : payload.crop,
        "Crop_Year"       : 2024,
        "Season"          : payload.season,
        "State"           : payload.state,
        "Area"            : payload.area,
        "Production"      : payload.area * 2,           # estimated
        "Annual_Rainfall" : payload.rainfall,
        "Fertilizer"      : payload.fertilizer,
        "Pesticide"       : payload.pesticide,
        "Avg_Temperature" : payload.temperature,
        "Max_Temperature" : payload.temperature + 5,    # estimated max
        "Min_Temperature" : payload.temperature - 5,    # estimated min
    }])

    # ── Label encode categorical columns (same as notebook) ──
    encoder = LabelEncoder()
    for col in ["Crop", "Season", "State"]:
        data[col] = encoder.fit_transform(data[col])

    # ── Create bin features (same as notebook) ──
    data["Rainfall_Bin"]   = pd.cut(data["Annual_Rainfall"], bins=5, labels=[0,1,2,3,4]).astype(int)
    data["Area_Bin"]       = pd.cut(data["Area"],            bins=5, labels=[0,1,2,3,4]).astype(int)
    data["Temp_Bin"]       = pd.cut(data["Avg_Temperature"], bins=5, labels=[0,1,2,3,4]).astype(int)
    data["Fertilizer_Bin"] = pd.cut(data["Fertilizer"],      bins=5, labels=[0,1,2,3,4]).astype(int)

    # ── Scale features (same as notebook) ──
    data_scaled = scaler.transform(data)

    # ── Predict ──
    result = ml_model.predict(data_scaled)
    return round(float(result[0]), 2)


def get_recommendation_text(payload: PredictionRequest, predicted_yield: float) -> str:
    """
    Returns farming advice based on input conditions.

    Rules:
      rainfall < 50   → suggest irrigation
      fertilizer < 20 → suggest more fertilizer
      pesticide > 50  → warn about overuse
      otherwise       → conditions are favorable
    """
    if payload.rainfall < 50:
        return "⚠️ Rainfall is very low. Consider supplemental irrigation."
    if payload.fertilizer < 20:
        return "⚠️ Fertilizer usage is below optimal. Consider increasing application."
    if payload.pesticide > 50:
        return "⚠️ High pesticide usage detected. Reduce to protect soil health."
    return "✅ Conditions look favorable. Maintain current farming practices."


@app.get(
    "/predictions/history",
    response_model=List[PredictionHistoryResponse],
    tags=["Predictions"]
)
def get_my_predictions(
    db             : Session = Depends(get_db),
    current_farmer : User    = Depends(get_current_farmer),
):
    """
    FLOW: Farmer Dashboard → My Predictions
    Returns the logged-in farmer's own prediction history.

    Token required: farmer role
    """
    return (
        db.query(PredictionHistory)
        .filter(PredictionHistory.user_id == current_farmer.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )


@app.get(
    "/predictions/history/all",
    response_model=List[PredictionHistoryResponse],
    tags=["Predictions"]
)
def get_all_predictions(
    db                 : Session = Depends(get_db),
    current_supervisor : User    = Depends(get_current_supervisor),
):
    """
    FLOW: Supervisor Dashboard → View All Predictions
    Returns predictions from ALL farmers.

    Token required: supervisor role
    """
    return (
        db.query(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )


# ==============================================================
# ── REPORT ROUTES ────────────────────────────────────────────
# FLOW:
#   Farmer     → My Reports (own only)
#   Supervisor → View All Reports (all farmers)
# ==============================================================

@app.post("/reports", response_model=ReportResponse, tags=["Reports"])
def create_report(
    payload      : ReportCreate,
    db           : Session = Depends(get_db),
    current_user : User    = Depends(get_current_user),
):
    """
    Create a new report. Both farmers and supervisors can do this.
    Token required: any role
    """
    report = Report(
        user_id               = current_user.id,
        report_title          = payload.report_title,
        report_type           = payload.report_type,
        report_description    = payload.report_description,
        file_path             = payload.file_path,
        prediction_history_id = payload.prediction_history_id,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    log_action(db, current_user.id, "CREATE_REPORT", f"Report created: {payload.report_title}")
    db.commit()

    return report


@app.get("/reports", response_model=List[ReportResponse], tags=["Reports"])
def list_reports(
    db           : Session = Depends(get_db),
    current_user : User    = Depends(get_current_user),
):
    """
    FLOW:
      Farmer     → My Reports (only sees own reports)
      Supervisor → View All Reports (sees all)

    Token required: any role
    """
    if current_user.role.role_name.lower() == "supervisor":
        # Supervisor sees ALL reports
        return db.query(Report).order_by(Report.created_at.desc()).all()

    # Farmer sees only their own reports
    return (
        db.query(Report)
        .filter(Report.user_id == current_user.id)
        .order_by(Report.created_at.desc())
        .all()
    )


@app.get("/reports/{report_id}", response_model=ReportResponse, tags=["Reports"])
def get_single_report(
    report_id    : int,
    db           : Session = Depends(get_db),
    current_user : User    = Depends(get_current_user),
):
    """
    Get one report by ID.
    Farmer can only view their own.
    Supervisor can view any.

    Raises:
      404 → report not found
      403 → farmer trying to view another's report
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    is_owner      = (report.user_id == current_user.id)
    is_supervisor = (current_user.role.role_name.lower() == "supervisor")

    if not (is_owner or is_supervisor):
        raise HTTPException(status_code=403, detail="Not authorized to view this report")

    return report


@app.delete("/reports/{report_id}", response_model=Message, tags=["Reports"])
def delete_report(
    report_id    : int,
    db           : Session = Depends(get_db),
    current_user : User    = Depends(get_current_user),
):
    """
    Delete a report by ID.
    Farmer can only delete their own.
    Supervisor can delete any.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    is_owner      = (report.user_id == current_user.id)
    is_supervisor = (current_user.role.role_name.lower() == "supervisor")

    if not (is_owner or is_supervisor):
        raise HTTPException(status_code=403, detail="Not authorized to delete this report")

    db.delete(report)
    db.commit()

    log_action(db, current_user.id, "DELETE_REPORT", f"Report deleted: ID {report_id}")
    db.commit()

    return Message(message=f"Report {report_id} deleted successfully ✅")


# ==============================================================
# ── SUPERVISOR ROUTES ────────────────────────────────────────
# FLOW: Role Check → ROLE = SUPERVISOR → Supervisor Dashboard
# All routes here require supervisor token.
# ==============================================================

@app.get("/dashboard/stats", response_model=DashboardStats, tags=["Supervisor"])
def dashboard_stats(
    db                 : Session = Depends(get_db),
    current_supervisor : User    = Depends(get_current_supervisor),
):
    """
    FLOW: Supervisor Dashboard → Dashboard & Overview
    Returns total counts for dashboard summary cards.

    Token required: supervisor role
    """
    farmer_role     = db.query(Role).filter(Role.role_name == "farmer").first()
    supervisor_role = db.query(Role).filter(Role.role_name == "supervisor").first()

    total_farmers     = db.query(User).filter(User.role_id == farmer_role.id).count()     if farmer_role     else 0
    total_supervisors = db.query(User).filter(User.role_id == supervisor_role.id).count() if supervisor_role else 0
    total_predictions = db.query(PredictionHistory).count()
    total_reports     = db.query(Report).count()

    return DashboardStats(
        total_farmers     = total_farmers,
        total_supervisors = total_supervisors,
        total_predictions = total_predictions,
        total_reports     = total_reports,
    )


@app.get("/supervisor/farmers", tags=["Supervisor"])
def get_all_farmers(
    db                 : Session = Depends(get_db),
    current_supervisor : User    = Depends(get_current_supervisor),
):
    """
    FLOW: Supervisor Dashboard → Manage Farmers → View Farmers
    Returns list of all registered farmers.

    Token required: supervisor role
    """
    farmer_role = db.query(Role).filter(Role.role_name == "farmer").first()
    return (
        db.query(User)
        .filter(User.role_id == farmer_role.id)
        .order_by(User.created_at.desc())
        .all()
    )


@app.delete("/supervisor/farmers/{farmer_id}", response_model=Message, tags=["Supervisor"])
def delete_farmer(
    farmer_id          : int,
    db                 : Session = Depends(get_db),
    current_supervisor : User    = Depends(get_current_supervisor),
):
    """
    FLOW: Supervisor Dashboard → Manage Farmers → Delete Farmer
    Deletes farmer + all their predictions, history, reports (cascade).

    Token required: supervisor role

    Raises:
      404 → farmer not found
      400 → trying to delete a supervisor account
    """
    farmer = db.query(User).filter(User.id == farmer_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")

    if farmer.role.role_name.lower() != "farmer":
        raise HTTPException(status_code=400, detail="Can only delete farmer accounts here")

    name = farmer.username
    db.delete(farmer)
    db.commit()

    log_action(
        db, current_supervisor.id,
        "DELETE_FARMER",
        f"Supervisor {current_supervisor.username} deleted farmer: {name}"
    )
    db.commit()

    return Message(message=f"Farmer '{name}' deleted successfully ✅")


@app.get(
    "/supervisor/analytics",
    response_model=AnalyticsResponse,
    tags=["Supervisor"]
)
def get_analytics(
    db                 : Session = Depends(get_db),
    current_supervisor : User    = Depends(get_current_supervisor),
):
    """
    FLOW: Supervisor Dashboard → Analytics & Statistics
    Returns system-wide analytics data.

    Token required: supervisor role
    """
    total_predictions = db.query(PredictionHistory).count()
    total_reports     = db.query(Report).count()

    # Most frequently predicted crop
    top_crop = (
        db.query(
            PredictionHistory.crop,
            func.count(PredictionHistory.crop).label("count")
        )
        .group_by(PredictionHistory.crop)
        .order_by(func.count(PredictionHistory.crop).desc())
        .first()
    )

    # Average yield
    avg = db.query(func.avg(PredictionHistory.predicted_yield)).scalar()

    return AnalyticsResponse(
        total_predictions   = total_predictions,
        total_reports       = total_reports,
        most_predicted_crop = top_crop[0] if top_crop else None,
        average_yield       = round(float(avg), 2) if avg else 0.0,
    )

@app.get("/supervisor/model-status", tags=["Supervisor"])
def get_model_status(
    current_supervisor: User = Depends(get_current_supervisor),
):
    """
    Returns ML model monitoring information.
    Token required: supervisor role.
    """

    if ml_model is not None and scaler is not None:
        return {
            "model_status": "Loaded",
            "model_file": "crop_yield_model.pkl",
            "scaler_file": "scaler.pkl",
            "prediction_api": "Active",
            "prediction_method": "XGBoost ML Model"
        }

    return {
        "model_status": "Fallback",
        "model_file": "Not Loaded",
        "scaler_file": "Not Loaded",
        "prediction_api": "Active",
        "prediction_method": "Fallback Formula"
    }
@app.post("/supervisor/upload-dataset", tags=["Supervisor"])
def upload_dataset(
    file: UploadFile = File(...),
    current_supervisor: User = Depends(get_current_supervisor),
):
    """
    Supervisor can upload a dataset file.
    """

    allowed_extensions = [".csv", ".xlsx", ".xls"]

    filename = file.filename or ""
    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only CSV, XLSX and XLS files are allowed."
        )

    upload_dir = "uploads/datasets"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Dataset uploaded successfully",
        "filename": filename,
        "file_path": file_path
    }

@app.get(
    "/supervisor/logs",
    response_model=List[SystemLogResponse],
    tags=["Supervisor"]
)
def get_system_logs(
    db                 : Session = Depends(get_db),
    current_supervisor : User    = Depends(get_current_supervisor),
):
    """
    FLOW: Supervisor Dashboard → System Logs
    Returns all system activity logs, newest first.

    Token required: supervisor role
    """
    return (
        db.query(SystemLog)
        .order_by(SystemLog.created_at.desc())
        .all()
    )


@app.get("/users", response_model=UserList, tags=["Supervisor"])
def list_all_users(
    db                 : Session = Depends(get_db),
    current_supervisor : User    = Depends(get_current_supervisor),
):
    """
    Returns ALL users (farmers + supervisors).
    Token required: supervisor role
    """
    users = db.query(User).order_by(User.created_at.desc()).all()
    return UserList(users=users)

@app.get("/my-predictions", tags=["Predictions"])
def get_my_predictions(
    db: Session = Depends(get_db),
    current_farmer: User = Depends(get_current_farmer)
):

    predictions = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.user_id == current_farmer.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )

    return predictions
@app.get("/supervisor/profile", tags=["Supervisor"])
def get_supervisor_profile(
    current_supervisor: User = Depends(get_current_supervisor)
):
    return {
        "id": current_supervisor.id,
        "full_name": current_supervisor.full_name,
        "username": current_supervisor.username,
        "email": current_supervisor.email,
        "phone": current_supervisor.phone,
        "role": "Supervisor",
        "state": current_supervisor.state,
        "district": current_supervisor.district,
        "village": current_supervisor.village,
        "created_at": current_supervisor.created_at
    }