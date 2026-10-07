# ==============================================================
# File Name : schemas.py
# Project   : Crop Yield Prediction System
# Description:
# Pydantic schemas for validating request data and
# shaping response data.
#
# REQUEST schemas → validate what comes IN from client
# RESPONSE schemas → shape what goes OUT to client
#
# Every API endpoint uses these schemas.
# ==============================================================

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr


# ==============================================================
# ── ROLE SCHEMAS ──
# ==============================================================

class RoleResponse(BaseModel):
    """Used when returning role info in responses."""
    id        : int
    role_name : str

    class Config:
        from_attributes = True   # allows reading SQLAlchemy model directly


# ==============================================================
# ── USER / AUTH SCHEMAS ──
# ==============================================================

class FarmerRegister(BaseModel):
    """
    Schema for farmer self-registration.
    Used in: POST /register/farmer

    Anyone can register as a farmer.
    Supervisor role CANNOT be selected here (see diagram).
    """
    full_name : str
    username  : str
    email     : EmailStr       # validates email format automatically
    phone     : str
    password  : str            # plain text — will be hashed before saving
    state     : str
    district  : str
    village   : Optional[str]   = None
    land_area : Optional[float] = None


class SupervisorCreate(FarmerRegister):
    """
    Schema for creating a supervisor.
    Same fields as FarmerRegister.
    Used in: POST /register/supervisor
    ONLY an existing supervisor can call this endpoint.
    """
    pass


class FarmerProfileUpdate(BaseModel):
    """
    Schema for updating farmer profile.
    ALL fields are optional — only provided fields are updated.
    Used in: PUT /farmer/profile/update

    Example: send only {"phone": "9999999999"} to update just phone.
    """
    full_name : Optional[str]   = None
    phone     : Optional[str]   = None
    state     : Optional[str]   = None
    district  : Optional[str]   = None
    village   : Optional[str]   = None
    land_area : Optional[float] = None


class UserResponse(BaseModel):
    """
    Schema for returning user data.
    Password is intentionally NOT included.
    Used in: registration responses, user listing.
    """
    id        : int
    full_name : str
    username  : str
    email     : EmailStr
    phone     : str
    state     : str
    district  : str
    village   : Optional[str]   = None
    land_area : Optional[float] = None
    role_id   : int
    is_active : bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserList(BaseModel):
    """
    Returns a list of users.
    Used in: GET /users (supervisor only)
    """
    users: List[UserResponse]


# ==============================================================
# ── JWT TOKEN SCHEMAS ──
# ==============================================================

class Token(BaseModel):
    """
    Returned after successful login.
    Used in: POST /login response

    Client must send this token as:
        Authorization: Bearer <access_token>
    """
    access_token : str
    token_type   : str   # always "bearer"


class TokenData(BaseModel):
    """
    Data stored inside the JWT token payload.
    Used internally to identify the logged-in user.
    """
    user_id  : Optional[int] = None
    username : Optional[str] = None
    role     : Optional[str] = None   # "farmer" or "supervisor"


# ==============================================================
# ── PREDICTION SCHEMAS ──
# ==============================================================

class PredictionRequest(BaseModel):
    """
    Input data for crop yield prediction.
    Used in: POST /predict (farmer only)

    These values are sent to the ML model.

    crop        → crop name e.g. "Rice", "Wheat"
    season      → "Kharif", "Rabi", "Zaid", "Whole Year"
    state       → Indian state e.g. "Punjab"
    area        → farm area in hectares
    rainfall    → annual rainfall in mm
    fertilizer  → fertilizer used in kg/hectare
    pesticide   → pesticide used in kg/hectare
    temperature → average temperature in Celsius
    """
    crop        : str
    season      : str
    state       : str
    area        : float
    rainfall    : float
    fertilizer  : float
    pesticide   : float
    temperature : float


class PredictionResponse(BaseModel):
    """
    Result returned after ML model predicts.
    Used in: POST /predict response

    predicted_yield → estimated yield in tonnes/hectare
    recommendation  → farming advice based on input values
    """
    predicted_yield : float
    recommendation  : Optional[str] = None


class PredictionHistoryResponse(BaseModel):
    """
    Returns a single prediction history record.
    Used in:
        GET /predictions/history       (farmer sees own)
        GET /predictions/history/all   (supervisor sees all)
    """
    id                : int
    user_id           : int
    crop              : str
    season            : str
    state             : str
    area              : float
    rainfall          : float
    fertilizer        : float
    pesticide         : float
    temperature       : float
    predicted_yield   : float
    recommendation    : Optional[str] = None
    prediction_status : str
    created_at        : datetime

    class Config:
        from_attributes = True


# ==============================================================
# ── REPORT SCHEMAS ──
# ==============================================================

class ReportCreate(BaseModel):
    """
    Input data for creating a report.
    Used in: POST /reports

    report_title          → title e.g. "Rice Yield July 2024"
    report_type           → e.g. "Crop Yield Prediction"
    report_description    → optional details
    file_path             → optional path to saved PDF/CSV
    prediction_history_id → optional link to a prediction record
    """
    report_title          : str
    report_type           : str = "Crop Yield Prediction"
    report_description    : Optional[str] = None
    file_path             : Optional[str] = None
    prediction_history_id : Optional[int] = None


class ReportResponse(ReportCreate):
    """
    Returned when viewing a report.
    Extends ReportCreate with DB id, user_id, timestamp.
    Used in: GET /reports, GET /reports/{id}
    """
    id         : int
    user_id    : int
    created_at : datetime

    class Config:
        from_attributes = True


# ==============================================================
# ── DASHBOARD / ANALYTICS SCHEMAS ──
# ==============================================================

class DashboardStats(BaseModel):
    """
    High-level counts for supervisor dashboard.
    Used in: GET /dashboard/stats (supervisor only)
    """
    total_farmers     : int
    total_supervisors : int
    total_predictions : int
    total_reports     : int


class AnalyticsResponse(BaseModel):
    """
    Detailed analytics for supervisor.
    Used in: GET /supervisor/analytics (supervisor only)
    """
    total_predictions   : int
    total_reports       : int
    most_predicted_crop : Optional[str] = None
    average_yield       : float


# ==============================================================
# ── SYSTEM LOG SCHEMA ──
# ==============================================================

class SystemLogResponse(BaseModel):
    """
    Returns a system log entry.
    Used in: GET /supervisor/logs (supervisor only)

    action  → short label e.g. "LOGIN", "PREDICTION"
    details → full description of what happened
    """
    id         : int
    user_id    : Optional[int] = None
    action     : str
    details    : Optional[str] = None
    created_at : datetime

    class Config:
        from_attributes = True


# ==============================================================
# ── COMMON RESPONSE ──
# ==============================================================

class Message(BaseModel):
    """
    Simple message response for success/info.
    Example: {"message": "Profile updated successfully"}
    """
    message: str
