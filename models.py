# ==============================================================
# File Name : models.py
# Project   : Crop Yield Prediction System
# Description:
# Defines all database tables using SQLAlchemy ORM.
# Each class = one table in PostgreSQL.
#
# Tables created:
#   roles              → farmer / supervisor roles
#   users              → all registered users
#   predictions        → latest prediction per farmer
#   prediction_history → full history of all predictions
#   reports            → reports linked to predictions
#   system_logs        → logs every action in the system
# ==============================================================

from sqlalchemy import (
    Column, Integer, String,
    Float, Boolean, DateTime,
    ForeignKey, Text
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

# ==============================================================
# TABLE 1: roles
# Stores the two roles: farmer and supervisor
# Seeded automatically at server startup
 
# id | role_name   | description
# ---+-------------+------------
#  1 | supervisor  | Full access
#  2 | farmer      | Limited access
# ==============================================================
class Role(Base):
    __tablename__ = "roles"

    id          = Column(Integer, primary_key=True, index=True)
    role_name   = Column(String(50), unique=True, nullable=False)  # "farmer" or "supervisor"
    description = Column(String(255), nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    # One role → many users
    users = relationship("User", back_populates="role", cascade="all, delete")


# ==============================================================
# TABLE 2: users
# Stores all registered farmers and supervisors.
#
# FLOW (from diagram):
#   - Farmer registers themselves → role_id = 2 (farmer)
#   - Supervisor created by existing supervisor → role_id = 1
#   - Login uses username + password
# ==============================================================
class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    full_name       = Column(String(100), nullable=False)
    username        = Column(String(50), unique=True, nullable=False)   # used for login
    email           = Column(String(100), unique=True, nullable=False)
    phone           = Column(String(20), nullable=False)
    hashed_password = Column(String, nullable=False)                    # bcrypt hash, never plain text

    # Location fields
    state    = Column(String(100), nullable=False)
    district = Column(String(100), nullable=False)
    village  = Column(String(100), nullable=True)
    land_area= Column(Float, nullable=True)                             # in acres/hectares

    # Account status — False = banned/inactive
    is_active = Column(Boolean, default=True)

    # Links to roles table
    role_id  = Column(Integer, ForeignKey("roles.id"), nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    role               = relationship("Role", back_populates="users")
    predictions        = relationship("Prediction", back_populates="user", cascade="all, delete")
    prediction_history = relationship("PredictionHistory", back_populates="user", cascade="all, delete")
    reports            = relationship("Report", back_populates="user", cascade="all, delete")
    logs               = relationship("SystemLog", back_populates="user", cascade="all, delete")


# ==============================================================
# TABLE 3: predictions
# Stores the LATEST prediction for each farmer.
# Gets a new record each time farmer predicts.
#
# FLOW (from diagram):
#   Farmer Dashboard → Predict Crop Yield → POST /predict
#   → ML Model → save result here
# ==============================================================
class Prediction(Base):
    __tablename__ = "predictions"

    id      = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Input features sent to ML model
    crop        = Column(String(100), nullable=False)   # e.g. "Rice"
    season      = Column(String(100), nullable=False)   # e.g. "Kharif"
    state       = Column(String(100), nullable=False)   # e.g. "Punjab"
    area        = Column(Float, nullable=False)          # hectares
    rainfall    = Column(Float, nullable=False)          # mm
    fertilizer  = Column(Float, nullable=False)          # kg/hectare
    pesticide   = Column(Float, nullable=False)          # kg/hectare
    temperature = Column(Float, nullable=False)          # Celsius

    # ML model output
    predicted_yield = Column(Float, nullable=False)     # tonnes/hectare
    recommendation  = Column(Text, nullable=True)       # farming advice

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="predictions")


# ==============================================================
# TABLE 4: prediction_history
# Stores ALL predictions ever made (never deleted/overwritten).
#
# FLOW (from diagram):
#   Farmer → My Predictions → GET /predictions/history
#   Supervisor → View All Predictions → GET /predictions/history/all
# ==============================================================
class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id      = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Same input fields as Prediction table
    crop        = Column(String(100), nullable=False)
    season      = Column(String(100), nullable=False)
    state       = Column(String(100), nullable=False)
    area        = Column(Float, nullable=False)
    rainfall    = Column(Float, nullable=False)
    fertilizer  = Column(Float, nullable=False)
    pesticide   = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)

    # ML output
    predicted_yield   = Column(Float, nullable=False)
    recommendation    = Column(Text, nullable=True)
    prediction_status = Column(String(50), default="Completed")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="prediction_history")


# ==============================================================
# TABLE 5: reports
# Stores reports created by farmers or supervisors.
#
# FLOW (from diagram):
#   Farmer → My Reports → GET /reports (own only)
#   Supervisor → View All Reports → GET /reports (all)
# ==============================================================
class Report(Base):
    __tablename__ = "reports"

    id      = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Optional link to a specific prediction
    prediction_history_id = Column(Integer, ForeignKey("prediction_history.id"), nullable=True)

    report_title       = Column(String(200), nullable=False)
    report_type        = Column(String(100), default="Crop Yield Prediction")
    report_description = Column(Text, nullable=True)
    file_path          = Column(String(255), nullable=True)  # path to saved PDF/CSV

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user       = relationship("User", back_populates="reports")
    prediction = relationship("PredictionHistory")


# ==============================================================
# TABLE 6: system_logs
# Logs every important action in the system.
#
# FLOW (from diagram):
#   Supervisor → System Logs → GET /supervisor/logs
#
# Logged events:
#   LOGIN, REGISTER, PREDICTION, CREATE_REPORT,
#   DELETE_FARMER, PROFILE_UPDATE, DELETE_REPORT
# ==============================================================
class SystemLog(Base):
    __tablename__ = "system_logs"

    id      = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable for system events

    action  = Column(String(100), nullable=False)   # e.g. "PREDICTION", "LOGIN"
    details = Column(Text, nullable=True)           # full description

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="logs")
