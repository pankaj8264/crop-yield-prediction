# ==============================================================
# File Name : auth.py
# Project   : Crop Yield Prediction System
# Description:
# Handles all authentication and authorization.
#
# FLOW (from diagram):
#   1. User sends username + password to POST /login
#   2. authenticate_user() checks credentials
#   3. create_access_token() creates a JWT token
#   4. Token is returned to client
#   5. Client sends token in every future request
#   6. get_current_user() reads and verifies the token
#   7. get_current_farmer() / get_current_supervisor()
#      checks the role and allows or blocks access
# Role Rules (from diagram):
#   - Farmer role  → access to farmer dashboard only
#   - Supervisor   → full system access
#   - Wrong role   → 403 Forbidden
# ==============================================================

from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from database import get_db
from models import User

# ==============================================================
# JWT Settings
# SECRET_KEY signs the token — keep this secret in production
# ==============================================================
SECRET_KEY = "crop_yield_prediction_super_secret_key_2026"
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24   # 24 hours

# ==============================================================
# Password Hashing
# bcrypt is the industry standard for hashing passwords
# ==============================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ==============================================================
# OAuth2 Scheme
# Tells FastAPI the token comes from POST /login
# Used by Swagger UI to show the "Authorize" button
# ==============================================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ==============================================================
# FUNCTION 1: hash_password
# Converts plain text password to bcrypt hash.
# Called during registration — we NEVER store plain passwords.
#
# Example:
#   hash_password("123456") → "$2b$12$..."
# ==============================================================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# ==============================================================
# FUNCTION 2: verify_password
# Compares plain password with stored bcrypt hash.
# Returns True if match, False if wrong.
# Called during login.
#
# Example:
#   verify_password("123456", "$2b$12$...") → True
# ==============================================================
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)



# ==============================================================
# FUNCTION 3: authenticate_user
# Looks up user by username and checks their password.
# Returns User object if valid, None if invalid.
# Called inside POST /login endpoint.
#
# Flow:
#   1. Find user by username in DB
#   2. If not found → return None
#   3. Check password → if wrong → return None
#   4. All good → return User
# ==============================================================
def authenticate_user(username: str, password: str, db: Session):
    # Find user
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None   # username not found

    # Check password
    if not verify_password(password, user.hashed_password):
        return None   # wrong password

    return user       # credentials valid


# ==============================================================
# FUNCTION 4: create_access_token
# Creates a signed JWT token with user info + expiry.
# Returned to client after successful login.
#
# Token contains:
#   { user_id, username, role, exp }
#
# Example:
#   create_access_token({"user_id": 1, "username": "Rahul", "role": "farmer"})
#   → "eyJhbGci..."
# ==============================================================
def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire  = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ==============================================================
# FUNCTION 5: verify_access_token
# Decodes and validates a JWT token.
# Returns the payload (dict) if valid.
# Raises 401 if token is invalid or expired.
# ==============================================================
def verify_access_token(token: str) -> dict:
    try:
        print("=" * 50)
        print("TOKEN RECEIVED:", token)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("PAYLOAD:", payload)
        print("=" * 50)
        return payload

    except JWTError as e:
        print("JWT ERROR:", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or Expired Token"
        )

# ==============================================================
# DEPENDENCY 1: get_current_user
# Used in ALL protected routes.
#
# Flow:
#   1. Extract Bearer token from Authorization header
#   2. Verify and decode the token
#   3. Get user_id from token payload
#   4. Load user from database
#   5. Return User object to the route
#
# Raises:
#   401 → token missing, invalid, or expired
#   404 → user not found in DB
#
# Usage in route:
#   current_user: User = Depends(get_current_user)
# ==============================================================
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    # Step 1 & 2: verify token
    payload = verify_access_token(token)

    # Step 3 & 4: load user
    user = db.query(User).filter(
        User.id == payload.get("user_id")
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


# ==============================================================
# DEPENDENCY 2: get_current_supervisor
# Builds on get_current_user.
# Checks role is "supervisor".
#
# FLOW (from diagram):
#   Role Check → ROLE = SUPERVISOR → Supervisor Dashboard
#   Wrong role → 403 Forbidden
#
# Raises:
#   403 → if logged in as farmer
#
# Usage in route:
#   supervisor: User = Depends(get_current_supervisor)
# ==============================================================
def get_current_supervisor(
    current_user: User = Depends(get_current_user)
) -> User:
    if current_user.role.role_name.lower() != "supervisor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Supervisor role required."
        )
    return current_user


# ==============================================================
# DEPENDENCY 3: get_current_farmer
# Builds on get_current_user.
# Checks role is "farmer".
#
# FLOW (from diagram):
#   Role Check → ROLE = FARMER → Farmer Dashboard
#   Wrong role → 403 Forbidden
#
# Raises:
#   403 → if logged in as supervisor
#
# Usage in route:
#   farmer: User = Depends(get_current_farmer)
# ==============================================================
def get_current_farmer(
    current_user: User = Depends(get_current_user)
) -> User:
    if current_user.role.role_name.lower() != "farmer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Farmer role required."
        )
    return current_user
