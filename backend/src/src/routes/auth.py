from fastapi import APIRouter, Request, Depends, HTTPException, status
from pydantic import BaseModel
from models.UserModel import UserModel
import jwt
from datetime import datetime, timedelta
import os

auth_router = APIRouter(
    prefix="/api/v1/auth",
    tags=["auth"],
)

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-super-secret-jwt-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "student"

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@auth_router.post("/register")
async def register(request: Request, user_req: RegisterRequest):
    user_model = await UserModel.create_instance(request.app.db_client)
    existing_user = await user_model.get_user_by_email(user_req.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = await user_model.create_user(
        name=user_req.name,
        email=user_req.email,
        password=user_req.password,
        role=user_req.role
    )
    
    # Sanitize user dict
    user["id"] = str(user.pop("_id", user.get("id")))
    user.pop("password_hash", None)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["id"], "role": user["role"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user": user}

@auth_router.post("/login")
async def login(request: Request, req: LoginRequest):
    user_model = await UserModel.create_instance(request.app.db_client)
    user = await user_model.get_user_by_email(req.email)
    if not user or not user_model.verify_password(req.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    # Sanitize user dict
    user["id"] = str(user.pop("_id", user.get("id")))
    user.pop("password_hash", None)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["id"], "role": user["role"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user": user}

@auth_router.get("/me")
async def get_me(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
        
    user_model = await UserModel.create_instance(request.app.db_client)
    user = await user_model.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Sanitize user dict
    user["id"] = str(user.pop("_id", user.get("id")))
    user.pop("password_hash", None)

    return {"user": user}
