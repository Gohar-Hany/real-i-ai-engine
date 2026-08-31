from pydantic import BaseModel, Field
from typing import Optional
import bcrypt
from bson import ObjectId

class UserSchema(BaseModel):
    id: Optional[str] = Field(alias="_id")
    name: str
    email: str
    password_hash: str
    role: str = "student"
    avatar: Optional[str] = None

class UserModel:
    def __init__(self, db_client):
        self.collection = db_client["users"]

    @classmethod
    async def create_instance(cls, db_client):
        return cls(db_client)

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        user = await self.collection.find_one({"email": email})
        if user:
            user["id"] = str(user["_id"])
        return user

    async def get_user_by_id(self, user_id: str) -> Optional[dict]:
        user = await self.collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user["id"] = str(user["_id"])
        return user

    async def create_user(self, name: str, email: str, password: str, role: str = "student", avatar: str = None) -> dict:
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = {
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "role": role,
            "avatar": avatar
        }
        result = await self.collection.insert_one(new_user)
        new_user["id"] = str(result.inserted_id)
        return new_user

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

    async def get_all_users(self) -> list:
        """Fetch all users, excluding password_hash."""
        users = []
        async for user in self.collection.find({}, {"password_hash": 0}):
            user["id"] = str(user.pop("_id"))
            users.append(user)
        return users

    async def update_user_role(self, user_id: str, new_role: str) -> bool:
        """Update a user's role. Returns True if a document was modified."""
        result = await self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"role": new_role}}
        )
        return result.modified_count > 0


