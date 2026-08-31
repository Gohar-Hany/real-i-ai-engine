from pydantic import BaseModel, Field
from typing import Optional
from bson.objectid import ObjectId

class InstructorGuideline(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    project_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    priority: str = Field("High")
    status: str = Field("Pending")
    notes: Optional[str] = ""
    created_at: Optional[str] = None
    is_active: bool = True

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [
                    ("task_id", 1)
                ],
                "name": "task_id_index_1",
                "unique": True
            },
            {
                "key": [
                    ("project_id", 1)
                ],
                "name": "project_id_index_1",
                "unique": False
            }
        ]
