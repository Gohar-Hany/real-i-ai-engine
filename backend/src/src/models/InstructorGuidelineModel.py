from .BaseDataModel import BaseDataModel
from .db_schemes import InstructorGuideline
from .enums.DataBaseEnum import DataBaseEnum

class InstructorGuidelineModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_GUIDELINE_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_GUIDELINE_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_GUIDELINE_NAME.value]
            indexes = InstructorGuideline.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"],
                    name=index["name"],
                    unique=index["unique"]
                )

    async def create_or_update_guideline(self, guideline: InstructorGuideline):
        """
        Upserts a guideline by task_id.
        If it already exists, updates its fields.
        """
        data = guideline.dict(by_alias=True, exclude_unset=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]

        result = await self.collection.update_one(
            {"task_id": guideline.task_id},
            {"$set": data},
            upsert=True
        )
        
        if result.upserted_id:
            guideline.id = result.upserted_id
        return guideline

    async def get_active_guidelines(self, project_id: str) -> list:
        """
        Returns all active guidelines for a given project_id.
        """
        cursor = self.collection.find({
            "project_id": project_id,
            "is_active": True
        }).sort("created_at", -1)
        
        guidelines = []
        async for doc in cursor:
            guidelines.append(InstructorGuideline(**doc))
        return guidelines

    async def deactivate_guideline(self, task_id: str) -> bool:
        """
        Deactivates a guideline by setting is_active to False.
        """
        result = await self.collection.update_one(
            {"task_id": task_id},
            {"$set": {"is_active": False}}
        )
        return result.modified_count > 0

    async def get_all_guidelines(self) -> list:
        """
        Returns all guidelines from the database sorted by creation time.
        """
        cursor = self.collection.find({}).sort("created_at", -1)
        guidelines = []
        async for doc in cursor:
            guidelines.append(InstructorGuideline(**doc))
        return guidelines

    async def delete_guideline(self, task_id: str) -> bool:
        """
        Deletes a guideline by task_id.
        """
        result = await self.collection.delete_one({"task_id": task_id})
        return result.deleted_count > 0

