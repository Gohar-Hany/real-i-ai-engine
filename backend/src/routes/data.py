from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
import aiofiles
from models import ResponseSignal
import logging
from .schemes.data import ProcessRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.db_schemes import DataChunk, Asset
from models.enums.AssetTypeEnum import AssetTypeEnum

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: str, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
        
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    # validate the file properties
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )

    # store the assets into the database
    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    asset_resource = Asset(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path)
    )

    asset_record = await asset_model.create_asset(asset=asset_resource)

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                "file_id": str(asset_record.id),
                "asset_name": file_id,
            }
        )

@data_router.post("/process/{project_id}")
async def process_endpoint(request: Request, project_id: str, process_request: ProcessRequest):

    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    asset_model = await AssetModel.create_instance(
            db_client=request.app.db_client
        )

    project_files_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id,
            asset_name=process_request.file_id
        )

        if asset_record is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.FILE_ID_ERROR.value,
                }
            )

        project_files_ids = {
            asset_record.id: asset_record.asset_name
        }
    
    else:
        

        project_files = await asset_model.get_all_project_assets(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,
        )

        project_files_ids = {
            record.id: record.asset_name
            for record in project_files
        }

    if len(project_files_ids) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.NO_FILES_ERROR.value,
            }
        )
    
    process_controller = ProcessController(project_id=project_id)

    no_records = 0
    no_files = 0

    chunk_model = await ChunkModel.create_instance(
                        db_client=request.app.db_client
                    )

    if do_reset == 1:
        _ = await chunk_model.delete_chunks_by_project_id(
            project_id=project.id
        )

    for asset_id, file_id in project_files_ids.items():

        file_content = process_controller.get_file_content(file_id=file_id)

        if file_content is None:
            logger.error(f"Error while processing file: {file_id}")
            continue

        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )

        if file_chunks is None or len(file_chunks) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.PROCESSING_FAILED.value
                }
            )

        file_chunks_records = [
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=i+1,
                chunk_project_id=project.id,
                chunk_asset_id=asset_id
            )
            for i, chunk in enumerate(file_chunks)
        ]

        no_records += await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_files += 1

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks": no_records,
            "processed_files": no_files
        }
    )

@data_router.get("/projects")
async def get_projects(request: Request):
    try:
        project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
        projects, _ = await project_model.get_all_projects(page=1, page_size=1000)
        return [
            {
                "_id": str(p.id),
                "project_id": p.project_id
            }
            for p in projects
        ]
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@data_router.get("/assets")
async def get_assets(request: Request):
    try:
        cursor = request.app.db_client["assets"].find({})
        assets = []
        async for doc in cursor:
            project_id_str = "Unknown"
            project_doc = await request.app.db_client["projects"].find_one({"_id": doc.get("asset_project_id")})
            if project_doc:
                project_id_str = project_doc.get("project_id", "Unknown")
            
            assets.append({
                "_id": str(doc.get("_id")),
                "asset_name": doc.get("asset_name"),
                "asset_type": doc.get("asset_type"),
                "asset_size": doc.get("asset_size"),
                "project": project_id_str
            })
        return assets
    except Exception as e:
        logger.error(f"Error listing assets: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@data_router.delete("/projects/{project_id}")
async def delete_project(request: Request, project_id: str):
    try:
        # Delete project by project_id
        result = await request.app.db_client["projects"].delete_one({"project_id": project_id})
        if result.deleted_count == 0:
            return JSONResponse(status_code=404, content={"detail": "Project not found"})
        # Optionally delete related assets and chunks
        return JSONResponse(status_code=200, content={"status": "success", "message": "Project deleted"})
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

from bson import ObjectId

@data_router.delete("/assets/{asset_id}")
async def delete_asset(request: Request, asset_id: str):
    try:
        result = await request.app.db_client["assets"].delete_one({"_id": ObjectId(asset_id)})
        if result.deleted_count == 0:
            return JSONResponse(status_code=404, content={"detail": "Asset not found"})
        return JSONResponse(status_code=200, content={"status": "success", "message": "Asset deleted"})
    except Exception as e:
        logger.error(f"Error deleting asset: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})
