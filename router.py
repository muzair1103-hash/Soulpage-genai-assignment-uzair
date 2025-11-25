from fastapi import APIRouter, HTTPException
from fastapi import APIRouter, File, HTTPException, UploadFile
from services import process_uploads, index_file, ask_service
from typing import Any
from logger import logger

router = APIRouter()


@router.post(
    "/upload", response_model=dict[str, Any], operation_id="upload_file_operation"
)
async def upload_files(
    knowledge_name: str,
    user_id: str,
    files: list[UploadFile] = File(...),
):
    try:
        return await process_uploads(
            knowledge_name=knowledge_name, files=files, user_id=user_id
        )
    except Exception as e:
        logger.error(f"Upload failed for {knowledge_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/index-file", response_model=dict[str, Any], operation_id="index_file_operation"
)
async def index_files(
    knowledge_name: str,
    user_id: str,
):
    try:
        return await index_file(knowledge_name=knowledge_name, user_id=user_id)

    except Exception as e:
        logger.error(f"Indexing failed for {knowledge_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_model=str, operation_id="ask_operation")
async def ask_router(knowledge_name: str, user_id: str, query: str):
    try:
        return await ask_service(
            knowledge_name=knowledge_name, user_id=user_id, query=query
        )

    except Exception as e:
        logger.error(f"Ask failed for {knowledge_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
