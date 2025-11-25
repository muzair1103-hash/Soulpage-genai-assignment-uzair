from fastapi import UploadFile
from typing import Any

from fastapi import HTTPException, UploadFile

from logger import logger
from backend import save_uploaded_file, index_all_pdfs, ask


async def process_uploads(
    knowledge_name: str, files: list[UploadFile], user_id: str
) -> dict[str, Any]:
    uploaded_files: list[str] = []
    failed_files: list[str] = []

    for file in files:
        filename, success = await save_uploaded_file(
            knowledge_name=knowledge_name, file=file, user_id=user_id
        )
        if success:
            uploaded_files.append(filename)
        else:
            failed_files.append(filename)
    return {
        "success": len(failed_files) == 0,
        "message": "Files processed.",
        "uploaded_files": uploaded_files,
        "failed_files": failed_files,
    }


async def index_file(knowledge_name: str, user_id: str) -> dict[str, Any]:
    try:
        return await index_all_pdfs(knowledge_name=knowledge_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Index failed for {knowledge_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def ask_service(knowledge_name: str, user_id: str, query: str):
    try:
        return await ask(knowledge_name=knowledge_name, user_id=user_id, query=query)

    except Exception as e:
        logger.error(f"Ask failed for {knowledge_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
