from router import router
from fastapi import FastAPI

app = FastAPI()

app.include_router(router, prefix="/rag", tags=["RAG"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
