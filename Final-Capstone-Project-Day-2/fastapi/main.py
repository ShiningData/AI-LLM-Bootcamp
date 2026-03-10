from fastapi import FastAPI, File, UploadFile, Form
from my_nodes import agent, config
from models import ChatRequest
from ingest_documents_vectordb import ingest_files
from dotenv import load_dotenv
from typing import List, Optional


load_dotenv()


app = FastAPI(title="AILLM3 Final Project")


@app.post("/chat")
async def answer_request(request: ChatRequest):
    try:
        # Get the final result from streaming
        final_result = None
        for step in agent.stream(
            {"messages": [{"role": "user", "content": request.message}]}, 
            stream_mode="values",
            config=config
        ):
            final_result = step

        # Return the last AI message
        return {"response": final_result["messages"][-1].content}

    except Exception as e:
        return {"error": f"Failed to process request: {str(e)}"}


@app.post("/ingest")
async def ingest_documents_endpoint(
    files: List[UploadFile] = File(..., description="Select files to ingest"),
    collection_name: Optional[str] = Form("my_documents", description="Collection name"),
    difficulty: Optional[str] = Form("middle", description="Difficulty level"),
    main_language: Optional[str] = Form("Python", description="Main programming language")
):
    try:
        result = await ingest_files(
            files=files,
            collection_name=collection_name,
            difficulty=difficulty,
            main_language=main_language
        )
        return result
    except Exception as e:
        return {"status": "error", "message": f"Failed to ingest documents: {str(e)}"}
