from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str

class IngestRequest(BaseModel):
    directory_path: str
    collection_name: Optional[str] = "my_documents"
    difficulty: Optional[str] = "middle"
    main_language: Optional[str] = "Python"