from pydantic import BaseModel
from typing import List

class TranscriptRequest(BaseModel):
    room_id: str
    transcript: str

class QueryRequest(BaseModel):
    room_id: str
    query: str

class SmartQueryRequest(BaseModel):
    query: str
    user_id: str
    room_ids: List[str] = []   # empty list = 0 meetings selected
