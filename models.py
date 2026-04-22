from pydantic import BaseModel

class TranscriptRequest(BaseModel):
    room_id: str
    transcript: str
    mixed_transcript: str

class QueryRequest(BaseModel):
    room_id: str
    query: str
