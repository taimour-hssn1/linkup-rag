from pydantic import BaseModel

class TranscriptRequest(BaseModel):
    room_id: int
    transcript: str