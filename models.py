from pydantic import BaseModel

class TranscriptRequest(BaseModel):
    meeting_id: int
    transcript: str