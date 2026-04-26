from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

db = os.getenv("PINECONE_DB")
pc = Pinecone(api_key=db)

try:
    pc.create_index(
        name="transcripts",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
except Exception as e:
    if "ALREADY_EXISTS" not in str(e):
        raise e

index = pc.Index("transcripts")
