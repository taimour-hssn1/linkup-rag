from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

db = os.getenv("PINECONE_DB")
pc = Pinecone(api_key=db)

existing_indexes = pc.list_indexes().names()
if "transcripts" not in existing_indexes:
    pc.create_index(
        name="transcripts",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index("transcripts")
