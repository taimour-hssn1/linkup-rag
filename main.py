from fastapi import FastAPI
from dotenv import load_dotenv
import os

from models import TranscriptRequest, QueryRequest, SmartQueryRequest
from database import index
from llm_config import embeddings_model, groq_chat, output_parser
from agent_service import smart_router, run_parallel_subagents, orchestrate, query_summary

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from templates import CHUNK_SUMMARY_PROMPT, FINAL_SUMMARY_PROMPT

load_dotenv()
app = FastAPI()

@app.post("/generate-summary")
def generate_summary(req: TranscriptRequest):
    docs = [Document(page_content=req.transcript)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(docs)
    chunk_texts = [chunk.page_content for chunk in chunks]

    embeddings = embeddings_model.embed_documents(chunk_texts)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunk_texts, embeddings)):
        vectors.append({
            "id": f"{req.room_id}-chunk-{i}",
            "values": embedding,
            "metadata": {
                "room_id": req.room_id,
                "chunk_index":i,
                "chunk_text": chunk
            }
        })

    index.upsert(vectors=vectors)
    print(f"Stored {len(vectors)} chunk in pinecone")
    print(req.transcript)

    chain = CHUNK_SUMMARY_PROMPT | groq_chat | output_parser

    chunk_summaries = []
    for chunk_text in chunk_texts:
        chunk_summary = chain.invoke({"chunk": chunk_text})
        chunk_summaries.append(chunk_summary)

    combined_text = "\n\n".join(chunk_summaries)
    final_chain = FINAL_SUMMARY_PROMPT | groq_chat | output_parser
    final_summary = final_chain.invoke({"combined": combined_text})
    print(final_summary)

    return {
        "summary_content": final_summary
    }

    # 1. Create a PDF and save it locally with the room_id name
    # pdf_filename = f"{req.room_id}.pdf"
    # try:
    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.set_font("Arial", size=12)
    #     # Fix encoding issues that FPDF might encounter with emojis or special markdown
    #     encoded_summary = final_summary.encode('latin-1', 'replace').decode('latin-1')
    #     pdf.multi_cell(0, 10, txt=encoded_summary)
    #     pdf.output(pdf_filename)
    #     print(f"Successfully saved PDF to {pdf_filename}")
    # except Exception as e:
    #     print(f"Error creating PDF: {e}")

    # 2. AWS S3 Upload (Commented out until credentials are available)
    # import boto3
    # s3_client = boto3.client(
    #     's3',
    #     aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    #     aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    #     region_name='us-east-1'
    # )
    # bucket_name = "my-linkup-summaries-bucket"
    # try:
    #     s3_client.upload_file(pdf_filename, bucket_name, pdf_filename)
    #     s3_url = f"https://{bucket_name}.s3.amazonaws.com/{pdf_filename}"
    #     # NOTE: Later you will return s3_url instead of the raw text summary
    #     # return {"s3_link": s3_url}
    # except Exception as e:
    #     print(f"Failed to upload to S3: {e}")

    # return {
    #     "s3_link": pdf_filename   # keep same key for Spring Boot
    # }

@app.post("/query")
def query_smart(req: SmartQueryRequest):  
    count = len(req.room_ids)

    print(req)
    if count == 1:
        # Path 1: Direct subagent, no routing needed
        qreq = QueryRequest(
            room_id=req.room_ids[0],
            query=req.query
        )
        result = query_summary(qreq)
        return {"response": result}

    elif count > 1:
        # Path 2: Parallel subagents + orchestrator
        results = run_parallel_subagents(req.room_ids, req.query)
        return {"response": orchestrate(results, req.query)}

    else:
        # Path 3: 0 meetings = Router decides
        resolved_ids = smart_router(req.query, req.user_id)
        
        if not resolved_ids:
            return {"response": "Meeting not available. Please specify a more precise date, meeting title, or select meetings manually."}
        
        print(resolved_ids)
        if len(resolved_ids) == 1:
            qreq = QueryRequest(
                room_id=resolved_ids[0],
                query=req.query
            )
            result = query_summary(qreq)
            return {"response": result}
        
        # Parallel pass if routing resulted in >1 meetings
        results = run_parallel_subagents(resolved_ids, req.query)
        return {"response": orchestrate(results, req.query)}
