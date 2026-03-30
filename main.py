from fastapi import FastAPI
from models import TranscriptRequest

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from database import index

from dotenv import load_dotenv
import os
import requests
from fpdf import FPDF


app = FastAPI()
load_dotenv()
# Initialize the client using the provided key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@app.post("/generate-summary")
def generate_summary(req: TranscriptRequest):

    docs = [Document(page_content=req.transcript)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    

    chunks = text_splitter.split_documents(docs)
    chunk_texts = [chunk.page_content for chunk in chunks]

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = embeddings_model.embed_documents(chunk_texts)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunk_texts, embeddings)):
        vectors.append({
            "id": f"{req.room_id}-chunk-{i}",
            "values": embedding,
            "metadata": {
                "chunk_index":i,
                "room_id": req.room_id,
                "chunk_text": chunk
            }
        })
    index.upsert(vectors=vectors)
    print(f"Stored {len(vectors)} chunk in pinecone")

    groq_chat = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    print(req.transcript)
    output_parser = StrOutputParser()

    chunk_prompt = ChatPromptTemplate.from_template(
        "Summarize this part of a meeting transcript briefly:\n{chunk}"
    )

    chain = chunk_prompt | groq_chat | output_parser

    chunk_summaries = []
    for chunk_text in chunk_texts:
        chunk_summary = chain.invoke({"chunk": chunk_text})
        chunk_summaries.append(chunk_summary)

    combined_text = "\n\n".join(chunk_summaries)
    final_prompt = ChatPromptTemplate.from_template(
        "You have been given partial summaries of a meeting. "
        "Combine them into one clean, coherent final summary:\n{combined}"
    )

    final_chain = final_prompt | groq_chat | output_parser
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