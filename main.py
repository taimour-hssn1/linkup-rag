from fastapi import FastAPI
from models import TranscriptRequest

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

    groq_chat = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )
    print(req.transcript)
    prompt = ChatPromptTemplate.from_template(
        "Summarize this meeting:\n{transcript}"
    )

    output_parser = StrOutputParser()

    chain = prompt | groq_chat | output_parser

    summary = chain.invoke({"transcript": req.transcript})
    print(summary)

    # 1. Create a PDF and save it locally with the meeting_id name
    pdf_filename = f"{req.meeting_id}.pdf"
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # Fix encoding issues that FPDF might encounter with emojis or special markdown
        encoded_summary = summary.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=encoded_summary)
        pdf.output(pdf_filename)
        print(f"Successfully saved PDF to {pdf_filename}")
    except Exception as e:
        print(f"Error creating PDF: {e}")

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

    return {
        "s3_link": pdf_filename   # keep same key for Spring Boot
    }