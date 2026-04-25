import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

groq_chat = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)
output_parser = StrOutputParser()
