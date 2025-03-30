from fastapi import FastAPI, Form, File, UploadFile, HTTPException
import openai
import os
import zipfile
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API Key
if not OPENAI_API_KEY:
    raise ValueError("⚠️ OpenAI API key not found in environment variables.")

# Initialize FastAPI app
app = FastAPI()
openai.api_key = OPENAI_API_KEY  

async def query_openai(question: str) -> str:
    """Query OpenAI API to get an answer."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ OpenAI API Error: {str(e)}"

async def process_file(file: UploadFile) -> str:
    """Extract question from CSV inside ZIP and get an answer from OpenAI."""
    if not file.filename.endswith(".zip"):
        return "⚠️ Only ZIP files are supported."

    try:
        zip_content = await file.read()
        with zipfile.ZipFile(BytesIO(zip_content), 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print("Extracted files:", file_list)  # Debugging print

            # Ensure there is exactly one CSV file
            csv_files = [f for f in file_list if f.endswith('.csv')]
            if len(csv_files) != 1:
                return "⚠️ ZIP must contain exactly one CSV file."

            # Read CSV file
            with zip_ref.open(csv_files[0]) as csv_file:
                df = pd.read_csv(csv_file)
                print("CSV columns:", df.columns)  
                
                if "answer" not in df.columns:
                    return "⚠️ 'answer' column not found in CSV."
                
                extracted_question = str(df["answer"].iloc[0]) 
                return await query_openai(extracted_question)  

    except zipfile.BadZipFile:
        return "⚠️ Invalid ZIP file format."
    except Exception as e:
        return f"⚠️ File processing error: {str(e)}"

@app.get("/")
async def root():
    """Simple check endpoint"""
    return {"message": "FastAPI is running!"}

@app.post("/api/")
async def answer_question(question: str = Form(...), file: UploadFile = File(None)):
    """Handles both text-based and file-based questions."""
    if file:
        file_answer = await process_file(file)
        if not file_answer.startswith("⚠️"):  
            return {"answer": file_answer}
    
    # Process text-based question
    openai_answer = await query_openai(question)
    return {"answer": openai_answer}
