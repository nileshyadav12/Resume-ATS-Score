from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import os
import fitz  
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import json
import re

app = FastAPI(
    title="Resume ATS Score",
    version="1.0",
    description="You can ask any type of question for a custom prompt; otherwise, it will return a null value."
)

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI is running!"}

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAVE_DIR = "extracted_resumes"
os.makedirs(SAVE_DIR, exist_ok=True)

# This API should not be used. Please note that I do not have another one, and the token has expired.
model_name = "gemini-1.5-flash-001"
api_key = "AIzaSyCXtX-vjh2a6zL9qmDTi5xByVhAh3_w9g8"

genai.configure(api_key=api_key)
llm = genai.GenerativeModel(model_name=model_name)

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...), custom_prompt: str = Form("")) -> JSONResponse:
    if not file.filename.endswith('.pdf'):
        return JSONResponse(status_code=400, content={"error": "File must be a PDF."})

    file_path = os.path.join(SAVE_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    pdf_document = fitz.open(file_path)
    extracted_text = ""
    for page in pdf_document:
        extracted_text += page.get_text()
    pdf_document.close()

    async def extract_resume_details_with_llm(extracted_text: str):
        prompt = f"""
Extract the following information from the resume text below and return it as a clean JSON object with key-value pairs:

1. Full Name
2. Date of Birth (Optional)
3. Email
4. Phone Number
5. Skills
6. Work Experience
7. Education
8. Certifications
9. Project Name (Optional)
Resume text:

{extracted_text}
"""
        completion = llm.generate_content(prompt, generation_config={'temperature': 0})
        cleaned_response = re.sub(r'```json\n|\n```', '', completion.text).strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from LLM", "raw_response": cleaned_response}

    resume_details = await extract_resume_details_with_llm(extracted_text)

    def calculate_ats_score(resume_details, job_description):
        skills = resume_details.get("Skills", [])
        work_experience = resume_details.get("Work Experience", [])
        education = resume_details.get("Education", [])
        certifications = resume_details.get("Certifications", [])

        job_description_keywords = job_description.lower().split()
        skills = [skill.lower() for skill in skills]
        work_experience_text = ' '.join([exp.get("Job Title", "").lower() for exp in work_experience])
        education_text = ' '.join([edu.get("Degree", "").lower() for edu in education])
        certifications_text = ' '.join(certifications).lower()

        skills_score = sum(2 for skill in skills if skill in job_description_keywords)
        experience_score = sum(1.5 for word in work_experience_text.split() if word in job_description_keywords)
        education_score = sum(1.5 for word in education_text.split() if word in job_description_keywords)
        certifications_score = sum(1.5 for word in certifications_text.split() if word in job_description_keywords)

        total_score = skills_score + experience_score + education_score + certifications_score
        total_keywords = len(job_description_keywords)

        return (total_score / total_keywords) * 100 if total_keywords else 0

    job_description = "software engineer python machine learning"
    ats_score = calculate_ats_score(resume_details, job_description)

    if custom_prompt:
        custom_answer_data = await custom_answer(extracted_text, custom_prompt)
        response_content = {
            "filename": file.filename,
            "resume_details": resume_details,
            "ats_score": ats_score,
            "custom_answer": custom_answer_data
        }
    else:
        response_content = {
            "filename": file.filename,
            "resume_details": resume_details,
            "ats_score": ats_score,
            "custom_answer": "No custom prompt provided!"
        }

    return JSONResponse(content=response_content)

async def custom_answer(extracted_text, custom_prompt):
    prompt = f"""
Resume text: {extracted_text}

Question: {custom_prompt}

Please provide a detailed response based on the resume.
"""
    completion = llm.generate_content(prompt, generation_config={'temperature': 0})
    return completion.text

@app.get("/files/{filename}")
async def get_file(filename: str):
    file_path = os.path.join(SAVE_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return JSONResponse(status_code=404, content={"error": "File not found."})
