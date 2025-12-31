from fastapi import FastAPI, UploadFile, File, HTTPException
from app.services.parser import extract_text
from app.services.chunker import chunk_text
from app.services.vectordb import get_vectorstore
from app.services.ai_engine import (
    calculate_match_score,
    find_missing_skills,
    improve_resume,
    generate_interview_questions
)

app = FastAPI(title="AI Resume Coach")

@app.post("/upload/resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        chunks = chunk_text(text)

        vectordb = get_vectorstore("resume")
        vectordb.add_texts(chunks)

        return {
            "message": "Resume stored successfully",
            "chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload/jd")
async def upload_jd(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        chunks = chunk_text(text)

        vectordb = get_vectorstore("job_description")
        vectordb.add_texts(chunks)

        return {
            "message": "JD stored successfully",
            "chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
      
@app.get("/search/{source}")
def semantic_search(source: str, query: str):
    if source not in ["resume", "jd"]:
        raise HTTPException(status_code=400, detail="source must be resume or jd")

    vectordb = get_vectorstore(source)
    results = vectordb.similarity_search(query, k=3)

    return {
        "query": query,
        "results": [r.page_content for r in results]
    }
    
@app.get("/ai/match-score")
def match_score():
    return calculate_match_score()


@app.get("/ai/missing-skills")
def missing_skills():
    return find_missing_skills()


@app.get("/ai/improve-resume")
def improve():
    return improve_resume()


@app.get("/ai/interview-questions")
def interview_questions():
    return generate_interview_questions()

