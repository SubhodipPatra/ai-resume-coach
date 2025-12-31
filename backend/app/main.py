from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from app.services.parser import extract_text
from app.services.chunker import chunk_text
from app.services.vectordb import get_vectorstore
from app.services.ai_engine import (
    calculate_match_score,
    find_missing_skills,
    improve_resume,
    generate_interview_questions
)

# Import the interview bot functions
from app.services.interview_bot import (
    generate_question,
    evaluate_answer,
    get_history,
    reset_session
)

app = FastAPI(title="AI Resume Coach & Interview Bot")


# Resume & Job Description APIs

@app.post("/upload/resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        chunks = chunk_text(text)

        vectordb = get_vectorstore("resume")
        vectordb.add_texts(chunks)

        return {"message": "Resume stored successfully", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload/jd")
async def upload_jd(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        chunks = chunk_text(text)

        vectordb = get_vectorstore("job_description")
        vectordb.add_texts(chunks)

        return {"message": "JD stored successfully", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# Semantic Search

@app.get("/search/{source}")
def semantic_search(source: str, query: str):
    if source not in ["resume", "jd"]:
        raise HTTPException(status_code=400, detail="source must be resume or jd")

    vectordb = get_vectorstore(source)
    results = vectordb.similarity_search(query, k=3)

    return {"query": query, "results": [r.page_content for r in results]}



# AI-driven Resume Analysis

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



# Interactive Interview Endpoints

@app.get("/chatbot/next-question")
def next_question_endpoint():
    result = generate_question()
    return result 


@app.post("/chatbot/answer")
def answer_endpoint(user_answer: str = Body(..., embed=True)):
    result = evaluate_answer(user_answer)
    return result


@app.get("/chatbot/status")
def status_endpoint():
    from app.services.interview_bot import get_session_status
    return get_session_status()
