from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.services.vectordb import get_vectorstore

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

def get_resume_jd_context():
    resume_db = get_vectorstore("resume_docs")
    jd_db = get_vectorstore("job_description")

    resume_docs = resume_db.similarity_search("resume content", k=5)
    jd_docs = jd_db.similarity_search("job description", k=5)

    resume_text = "\n".join([d.page_content for d in resume_docs])
    jd_text = "\n".join([d.page_content for d in jd_docs])

    return resume_text, jd_text
def calculate_match_score():
    resume, jd = get_resume_jd_context()

    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template="""
You are an ATS system.

Compare the resume with the job description.
Return:
1. Match percentage (0–100)
2. Short justification

Resume:
{resume}

Job Description:
{jd}

Respond in JSON:
{{
  "match_percentage": number,
  "reason": "text"
}}
"""
    )

    response = llm.invoke(
        prompt.format(resume=resume, jd=jd)
    )

    return response.content
def find_missing_skills():
    resume, jd = get_resume_jd_context()

    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template="""
Extract skills required in the JD but missing in the resume.

Resume:
{resume}

Job Description:
{jd}

Return JSON:
{{
  "missing_skills": ["skill1", "skill2"]
}}
"""
    )

    response = llm.invoke(
        prompt.format(resume=resume, jd=jd)
    )

    return response.content
def improve_resume():
    resume, jd = get_resume_jd_context()

    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template="""
Improve the resume bullets to better match the JD.
Focus on:
- Action verbs
- Metrics
- ATS keywords

Resume:
{resume}

Job Description:
{jd}

Return JSON:
{{
  "improvements": [
    "Improved bullet 1",
    "Improved bullet 2"
  ]
}}
"""
    )

    response = llm.invoke(
        prompt.format(resume=resume, jd=jd)
    )

    return response.content
def generate_interview_questions():
    resume, jd = get_resume_jd_context()

    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template="""
Generate 50 interview questions based on resume and JD.
Include:
- Technical
- Behavioral
- Scenario-based

Resume:
{resume}

Job Description:
{jd}

Return JSON:
{{
  "questions": [
    "Question 1",
    "Question 2"
  ]
}}
"""
    )

    response = llm.invoke(
        prompt.format(resume=resume, jd=jd)
    )

    return response.content
