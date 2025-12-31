import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from app.services.vectordb import get_vectorstore

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=500
)


from app.models.interview_session import session

def get_resume_jd_context():
    try:
        print("DEBUG: Getting vectorstores...")
        
       
        resume_db = get_vectorstore("resume")  
        jd_db = get_vectorstore("job_description")  
        
        print(f"DEBUG: Resume DB: {resume_db}")
        print(f"DEBUG: JD DB: {jd_db}")
        

        resume_docs = resume_db.similarity_search("skills experience education", k=5)
        jd_docs = jd_db.similarity_search("requirements qualifications responsibilities", k=5)
        
        print(f"DEBUG: Found {len(resume_docs)} resume docs")
        print(f"DEBUG: Found {len(jd_docs)} JD docs")
        
        resume_text = "\n".join([d.page_content for d in resume_docs])
        jd_text = "\n".join([d.page_content for d in jd_docs])
        
        print(f"DEBUG: Resume text length: {len(resume_text)}")
        print(f"DEBUG: JD text length: {len(jd_text)}")
        
        if not resume_text and not jd_text:
            print("ERROR: Both resume and JD are empty")
            return "No resume content found", "No JD content found"
            
        return resume_text, jd_text
        
    except Exception as e:
        print(f"ERROR in get_resume_jd_context: {str(e)}")
        return f"Error: {str(e)}", f"Error: {str(e)}"



def generate_question():
    try:
        print(f"\n=== DEBUG: Starting generate_question ===")
        print(f"Session questions count: {len(session.questions)}")
        print(f"Session max_questions: {session.max_questions}")
        
       
        if len(session.questions) >= session.max_questions:
            print(f"DEBUG: Max questions reached ({session.max_questions})")
            return {"error": "Maximum number of questions reached"}
        
        print("DEBUG: Getting context...")
        resume, jd = get_resume_jd_context()
        
        print(f"DEBUG: Resume preview: {resume[:100]}...")
        print(f"DEBUG: JD preview: {jd[:100]}...")

        if "Error:" in resume or "No resume" in resume or len(resume) < 10:
            print("ERROR: No valid resume content")
            return {"error": "No resume content found. Please upload a resume first."}
        
        if "Error:" in jd or "No JD" in jd or len(jd) < 10:
            print("ERROR: No valid JD content")
            return {"error": "No job description found. Please upload a JD first."}
        
        prompt = PromptTemplate(
            input_variables=["resume", "jd"],
            template="""
You are an experienced technical interviewer. Based on the following Resume and Job Description, generate ONE interview question.

Focus on:
1. Technical skills mentioned in both
2. Experience gaps between resume and JD
3. Behavioral aspects relevant to the role

Resume:
{resume}

Job Description:
{jd}

IMPORTANT: Return ONLY valid JSON with this exact format:
{{
  "question": "Your generated question here"
}}

Do not include any other text, markdown, or formatting.
"""
        )
        
        print("DEBUG: Formatting prompt...")
        formatted_prompt = prompt.format(resume=resume[:3000], jd=jd[:3000])  
        
        print("DEBUG: Calling LLM...")
        response = llm.invoke(formatted_prompt)
        
        print(f"DEBUG: LLM response type: {type(response)}")
        print(f"DEBUG: LLM response content: {response.content}")

        content = response.content.strip()
        

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        print(f"DEBUG: Cleaned content: {content}")
        
        try:
            question_obj = json.loads(content)
            question = question_obj.get("question", "")
            
            if not question:
                print("ERROR: Question key is empty")
                return {"error": "Failed to generate question"}
                
            print(f"DEBUG: Generated question: {question}")
            

            session.questions.append(question)
            print(f"DEBUG: Added to session. Total questions: {len(session.questions)}")
            
            return {"question": question}
            
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON parsing failed: {e}")
            print(f"ERROR: Raw content: {content}")
            

            if '"question"' in content:
                import re
                match = re.search(r'"question"\s*:\s*"([^"]+)"', content)
                if match:
                    question = match.group(1)
                    session.questions.append(question)
                    return {"question": question}
            
            return {"error": f"Failed to parse LLM response: {str(e)}"}
            
    except Exception as e:
        print(f"ERROR in generate_question: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Internal error: {str(e)}"}


def evaluate_answer(user_answer: str):
    try:
        if not session.questions:
            return {"error": "No question has been asked yet. Please get a question first."}
        
        question = session.questions[-1]
        print(f"DEBUG: Evaluating answer for question: {question}")
        print(f"DEBUG: User answer: {user_answer}")
        
        prompt = PromptTemplate(
            input_variables=["question", "user_answer"],
            template="""
You are a technical interviewer evaluating an answer.

Question: {question}

Candidate's Answer: {user_answer}

Analyze the answer and provide feedback in JSON format with these exact keys:

{{
  "is_correct": true/false,
  "score": 0-100,
  "strengths": ["list", "of", "strengths"],
  "improvements": ["areas", "to", "improve"],
  "suggested_better_answer": "optional better answer"
}}

Focus on:
1. Technical accuracy
2. Completeness
3. Relevance to the question
4. Communication clarity
"""
        )
        
        response = llm.invoke(prompt.format(question=question, user_answer=user_answer))
        content = response.content.strip()
        

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        try:
            result = json.loads(content)
        except:
            result = {
                "is_correct": False,
                "score": 0,
                "strengths": [],
                "improvements": ["Could not evaluate"],
                "suggested_better_answer": ""
            }
        
       
        session.history.append({
            "question": question,
            "answer": user_answer,
            "evaluation": result
        })
        
        print(f"DEBUG: Evaluation result: {result}")
        return result
        
    except Exception as e:
        print(f"ERROR in evaluate_answer: {str(e)}")
        return {"error": str(e)}



def get_history():
    return {
        "total_questions": len(session.questions),
        "history": session.history
    }



def reset_session():
    session.questions = []
    session.history = []
    session.current_index = 0
    print("DEBUG: Session reset")
    return {"message": "Session reset successfully"}



def get_session_status():
    return {
        "questions_asked": len(session.questions),
        "max_questions": session.max_questions,
        "current_index": session.current_index
    }