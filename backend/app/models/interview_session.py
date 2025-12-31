
from typing import List, Dict
from pydantic import BaseModel

class InterviewSession:
    def __init__(self, max_questions: int = 5):
        self.questions: List[str] = []
        self.history: List[Dict] = []
        self.max_questions = max_questions
        self.current_index = 0  

session = InterviewSession()
