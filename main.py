from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from langchain_core.messages import HumanMessage
from chatbot import run_agent

app = FastAPI()

# ✅ Allow all CORS (or tighten it later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request schema
class QuestionRequest(BaseModel):
    question: str

# ✅ Response schema
class AnswerResponse(BaseModel):
    answer: str

# ✅ Root route (optional)
@app.get("/")
async def root():
    return {"message": "Chatbot API is running!"}

# ✅ Chatbot endpoint
@app.post("/get_answer", response_model=AnswerResponse)
async def get_answer(request: QuestionRequest):
    user_question = request.question.strip()

    if not user_question:
        return {"answer": "Please enter a valid question."}

    session_id = str(uuid.uuid4())

    try:
        response = run_agent(HumanMessage(content=user_question), session_id)

        if isinstance(response, dict) and "output" in response:
            answer = response["output"]
        else:
            answer = str(response) if response else "Sorry, I couldn't find an answer."

        return {"answer": answer}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"answer": "I apologize, but I encountered an error processing your request. Please try again."}
