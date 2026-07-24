import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from engine.rag_engine import RAGEngine

# .env se variables load 
load_dotenv()

# FastAPI app initialize karo
app = FastAPI(
    title="Enterprise AI Knowledge Agent API",
    description="Backend API for RAG-based AI Agent built with FastAPI and LangChain.",
    version="1.0.0"
)

# Global level par RAGEngine initialize 
print("Initializing RAG Engine and ingesting documents...")
try:
    rag_engine = RAGEngine(data_dir=".", api_key="OPENAI_API_KEY")
    rag_engine.ingest_documents()
    rag_chain = rag_engine.get_conversational_chain()
    print("RAG Engine is ready!")
except Exception as e:
    print(f"Error initializing RAG Engine: {e}")
    rag_chain = None

# Request body ke liye Pydantic model (Data validation)
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_user_session"

class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str

@app.get("/")
def home():
    return {"message": "AI Knowledge Agent API is running successfully!"}

@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG Engine is not initialized properly.")
    
    try:
        config = {"configurable": {"session_id": request.session_id}}
        
        # Invoke the chain
        result = rag_chain.invoke({"input": request.question}, config=config)
        
        # LangChain history chain gives a dict in return which contains the 'answer' key
        if isinstance(result, dict):
            answer_text = result.get("answer", str(result))
        else:
            answer_text = str(result)
        
        return ChatResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))