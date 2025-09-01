from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from groq import Groq
from typing import Optional, Dict, List
import json

# Initialize FastAPI app
app = FastAPI(
    title="Teacher Assistant Chatbot",
    description="A chatbot that assists teachers using their profile information",
    version="1.0.2"
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    profile_url: Optional[str] = None  # ✅ Optional full URL
    instructor_id: Optional[int] = None  # ✅ Optional instructor_id

class ChatResponse(BaseModel):
    response: str

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"))

# Conversation store
conversations: Dict[str, List[Dict[str, str]]] = {}

async def fetch_instructor_profile(profile_url: str) -> dict:
    """Fetch instructor profile from the given API URL"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(profile_url)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error fetching instructor profile: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"API returned error: {e.response.text}")

def create_system_prompt(instructor_data: dict) -> str:
    """Create a system prompt using JSON profile info"""
    instructor_json = json.dumps(instructor_data, indent=2)

    return f"""
    You are an AI assistant helping a teacher and replying to queries relevant to their teaching.
    Guide them using professional, supportive, and actionable advice.

    Use the following instructor profile data (JSON) to answer questions:

    {instructor_json}

    Rules:
    - Remember the conversation history to maintain context.
    - Only use information from this JSON or previous messages when answering.
    - Do not make up details that are not in the JSON or history.
    - If the JSON or history does not contain the answer, say "That information is not available in the profile."
    - Keep answers short, professional, and directly based on JSON values.
    """

async def generate_chat_response(message: str, profile_key: str, instructor_data: dict) -> str:
    """Generate response using Groq API with memory"""
    try:
        # Initialize conversation for this profile if not exists
        if profile_key not in conversations:
            conversations[profile_key] = [
                {"role": "system", "content": create_system_prompt(instructor_data)}
            ]

        # Add user message
        conversations[profile_key].append({"role": "user", "content": message})

        # Generate response
        chat_completion = groq_client.chat.completions.create(
            messages=conversations[profile_key],
            model="llama-3.3-70b-versatile",
            temperature=0.8,
            max_tokens=100
        )

        bot_response = chat_completion.choices[0].message.content

        # Save assistant response
        conversations[profile_key].append({"role": "assistant", "content": bot_response})

        return bot_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_teacher_assistant(request: ChatRequest):
    """Main endpoint for chatting with the teacher assistant"""

    # ✅ Determine which URL to use
    if request.profile_url:
        profile_url = request.profile_url
    elif request.instructor_id:
        profile_url = f"https://ace.prismaticcrm.com/api/instructor-profile/{request.instructor_id}"
    else:
        raise HTTPException(status_code=400, detail="Either profile_url or instructor_id must be provided")

    instructor_data = await fetch_instructor_profile(profile_url)
    bot_response = await generate_chat_response(request.message, profile_url, instructor_data)

    return ChatResponse(response=bot_response)

@app.get("/")
async def root():
    return {
        "message": "Teacher Assistant Chatbot API",
        "endpoints": {
            "POST /chat": "Main chat endpoint - send messages to the teacher assistant",
            "GET /docs": "Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "teacher-assistant-chatbot"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
