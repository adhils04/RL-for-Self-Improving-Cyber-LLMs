import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time

# Add src to python path to load the real model if needed later
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defender_policy.baseline import ReviewOneBaselineDefender

app = FastAPI(title="MAPPO UI Server")

# Allow CORS for the frontend HTML file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

# Initialize the exact same Defender used in the MAPPO runs
defender = ReviewOneBaselineDefender()

@app.post("/chat")
async def chat_with_llm(request: PromptRequest):
    """
    Receives an adversarial prompt from the UI and returns the Defender's response.
    """
    # Pass the adversarial prompt as the untrusted content injected into a benign task
    benign_task = "Please summarize the contents of this document."
    
    # Run the exact same defense logic as the Kaggle project
    outcome = defender.respond(user_task=benign_task, untrusted_content=request.prompt)
    
    # Format the response clearly for the UI
    status_tag = f"[{outcome.status.upper()}]"
    final_response = f"{status_tag} {outcome.response}"
    
    return {"response": final_response}

if __name__ == "__main__":
    import uvicorn
    print("[*] Starting MAPPO UI Server on http://127.0.0.1:8000")
    uvicorn.run("ui_server:app", host="127.0.0.1", port=8000, reload=True)
