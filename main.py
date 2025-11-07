import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import db, create_document, get_documents
from schemas import UserAccount, ChatSession, GenerationRequest, ChatMessage, Tier
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Simple Auth via API Key
# ------------------------

def require_user(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> UserAccount:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    # Look up user by api_key
    user_docs = list(db["useraccount"].find({"api_key": x_api_key}).limit(1)) if db else []
    if not user_docs:
        raise HTTPException(status_code=401, detail="Invalid API key")
    doc = user_docs[0]
    return UserAccount(email=doc.get("email"), name=doc.get("name"), tier=doc.get("tier", "Starter"), api_key=doc.get("api_key"))

# ------------------------
# Capability gating
# ------------------------

TIER_MODELS: Dict[Tier, List[str]] = {
    "Starter": ["community"],
    "Plus": ["community", "deepseek", "grok"],
    "Pro": ["community", "deepseek", "grok", "gpt-4", "claude-3", "gemini-1.5"],
    "Enterprise": ["community", "deepseek", "grok", "gpt-4", "claude-3", "gemini-1.5", "ernie"],
}

TIER_TOOLS: Dict[Tier, Dict[str, List[str]]] = {
    "Starter": {"image": ["nano-bana-lite"], "video": []},
    "Plus": {"image": ["nano-bana", "sea-dream"], "video": ["video-lite"]},
    "Pro": {"image": ["nano-bana-pro", "sea-dream"], "video": ["video-standard"]},
    "Enterprise": {"image": ["nano-bana-pro", "sea-dream"], "video": ["video-studio"]},
}

# ------------------------
# External API placeholders (configure keys in env)
# ------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
ERNIE_API_KEY = os.getenv("ERNIE_API_KEY")

# ------------------------
# Helpers
# ------------------------

def call_model(model: str, prompt: str) -> str:
    """Dispatch to a third-party model. For demo, mock responses if keys missing."""
    try:
        if model == "gpt-4" and OPENAI_API_KEY:
            # Example minimal call; replace with actual OpenAI Chat Completions as needed
            return f"[GPT-4] {prompt}"
        if model == "claude-3" and ANTHROPIC_API_KEY:
            return f"[Claude] {prompt}"
        if model == "gemini-1.5" and GOOGLE_API_KEY:
            return f"[Gemini] {prompt}"
        if model == "deepseek" and DEEPSEEK_API_KEY:
            return f"[DeepSeek] {prompt}"
        if model == "grok" and GROK_API_KEY:
            return f"[Grok] {prompt}"
        if model == "ernie" and ERNIE_API_KEY:
            return f"[ERNIE] {prompt}"
        # community fallback
        return f"[Community] {prompt}"
    except Exception as e:
        return f"[{model} error] {str(e)[:120]}"

def synthesize_panel(responses: List[str]) -> str:
    """Simple synthesis: extract consensus and summarize."""
    if not responses:
        return "No responses available."
    head = responses[0]
    if len(responses) == 1:
        return head
    # naive merge
    bullets = "\n".join([f"• {r}" for r in responses])
    return f"Consensus summary based on {len(responses)} models:\n{bullets}"

# ------------------------
# Routes
# ------------------------

@app.get("/")
def root():
    return {"ok": True}

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    selected_models: Optional[List[str]] = None

@app.post("/api/chat")
def multi_model_chat(req: ChatRequest, user: UserAccount = Depends(require_user)):
    models_allowed = set(TIER_MODELS.get(user.tier, []))
    models = req.selected_models or list(models_allowed)
    # filter by tier
    models = [m for m in models if m in models_allowed]
    if not models:
        raise HTTPException(status_code=403, detail="No models available at your tier")

    # Determine final user prompt
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message provided")
    prompt = user_msgs[-1]

    # Call models and synthesize
    panel_responses = [call_model(m, prompt) for m in models]
    final = synthesize_panel(panel_responses)

    # Audit
    try:
        create_document("auditlog", {"user_key": user.api_key, "action": "chat", "detail": f"models={models}"})
    except Exception:
        pass

    return {"models_used": models, "panel": panel_responses, "final": final}

class ImageRequest(BaseModel):
    tool: Optional[str] = None
    prompt: str

@app.post("/api/generate/image")
def generate_image(req: ImageRequest, user: UserAccount = Depends(require_user)):
    allowed_tools = TIER_TOOLS.get(user.tier, {}).get("image", [])
    tool = req.tool or (allowed_tools[0] if allowed_tools else None)
    if tool not in allowed_tools:
        raise HTTPException(status_code=403, detail="Tool not available for your tier")

    # Mock generation URL
    url = f"https://images.example/{tool}/render?prompt={requests.utils.quote(req.prompt)}"
    try:
        create_document("auditlog", {"user_key": user.api_key, "action": "image", "detail": f"tool={tool}"})
    except Exception:
        pass

    return {"tool": tool, "url": url}

class VideoRequest(BaseModel):
    tool: Optional[str] = None
    prompt: str

@app.post("/api/generate/video")
def generate_video(req: VideoRequest, user: UserAccount = Depends(require_user)):
    allowed_tools = TIER_TOOLS.get(user.tier, {}).get("video", [])
    tool = req.tool or (allowed_tools[0] if allowed_tools else None)
    if tool not in allowed_tools:
        raise HTTPException(status_code=403, detail="Tool not available for your tier")

    url = f"https://videos.example/{tool}/render?prompt={requests.utils.quote(req.prompt)}"
    try:
        create_document("auditlog", {"user_key": user.api_key, "action": "video", "detail": f"tool={tool}"})
    except Exception:
        pass

    return {"tool": tool, "url": url}

class SignupRequest(BaseModel):
    email: str
    name: Optional[str] = None

@app.post("/api/auth/signup")
def signup(req: SignupRequest):
    import secrets
    key = secrets.token_hex(16)
    user = UserAccount(email=req.email, name=req.name, tier="Starter", api_key=key)
    try:
        create_document("useraccount", user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"api_key": key, "tier": user.tier}

class UpgradeRequest(BaseModel):
    tier: Tier

@app.post("/api/billing/upgrade")
def upgrade(req: UpgradeRequest, user: UserAccount = Depends(require_user)):
    # In real life, integrate with Stripe/Paddle; here we persist tier.
    try:
        db["useraccount"].update_one({"api_key": user.api_key}, {"$set": {"tier": req.tier}})
        create_document("auditlog", {"user_key": user.api_key, "action": "upgrade", "detail": f"tier={req.tier}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "tier": req.tier}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = getattr(db, 'name', '✅ Connected')
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
