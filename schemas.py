"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
Each Pydantic model corresponds to a MongoDB collection where the
collection name is the lowercase of the class name.

Example: class UserAccount -> collection "useraccount"
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal

# Access tiers used to gate models/tools
Tier = Literal["Starter", "Plus", "Pro", "Enterprise"]

class UserAccount(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    tier: Tier = Field("Starter", description="Subscription tier")
    api_key: str = Field(..., description="User API key for simple header-based auth")

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatSession(BaseModel):
    user_key: str
    messages: List[ChatMessage]
    selected_models: Optional[List[str]] = None

class GenerationRequest(BaseModel):
    user_key: str
    kind: Literal["image", "video"]
    tool: Optional[str] = None
    prompt: str

# Optional: Simple audit log for requests
class AuditLog(BaseModel):
    user_key: str
    action: Literal["chat", "image", "video", "upgrade"]
    detail: Optional[str] = None
