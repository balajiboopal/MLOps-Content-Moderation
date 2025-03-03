from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.ml_models.text_classifier import predict_toxicity
import logging

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextInput(BaseModel):
    text: str

class ModeratedContent(BaseModel):
    text: str
    toxicity_score: float
    moderation_result: str

@app.post("/moderate", response_model=ModeratedContent)
async def moderate_text(input: TextInput):
    """Moderates input text and returns toxicity score."""
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    toxicity_score = predict_toxicity(input.text)
    moderation_result = "toxic" if toxicity_score > 0.5 else "non-toxic"

    logger.info(f"Moderated text: '{input.text}' with score {toxicity_score}")
    return ModeratedContent(text=input.text, toxicity_score=toxicity_score, moderation_result=moderation_result)

@app.get("/")
async def root():
    """Root API check."""
    return {"message": "Welcome to the Content Moderation API"}
