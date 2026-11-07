
import os
import secrets

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field

from app.inference import generate_response


app = FastAPI(
    title="Avrum Inference API",
    description="API for Avrum's fine-tuned agricultural AI model",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    max_new_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)


class GenerateResponse(BaseModel):
    response: str


def verify_api_key(x_api_key: str | None = Header(default=None)):
    expected_key = os.getenv("INFERENCE_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Inference API key has not been configured.",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "avrum-inference",
    }


@app.post(
    "/v1/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(verify_api_key)],
)
def generate(request: GenerateRequest):
    response = generate_response(
        question=request.question,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )

    return GenerateResponse(response=response)
