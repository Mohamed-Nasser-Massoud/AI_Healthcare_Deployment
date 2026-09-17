from fastapi import APIRouter, HTTPException

from app import model_loader
from app.schemas import AskRequest, AskResponse

router = APIRouter(tags=["rag"])


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = model_loader.ask_rag(request.question, k=request.k)
    except Exception as e:
        # The LLM call is the one part of this whole API that depends on an
        # external service being up and reachable. That's a fundamentally
        # different failure mode from a bad request or a model bug, so it
        # gets its own status code (502 Bad Gateway: "the upstream service
        # failed", not "your request was wrong") rather than an unhandled
        # 500 or a silently wrong answer.
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider request failed: {e}",
        )
    return AskResponse(answer=result["answer"], sources=result["sources"])
