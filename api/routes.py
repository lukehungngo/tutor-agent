from fastapi import APIRouter
from services.planner import generate_subqueries
from services.researcher import ai_research
from services.publisher import generate_learning_report

router = APIRouter()

@router.get("/api/learning")
async def get_learning_path(task: str):
    """AI-powered learning career generator."""
    subqueries_result = await generate_subqueries(task)
    research_results = ai_research(subqueries_result["sub_queries"])
    report = generate_learning_report(task, research_results)

    return {"message": "Learning path generated!", "data": report}
