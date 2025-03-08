from typing import Dict, List, TypedDict, Optional, Any
from pydantic import BaseModel, Field
from langchain.docstore.document import Document

# Define the response model for the RAG system
class QueryResponse(BaseModel):
    """Response model for the query answering system."""
    answer: str = Field(description="The answer to the user's query")
    source: str = Field(description="The source of the information (vector DB or web search)")
    confidence: float = Field(description="Confidence score from 0 to 1")

# Define the state for the LangGraph workflow
class State(TypedDict):
    """State definition for the RAG workflow."""
    query: str
    retrieval_context: Optional[List[Document]]
    web_search_results: Optional[List[Document]]
    memory: List[Dict[str, str]]
    response: Optional[Dict[str, Any]]
    needs_web_search: bool