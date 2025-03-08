import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph

from src.agent_state import State, QueryResponse
from src.utils.prompts import CONTEXT_ANALYSIS_PROMPT, ANSWER_GENERATION_PROMPT
from src.tools import setup_web_search

class RAGSystem:
    """Class for managing the RAG system with langchain and langgraph."""
    
    def __init__(self, openai_api_key):
        """Initialize the RAG system with OpenAI API key."""
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.2,
            api_key=openai_api_key,
        )
        self.json_llm = self.llm.bind(response_format={"type": "json_object"})
        self.memory = []
        self.graph = None
    
    def create_graph(self, retriever):
        """Create the LangGraph workflow."""
        web_search = setup_web_search()
        json_parser = JsonOutputParser(pydantic_object=QueryResponse)
        
        async def retrieve_context(state: State) -> State:
            query = state["query"]
            context = await retriever(query)
            state["retrieval_context"] = context
            return state
        
        async def analyze_context(state: State) -> State:
            query = state["query"]
            context = state["retrieval_context"]
            
            if not context or len(context) == 0:
                state["needs_web_search"] = True
                return state
            
            chain = CONTEXT_ANALYSIS_PROMPT | self.json_llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "query": query,
                "context": "\n\n".join([doc.page_content for doc in context])
            })
            
            state["needs_web_search"] = not result['sufficient']
            return state
        
        async def perform_web_search(state: State) -> State:
            if state["needs_web_search"]:
                search_results = web_search(state["query"])
                state["web_search_results"] = search_results
            return state
        
        async def generate_answer(state: State) -> State:
            query = state["query"]
            memory = state["memory"]
            
            documents = []
            source_type = "vector_db"
            source_details = {}
            
            if state["retrieval_context"]:
                documents.extend(state["retrieval_context"])
                if len(state["retrieval_context"]) > 0:
                    first_doc = state["retrieval_context"][0]
                    if first_doc.metadata and "video_url" in first_doc.metadata:
                        source_details["video_url"] = first_doc.metadata["video_url"]
            
            if state["needs_web_search"] and state["web_search_results"]:
                documents.extend(state["web_search_results"])
                source_type = "web_search" if not state["retrieval_context"] else "vector_db_and_web_search"
                
                if state["web_search_results"] and len(state["web_search_results"]) > 0:
                    search_content = state["web_search_results"][0].page_content
                    if "link:" in search_content:
                        link_start = search_content.find("link:") + 6
                        link_end = search_content.find(",", link_start) if "," in search_content[link_start:] else len(search_content)
                        url = search_content[link_start:link_end].strip()
                        source_details["webpage_url"] = url
            
            if not documents:
                state["response"] = {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "source": "none",
                    "confidence": 0.0
                }
                return state
            
            chat_history = []
            for message in memory:
                if message["role"] == "human":
                    chat_history.append(HumanMessage(content=message["content"]))
                elif message["role"] == "ai":
                    chat_history.append(AIMessage(content=message["content"]))
            
            source_info = source_type
            if source_details:
                if "video_url" in source_details and source_type.startswith("vector_db"):
                    source_info = f"YouTube Video: {source_details['video_url']}"
                elif "webpage_url" in source_details and "web_search" in source_type:
                    source_info = f"Web Search: {source_details['webpage_url']}"
            
            chain = ANSWER_GENERATION_PROMPT | self.json_llm | json_parser
            
            result = await chain.ainvoke({
                "query": query,
                "context": "\n\n".join([doc.page_content for doc in documents]),
                "chat_history": chat_history if chat_history else "No previous conversation.",
                "source_info": source_info
            })
            
            state["response"] = result
            return state
        
        workflow = StateGraph(State)
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("analyze_context", analyze_context)
        workflow.add_node("web_search", perform_web_search)
        workflow.add_node("generate_answer", generate_answer)
        
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "analyze_context")
        workflow.add_edge("analyze_context", "web_search")
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    async def process_query(self, query: str):
        self.memory.append({"role": "human", "content": query})
        
        state = {
            "query": query,
            "retrieval_context": None,
            "web_search_results": None,
            "memory": self.memory.copy(),
            "response": None,
            "needs_web_search": False
        }
        
        try:
            result = await self.graph.ainvoke(state)
            if result["response"]:
                response_content = json.dumps(result["response"])
                self.memory.append({"role": "ai", "content": response_content})
                return result["response"]
            else:
                fallback_response = {
                    "answer": "I encountered an error processing your request.",
                    "source": "error",
                    "confidence": 0.0
                }
                self.memory.append({"role": "ai", "content": json.dumps(fallback_response)})
                return fallback_response
        except Exception as e:
            fallback_response = {
                "answer": f"I encountered an error processing your request: {str(e)}",
                "source": "error",
                "confidence": 0.0
            }
            self.memory.append({"role": "ai", "content": json.dumps(fallback_response)})
            return fallback_response