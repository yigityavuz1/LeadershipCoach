from typing import List
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.docstore.document import Document
from src.utils.elevenlabs_wrapper import ElevenLabsText2SpeechTool, ElevenLabsModel
import os

def setup_web_search():
    """Setup DuckDuckGo web search tool."""
    search_tool = DuckDuckGoSearchResults(num_results=3)
    
    def search(query: str) -> List[Document]:
        search_results = search_tool.invoke(query)
        documents = []
        
        for result in search_results.split('\n'):
            if result.strip():
                documents.append(Document(page_content=result))
        
        return documents
    
    return search

def setup_text_to_speech():
    """Setup ElevenLabs text-to-speech tool."""
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY") or "YOUR_ELEVENLABS_API_KEY"
    tts_tool = ElevenLabsText2SpeechTool(
        elevenlabs_api_key=elevenlabs_api_key,
        model=ElevenLabsModel.MULTI_LINGUAL,  # or whichever model you prefer
        voice="JBFqnCBsd6RMkjVDRZzb"          # your own voice ID
    )
    return tts_tool