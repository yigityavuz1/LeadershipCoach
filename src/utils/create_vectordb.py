import asyncio
import streamlit as st
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.docstore.document import Document
import os
class VectorDatabase:
    """Class for managing Weaviate vector database operations."""
    
    def __init__(self, hf_token):
        """Initialize the vector database with API token."""
        self.hf_token = hf_token
        self.client = None
        self.collection_name = "transcript_index"
        self.embeddings = None
        self._connection_lock = asyncio.Lock()
        self._is_connected = False
        
        # Get Weaviate connection details from environment variables
        self.weaviate_host = os.environ.get("WEAVIATE_HOST", "weaviate")
        self.weaviate_port = os.environ.get("WEAVIATE_PORT", "8080")
                
    async def connect(self):
        """Connect to Weaviate database."""
        async with self._connection_lock:
            if self._is_connected and self.client:
                return self.client
                
            self.client = weaviate.use_async_with_local(
                headers={"X-HuggingFace-Api-Key": self.hf_token},
                host=self.weaviate_host,
            )
            await self.client.connect()
            await self.client.is_ready()
            
            # Initialize embeddings
            self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                model_name="BAAI/bge-m3",
                api_key=self.hf_token,
            )
            
            self._is_connected = True
            return self.client
    
    async def ensure_connected(self):
        """Ensure the client is connected before operations."""
        if not self._is_connected or not self.client:
            await self.connect()
    
    async def collection_exists(self):
        """Check if the collection exists."""
        await self.ensure_connected()
        try:
            collections = await self.client.collections.list_all()
            return bool(collections.get(self.collection_name.capitalize()))
        except Exception as e:
            st.warning(f"Error checking collection existence: {str(e)}")
            return False
    
    async def create_collection(self):
        """Create Weaviate collection if it doesn't exist."""
        await self.ensure_connected()
        
        if await self.collection_exists():
            return True
        
        # Define vectorizer configuration
        vectorizer_config = [
            Configure.NamedVectors.text2vec_huggingface(
                name="page_content_vectorizer",
                source_properties=["page_content"],
                model="BAAI/bge-m3",
            )
        ]
        
        try:
            await self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=vectorizer_config,
                properties=[
                    Property(
                        name="page_content",
                        data_type=DataType.TEXT,
                        description="Transcript chunk text"
                    ),
                    Property(
                        name="page_content_vector",
                        data_type=DataType.NUMBER_ARRAY,
                        description="Embedding for the transcript chunk text"
                    ),
                    Property(
                        name="video_url",
                        data_type=DataType.TEXT,
                        description="YouTube video URL"
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.NUMBER,
                        description="Index of the chunk"
                    ),
                    Property(
                        name="chunk_length",
                        data_type=DataType.NUMBER,
                        description="Number of characters in this chunk"
                    ),
                ],
            )
            return True
        except Exception as e:
            if "ResourceNameAlreadyInUse" in str(e):
                return True  # Collection already exists
            else:
                raise e
    
    async def embed_documents(self, documents):
        """Create embeddings for documents."""
        await self.ensure_connected()
        
        texts = [doc.page_content or "" for doc in documents]
        vectors = []
        batch_size = 10
        
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
            try:
                batch_vectors = await self.embeddings.aembed_documents(batch_texts)
                vectors.extend(batch_vectors)
            except Exception as e:
                await asyncio.sleep(5)  # Wait before retry
                batch_vectors = await self.embeddings.aembed_documents(batch_texts)
                vectors.extend(batch_vectors)
        
        return vectors
    
    async def upload_documents(self, documents, vectors):
        """Upload documents and vectors to Weaviate."""
        await self.ensure_connected()
        
        doc_dicts = []
        
        for i, doc in enumerate(documents):
            meta = doc.metadata or {}
            data_obj = {
                "page_content": doc.page_content or "",
                "page_content_vector": vectors[i],
                "video_url": str(meta.get("video_url", "")),
            }
            
            # Handle chunk_index & chunk_length
            chunk_idx = meta.get("chunk_index")
            chunk_len = meta.get("chunk_length")
            if chunk_idx is not None:
                data_obj["chunk_index"] = int(chunk_idx)
            if chunk_len is not None:
                data_obj["chunk_length"] = int(chunk_len)
            
            doc_dicts.append(data_obj)
        
        # Insert documents in batches
        collection = self.client.collections.get(self.collection_name)
        inserted_count = 0
        batch_size = 5
        
        for start_idx in range(0, len(doc_dicts), batch_size):
            chunk = doc_dicts[start_idx : start_idx + batch_size]
            await collection.data.insert_many(chunk)
            inserted_count += len(chunk)
        
        return inserted_count
    
    async def create_retriever(self):
        """Create a retriever function for the Weaviate collection."""
        await self.ensure_connected()
        collection = self.client.collections.get(self.collection_name)
        
        async def retrieve(query: str, k: int = 3):
            """Retrieve relevant documents from Weaviate."""
            await self.ensure_connected()
            
            query_embedding = await self.embeddings.aembed_query(query)
            
            results = await collection.query.hybrid(
                query=query, vector=query_embedding, limit=k, alpha=0.5,
            )
            
            documents = []
            seen_content = set()  # To deduplicate results
            
            for obj in results.objects:
                content = obj.properties['page_content']
                if content not in seen_content:
                    seen_content.add(content)
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "video_url": obj.properties['video_url'],
                                "chunk_index": obj.properties['chunk_index'],
                            }
                        )
                    )
            
            return documents
        
        return retrieve