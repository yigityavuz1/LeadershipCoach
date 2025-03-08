import os
import uuid
import asyncio
import streamlit as st
from pytubefix import Playlist
import nest_asyncio
from dotenv import load_dotenv

from src.utils.get_transcriptions import YouTubeDownloader, AudioTranscriber, TranscriptionProcessor
from src.utils.create_vectordb import VectorDatabase
from src.rag_graph import RAGSystem
from src.tools import setup_text_to_speech

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set up the event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

class YoutubeRAGApp:
    """Main application class that orchestrates all components."""
    
    def __init__(self):
        load_dotenv()
        
        self.playlist_url = os.environ.get("YOUTUBE_PLAYLIST_URL")
        self.hf_token = os.environ.get("HF_SERVERLESS_INFERENCE_TOKEN")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        self.downloader = YouTubeDownloader()
        self.transcriber = AudioTranscriber()
        self.processor = TranscriptionProcessor()
        self.vector_db = VectorDatabase(self.hf_token)
        self.rag_system = RAGSystem(self.openai_api_key)
        
        self.is_ready = False
        self.retriever = None

        # Initialize text-to-speech tool
        self.tts_tool = setup_text_to_speech()
    
    async def setup(self):
        if self.is_ready:
            return
        try:
            with st.status("Connecting to Weaviate database...", expanded=False) as status:
                await self.vector_db.connect()
                status.update(label="Connected to database", state="complete")
            
            collection_exists = await self.vector_db.collection_exists()
            
            if not collection_exists:
                await self._process_all()
            else:
                with st.status("Collection exists. Checking if updates are needed...", expanded=False) as status:
                    await self._check_and_update()
                    status.update(label="Updates completed", state="complete")
            
            with st.status("Setting up RAG system...", expanded=False) as status:
                self.retriever = await self.vector_db.create_retriever()
                self.rag_system.graph = self.rag_system.create_graph(self.retriever)
                status.update(label="RAG system ready", state="complete")
            
            self.is_ready = True
            st.success("System is ready! You can now ask questions about the YouTube content.")
            
        except Exception as e:
            st.error(f"Error during setup: {str(e)}")
            raise e
    
    async def _process_all(self):
        with st.status("Creating Weaviate collection...", expanded=False) as status:
            success = await self.vector_db.create_collection()
            if success:
                status.update(label="Collection created or already exists", state="complete")
            else:
                status.update(label="Failed to create collection", state="error")
                return
        
        with st.status("Downloading videos from YouTube...", expanded=False) as status:
            downloaded_files = self.downloader.download_playlist(self.playlist_url)
            status.update(label=f"Downloaded {len(downloaded_files)} videos", state="complete")
        
        with st.status("Transcribing videos...", expanded=False) as status:
            for i, (video_url, audio_file) in enumerate(downloaded_files):
                video_id = self.downloader.get_video_id(video_url)
                status.update(label=f"Transcribing video {i+1}/{len(downloaded_files)}")
                transcript = self.transcriber.transcribe_audio(audio_file)
                self.transcriber.save_transcript(transcript, video_id)
            status.update(label="Transcription complete", state="complete")
        
        with st.status("Processing transcripts...", expanded=False) as status:
            documents = self.processor.process_transcripts()
            status.update(label=f"Processed {len(documents)} document chunks", state="complete")
        
        with st.status("Creating embeddings and uploading to database...", expanded=False) as status:
            vectors = await self.vector_db.embed_documents(documents)
            status.update(label="Embeddings created, uploading to database...")
            count = await self.vector_db.upload_documents(documents, vectors)
            status.update(label=f"Uploaded {count} documents to database", state="complete")
    
    async def _check_and_update(self):
        playlist = Playlist(self.playlist_url)
        new_videos = []
        
        for video_url in playlist.video_urls:
            video_id = self.downloader.get_video_id(video_url)
            if not self.transcriber.transcript_exists(video_id):
                new_videos.append(video_url)
        
        if new_videos:
            with st.status(f"Found {len(new_videos)} new videos to process", expanded=False) as status:
                for i, video_url in enumerate(new_videos):
                    video_id = self.downloader.get_video_id(video_url)
                    status.update(label=f"Processing new video {i+1}/{len(new_videos)}")
                    
                    if not self.downloader.audio_exists(video_url):
                        audio_file = self.downloader.download_audio(video_url)
                    else:
                        audio_file = os.path.join(self.downloader.download_path, f"{video_id}.mp4")
                    
                    transcript = self.transcriber.transcribe_audio(audio_file)
                    self.transcriber.save_transcript(transcript, video_id)
                
                status.update(label=f"Processed {len(new_videos)} new videos", state="complete")
            
            with st.status("Processing and uploading new transcripts...", expanded=False) as status:
                documents = self.processor.process_transcripts()
                if documents:
                    vectors = await self.vector_db.embed_documents(documents)
                    count = await self.vector_db.upload_documents(documents, vectors)
                    status.update(label=f"Uploaded {count} new document chunks", state="complete")
                else:
                    status.update(label="No new document chunks to upload", state="complete")
        else:
            st.info("All videos are already processed. No updates needed.")
    
    async def process_query(self, query):
        if not self.is_ready:
            return {"answer": "System is still initializing. Please wait.", "source": "error", "confidence": 0.0}
        
        await self.vector_db.ensure_connected()
        return await self.rag_system.process_query(query)


def get_app_instance():
    """Create or get the app instance."""
    # Uncomment the @st.cache_resource decorator to cache the app instance
    # @st.cache_resource
    return YoutubeRAGApp()

async def main():
    st.set_page_config(
        page_title="LeadershipCoach",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("🎥 LeadershipCoach")
    st.write("Ask questions about the content from the leadership videos.")
    
    app = get_app_instance()
    await app.setup()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display existing chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.write(message["content"]["answer"])
                source = message["content"]["source"]
                confidence = message["content"]["confidence"]
                
                with st.expander("Source details"):
                    st.write(f"**Source:** {source}")
                    st.write(f"**Confidence:** {confidence:.2f}")

                # Display a "Play" button for the final text response
                final_text = message["content"]["answer"]
                if final_text.strip():
                    # Use a unique key with message index AND position identifier
                    play_key = f"play_audio_history_{i}"
                    if st.button("Play", key=play_key):
                        # Convert final_text to speech
                        with st.spinner("Converting text to speech..."):
                            audio_file = app.tts_tool.run(final_text)
                            with open(audio_file, "rb") as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format="audio/mp3")

            else:
                st.write(message["content"])
    
    # Input box for new questions
    if prompt := st.chat_input("Ask a question about the YouTube content:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = await app.process_query(prompt)
                except Exception as e:
                    response = {
                        "answer": f"I encountered an error processing your request: {str(e)}",
                        "source": "error",
                        "confidence": 0.0
                    }
            st.write(response["answer"])
            
            source = response["source"]
            confidence = response["confidence"]
            with st.expander("Source details"):
                st.write(f"**Source:** {source}")
                st.write(f"**Confidence:** {confidence:.2f}")

            # TTS for newly generated message
            final_text = response["answer"]
            if final_text.strip():
                # Use a completely different key pattern for the new message
                play_key = f"play_audio_new_{len(st.session_state.messages)}"
                if st.button("Play", key=play_key):
                    audio_file = app.tts_tool.run(final_text)
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")
        
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    loop.run_until_complete(main())