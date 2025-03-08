import os
import streamlit as st
import whisper
from pytubefix import Playlist, YouTube
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import torch

# Ensure PyTorch doesn't try to access unset paths
torch.classes.__path__ = []

class YouTubeDownloader:
    """Class for downloading YouTube videos."""
    
    def __init__(self, download_path="audio_files"):
        """Initialize the downloader with a download path."""
        self.download_path = download_path
        if not os.path.exists(download_path):
            os.makedirs(download_path)
    
    def download_playlist(self, playlist_url):
        """Download all videos from a YouTube playlist."""
        playlist = Playlist(playlist_url)
        downloaded_files = []
        
        for video_url in playlist.video_urls:
            try:
                audio_file = self.download_audio(video_url)
                downloaded_files.append((video_url, audio_file))
            except Exception as e:
                st.error(f"Error downloading {video_url}: {e}")
        
        return downloaded_files
    
    def download_audio(self, video_url):
        """Download audio from a single YouTube video."""
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        video_id = self.get_video_id(video_url)
        file_path = os.path.join(self.download_path, f"{video_id}.mp4")
        
        # Check if file already exists
        if os.path.exists(file_path):
            return file_path
            
        # Download the file
        downloaded_file = audio_stream.download(output_path=self.download_path)
        return downloaded_file
    
    def get_video_id(self, video_url):
        """Extract the video ID from a YouTube URL."""
        return YouTube(video_url).video_id
    
    def audio_exists(self, video_url):
        """Check if the audio for a video has already been downloaded."""
        video_id = self.get_video_id(video_url)
        return os.path.exists(os.path.join(self.download_path, f"{video_id}.mp4"))


class AudioTranscriber:
    """Class for transcribing audio files."""
    
    def __init__(self, model_name="base", transcripts_path="transcripts"):
        """Initialize the transcriber with a model and output path."""
        self.model = None  # Load the model lazily
        self.model_name = model_name
        self.transcripts_path = transcripts_path
        
        if not os.path.exists(transcripts_path):
            os.makedirs(transcripts_path)
    
    def load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            self.model = whisper.load_model(self.model_name)
    
    def transcribe_audio(self, audio_file, language="tr"):
        """Transcribe an audio file using Whisper."""
        self.load_model()
        result = self.model.transcribe(audio_file, language=language)
        return result["text"]
    
    def save_transcript(self, transcript, video_id):
        """Save transcript to a file."""
        transcript_filename = os.path.join(self.transcripts_path, f"{video_id}.txt")
        
        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        return transcript_filename
    
    def transcript_exists(self, video_id):
        """Check if a transcript already exists for a video."""
        transcript_filename = os.path.join(self.transcripts_path, f"{video_id}.txt")
        return os.path.exists(transcript_filename)
    
    def get_transcript(self, video_id):
        """Get the transcript for a video if it exists."""
        transcript_filename = os.path.join(self.transcripts_path, f"{video_id}.txt")
        
        if os.path.exists(transcript_filename):
            with open(transcript_filename, "r", encoding="utf-8") as f:
                return f.read()
        
        return None


class TranscriptionProcessor:
    """Class for processing transcriptions into chunks."""
    
    def __init__(self, chunk_size=4000, chunk_overlap=400):
        """Initialize the processor with chunking parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    
    def process_transcripts(self, transcripts_dir="transcripts"):
        """Process transcript files into chunks with metadata."""
        documents = []
        
        for file_name in os.listdir(transcripts_dir):
            if file_name.endswith(".txt"):
                video_id = os.path.splitext(file_name)[0]
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                file_path = os.path.join(transcripts_dir, file_name)
                
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                
                chunks = self.splitter.split_text(content)
                
                for idx, chunk in enumerate(chunks):
                    metadata = {
                        "video_url": video_url, 
                        "chunk_index": idx, 
                        "chunk_length": len(chunk)
                    } 
                    doc = Document(page_content=chunk, metadata=metadata)
                    documents.append(doc)
        
        return documents