import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

model = SentenceTransformer("all-MiniLM-L6-v2")

## step 1 indexing (document ingestion )

video_id = "rtpTvvp3OZA" 

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id,languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
