import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

model = SentenceTransformer("all-MiniLM-L6-v2")

## step 1 indexing (document ingestion )

video_id = "Gfr50f6ZBvo" 

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id,languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")


#step1 B indexing (text splitting)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])
 
# print(chunks[0])

#ste1 c & d indexing(embedding generation and storing in vector store )


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embeddings)





### STEP2 RETRIEVAL

retriever = vector_store.as_retriever(search_type = 'similarity'  , search_kwargs={"k" : 4})

result = retriever.invoke("what is deepmind")

print(result)

