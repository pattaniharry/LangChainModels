import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

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




## note : - in retriver we always give input a query and output is always a list of documents 
## step 3    augmentation 
# 

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0
)   
 
prompt = PromptTemplate (
    template = """You are a helpful assitant.
                 Answer ONLY from the provided transcript context.
                 If the answer is not explicitly stated, respond exactly: "I don't know." 
                 
                 {context}
                 Question:{question}

                 """,
                 input_variables = ['context', 'question']
)


question = "is the topic of Blender discusssed in the video?"
retrived_docs = retriever.invoke(question)


context_text = "\n\n".join(doc.page_content for doc in retrived_docs)

final_prompt = prompt.invoke({"context":context_text,"question":question})

print(final_prompt)
