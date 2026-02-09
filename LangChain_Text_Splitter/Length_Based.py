from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("AI.txt")
docs = loader.load()     # <-- documents

splitter = CharacterTextSplitter(
    chunk_size=100, # defines the size of chunk 
    chunk_overlap=0,# defines the overlap between chunks 
    separator="^ ^ "
)

result = splitter.split_documents(docs)

print(result)


