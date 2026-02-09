from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('Data_Solutions_Associate_Intern.pdf')

docs = loader.load()

#print(len(docs)) # returns  number pages in the pdf 


print(docs[0].page_content) # return the page content of the first page

print(docs[1].metadata) # return the metadata of the second page
