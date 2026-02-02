from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate ,PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
)




#we are using prompt and chaining first one we will ask for detailed prompt 

template1 = PromptTemplate (
    template  = "provide a detailed explanation of the following topic: {topic}",
    input_variables = ["topic"]
)

#second one we will ask for summary using that detailed prompt 


template2 = PromptTemplate (
    template  = "Provide only 5 line summary from given data: {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()


chain = template1 | llm | parser | template2 | llm | parser

result = chain.invoke({"topic": "Black Hole"})

print (result)