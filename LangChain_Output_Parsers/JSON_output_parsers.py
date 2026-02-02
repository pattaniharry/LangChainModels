from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate ,PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

parser = JsonOutputParser()


template = PromptTemplate (
    template = "give me the name , age and city of 2 fictional person \n {format_instructions}",
    input_variables = [] ,
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

chain = template | llm | parser

result = chain.invoke({})

print (result)