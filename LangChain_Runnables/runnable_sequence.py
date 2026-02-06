from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template ="Write a joke about {topic}",
    input_variables = ['topic'],
)

llm = ChatGroq(
    model = "llama-3.1-8b-instant"
)

parser = StrOutputParser()

chain = RunnableSequence(prompt , llm , parser)

result = chain.invoke({'topic': 'Groq'})

print(result)