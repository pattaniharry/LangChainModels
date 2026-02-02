from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate ,PromptTemplate
from dotenv import load_dotenv

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


prompt1 = template1.invoke({"topic": "Black Hole"})

prompt2 = template2.invoke({"text": prompt1})
response = llm.invoke(prompt2)

print(response.content)

