from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

model1 = ChatGroq(
    model = "llama-3.1-8b-instant",
)

prompt1 = PromptTemplate(
    template = "generate short and simple notes form the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = "generate a quiz with 5 questions from the following notes \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template = "merge the provided notes and quiz into a single study guide \n Notes: {notes} \n Quiz: {quiz}",
    input_variables=["notes", "quiz"]
)


parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model1 | parser
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'text': "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. It involves the conversion of carbon dioxide and water into glucose and oxygen. This process is essential for the survival of life on Earth as it provides oxygen for respiration and organic compounds for food."})

print (result)

chain.get_graph().print_ascii()
