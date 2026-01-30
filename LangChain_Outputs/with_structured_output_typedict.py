from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = ChatGroq(
    model = "llama-3.1-8b-instant"

)

#schema 

class Review(TypedDict):

    summary : str
    sentiment : str


structured_review = llm.with_structured_output(Review)

result = structured_review.invoke("""I'm thoroughly impressed with the new X500 wireless earbuds - their crystal-clear sound and long-lasting battery life make them a top choice for anyone looking to upgrade their audio experience. The sleek design is also a bonus, making them a stylish accessory that complements any outfit. Overall, I'd highly recommend the X500 earbuds to anyone seeking a reliable and high-quality listening solution.""")
print(result)
