from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict , Annotated

load_dotenv()

llm = ChatGroq(
    model = "llama-3.1-8b-instant"

)

#schema 

class Review(TypedDict):

    summary : Annotated[str, "A brief summary of the review content."]
    sentiment : Annotated[str, "The overall sentiment of the review, e.g., positive, negative, neutral."]


structured_review = llm.with_structured_output(Review)

result = structured_review.invoke("""
Directed by Christopher Nolan, 'Inception' is a thought-provoking science fiction action film that delves into the concept of shared dreaming. The movie follows Cobb, a skilled thief who specializes in entering people's dreams and stealing their secrets, played by Leonardo DiCaprio. Cobb is offered a chance to redeem himself by performing an inception - planting an idea in someone's mind instead of stealing one.

**Plot**

The story begins with Cobb and his team, consisting of Arthur (Joseph Gordon-Levitt), Ariadne (Ellen Page), Eames (Tom Hardy), and Saito (Ken Watanabe), who propose a plan to perform an inception on the CEO of a powerful corporation, Robert Fischer (Cillian Murphy). Saito convinces Cobb that he will help him clear his name and return to the United States by successfully performing the inception.

As they dive into the world of dream-sharing, the team encounters various obstacles, including projections, the hotel corridor maze, and the top, a dream within a dream. Cobb's subconscious, represented by his deceased wife Mal (Marion Cotillard), poses a significant threat to the team's success.

The film takes the audience on a thrilling ride through multiple levels of dreams, blurring the lines between reality and fantasy. Cobb's emotional journey, driven by his grief and guilt, is interwoven with the plot, adding depth to the narrative.

**Themes**s

1. **Reality and Perception**: 'Inception' explores the concept of reality and perception, leaving the audience questioning what is real and what is a dream. This theme is expertly woven throughout the film, making it difficult to distinguish between the different levels of reality.
2. **Grief and Guilt**: Cobb's emotional journey serves as the emotional core of the film. His grief and guilt over the death of his wife Mal are palpable, and his desire to return to the United States is driven by his desire to see his children.
3. **Identity and Humanity**: The film raises questions about identity and humanity, particularly in the context of shared dreaming. The team's actions and interactions in the dream world mirror their real-world personalities, highlighting the blurred lines between reality and fantasy.

**Performances**

The cast delivers outstanding performances, with Leonardo DiCaprio being the standout. His portrayal of Cobb's emotional turmoil is raw and captivating, making his character's journey relatable and engaging.

Supporting performances include Joseph Gordon-Levitt as Arthur, who brings a sense of calm and rationality to the team; Ellen Page as Ariadne, who shines as the architect of the dream world; and Tom Hardy as Eames, who brings a sense of humor and charm to the role.

**Visuals and Cinematography**

The film's visuals are breathtaking, with stunning action sequences and mind-bending dreamscapes. Christopher Nolan's direction is masterful, using the camera to create a sense of disorientation and unease. The film's use of practical effects and CGI creates a seamless and immersive experience.

**Score**

The score, composed by Hans Zimmer, is a masterpiece. The use of a ticking clock to symbolize the progression of time within the dream world is particularly effective. The score perfectly complements the film's themes and actions, adding an emotional depth to the narrative.

**Conclusion**

'Inception' is a thought-provoking and visually stunning film that delves into the concept of shared dreaming. With a complex plot, outstanding performances, and breathtaking visuals, the film is a must-see for fans of science fiction and action movies. Christopher Nolan's direction is masterful, and the film's themes of reality, grief, and identity will leave you questioning the nature of reality and humanity.

**Rating:** 9.5/10

**Recommendation:** 'Inception' is a must-see for fans of science fiction, action movies, and philosophical themes. If you enjoy complex narratives and thought-provoking ideas, this film is a must-watch.""")
print(result)

# review = llm.invoke("a detailed review of the moview inception ")
# print(review.content)