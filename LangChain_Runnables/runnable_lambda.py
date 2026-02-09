from langchain_core.runnables import RunnableLambda

def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)

result = runnable_word_counter.invoke("this is a simple string , is it ?")

# result - RunnableLambda(lambda x : len(x.split())

print(result)

