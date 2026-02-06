from langchain_community.document_loaders import TextLoader

loader = TextLoader('cricket.txt', encoding='utf8')

docs = loader.load()

print(docs)
print(type(docs))

print(len(docs))

print(docs[0])


