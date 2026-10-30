from langchain_community.document_loaders import TextLoader


loader  = TextLoader('fav_shows.txt' , encoding= 'utf-8')

docs = loader.load()

print(type(docs))
print(len(docs))
print('\n \n \n ________________THIS IS THE page_content  __________________________________________________________________')
print(docs[0].page_content)
print('\n \n \n ________________THIS IS THE metadata __________________________________________________________________')
print(docs[0].metadata)

