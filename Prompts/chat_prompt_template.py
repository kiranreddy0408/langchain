from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI

chat_temp=ChatPromptTemplate([
    ('system',"your are {domain} expert"),
    ('human',"Explain me this {topic}")
])

model=GoogleGenerativeAI(model="gemini-2.0-flash")
# method 1
# res=model.invoke(chat_temp.invoke({'domain':'biology','topic':'eyes'}))

# method 2 using chain
chain=chat_temp | model

res=chain.invoke({'domain':'biology','topic':'eyes'})

print(res)