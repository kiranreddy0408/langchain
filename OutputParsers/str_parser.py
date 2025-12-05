from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

parser=StrOutputParser()

template1=PromptTemplate(
    template="Explain about {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="Summarize this topic in 5 lines.\\n{text}",
    input_variables=['text']
)

chain= template1 | model |parser| template2|model |parser
res=chain.invoke({'topic':"laws of motion"})
print(res)