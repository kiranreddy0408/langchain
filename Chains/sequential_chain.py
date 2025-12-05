from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
# from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

model1= ChatGoogleGenerativeAI(model="gemini-1.5-flash")
model2= ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt1=PromptTemplate(
    template="Explain breifly about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="summarize the text in 5 points.\n {text}",
    input_variables=['text']
)



parser=StrOutputParser()

chain= prompt1 | model1 | parser | model2 | parser


res= chain.invoke({'topic':"langchain"})

print(res)