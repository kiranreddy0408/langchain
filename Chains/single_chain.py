from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt=PromptTemplate(
    template="captial city of {country}",
    input_variables=['country']
)
parser=StrOutputParser()

chain= prompt | model | parser

res= chain.invoke({'country':"USA"})

print(res)