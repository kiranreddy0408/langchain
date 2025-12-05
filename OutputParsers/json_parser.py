from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import json
  
  
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

### It lacks the structured schema defining capability instead 
# use "StructuredOutputParser"
parser=JsonOutputParser()

template=PromptTemplate(
    template="generate 5 multiple choice questions on {topic}. For each question, provide the question, exactly 4 options, and the correct answer.\\n{format_instructions}",
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain=template | model | parser

res=chain.invoke({'topic':"laws of motion"})

print(json.dumps(res))