from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class Notes(BaseModel):
    Intro:str=Field(description="Introduction of the topic")
    Explaination:str=Field(description="Detail explaination of the topic")
    no_of_days:int=Field(description="No of days to cover")
    Conclusion:str=Field(description="Conclusion of the topic")

parser=PydanticOutputParser(pydantic_object=Notes)

prompt=PromptTemplate(
    template="generate notes for the {topic} in the following format\\n.{format_instructions}",
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions}
)

chain= prompt | model | parser
res=chain.invoke({'topic':"langchain"})

print(res.model_dump_json())

print(type(res))
