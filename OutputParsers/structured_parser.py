from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

schema = [
    ResponseSchema(name="Que", description="The question"),
    ResponseSchema(name="Options", description="List of options"),
    ResponseSchema(name="ans", description="The correct answer")
]

parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
      template="""
Generate 3 multiple choice question on the topic: {topic}

Include:
- "Que": the question
- "Options": a list of exactly 4 answer choices (e.g., ["A", "B", "C", "D"])
- "ans": the correct answer string (should match one of the 4 options)

{format_instructions}
""",
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain= prompt | model |parser
res=chain.invoke({'topic':"laws of motion"})

print(res)