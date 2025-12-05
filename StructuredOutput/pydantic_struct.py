from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class TestGeneration(BaseModel):
    Que: str = Field(..., description="The question to be asked in the quiz.")
    Options: List[str] = Field(..., min_items=4, max_items=4, description="A list of exactly 4 options for the question.")
    ans: str = Field(..., description="The correct answer to the question, which must match one of the options.")

structured_model=model.with_structured_output(TestGeneration)

res = structured_model.invoke("generate 5 questions quiz on laws of motion, each with 4 options and answers")

# Assuming the response contains multiple questions, parse and print them
if isinstance(res, list):
    for idx, question in enumerate(res, start=1):
        print(f"Question {idx}: {question.model_dump_json()}")
else:
    print(res.model_dump_json())