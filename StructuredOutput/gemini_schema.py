from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import re

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

### Use json output parser(see OutputParser\json_parser.py file code)
schema = """{
    "type": "object",
    "properties": {
        "Que": {"type": "string", "description": "The question"},
        "Options": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of options"
        },
        "ans": {"type": "string", "description": "The correct answer"}
    },
    "required": ["Que", "Options", "ans"]
}"""

prompt = "generate 2 multiple choice questions on Newton's laws of motion. For each question, provide the question, exactly 4 options, and the correct answer. Return the result as a JSON array of objects, where each object conforms to the following schema:\n\n"
prompt += schema

res = model.invoke(prompt)
match = re.search(r"```json\s*(.*?)\s*```", res.content, re.DOTALL)


# print(type(res.content))
if match:
    json_string = match.group(1).strip()
    
    # Step 2: Parse the extracted JSON string into a Python object
    try:
        json_object = json.loads(json_string)
        print(json_object)
        # print(f"Type of the object: {type(json_object)}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"The string that caused the error was: '{json_string}'")

