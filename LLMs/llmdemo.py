# -*- coding: utf-8 -*-
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv


load_dotenv()



llm = GoogleGenerativeAI(model="gemini-1.5-flash")

response = llm.invoke("Captial city of india")
print(response)
