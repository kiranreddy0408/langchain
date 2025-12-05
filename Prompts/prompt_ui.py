from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
from dotenv import load_dotenv

st.header("Research Tool")
load_dotenv()
template=load_prompt('research_guide_template.json')
paper_input=st.selectbox(label="Select the explanation style", options=["Attention is all you need", "BERT a new transformer", "Langchain: A framework for LLMs"])
style_input=st.selectbox(label="Select the explanation style", options=["Mathametical", "Coding", "Analogy"])
length_input=st.selectbox(label="Select the explanation length", options=["Short", "Medium", "Long"])
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

if st.button(label="Send"):
    chain=template | model
    res=chain.invoke({
        'paper_input':{paper_input},
        'style_input':{style_input},
        'length_input':{length_input}

    })
    st.write(res.content)