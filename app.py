import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#tracking langsmith
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

#prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assisant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

#streamlit framework
st.title("Langchain Demo with GEMMA 2B")
input_text=st.text_input("What question you have in mind?")

#ollama
llm= Ollama(model="gemma:2b")
output_parser=StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
     response = chain.invoke({"question": input_text})
     st.write(response)