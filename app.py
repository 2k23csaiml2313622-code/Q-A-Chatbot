import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# LangSmith Tracking 
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With GROQ"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, llm_model, temperature, max_tokens):
    llm = ChatGroq(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    answer = chain.invoke({"question": question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot With Groq")

##slidebar
st.sidebar.title("Settings")
## drop down to select various llm models
llm=st.sidebar.selectbox("Select a Groq model",["llama-3.1-8b-instant","llama-3.1-70b-versatile","gemma2-9b-it"])
temperature=st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens= st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and write your question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide your query")
