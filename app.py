import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# =======================
# GROQ API KEY HANDLING
# =======================

GROQ_API_KEY = None

# 1️⃣ Try Streamlit Secrets (Cloud)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# 2️⃣ Fallback to local environment (.env or system env)
if GROQ_API_KEY is None:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 3️⃣ Stop safely if missing
if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not found. Add it to Streamlit Secrets to continue.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# =======================
# PROMPT TEMPLATE
# =======================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)


# =======================
# RESPONSE GENERATOR
# =======================

def generate_response(question, llm_model, temperature, max_tokens):
    llm = ChatGroq(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    return chain.invoke({"question": question})


# =======================
# STREAMLIT UI
# =======================

st.title("Enhanced Q&A Chatbot With Groq")

st.sidebar.title("Settings")

llm = st.sidebar.selectbox(
    "Select a Groq model",
    ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "gemma2-9b-it"]
)

temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.7
)

max_tokens = st.sidebar.slider(
    "Max Tokens", min_value=50, max_value=300, value=150
)

st.write("Go ahead and write your question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide your query")
