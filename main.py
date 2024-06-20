import streamlit as st
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
import os
import pandas as pd
from matplotlib import pyplot as plt
from summarize import evaluate, summarize


load_dotenv()

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="Summarize Test",
    page_icon="📊",
)

st.write("# Summarizer")

st.sidebar.write("## Setup")

# Step 1 - Get OpenAI API key

# Intialize llm
langchain_llm = None


openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY")

st.sidebar.write("## LLM Provider")
st.sidebar.write("### Choose a Provider")

providers = [
    {"label": "Select a Provider"},
    {"label": "Gemini"},
    {"label": "OpenAI"},
]

selected_provider_label = st.sidebar.selectbox(
    'Choose a provider',
    options=[provider["label"] for provider in providers],
    index=0
)

selected_provider = providers[[
        provider["label"] for provider in providers].index(selected_provider_label)]["label"]


if selected_provider == "Gemini":
    if not gemini_key:
        gemini_key = st.sidebar.text_input("Enter OpenAI API key:")
        if gemini_key:
            display_key = gemini_key[:1] + "*" * (len(gemini_key) - 5) + gemini_key[-2:]
            st.sidebar.write(f"Current key: {display_key}")
            langchain_llm = ChatGoogleGenerativeAI(model="gemini-pro")
        else:
            st.sidebar.write("Please enter Gemini API key.")
    else:
        display_key = gemini_key[:1] + "*" * (len(gemini_key) - 5) + gemini_key[-2:]
        langchain_llm = ChatGoogleGenerativeAI(model="gemini-pro")
    st.sidebar.write(f"Gemini API key loaded from environment variable: {display_key}")

if selected_provider == "OpenAI":
    if not openai_key:
        openai_key = st.sidebar.text_input("Enter OpenAI API key:")
        if openai_key:
            display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
            st.sidebar.write(f"Current key: {display_key}")
            langchain_llm = ChatOpenAI(model="gpt-3.5-turbo")
        else:
            st.sidebar.write("Please enter OpenAI API key.")
    else:
        display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
        langchain_llm = ChatOpenAI(model="gpt-3.5-turbo")
    st.sidebar.write(f"OpenAI API key loaded from environment variable: {display_key}")

st.markdown(
    """
    Quick summarizer using langchain

   ----
""")
st.sidebar.write("### Document")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])


if not uploaded_file:
    st.info("To continue, upload a document for summarization.")

get_score = False

if uploaded_file is not None:
    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    if uploaded_file:
        pdf = PyMuPDFLoader(uploaded_file.name)  
    answer = summarize(langchain_llm, pdf)
    st.write(answer)
    get_score = st.button("Evaluate")

if get_score:
    score = evaluate(langchain_llm, pdf, answer)
    st.write(score)