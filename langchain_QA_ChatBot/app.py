# Q&A Chatbot

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

import streamlit as st


# Function to load Gemini Model and get response

def get_gemini_response(question):
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    response=llm.invoke(question)
    return response

#Initialize our streamlit app

st.set_page_config(page_title="Q&A Demo")

st.header("Langchain Application")

user_input = st.text_input("Ask a question:", key="input")
submit = st.button("Ask")

# If ask button is clicked

if submit:
    if user_input.strip() == "":
        st.error("Please enter a question!")
    else:
        response = get_gemini_response(user_input)
        st.subheader("Response:")
        st.write(response.content)