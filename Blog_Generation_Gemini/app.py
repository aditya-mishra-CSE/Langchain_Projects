import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API key
load_dotenv()
# import os

# Load Gemini API Key from HuggingFace Secrets
# api_key = os.getenv("GOOGLE_API_KEY")

# if not api_key:
#     st.error("❌ GOOGLE_API_KEY is missing! Please set it in HuggingFace Space Secrets.")
#     st.stop()

# Function to generate blog using Gemini
def generate_blog(topic, words, style):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        # api_key=api_key
    )

    template = """
    Write a professional blog post for the audience type: {style}
    Topic: {topic}
    Blog Length: {words} words
    
    Make the blog engaging, easy to read, structured, and helpful.
    """

    prompt = PromptTemplate(
        input_variables=["topic", "words", "style"],
        template=template
    )

    final_prompt = prompt.format(
        topic=topic,
        words=words,
        style=style
    )

    response = llm.invoke(final_prompt)
    return response.content


# ---------------------- STREAMLIT UI ----------------------

st.set_page_config(page_title="Gemini Blog Generator", page_icon="📝")

st.title("📝 Gemini Blog Generator")
st.write("Generate high-quality blogs instantly using Google's Gemini AI.")

topic = st.text_input("Enter Blog Topic")

col1, col2 = st.columns(2)

with col1:
    words = st.text_input("Number of Words", value="200")

with col2:
    style = st.selectbox(
        "Blog Audience",
        ["Common People", "Researchers", "Data Scientist", "Students"]
    )

if st.button("Generate Blog"):
    if not topic.strip():
        st.warning("⚠ Please enter a topic!")
    else:
        with st.spinner("Generating blog... ⏳"):
            blog = generate_blog(topic, words, style)
        st.success("✨ Blog Generated Successfully!")
        st.write(blog)

        # Option to download the blog
        st.download_button(
            label="Download Blog as TXT",
            data=blog,
            file_name=f"{topic.replace(' ', '_')}.txt",
            mime="text/plain"
        )
