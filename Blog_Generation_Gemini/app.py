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

# ------------------------ BLOG GENERATION LOGIC ------------------------

def generate_blog(topic, words, style):
    """
    Generates a blog using Google's Gemini model through LangChain.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # updated stable model
        temperature=0.4,
        # api_key = api_key
    )

    template = """
    Write a professional blog post for this audience: {style}
    Topic: {topic}
    Blog Length: {words} words

    Make the blog engaging, structured, clear, and helpful.
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


# ------------------------ STREAMLIT UI ------------------------

st.set_page_config(page_title="Gemini Blog Generator", page_icon="📝")

st.title("📝 Gemini Blog Generator")
st.write("Generate high-quality, structured blog posts using Google's Gemini AI.")

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

        st.download_button(
            label="Download Blog as TXT",
            data=blog,
            file_name=f"{topic.replace(' ', '_')}.txt",
            mime="text/plain"
        )
