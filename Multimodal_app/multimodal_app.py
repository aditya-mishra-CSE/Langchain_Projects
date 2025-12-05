import streamlit as st
import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = GEMINI_API_KEY)

st.title("AI Image Generator")
user_prompt = st.text_input("What do you want to generate image for?")

if st.button("Generate Image"):
    if not user_prompt:
        st.warning("Please enter the prompt")
    else:
        try:
            with st.spinner("Generating Image..."):
                response = client.models.generate_content(
                    model = "gemini-2.0-flash-exp-image-generation",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=['Text', 'Image']
                    )
                )
                st.subheader("Generated Image")
                for part in response.candidates[0].contents.parts:
                    if part.text is not None:
                        st.write(part.text)
                    elif part.inline_data is not None:
                        image = Image.open(BytesIO((part.inline_data.data)))
                        image.save('gemini-native-image.png')
                        image.show()
        except Exception as e:
            st.error(f"Error generating image: {e}")


st.title("AI Image Caption Generator")

uploaded_image = st.file_uploader("Upload an image for caption generation")

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image")

    if st.button("Generate Caption"):
        try:
            with st.spinner("Generating caption..."):
                response = client.models.generate_content(
                    model = "gemini-2.0-flash",
                    contents= ["What is this image?", image]
                )
                st.subheader("Generated Caption: ")
                st.write(response.text)

        except Exception as e:
            st.error("Error generating caption")



st.title("AI Youtube Video Summarizer")
youtube_url = st.text_input("Enter youtube video url")

if st.button("Summarize Video"):
    if not youtube_url:
        st.warning("No Youtube Url Present!")
    else:
        try:
            with st.spinner("Generating summary..."):
                response = client.models.generate_content(
                    model='models/gemini-2.5-flash',
                    contents=types.Content(
                        parts=[
                            types.Part(
                                file_data=types.FileData(file_uri=youtube_url)
                            ),
                            types.Part(text='Please summarize the video in 3 sentences.')
                        ]
                    )
                )
            st.subheader("Video summary")
            st.write(response.text)
        except Exception as e:
            st.error("Error generatoing summary")  
