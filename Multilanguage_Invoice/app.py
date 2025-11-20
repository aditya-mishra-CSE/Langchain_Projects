# from dotenv import load_dotenv

# load_dotenv() # Load all the environment variables from .env

# import streamlit as st 
# import os
# from PIL import Image
# import google.generativeai as genai

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ##Function to load Gemini Pro Vision
# model = genai.GenerativeModel('gemini-1.5-flash')

# def get_gemini_response(input, image, prompt):
#     response = model.generate_content([input, image[0], prompt])
#     return response.text

# def input_image_details(uploaded_file):
#     if uploaded_file is not None:
#         #Read the file into bytes
#         bytes_data = uploaded_file.getvalue()

#         image_parts = [
#             {
#                 "mime_type": uploaded_file.type, #Get the mime type of the uoploaded file
#                 "data": bytes_data
#             }
#         ]
#         return image_parts
#     else:
#         raise FileNotFoundError("No file uploaded")


# ##initilize our streamlit app

# st.set_page_config(page_title="MultiLanguage Invoive Extractor")

# st.header("MultiLanguage Invoice Extractor")

# input = st.text_input("Input Prompt: ", key="input")
# uploaded_file = st.file_uploader("Choose an image: ", type=["jpg", "jpeg", "png"])

# image=""
# if uploaded_file is not None:
#     image=Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)

# submit = st.button("Tell me about the Invoice")

# input_prompt= """
# You are an expert in understanding invoices. We will upload a image as invoice
# and you will have to answer any questions based on the uploaded invoice image
# """

# #If submit button is clicked
# if submit:
#     image_data=input_image_details(uploaded_file)
#     response=get_gemini_response(input_prompt, image_data, input)
#     st.subheader("The Response is")
#     st.write(response)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(system_prompt, uploaded_file, user_question):
    image_bytes = uploaded_file.getvalue()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            system_prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=uploaded_file.type
            ),
            user_question
        ]
    )
    return response.text


def input_image_details(uploaded_file):
    if uploaded_file is not None:
        return uploaded_file
    else:
        raise FileNotFoundError("No file uploaded")


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="MultiLanguage Invoice Extractor")
st.header("MultiLanguage Invoice Extractor")

user_input = st.text_input("Ask about the invoice:")
uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Invoice", use_column_width=True)

submit = st.button("Tell me about the Invoice")

system_prompt = """
You are an expert in understanding invoices.
Analyze the uploaded invoice image and answer the user's question accurately.
"""

if submit:
    response = get_gemini_response(system_prompt, uploaded_file, user_input)
    st.subheader("Response")
    st.write(response)

