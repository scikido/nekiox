import streamlit as st 
import os
import base64
from qna import * 
from pathlib import Path
from csvfile import *
from llamaparser import *


def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size


def main():
    uploaded_file = st.file_uploader("Upload PDF here", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = ""+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
        
        # save_folder = 'docs/'
        # save_path = Path(save_folder, uploaded_file.name)
        # with open(save_path, mode='wb') as w:
        #     w.write(uploaded_file.getvalue())
        
        text  = parse_using_llama(uploaded_file.name)
        print("text extracted!")
        vectorstore, llm = processing_embedding(text)
        print("Vectorstore and LLM created!")
        st.write(text)
        

        with st.sidebar:
            messages = st.container(height=300)
            if prompt := st.chat_input("Say something"):
                messages.chat_message("user").write(prompt)
                answer = answer_question(vectorstore, llm, prompt)
                messages.chat_message("assistant").write(f"Echo: {answer}")

            pdf_to_csv(uploaded_file.name)
            with open('output.csv') as f:
                st.download_button('Download CSV', f)


if __name__ == "__main__":
    main()