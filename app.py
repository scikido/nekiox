import streamlit as st 
import os
import base64
from qna import * 
from pathlib import Path
from llamaparser import *
from rl import *


def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size


def main():
    st.set_page_config(layout="wide")
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
        vectorstore, llm, embeddings_hf = processing_embedding(text)
        print("Vectorstore and LLM created!")
        response_1, response_2 = answer_question(vectorstore, llm, "Give all the data in json format")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(response_1)

        with col2:
            st.write(response_2)

        with col3:
            genre = st.radio(
                "Which response is better (1/2)?",
                ["1", "2"], index=None)
            
            with open("response1.json", 'w') as f:
                f.write(str(response_1))

            with open("response2.json", 'w') as f:
                f.write(str(response_2))

            # st.download_button(
            # label="Download Response 1",
            # data= response1.json,
            # file_name='response1.json')

            # st.download_button(
            # label="Download Response 2",
            # data=response2.json,
            # file_name='response2.json')
            if genre:
                reinforcement(text, genre)
        
if __name__ == "__main__":
    main()