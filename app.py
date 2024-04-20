import streamlit as st 
import os
import base64
from qna import * 
from pathlib import Path
from myparser import *
from rl import *
import random
import string
from llamaparser import *

def generate_random_filename(length=10, extension=''):
    # Generate a random string of letters
    letters = string.ascii_letters
    random_letters = ''.join(random.choice(letters) for _ in range(length))
    # Concatenate the random letters with the extension
    filename = random_letters + extension
    return filename



def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size


def main():
    st.set_page_config(layout="wide")
    st.title("PDF Extractor - NekoX")
    uploaded_file = st.file_uploader("Upload PDF here", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = ""+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
        
        save_folder = 'docs/'
        save_path = Path(save_folder, uploaded_file.name)
        with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())

        random_filename = generate_random_filename(8, '.txt')
        parsing_with_docs(random_filename)
        time.sleep(10)
        # os.mkdir(random_filename)
        text1  = getting_text(r"C:\Users\DELL\Downloads\Logithon - NekoX\new_outputs\\"+random_filename + ".jsonl")
        text2 = parse_using_llama(uploaded_file.name)
        print("text extracted!")
        vectorstore1, llm1, embeddings_hf1 = processing_embedding(text1)
        vectorstore2, llm2, embeddings_hf2 = processing_embedding(text2)
        print("Vectorstore and LLM created!")
        response_1 = answer_question(vectorstore1, llm1, "Give all the data in json format")
        response_2 = answer_question(vectorstore2, llm2, "Give all the data in json format")
        library_name = "parsing_test_lib_5"
        output = parsing_documents_into_library(library_name)
        image_path = r"C:\Users\DELL\llmware_data\accounts\llmware\parsing_test_lib_5"+"\images"
        response_1["image"]= image_path
        response_2["image"]= image_path
        col1, col2, col3 = st.columns(3)

        with col1:
            response_1["query"] = "1"
            st.write(response_1)

        with col2:
            response_2["query"] = "2"
            st.write(response_2)

        with col3:
            genre = st.radio(
                "Which response is better (1/2)?",
                ["1", "2"], index=None)
            
            with open("response1.json", 'w') as f:
                f.write(str(response_1["result"]))

            with open("response2.json", 'w') as f:
                f.write(str(response_2["result"]))

            st.download_button(label="Download Response 1", data=open("response1.json", "rb"), file_name="response1.json")
            st.download_button(label="Download Response 2", data=open("response2.json", "rb"), file_name="response2.json")
            img = os.listdir(r"C:\Users\DELL\llmware_data\accounts\llmware\outputs\images")

            for i in img:
                st.image(r"C:\Users\DELL\llmware_data\accounts\llmware\outputs\images\\"+i)
            
            if genre:
                reinforcement(text1, genre)
        
if __name__ == "__main__":
    main()