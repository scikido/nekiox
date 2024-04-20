from llama_parse import LlamaParse
import streamlit as st
import time

def parse_using_llama(pdf):
    parser = LlamaParse(
        api_key="llx-D9kZUjUTnRxsPFwtmN50di1cpBdZcVLWoYP9DBegy0JAX7aG",
        result_type="markdown",
        verbose=True,
        language="en",
        num_workers=2,
    )

    documents = parser.load_data(pdf)
    return documents[0].text




# def main():
#     # docs_text = parse_using_llama(r"C:\Users\DELL\Downloads\table extract\aoc_3938_18_03_2020.pdf")
#     # print(docs_text)


# # Invoking the main function
# if __name__ == "__main__":
#     main()