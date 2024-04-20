from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_community import embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from google.colab import userdata
from transformers import AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import time
import textwrap
import gradio as gr
from langchain.chains import RetrievalQA
import os
from langchain_groq import ChatGroq

def model_embedding():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}

    embeddings_hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    return embeddings_hf

def processing_embedding(text):
    # loader = PyMuPDFLoader(pdf)
    # text = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    #chunks = text_splitter.split_documents(text)
    pages = text_splitter.split_text(text)
    chunks = text_splitter.create_documents(pages)

    embeddings_hf = model_embedding()

    

    vectorstore = Chroma.from_documents(
        documents = chunks,
        collection_name= "groq_embeds",
        embedding = embeddings_hf,
    )

    retriever = vectorstore.as_retriever()
    load_dotenv()


    groq_api_key = os.environ['GROQ_API_KEY']
    llm = ChatGroq(temperature=0, model_name = "mixtral-8x7b-32768" )
    return vectorstore, llm, embeddings_hf


def answer_question(vectorstore, llm, question):
    rag_template = """Use the following context to create json and do not write anything except json format:
    {context}
    Question: {question}
    """

    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": rag_prompt},
    )

    # response = qa_chain.invoke("Extract all the data from the given text and display it in a proper format.")
    # Get two responses with embeddings

    embeddings_hf = model_embedding()
    response = qa_chain.invoke(question + "accurately")
    # # response_1_embedding = embeddings_hf.embed_query(response_1)

    # # # Retrieve nearest neighbors or embeddings associated with specific documents
    # # nearest_neighbors_1, distances_1 = vectorstore.find_nearest_neighbors(response_1_embedding, k=5)

    # response_2 = qa_chain.invoke(question + "give all relevant data in json format")
    # # response_2_embedding = embeddings_hf.embed_query(response_2)
    # # nearest_neighbors_2, distances_2 = vectorstore.find_nearest_neighbors(response_2_embedding, k=5)

    return response
