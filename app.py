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
import os
from dotenv import load_dotenv
import time
import textwrap
import gradio as gr

loader = PyMuPDFLoader(r"C:\Users\DELL\Downloads\app\IPL-Winners-List-2008-2022.pdf")
text = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(text)

from transformers import AutoModel
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}

embeddings_hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

vectorstore = Chroma.from_documents(
    documents = chunks,
    collection_name= "groq_embeds",
    embedding = embeddings_hf,
)

retriever = vectorstore.as_retriever()
load_dotenv()
#from google.colab import userdata
import os
from langchain_groq import ChatGroq

groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(temperature=0, model_name = "mixtral-8x7b-32768" )

from langchain.chains import RetrievalQA

rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": rag_prompt},
)

response = qa_chain.invoke("Extract all the data from the given text and display it in a proper format.")

print(response['result'])