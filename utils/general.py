from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
import os
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings#to get embeddings
from langchain.vectorstores import Qdrant #vector database
from qdrant_client import QdrantClient
from langchain.llms import CTransformers#to get llm 
from langchain.text_splitter import RecursiveCharacterTextSplitter#splitting text into chunks
from langchain.chains import RetrievalQA#building Retrieval chain
from langchain.document_loaders import PyPDFLoader,  UnstructuredURLLoader #to read pdfs, urls


qdrant_url = "https://fc211df0-0c5f-4f4f-8132-61b0e6d206b7.us-east4-0.gcp.cloud.qdrant.io"
qdrant_api_key = "LbFjrbBsZgAaEdfpPtusjDfFpcdadiM2CWubWJdTV6CSacoQ9IxVpw"

def split_documents(documents, text_splitter):
    try:
        # Split documents using the specified text splitter
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        print(f"Error: Unable to split documents - {e}")
        return None
    

def load_documents(folder_path):
    try:
        print("Loading Files Started...")
        # Get a list of all files in the folder
        all_files = os.listdir(folder_path)

        # Filter for files with a .pdf extension
        pdf_files = [file for file in all_files if file.lower().endswith(".pdf")]

        # Initialize a list to store loaded documents
        loaded_documents = []

        # Load documents from each PDF file
        for pdf_file in pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            loader = PyPDFLoader(file_path)  # Replace with your actual document loader
            documents = loader.load()
            loaded_documents.extend(documents)
        print("Documents loaded succesfully.")
        return loaded_documents

    except FileNotFoundError:
        print(f"Error: Folder not found - {folder_path}")
    except Exception as e:
        print(f"Error: Unable to load documents - {e}")

    return None

def create_embeddings(model_name, model_kwargs=None, encode_kwargs=None):
    try:
        # Create Hugging Face BGE embeddings
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs or {},
            encode_kwargs=encode_kwargs or {}
        )
        return embeddings
    except Exception as e:
        print(f"Error: Unable to create embeddings - {e}")
        return None
    
def create_qdrant(texts, embeddings, url, prefer_grpc=False, collection_name="new_db"):
    try:
        # Create an instance of YourQdrant or replace it with your actual Qdrant class
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=url,
            prefer_grpc=prefer_grpc,
            collection_name=collection_name
        )
        return qdrant
    except Exception as e:
        print(f"Error: Unable to create Qdrant - {e}")
        return None
    
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        temperature = 0.2
        )
    return llm


# a custom prompt help us to assist our agent with better answer and make sure to not make up answers
custom_prompt_template = """Use the following pieces of information to answer the user’s question.
If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful and Caring answer:
"""

prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

def qa_bot_qdrant_response(context):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    #connect to the vector database
    client = QdrantClient(url=qdrant_url,api_key=qdrant_api_key)
    
    doc_store = Qdrant(
        client=client,
        collection_name="my_documents_new",
        embeddings=embeddings)

    llm = load_llm()
    qa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=doc_store.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )

    response = qa({'query': context})

    return response

class ConversationManager:
    def __init__(self):
        self.context = ""  # Initialize empty context for the conversation

    def update_context(self, new_context):
        self.context += " " + new_context  # Append new context to the existing context

    def get_response(self, query):
        self.update_context(query)  # Add the user's query to the conversation context
        response = qa_bot_qdrant_response(query)
        return response