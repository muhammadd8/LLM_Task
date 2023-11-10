from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
import os

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
    