from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.general import load_documents, create_embeddings, create_qdrant, split_documents


folder_path = "D:\Task\\"
loaded_documents = load_documents(folder_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)
# Split documents
texts = split_documents(loaded_documents, text_splitter)

if texts:
    print(f"Documents split successfully. Total chunks: {len(texts)}")

    # Specify Hugging Face BGE model details
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    # Create embeddings
    embeddings = create_embeddings(model_name, model_kwargs, encode_kwargs)

    if embeddings:
        print("Embeddings created successfully.")
        url = "http://localhost:6333"
        prefer_grpc = False
        collection_name = "new_db"
        # Create Qdrant instance
        qdrant = create_qdrant(texts, embeddings, url, prefer_grpc, collection_name)

        if qdrant:
            print("Qdrant instance created successfully.")
            # Proceed with using the 'qdrant' variable as needed
        else:
            print("Failed to create Qdrant instance.")
    else:
        print("Failed to create embeddings.")
else:
    print("Failed to split documents.")
