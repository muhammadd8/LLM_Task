from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
from utils.general import load_documents, create_embeddings, create_qdrant, split_documents


url = "http://localhost:6333"
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
else:
    print("Failed to create embeddings.")

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

query = "which version of YOLO was used?"

docs = db.similarity_search_with_score(query=query, k=2)

concatenated_text = ""
for i in docs:
    doc, score = i
    concatenated_text += doc.page_content

concatenated_text = concatenated_text + " " + query
print(concatenated_text)
# Load the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
# model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

# Tokenize the concatenated text
input_ids = tokenizer.encode(concatenated_text, return_tensors="pt")

# Generate text based on the input
generated_ids = model.generate(input_ids, max_length=150, num_beams=5, length_penalty=2.0, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

# Decode the generated text
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# # Print the generated text
print(generated_text)