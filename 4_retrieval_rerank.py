import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import torch
from pinecone import Pinecone, ServerlessSpec
import time
import config as conf
import util as ut

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HuggingFace_API_KEY = os.getenv('HuggingFace_API_KEY')
ENVIRONMENT = os.getenv('ENVIRONMENT')
CLOUD = os.getenv('CLOUD')
INDEX_NAME = os.getenv('INDEX_NAME')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CSV
df = pd.read_csv(conf.csv_file_path)

# 1. Convert the string back to a list of floats
df['chunk_emb'] = df['chunk_emb_str'].apply(lambda x: np.array(list(map(float, x.split(',')))))

# 2. Pinecone initialization
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = INDEX_NAME #Create a index
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes() #get existing index fron connection
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=384,  # dimensionality of minilm
        metric='cosine',
        spec=ServerlessSpec(
          cloud=CLOUD,
          region=ENVIRONMENT
      )
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

# 3. Pinecone data insertion

# Prepare data for upsert
vectors_to_upsert = []
for _, row in df.iterrows():
    vector = {
        'id': str(row['chunk_id']),
        'values': row['chunk_emb'].tolist(),  # Convert numpy array to list
        'metadata': {
            'web_link': row['web_link'],
            'chunk_text': row['chunk_text']
        }
    }
    vectors_to_upsert.append(vector)

# Upsert vectors in batches
batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i+batch_size]
    index.upsert(vectors=batch)

print("Vector database created successfully!")

# 4. Query expansion
# Initialize SentenceTransformer model
sentence_model = SentenceTransformer(conf.sentence_model_name, use_auth_token=HuggingFace_API_KEY)

#Rerank model initialization
rerank_tokenizer = AutoTokenizer.from_pretrained(conf.rerank_model_name, use_auth_token=HuggingFace_API_KEY)
rerank_model = AutoModelForSequenceClassification.from_pretrained(conf.rerank_model_name, use_auth_token=HuggingFace_API_KEY)

#LLM model initialization
# Load Meta-Llama-3-8B model and tokenizer

llama_tokenizer = AutoTokenizer.from_pretrained(conf.llama_model_name, use_auth_token=HuggingFace_API_KEY)
llama_model = AutoModelForCausalLM.from_pretrained(conf.llama_model_name,
                                             device_map="auto",
                                             torch_dtype=torch.float16,
                                             load_in_8bit=True,
                                             use_auth_token=HuggingFace_API_KEY
                                             )


first_question = True
while True:
    if first_question:
        question = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        question = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if question.lower() == "quit":
        break

    if question == "":
        continue

    first_question = False

    print("\nQUESTION: \"%s\"" % question)
    answer, results = ut.advanced_retrieval_and_ranking(question=question, index=index, sentence_model=sentence_model, 
                                                     llm_model=llama_model, llm_tokenizer=llama_tokenizer, 
                                                     rerank_tokenizer=rerank_tokenizer, rerank_model=rerank_model,
                                                     device=device, top_k=5)
    print("ANSWER: \"%s\"\n" % answer)

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for i, result in enumerate(results[:5], 1):
      print(f"{i}. Score: {result['score']:.4f}")
      print(f"   URL: {result['metadata']['web_link']}")
      print(f"   Text: {result['metadata']['chunk_text'][:100]}...")
      print("---")

