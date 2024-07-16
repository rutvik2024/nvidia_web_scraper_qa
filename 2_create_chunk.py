#import required libraries
from sentence_transformers import SentenceTransformer


import os
import json
from dotenv import load_dotenv

import config as conf
import util as ut

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
HuggingFace_API_KEY = os.getenv('HuggingFace_API_KEY')


#Load json file
json_file_path = conf.json_file_path

with open(json_file_path, 'r') as f:
    data = json.load(f)

# print(data) #json file has three fields: url, content and depth

#extract only content
all_content = [item['content'] for item in data]
# print("Total number of pages data I have extracted: ", len(all_content))


#To Create a semantic chunks
# 1.Initialize SentenceTransformer model
sentence_model = SentenceTransformer(conf.sentence_model_name, use_auth_token=HuggingFace_API_KEY)

# 2.Create semantic chunks for each document
all_chunks = []
for item in data:
    content = item['content']
    web_link = item['url']  # 'url' is a field in your JSON data
    chunks = ut.create_semantic_chunks(text=content, web_link=web_link, model=sentence_model, max_chunk_length=128, min_chunks=2)
    all_chunks.extend(chunks)

# print("First Chunks: ",all_chunks[0])

#Save Chunk to csv file
ut.save_chunks_to_csv(chunks=all_chunks, output_file=conf.csv_file_path)

