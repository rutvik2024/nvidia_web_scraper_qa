import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import config as conf
import util as ut
import os

# Access the environment variables
HuggingFace_API_KEY = os.getenv('HuggingFace_API_KEY')

#Create embedding and store to csv file
# 1. Take path of csv file
csv_file_path = conf.csv_file_path

# 2. load csv file
df = pd.read_csv(csv_file_path)

# 3. remove stop words and punctuation from text
# print("Chunk before cleaning: ",df['chunk_text'][0])
df['chunk_text'] = df['chunk_text'].apply(ut.clean_text_num)
# print("Chunk after cleaning: ",df['chunk_text'][0])

sentence_model = SentenceTransformer(conf.sentence_model_name, use_auth_token=HuggingFace_API_KEY)

# Generate embeddings
df['chunk_emb'] = df['chunk_text'].apply(lambda x: sentence_model.encode(x).tolist())

# When saving to CSV, convert the list to a string
df['chunk_emb_str'] = df['chunk_emb'].apply(lambda x: ','.join(map(str, x)))

# 4. Save to CSV
df.to_csv(conf.csv_file_path, index=False)

print(f"Embeddings added and CSV file saved successfully at {conf.csv_file_path}.")