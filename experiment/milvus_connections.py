import pymilvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import pandas as pd
import config as conf
import numpy as np
import time

#load csv file
df = pd.read_csv(conf.csv_file_path)
# df_copy = df.copy()
print(df)

print("Before conversion type of chunk_emb is:",type(df['chunk_emb'][0]))
# In our new_embedding.csv file all the embeddings are stored in string format so I have to convert them into numpy array.
# Convert the string back to a list of floats
df['chunk_emb'] = df['chunk_emb_str'].apply(lambda x: np.array(list(map(float, x.split(',')))))
print("After conversion type of chunk_emb is:",type(df['chunk_emb'][0]), " its shape is :",df['chunk_emb'][0].shape) #Shape will be (384,) because 'all-MiniLM-L6-v2' sentence transformer used which uses 384 embedding dimension for each chunk_text

# print("Connecting with localhost MILVUS")
# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
print("Connect with localhost MILVUS at port 19350")


# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="web_link", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Dimension of all-MiniLM-L6-v2 embeddings
]
schema = CollectionSchema(fields, "NVIDIA CUDA Documentation Chunks VectorDB")

print("Schema is created")

print("Create collection")
# Create collection
collection_name = "cuda_chunks"
collection = Collection(collection_name, schema)
print("collection Created")

print("Creating HNSW for embedding")
# Create HNSW index
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 8, "efConstruction": 64}
}
collection.create_index(field_name="embedding", index_params=index_params)
print("Created HNSW for embedding")

print("Data Insertion start")
# Insert data into Milvus
insert_data = [
    [web_link for web_link in df['web_link']],
    [text for text in df['chunk_text']],
    [embedding for embedding in df['chunk_emb']],
]
collection.insert(insert_data)

print(f"Inserted {len(df)} chunks into Milvus collection '{collection_name}'")

print("collection flushing start")
# Flush the collection to ensure data is written to disk
collection.flush()
print("collection flushing end")

print("collection load start")
#To see data on attu UI for which we have to load data to memory.
collection.load()
print("collection flushing start")

# Get the collection stats
print(collection.stats())

# Disconnect from Milvus
connections.disconnect("default")

print("Data chunking and vector database creation completed.")
