import nltk
from nltk.tokenize import sent_tokenize
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import torch

import os
import csv
import re


# Download necessary NLTK data
nltk.download('stopwords') #For stop word removal
nltk.download('punkt') #To create tokenizer

#Save csv file
def save_chunks_to_csv(chunks, output_file='chunks.csv'):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['chunk_id', 'web_link', 'chunk_text'])

        # Write each chunk
        for i, (chunk, link) in enumerate(chunks):
            writer.writerow([i, link, chunk])

    print(f"Chunks have been saved to {output_file}") 

# To create semantic chunks
def create_semantic_chunks(text, web_link, model, max_chunk_length=256, min_chunks=2):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_length:
            if current_chunk:
                chunks.append((" ".join(current_chunk), web_link))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append((" ".join(current_chunk), web_link))

    # Ensure there are at least min_chunks
    if len(chunks) < min_chunks:
        # Split the last chunk into smaller chunks
        last_chunk, last_link = chunks[-1]
        sentences = sent_tokenize(last_chunk)
        while len(chunks) < min_chunks:
            if len(sentences) <= 1:
                break
            chunks.pop()
            chunks.extend([(" ".join(sentences[:len(sentences)//2]), last_link),
                           (" ".join(sentences[len(sentences)//2:]), last_link)])
            sentences = sent_tokenize(" ".join(sentences[len(sentences)//2:]))

    # Generate embeddings for chunks
    chunk_texts = [chunk[0] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_texts)

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
    clustering.fit(chunk_embeddings)

    # Combine chunks based on clustering results
    combined_chunks = []
    for cluster_id in range(max(clustering.labels_) + 1):
        cluster_chunks = [chunk for chunk, label in zip(chunks, clustering.labels_) if label == cluster_id]
        combined_text = " ".join([chunk[0] for chunk in cluster_chunks])
        combined_link = cluster_chunks[0][1]  # Use the link from the first chunk in the cluster
        combined_chunks.append((combined_text, combined_link))

    return combined_chunks

# Function to remove punctuation from text
def clean_text_num(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])

    return text

# Query expansion code
def expand_query(question, index, model, top_k=3):
    query_embedding = model.encode(question).tolist()
    similar_vectors = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    expanded_terms = [result['metadata']['chunk_text'] for result in similar_vectors['matches']]
    expanded_query = question + " " + " ".join(expanded_terms)
    return expanded_query    

#Hybrid Retrieval
def hybrid_search(question, index, model, alpha=0.5, top_k=10):
    # Vector search
    query_vector = model.encode([question]).tolist()
    vector_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # BM25 search
    tokenized_corpus = [result['metadata']['chunk_text'].split() for result in vector_results['matches']]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Combine results
    combined_scores = {}
    for i, (score, result) in enumerate(zip(bm25_scores, vector_results['matches'])):
        doc_id = result['id']
        combined_scores[doc_id] = alpha * score + (1 - alpha) * result['score']

    # Sort results
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    # print("Sorted results:", sorted_results)

    # Retrieve the original results in the new order
    final_results = []
    for doc_id, _ in sorted_results:
        for result in vector_results['matches']:
            if result['id'] == doc_id:
                final_results.append(result)
                break

    return final_results

#Re-ranking
def rerank(question, documents, tokenizer, model, top_k=5):
    pairs = [[question, doc['metadata']['chunk_text']] for doc in documents]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)

    ranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    # print(ranked_results)
    return [doc for doc, _ in ranked_results[:top_k]]

#Question answer 

#Question answer 
def answer_question(question, context, llm_model, llm_tokenizer, device, max_length=200):
  prompt = f"""
    You are a helpful assitant. Use the following context to answer the question. If the context doesn't contain relevant information then say "I don't know.
    Context: {context}
    Question: {question}
    Assistant: Based on the given context, I will answer the question to the best of my ability.\n
  """

  inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)

  with torch.no_grad():
    outputs = llm_model.generate(
        **inputs,
        max_length=len(inputs["input_ids"][0])+max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        pad_token_id=llm_tokenizer.eos_token_id,
        eos_token_id=llm_tokenizer.eos_token_id,
    )
  answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
  

  # Format the answer according to the desired format
  formatted_answer = answer.replace("\n\n", "\n- ").replace("\n", " ")
  formatted_answer = f"{formatted_answer}"

  formatted_answer = formatted_answer.split("Based on the given context, I will answer the question to the best of my ability. -    ")
  return formatted_answer[-1]


def advanced_retrieval_and_ranking(question, index, sentence_model, llm_model, llm_tokenizer, rerank_tokenizer, rerank_model, device, top_k=5):
    # Step 1: Query Expansion
    expanded_query = expand_query(question=question, index=index, model=sentence_model, top_k=3)
    # print(f"Expanded Query: {expanded_query}")

    # Step 2: Hybrid Retrieval
    initial_results = hybrid_search(question=expanded_query, index=index, model=sentence_model, alpha=0.5, top_k=top_k*2)
    # print(f"Initial Results: {initial_results}")

    # Step 3: Re-ranking
    final_results = rerank(question=question, documents=initial_results, tokenizer=rerank_tokenizer, model=rerank_model, top_k=top_k)

    # Step 4: Join the text of top results
    context = " ".join([result['metadata']['chunk_text'] for result in final_results])

    # print("Context:\n",context)
    answer = answer_question(question=question, context=context, llm_model=llm_model, llm_tokenizer=llm_tokenizer, device=device, max_length=200)

    return answer, final_results