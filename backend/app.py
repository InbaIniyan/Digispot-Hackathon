import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import torch

import psycopg2

# For MiniLM
from sentence_transformers import SentenceTransformer
# For BERT and T5
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel


from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

# Create a session with retries and timeout
session = requests.Session()
retry = Retry(
    total=5,  # total retries
    backoff_factor=0.3,  # exponential backoff factor
    status_forcelist=[500, 502, 503, 504],  # retry for specific HTTP status codes
)
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)

# Set timeout for requests
timeout = 60  # increase timeout value


# NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
stop_words = set(stopwords.words('english'))

# Parameters for speed improvements
MAX_PAGES = 20  # Limit the number of pages to process per website

# Load Embedding Models Globally
minilm_model = SentenceTransformer('all-MiniLM-L6-v2')
def get_minilm_embedding(text):
    return minilm_model.encode(text)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
def get_bert_embedding(text):
    tokens = bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy()[0]

t5_tokenizer = T5Tokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
t5_model = T5EncoderModel.from_pretrained("sentence-transformers/sentence-t5-base")
def get_t5_embedding(text):
    tokens = t5_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = t5_model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy()[0]

# Common Utility Functions
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = text.lower()
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if w not in stop_words]
    return " ".join(filtered_text)

def cluster_pages_with_evaluation(page_data, embedding_fn):
    embeddings = []
    for idx, page in enumerate(page_data):
        txt = clean_text(" ".join(page['headings']) + " " + page['content'] + " " + page['meta_description'])
        embeddings.append(embedding_fn(txt))

    if not embeddings:
        return {}, 0

    num_clusters = min(len(page_data) // 3, 5) if len(page_data) >= 3 else 1
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    labels = clustering.fit_predict(embeddings)
    
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(page_data[i])
    
    silhouette = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0
    return clusters, silhouette

def suggest_links(clusters):
    suggestions = []
    for cluster_id, pages in clusters.items():
        for i, page1 in enumerate(pages):
            for j in range(i + 1, len(pages)):
                page2 = pages[j]
                suggestions.append({
                    'source_page': page1,
                    'target_page': page2,
                    'reason': 'Same Topic Cluster'
                })
    return suggestions

def generate_report(suggestions, model_name):
    report = []
    for suggestion in suggestions:
        report.append({
            'Model': model_name,
            'Source Page': suggestion['source_page']['url'],
            'Target Page': suggestion['target_page']['url'],
            'Suggested Anchor Text': suggestion['target_page']['title'],
            'Reason': suggestion['reason']
        })
    return pd.DataFrame(report)

# Flask app setup and routes (you need to integrate this into your Flask backend)
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/generate_link_suggestions', methods=['POST'])
def generate_link_suggestions():
    data = request.get_json()
    websites = data['websites']
    model_name = data['model_name']  # Model name selected by the user (MiniLM, BERT, T5)
    
    combined_detailed_df = pd.DataFrame()
    summary_rows = []

    models = {
        'MiniLM': get_minilm_embedding,
        'BERT': get_bert_embedding,
        'T5': get_t5_embedding
    }
    
    # Your website URL processing and clustering code
    for website_url in websites:
        print(f"Analyzing: {website_url}")
        
        # Fetch the content of the website
        html = requests.get(website_url).text
        soup = BeautifulSoup(html, 'html.parser')
        
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            absolute_url = requests.compat.urljoin(website_url, href)
            if absolute_url.startswith(website_url):
                title = link.text.strip()
                links.append({'url': absolute_url, 'title': title})
        
        page_data = []
        for link in links[:MAX_PAGES]:
            try:
                page_html = requests.get(link['url']).text
                page_soup = BeautifulSoup(page_html, 'html.parser')
                content = page_soup.get_text(strip=True)
                meta_description = ""
                meta_tag = page_soup.find('meta', attrs={'name': 'description'})
                if meta_tag and meta_tag.get('content'):
                    meta_description = meta_tag.get('content')
                headings = [h.text.strip() for h in page_soup.find_all(['h1', 'h2', 'h3'])]
                page_data.append({
                    'url': link['url'],
                    'title': link['title'],
                    'content': content,
                    'meta_description': meta_description,
                    'headings': headings
                })
            except Exception as e:
                continue
        
        if page_data:
            # Use the evaluation-aware clustering function:
            clusters, silhouette = cluster_pages_with_evaluation(page_data, models[model_name])
            suggestions = suggest_links(clusters)
            report_df = generate_report(suggestions, model_name)
            combined_detailed_df = pd.concat([combined_detailed_df, report_df], ignore_index=True)

            num_clusters = len(clusters)
            total_suggestions = len(suggestions)
            avg_cluster_size = np.mean([len(pages) for pages in clusters.values()]) if clusters else 0

            summary_rows.append({
                'Website': website_url,
                'Model': model_name,
                'Pages Processed': len(page_data),
                'Clusters': num_clusters,
                'Total Suggestions': total_suggestions,
                'Avg Cluster Size': round(avg_cluster_size, 2),
                'Silhouette Score': round(silhouette, 3)
            })

    # Save data to CSV
    summary_df = pd.DataFrame(summary_rows)
    combined_detailed_df.to_csv("internal_link_suggestions_detailed.csv", index=False)

    # Now insert the data from the CSV into PostgreSQL

    # Establish a connection to your PostgreSQL database
    conn = psycopg2.connect(
        dbname="seo_tool", 
        user="postgres", 
        password="123", 
        host="localhost", 
        port="5432"
    )
    cursor = conn.cursor()

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('internal_link_suggestions_detailed.csv')

    # Loop through the rows of the DataFrame and insert into PostgreSQL
    for index, row in df.iterrows():
        try:
            print(f"Model: {row['Model']}, Source Page: {row['Source Page']}, Target Page: {row['Target Page']}")
            cursor.execute("""INSERT INTO internal_link_suggestions (model, source_page, target_page, suggested_anchor_text, reason) VALUES (%s, %s, %s, %s, %s)""", 
                (row['Model'], row['Source Page'], row['Target Page'], row['Suggested Anchor Text'], row['Reason']))
            print(f"Inserted row {index + 1}")
        except Exception as e:
            print(f"Error inserting row {index + 1}: {e}")

    # Commit the transaction
    conn.commit()

    # Close the connection
    cursor.close()
    conn.close()

    # Now, return the results as a JSON response
    return jsonify({
        "summary": summary_df.to_dict(orient="records"),
        "detailed": combined_detailed_df.to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True)
