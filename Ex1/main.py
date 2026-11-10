import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import ollama

products_df = pd.read_csv("products.csv")
faq_df = pd.read_csv("faq.csv")
reviews_df = pd.read_csv("reviews.csv")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_model = pipeline("sentiment-analysis")
ollama_model = "qwen2.5:0.5b"

products_df["text"] = (
    products_df["product_name"].fillna("") + " " +
    products_df["category"].fillna("") + " " +
    products_df["description"].fillna("")
)

faq_embeddings = embedding_model.encode(faq_df["question"].tolist())
product_embeddings = embedding_model.encode(products_df["text"].tolist())

vectorizer = TfidfVectorizer(stop_words="english")
product_tfidf = vectorizer.fit_transform(products_df["text"])

def find_faq(query):
    emb = embedding_model.encode(query)
    scores = cosine_similarity([emb], faq_embeddings)[0]
    i = np.argmax(scores)

    if scores[i] >= 0.4:
        return faq_df.iloc[i]["answer"]
    return None

def find_product(query):
    emb = embedding_model.encode(query)
    scores = cosine_similarity([emb], product_embeddings)[0]
    return products_df.iloc[np.argmax(scores)]


# Recommend products
def recommend(query, k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, product_tfidf)[0]
    indices = np.argsort(scores)[::-1][:k]

    return products_df.iloc[indices][
        ["product_name", "price"]
    ]


# Analyze reviews
def analyze_reviews(product_id):
    reviews = reviews_df[
        reviews_df["product_id"] == product_id
    ]["review_text"]

    for review in reviews:
        result = sentiment_model(review)[0]
        print(f"{review} → {result['label']}")


def product_qa(query):
    product = find_product(query)

    details = f"""
Product: {product['product_name']}
Category: {product['category']}
Price: ₹{product['price']}
Description: {product['description']}
"""
    response = ollama.chat(
        model=ollama_model,
        messages=[
            {
                "role": "system",
                "content": "You are an e-commerce assistant."
            },
            {
                "role": "user",
                "content": f"{details}\nQuestion: {query}"
            }
        ]
    )
    return response["message"]["content"]

while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = find_faq(query)
    if answer:
        print("Bot:", answer)
        continue
    if any(x in query.lower()
           for x in ["recommend", "suggest", "similar", "alternative"]):
        print("Bot:\n", recommend(query))
        continue
    if any(x in query.lower()
           for x in ["review", "reviews", "complaints", "feedback"]):

        product = find_product(query)
        print(f"Reviews for {product['product_name']}:")
        analyze_reviews(product["product_id"])
        continue
    print("Bot:", product_qa(query))