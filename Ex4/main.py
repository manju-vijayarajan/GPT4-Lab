import pandas as pd
import numpy as np
import faiss
import requests
import gradio as gr


OLLAMA_URL = "http://localhost:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "qwen2.5:0.5b"

database = pd.read_csv("database.csv")

database = database.fillna("")
database["text"] = (
    database["title"].astype(str)
    + " "
    + database["genre"].astype(str)
    + " "
    + database["summary"].astype(str)
    + " "
    + database["keywords"].astype(str)
)

def get_embedding(text):

    response = requests.post(
        f"{OLLAMA_URL}/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if "embeddings" not in data:
        raise ValueError(
            f"Ollama did not return embeddings: {data}"
        )

    return np.array(
        data["embeddings"][0],
        dtype="float32"
    )


print("Generating embeddings...")

embeddings = []

for text in database["text"]:
    embeddings.append(get_embedding(text))

embeddings = np.array(
    embeddings,
    dtype="float32"
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index created successfully.")


def recommend(user_interest, user_preferences, domain):

    if not user_interest.strip():
        return "Please enter your interests."

    if not user_preferences.strip():
        return "Please enter your preferences."

    if domain == "Books":

        filtered_database = database[
            database["domain"] == "Books"
        ]

    elif domain == "Movies":

        filtered_database = database[
            database["domain"] == "Movies"
        ]

    else:
        filtered_database = database

    if filtered_database.empty:
        return "No items available for the selected domain."

    query = (
        f"Interest: {user_interest}. "
        f"Preferences: {user_preferences}"
    )
    query_embedding = get_embedding(query)

    filtered_indices = filtered_database.index.tolist()

    filtered_embeddings = embeddings[
        filtered_indices
    ]

    filtered_index = faiss.IndexFlatL2(
        filtered_embeddings.shape[1]
    )

    filtered_index.add(filtered_embeddings)

    k = min(3, len(filtered_database))

    distances, indices = filtered_index.search(
        np.array([query_embedding], dtype="float32"),
        k=k
    )
    recommendations = filtered_database.iloc[
        indices[0]
    ]
    recommendation_text = ""

    for i, (_, item) in enumerate(
        recommendations.iterrows(),
        start=1
    ):
        recommendation_text += (
            f"{i}. {item['title']}\n"
            f"   Domain: {item['domain']}\n"
            f"   Genre: {item['genre']}\n"
            f"   Description: {item['summary']}\n\n"
        )
    prompt = f"""
You are a personalized recommendation assistant.

User interest:
{user_interest}

User preferences:
{user_preferences}

Selected domain:
{domain}

Recommended items:
{recommendation_text}

Explain why these recommendations match
the user's interests and preferences.

Keep the explanation concise and clear.
"""
    response = requests.post(
        f"{OLLAMA_URL}/generate",
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()
    explanation = data.get(
        "response",
        "No explanation generated."
    )

    return (
        "### Recommendations\n\n"
        + recommendation_text
        + "\n### Why these recommendations?\n\n"
        + explanation
    )

def main():

    interface = gr.Interface(

        fn=recommend,

        inputs=[

            gr.Textbox(
                label="Your Interests",
                placeholder=(
                    "Example: I want to learn "
                    "artificial intelligence"
                ),
                lines=3
            ),

            gr.Textbox(
                label="Your Preferences",
                placeholder=(
                    "Example: I prefer "
                    "Computer Science topics"
                ),
                lines=3
            ),

            gr.Dropdown(
                choices=[
                    "All",
                    "Books",
                    "Movies"
                ],
                value="All",
                label="Select Domain"
            )
        ],

        outputs=gr.Markdown(
            label="Recommendations"
        ),

        title="Personalized Recommendation System",

        description=(
            "Enter your interests and preferences "
            "to receive personalized recommendations."
        )
    )

    interface.launch()

if __name__ == "__main__":
    main()