import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import kagglehub

def build_db():
    
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    chunked_csv_path = "data/chunked_reviews.csv"

    df = pd.read_csv(chunked_csv_path)
    df = df.head(5000)

    client = chromadb.PersistentClient(path="vector_db")
    collection = client.get_or_create_collection(
        name="movie_reviews",
        metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding and inserting into vector DB...")
    for i, row in df.iterrows():
        text = row["chunk"]
        embedding = model.encode(text).tolist()
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            metadatas=[{"sentiment": row["sentiment"]}],
            documents=[text]
        )
    print("Vector database created successfully!")

if __name__ == "__main__":
    build_db()
