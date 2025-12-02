import pandas as pd
import kagglehub

CHUNK_SIZE = 500
OVERLAP = 100

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def run_chunking():
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    df = pd.read_csv(f"{path}/IMDB Dataset.csv")

    chunk_rows = []
    for idx, row in df.iterrows():
        review = str(row["review"])
        sentiment = row["sentiment"]
        chunks = chunk_text(review)
        for chunk in chunks:
            chunk_rows.append({
                "chunk": chunk,
                "sentiment": sentiment
            })

    chunk_df = pd.DataFrame(chunk_rows)
    chunk_df.to_csv("data/chunked_reviews.csv", index=False)
    print("Chunking complete! Saved to data/chunked_reviews.csv")

if __name__ == "__main__":
    run_chunking()
