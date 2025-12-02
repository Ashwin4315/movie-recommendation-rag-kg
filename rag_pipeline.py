import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv

load_dotenv()

client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

client_vectordb = chromadb.PersistentClient(path="vector_db")
collection = client_vectordb.get_collection("movie_reviews")

DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
sparql = SPARQLWrapper(DBPEDIA_ENDPOINT)


def query_movie_kg(movie_name: str):
    try:
        query = f"""
        SELECT ?movie ?label ?director ?genre ?releaseDate WHERE {{
            ?movie rdf:type dbo:Film .
            ?movie rdfs:label ?label .
            FILTER (lang(?label)='en')
            FILTER regex(str(?label), "^{movie_name}$", "i")
            OPTIONAL {{ ?movie dbo:director ?director }}
            OPTIONAL {{ ?movie dbo:genre ?genre }}
            OPTIONAL {{ ?movie dbo:releaseDate ?releaseDate }}
        }} LIMIT 1
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        if not results["results"]["bindings"]:
            return None

        row = results["results"]["bindings"][0]

        return {
            "title": row.get("label", {}).get("value", "N/A"),
            "director": row.get("director", {}).get("value", "N/A"),
            "genre": row.get("genre", {}).get("value", "N/A"),
            "year": row.get("releaseDate", {}).get("value", "N/A"),
        }

    except Exception as e:
        return {"error": str(e)}


def rag_answer(question: str):
    query_embedding = model_embed.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    docs = results["documents"][0]
    if len(docs) == 0:
        docs = ["No relevant documents found."]

    context = "\n\n".join(docs)

    movie_name = question.strip()
    kg_info = query_movie_kg(movie_name)

    if kg_info and "error" not in kg_info:
        kg_context = (
            f"Title: {kg_info.get('title')}, "
            f"Director: {kg_info.get('director')}, "
            f"Genre: {kg_info.get('genre')}, "
            f"Release Year: {kg_info.get('year')}"
        )
    else:
        kg_context = "No useful structured KG facts found."

    prompt = f"""
Answer the user query using these two sources:

1. Movie Review Evidence:
{context}

2. Knowledge Graph Facts:
{kg_context}

If the KG or reviews do not provide enough information, say so clearly.

Question: {question}
"""

    response = client_llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()

    return answer, docs, kg_info
