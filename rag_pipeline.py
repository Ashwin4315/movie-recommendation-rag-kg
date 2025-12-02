# rag_pipeline.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ LLM Client
client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2️⃣ Embedding model
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# 3️⃣ Vector DB
client_vectordb = chromadb.PersistentClient(path="vector_db")
collection = client_vectordb.get_collection("movie_reviews")

# 4️⃣ DBpedia endpoint
DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
sparql = SPARQLWrapper(DBPEDIA_ENDPOINT)

def query_movie_kg(movie_name: str):
    """Query DBpedia for movie info using regex match"""
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

        binding = results["results"]["bindings"][0]

        movie_info = {
            "movie": binding.get("label", {}).get("value", "N/A"),
            "director": binding.get("director", {}).get("value", "N/A"),
            "genre": binding.get("genre", {}).get("value", "N/A"),
            "year": binding.get("releaseDate", {}).get("value", "N/A"),
        }
        return movie_info

    except Exception as e:
        return {"error": f"KG query failed: {str(e)}"}

def rag_answer(question: str):
    """Retrieve answer using RAG and fetch KG info"""
    # 1️⃣ Embed question
    query_embedding = model_embed.encode(question).tolist()

    # 2️⃣ Retrieve top 5 chunks from Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    docs = results["documents"][0]
    if len(docs) == 0:
        docs = ["No relevant documents found."]

    context = "\n\n".join(docs)

    # 3️⃣ RAG prompt
    prompt = f"""
Use ONLY the following movie review evidence to answer the question.
If evidence is insufficient, say "Not enough information in reviews."

Evidence:
{context}

Question: {question}
"""

    response = client_llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()

    # 4️⃣ KG lookup (simple: use the question as movie name)
    movie_name = question.strip()
    kg_info = query_movie_kg(movie_name)

    return answer, docs, kg_info
