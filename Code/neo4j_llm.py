import ollama
import json
from neo4j import GraphDatabase
import re

STOPWORDS = {
    "nearest", "near", "closest", "find", "where",
    "what", "are", "is", "to", "the"
}

##### LLM ONE #####

def extract_query_intent(user_query):
    prompt = f"""
You are a query planner for a Neo4j knowledge graph built from triples.

Each triple has the form:
(Subject)-[Relationship]->(Object)

Subjects and Objects are strings.
Relationships describe meaning (e.g. serves, main_services_are, phone_number_is).

Task:
From the user query, extract relevant search terms and relationship names.

Return JSON in this EXACT format:
{{
  "keywords": [],
  "relationships": []
}}

Rules:
- If a word is similar to a relationship name, put it in "relationships" as the original relationship
Ex. words located or location should be put in the relationships list as "is_located_at"
- Use lowercase
- If a word isn't a relationship type, don't put it in the relationships list
- Keywords should be relevant search terms that aren't relationship types
- Words that describe intent are not graph entities. Do NOT include them as keywords.
- Output valid JSON only

User query:
"{user_query}"
"""

    response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": prompt}],
    options={
        "temperature": 0,
        "num_predict": 150,
        "stop": ["\n\n", "```"]
    }
)

    return json.loads(response["message"]["content"])


def extract_zipcode(keywords):
    for kw in keywords:
        if re.fullmatch(r"\d{5}", kw):
            return kw
    return None


def build_cypher(intent):
    cypher = """
    MATCH (s:node)-[r]->(o:node)
    """

    where_clauses = []
    params = {}

    if intent["relationships"]:
        where_clauses.append("type(r) IN $relationships")
        params["relationships"] = intent["relationships"]

    zipcode = extract_zipcode(intent["keywords"])

    filtered_keywords = [
        kw.lower()
        for kw in intent["keywords"]
        if kw.lower() not in STOPWORDS and kw != zipcode
    ]

    if filtered_keywords:
        where_clauses.append("""
        ANY(kw IN $keywords WHERE
            toLower(s.name) CONTAINS kw
        )
        """)
        params["keywords"] = filtered_keywords

    if zipcode:
        where_clauses.append(
            "toLower(o.name) CONTAINS $zipcode"
        )
        params["zipcode"] = zipcode

    if where_clauses:
        cypher += " WHERE " + " AND ".join(where_clauses)

    cypher += """
    RETURN s.name AS subject,
           type(r) AS relationship,
           o.name AS object
    LIMIT 25
    """

    return cypher, params


driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "honorskg")
)

def run_query(cypher, params):
    with driver.session() as session:
        result = session.run(cypher, params)
        return [record.data() for record in result]

 #Example usage
query = "Where is the nearest food pantry to 92501?"

intent = extract_query_intent(query)

print(intent)

cypher, params = build_cypher(intent)
results = run_query(cypher, params)

print(results)

##### LLM TWO #####

def generate_natural_response(user_query, triples):
    if not triples:
        return "I couldn’t find any matching services for your request."

    # Deduplicate by subject + object
    seen = set()
    unique_triples = []
    for t in triples:
        key = (t["subject"], t["object"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    facts = "\n".join(
        f"- {t['subject']} is located at {t['object']}"
        for t in unique_triples
    )

    prompt = f"""
You are answering a question using a knowledge graph.

STRICT RULES:
- Use only the provided facts
- Addresses MUST be copied character-for-character from the facts
- Do NOT abbreviate, truncate, reformat, or paraphrase addresses
- If unsure, copy the address exactly as written- Do not include analysis
- Do not include instruction text
- Output bullet points only
- Do NOT explain ambiguity.
- Do NOT assume distances or "nearest".
- Do NOT summarize to a single result.
- If multiple valid answers exist, LIST ALL OF THEM.
- If "nearest" is requested but distance data is missing, say so briefly at the end.

User question:
"{user_query}"

Facts:
{facts}

===
FINAL ANSWER (do not include instructions or explanations):

Answer:
"""

    response = ollama.chat(
    model="phi3",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0,
        "num_predict": 200,
        "stop": [
            "<|end_of_instruction|>",
            "<|assistant|>",
            "<|user|>",
            "FINAL ANSWER:"
        ]
    }
)

    return response["message"]["content"].strip()

final_answer = generate_natural_response(query, results)
print(final_answer)