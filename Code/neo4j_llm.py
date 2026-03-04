import ollama
import json
from neo4j import GraphDatabase
import re
from google import genai
import os
from dotenv import load_dotenv

load_dotenv('../Misc/.env')

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
- Only use the first question in the user query to extract keywords and relationships. Ignore any prompt that is just information.
- Output valid JSON only

User query:
"{user_query}"
"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_predict": 500,
                "stop": ["\n\n", "```"]
            }
        )
        
        # Debug: print the entire response object
        print(f"Full response object: {response}")
        print(f"Response text: '{response['message']['content']}'")
        print(f"Response text length: {len(response['message']['content'])}")
        
        response_text = response["message"]["content"].strip()
        
        if not response_text:
            print("Warning: Empty response from API")
            return {"keywords": [], "relationships": []}
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1])
            response_text = response_text.strip()
        
        print(f"Cleaned response text: '{response_text}'")
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        return {"keywords": [], "relationships": []}


def extract_zipcode(keywords):
    for kw in keywords:
        if re.fullmatch(r"\d{5}", kw):
            return kw
    return None


def build_cypher(intent):
    service_keywords = ["food pantry", "shelter", "mental health", "housing"]
    has_service_keyword = any(kw in intent["keywords"] for kw in service_keywords)

    if not intent:
        raise ValueError("Intent is empty — LLM may have failed.")
    
    if has_service_keyword:
        cypher = """
        MATCH (s:node)-[r1:main_services_are]->(service_type:node)
        MATCH (s)-[r2:is_located_at]->(address:node)
        """
        
        where_clauses = []
        params = {}
        
        zipcode = extract_zipcode(intent["keywords"])
        print(f"DEBUG: Extracted zipcode: {zipcode}")
        
        filtered_keywords = [
            kw.lower()
            for kw in intent["keywords"]
            if kw.lower() not in STOPWORDS and kw != zipcode
        ]
        print(f"DEBUG: Filtered keywords: {filtered_keywords}")
        
        # Match service type - search within array
        if filtered_keywords:
            where_clauses.append("""
            ANY(kw IN $keywords WHERE
                ANY(service IN service_type.name WHERE toLower(service) CONTAINS kw)
            )
            """)
            params["keywords"] = filtered_keywords
            print(f"DEBUG: Keywords: {filtered_keywords}")
        
        # Match zipcode
        if zipcode:
            cypher += """
                MATCH (s)-[r3:zipcode_is]->(zip:node)
                """
            where_clauses.append(
                "toInteger(zip.name) = $zipcode_int"
            )
            params["zipcode_int"] = int(zipcode)  # Convert to integer
        
        if where_clauses:
            cypher += " WHERE " + " AND ".join(where_clauses)
        
        cypher += """
        RETURN s.name AS subject,
               'is_located_at' AS relationship,
               address.name AS object
        LIMIT 25
        """
    
    print(f"DEBUG: Final Cypher:\n{cypher}")
    print(f"DEBUG: Parameters: {params}")
    
    return cypher, params

NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

def run_query(cypher, params):
    with driver.session() as session:
        result = session.run(cypher, params)
        return [record.data() for record in result]

 #Example usage
query = "Where is the nearest food pantry to 92501?  How far are the services from where I am? I am on 3931 Carter Ave."

intent = extract_query_intent(query)

print(intent)

cypher, params = build_cypher(intent)
results = run_query(cypher, params)

#print(results)

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
You are a social worker and you want to answer a question using the given content. Give a summary of the relevant information in the content that answers the question. Use only the information provided in the content, and do not make any assumptions or use any outside knowledge. If the content does not contain enough information to answer the question, say so.

STRICT RULES:
- Use only the provided facts
- Addresses MUST be copied character-for-character from the facts
- Do NOT abbreviate, truncate, reformat, or paraphrase addresses
- If unsure, copy the address exactly as written- Do not include analysis
- Do not include instruction text
- Do NOT explain ambiguity.
- Do NOT assume distances or "nearest".
- If multiple valid answers exist, LIST ALL OF THEM.
- Answer the first question first, and then use the rest of the content to add on to the answer.
- If the users location is mentioned, use it to calculate distance to the services. If not, say so briefly at the end.

User question:
"{user_query}"

Facts:
{facts}

===
FINAL ANSWER (do not include instructions or explanations):

Answer:
"""
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0,
            "max_output_tokens": 2000,  # Increased from 200 to be safe
            "stop_sequences": [
                "<|end_of_instruction|>",
                "<|assistant|>",
                "<|user|>",
                "FINAL ANSWER:"
            ]
        }
    )

    return response.text.strip()

final_answer = generate_natural_response(query, results)
print(final_answer)