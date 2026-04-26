import ollama
import json
from neo4j import GraphDatabase
import re
import os
from dotenv import load_dotenv

load_dotenv('../Misc/.env')

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
    if not intent or "keywords" not in intent:
        raise ValueError("Invalid intent")

    all_keywords = [k.lower() for k in intent["keywords"]]

    zipcode = None
    for k in all_keywords:
        if k.isdigit() and len(k) == 5:
            zipcode = k
            break

    entity_keywords = ["phone", "number", "contact", "address", "website", "url"]
    service_keywords = ["food", "pantry", "shelter", "mental", "health", "housing", "services"]
    geo_keywords = ["near", "nearest", "closest", "nearby"]

    is_entity_lookup = any(any(ek in kw for ek in entity_keywords) for kw in all_keywords)
    is_service_search = any(any(sk in kw for sk in service_keywords) for kw in all_keywords)
    has_geo_intent = any(gk in " ".join(all_keywords) for gk in geo_keywords) or any(k.isdigit() for k in all_keywords)

    STOPWORDS = {"nearest", "near", "closest", "find", "where", "what", "are", "is", "to", "the"}
    
    keywords = [
        k for k in all_keywords 
        if k not in STOPWORDS and k != zipcode
    ]

    params = {"keywords": keywords}

    # =========================================================
    # 1. ENTITY LOOKUP (phone, website, etc.)
    # =========================================================
    if is_entity_lookup:
        entity_words = {"phone", "number", "contact", "address", "website", "url"}
        name_keywords = [k for k in keywords if k not in entity_words]
        params["keywords"] = name_keywords if name_keywords else keywords

        wants_phone = any(k in " ".join(all_keywords) for k in ["phone", "number", "contact"])
        wants_website = any(k in all_keywords for k in ["website", "url"])
        wants_address = any(k in all_keywords for k in ["address", "location"])

        if wants_phone:
            cypher = """
            MATCH (o:node)-[:phone_number_is]->(phone:node)
            WHERE ANY(k IN $keywords WHERE toLower(o.name) CONTAINS k)
            RETURN o.name AS subject, phone.name AS phone
            LIMIT 5
            """
        elif wants_website:
            cypher = """
            MATCH (o:node)-[:website_is]->(website:node)
            WHERE ANY(k IN $keywords WHERE toLower(o.name) CONTAINS k)
            RETURN o.name AS subject, website.name AS website
            LIMIT 5
            """
        elif wants_address:
            cypher = """
            MATCH (o:node)-[:is_located_at]->(address:node)
            WHERE ANY(k IN $keywords WHERE toLower(o.name) CONTAINS k)
            RETURN o.name AS subject, address.name AS object
            LIMIT 5
            """
        else:
            cypher = """
            MATCH (o:node)
            WHERE ANY(k IN $keywords WHERE toLower(o.name) CONTAINS k)
            OPTIONAL MATCH (o)-[:phone_number_is]->(phone:node)
            OPTIONAL MATCH (o)-[:website_is]->(website:node)
            RETURN o.name AS subject, phone.name AS phone, website.name AS website
            LIMIT 10
            """

    # =========================================================
    # 2. SERVICE SEARCH WITH LOCATION (must come before plain service search!)
    # =========================================================
    elif is_service_search and has_geo_intent:  
        if zipcode:
            cypher = """
            MATCH (s:node)-[r1:main_services_are]->(service_type:node)
            MATCH (s)-[r2:is_located_at]->(address:node)
            MATCH (s)-[r3:zipcode_is]->(zip:node)
            WHERE ANY(kw IN $keywords WHERE toLower(service_type.name) CONTAINS kw)
              AND (zip.name = $zipcode_str OR zip.name = $zipcode_str + '.0')
              AND address.name CONTAINS $zipcode_str
              AND NOT toLower(s.name) CONTAINS "support services"
            RETURN s.name AS subject,
                   'is_located_at' AS relationship,
                   address.name AS object
            LIMIT 25
            """
            params["zipcode_str"] = str(int(float(zipcode)))
        else:
            cypher = """
            MATCH (s:node)-[:main_services_are]->(service:node)
            MATCH (s)-[:is_located_at]->(loc:node)
            WHERE ANY(k IN $keywords WHERE 
                    ANY(svc IN service.name WHERE toLower(svc) CONTAINS k)
                  )
            RETURN s.name AS subject,
                   'is_located_at' AS relationship,
                   loc.name AS object
            LIMIT 25
            """

    # =========================================================
    # 3. SERVICE SEARCH (without location)
    # =========================================================
    elif is_service_search:
        cypher = """
        MATCH (o:node)-[:main_services_are]->(service:node)
        OPTIONAL MATCH (o)-[:other_services_are]->(other:node)
        OPTIONAL MATCH (o)-[:is_located_at]->(loc:node)
        WHERE ANY(k IN $keywords WHERE 
                ANY(svc IN service.name WHERE toLower(svc) CONTAINS k)
              )
        RETURN
            o.name AS subject,
            service.name AS main_service,
            other.name AS other_service,
            loc.name AS location
        LIMIT 25
        """

    # =========================================================
    # 4. FALLBACK
    # =========================================================
    else:
        cypher = """
        MATCH (o:node)
        WHERE ANY(k IN $keywords WHERE toLower(o.name) CONTAINS k)
        RETURN o.name AS subject
        LIMIT 10
        """

    print(f"DEBUG Cypher:\n{cypher}")
    print(f"DEBUG Params: {params}")

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
query = "What is the phone number of Youth Emergency Services (SafeHouse)?"

intent = extract_query_intent(query)

print(intent)

cypher, params = build_cypher(intent)
results = run_query(cypher, params)
print(f"DEBUG Results: {results}")  # Add this

##### LLM TWO #####

def generate_natural_response(user_query, triples):
    if not triples:
        return "I couldn't find any matching services for your request."

    seen = set()
    unique = []

    for t in triples:
        subject = t.get("subject") or t.get("n", {}).get("name")
        obj = t.get("object")  # This is the address!
        phone = t.get("phone")
        website = t.get("website")

        if not subject:
            continue

        key = (subject, obj)
        if key in seen:
            continue
        seen.add(key)

        unique.append({
            "subject": subject,
            "object": obj,
            "phone": phone,
            "website": website
        })

    facts = "\n".join(
        f"{i+1}. {t['subject']}"
        + (f" — Address: {t['object']}" if t.get("object") else "")
        + (f" — Phone: {t['phone']}" if t.get("phone") else "")
        + (f" — Website: {t['website']}" if t.get("website") else "")
        for i, t in enumerate(unique)
    )

    prompt = f"""You are a helpful social worker. A user asked: "{user_query}"

The database returned exactly {len(unique)} results. You MUST include ALL {len(unique)} of them in your response, numbered exactly as shown below. Do not add, remove, or reorder any entries. Do not determine which is nearest. Copy all addresses exactly as written.

Results:
{facts}

Write a brief intro sentence, then list all {len(unique)} results exactly as numbered above:"""
    
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "num_predict": 1000
        }
    )
    return response["message"]["content"].strip()

final_answer = generate_natural_response(query, results)
print(final_answer)