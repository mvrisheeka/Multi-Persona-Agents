import json
import joblib
import numpy as np
from langchain_ollama import ChatOllama
from semantic_router import SemanticRouter
semantic_router = SemanticRouter("data/GroundTruth_train.xlsx")

ROUTER_EVAL_MODE = False

llm = ChatOllama(model="llama3", temperature=0)

PERSONAS = ["teacher", "parent", "senior", "friend", "counselor"]
 
# Load trained model
  
try:
    model = joblib.load("models/router_model.pkl")
    mlb = joblib.load("models/router_labels.pkl")
    HAS_MODEL = True
    print("[OK] Trained router loaded.")
except Exception:
    HAS_MODEL = False
    print("[WARN] No trained router found.")
    
  
# ML router (MAIN)
  
def trained_router(query: str):
    if not HAS_MODEL:
        return None
    Y_pred = model.predict([query])
    labels = mlb.inverse_transform(np.array(Y_pred))[0]
    personas = list(labels)
    if personas:
        return {"personas": personas[:2], "use_kb": "teacher" in personas}
    return None

  

# role request
def detect_role_request(query: str):
    q = query.lower()

    role_words = ["friend", "teacher", "parent", "senior", "counselor"]

    for role in role_words:
        if f"as a {role}" in q or f"like a {role}" in q:
            return role

    return None

# Rule based 1st priority

def detect_emotional_intent(query: str) -> bool:
    q = query.lower()

    emotional_patterns = [
        "i failed",
        "i am sad",
        "i feel bad",
        "feel really bad",
        "i feel terrible",
        "i am stressed",
        "i am worried",
        "i am scared",
        "i messed up",
        "i did badly",
        "i did bad",
        "i'm depressed",
        "im depressed",
        "i am depressed",
        "i am anxious",
        "i feel anxious",
        "i feel lost",
        "i dont know what to do",
        "i don't know what to do",
        "i feel hopeless",
        "i feel low"
    ]

    return any(p in q for p in emotional_patterns)


def detect_decision_intent(query: str) -> bool:
    q = query.lower()

    decision_patterns = [
        "what should i do",
        "what do i do",
        "what next",
        "what now",
        "next step",
        "how do i move forward",
        "how to move forward",
        "what about my future",
        "what should i do next",
        "guide me",
        "career",
        "my future",
        "my next step",
        "how should i proceed"
    ]

    return any(p in q for p in decision_patterns)


def detect_guidance_intent(query: str):
    q = query.lower()

    guidance_words = [
        "how to improve",
        "how can i improve",
        "how to maintain",
        "how to manage",
        "how should i",
        "tips to",
        "tips for",
        "guide me",
        "suggest",
        "recommend",
        "routine",
        "habit",
        "discipline",
        "posture",
        "stay active",
        "productivity",
        "daily schedule",
        "time management",
        "plan my day",
        "choose subjects",
        "choose elective",
        "busy mornings",
        "keep healthy",
        "lifestyle",
        "improve myself"
    ]

    return any(w in q for w in guidance_words)


def detect_future_uncertainty(query: str):
    q = query.lower()

    future_words = [
        "scope",
        "which field",
        "which branch",
        "which path",
        "what option",
        "options do i have",
        "after graduation",
        "after engineering",
        "higher studies",
        "masters or job",
        "what direction",
        "what path should i choose",
        "what should i choose",
        "what to do after",
        "my future"
    ]

    return any(w in q for w in future_words)

def rule_router(query: str):

    emotional = detect_emotional_intent(query)
    guidance = detect_guidance_intent(query)
    professional = detect_professional_prep(query)
    career = detect_career(query)
    lifestyle = detect_lifestyle(query)
    learning = detect_learning(query)
    casual = detect_casual(query)
    future = detect_future_uncertainty(query)


    personas = []

    #  Emotional (highest priority)
    if emotional:
        personas.append("counselor")

    # Career / professional growth
    if professional or career or future:
        personas.append("senior")

    # Life guidance & habits
    if guidance or lifestyle:
        personas.append("parent")

    # Academic learning
    # Only add teacher if NOT emotional guidance
    if learning and not emotional and not guidance:
        personas.append("teacher")

    # Casual conversation
    if casual and not personas:
        personas.append("friend")

    # finalize
    
    if not personas:
        return None

    # remove duplicates while preserving order
    unique = []
    for p in personas:
        if p not in unique:
            unique.append(p)

    return {
        "personas": unique[:2],
        "use_kb": "teacher" in unique
    }

def detect_learning(query: str):
    q = query.lower()
    learning_words = [
        "explain", "what is", "define", "how does", "concept",
        "example", "learn", "study", "topic", "question",
        "mcq", "syllabus", "data structure", "algorithm",
        "network", "dbms", "os", "machine learning"
    ]
    return any(w in q for w in learning_words)

def detect_professional_prep(query: str):
    q = query.lower()

    prep_words = [
        "skills",
        "skillset",
        "certification",
        "certifications",
        "resume",
        "cv",
        "portfolio",
        "interview",
        "interviews",
        "communication",
        "confidence",
        "job role",
        "career path",
        "software engineer",
        "data scientist",
        "placement preparation",
        "industry",
        "company expectations"
    ]

    return any(w in q for w in prep_words)

def detect_career(query: str):
    q = query.lower()
    career_words = [
        "job", "career", "placement", "interview",
        "future", "industry", "company", "resume",
        "use in jobs", "helps in placements"
    ]
    return any(w in q for w in career_words)


def detect_lifestyle(query: str):
    q = query.lower()
    lifestyle_words = [
        "eat", "food", "diet", "health", "routine",
        "daily", "sleep", "not sleeping", "habit",
        "schedule", "time management", "stay healthy"
    ]
    return any(w in q for w in lifestyle_words)


def detect_casual(query: str):
    q = query.lower()
    casual_words = ["bored", "talk to me", "chat", "idk", "nothing much","keep me company",
"stay with me",
"talk with me",
"be with me",
"chat with me"
]
    return any(w in q for w in casual_words)

# STRICT LLM FALLBACK(not using anymore , after adding semantic routing)

def llm_router(query: str):
    prompt = f"""
You are an intent classifier for a multi-persona assistant.

Pick the best ONE or TWO personas.

Personas:
- teacher → academic, technical, exams
- senior → career, future guidance
- counselor → emotional support
- friend → casual conversation
- parent → family-style advice

Return STRICT JSON only:
{{"personas": ["p1", "p2"], "use_kb": true/false}}

Query:
{query}
"""

    try:
        raw = llm.invoke(prompt).content.strip()
        result = json.loads(raw)
        personas = result.get("personas", [])
        use_kb = bool(result.get("use_kb", False))
    except Exception:
        personas = []
        use_kb = False

    if personas:
        return {"personas": personas[:2], "use_kb": use_kb}

    return {"personas": ["friend"], "use_kb": False}


# MAIN

def classify_intent(query: str):
    role = detect_role_request(query)
    if role:
        return {
            "personas": [role],
            "use_kb": role == "teacher",
            "role_locked": True
        }

    emotional = detect_emotional_intent(query)
    decision = detect_decision_intent(query)

    # Emotional + decision → counselor + senior
    if emotional and decision:
        return {
            "personas": ["counselor", "senior"],
            "use_kb": False
        }

    # Emotional only → counselor
    if emotional:
        return {
            "personas": ["counselor"],
            "use_kb": False
        }

    # Otherwise normal routing
    result = rule_router(query)
    if result:
        return result

    # trained ML prior
    result = trained_router(query)
    if result:
        return result

    # semantic fallback
    
    semantic_result = semantic_router.predict(query)

    if semantic_result is not None:
        persona, score = semantic_result
        print(f"[SEMANTIC ROUTER] matched -> {persona} (score={score:.2f})")

        return {
            "personas": [persona],
            "use_kb": persona == "teacher"
        }

    # final fallback (LLM)
    
    if ROUTER_EVAL_MODE:
        # during evaluation do NOT call LLM
        return {
            "personas": ["friend"],
            "use_kb": False
        }

    return llm_router(query)
