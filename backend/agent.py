from typing import Dict, List, TypedDict, Annotated, Optional, Any
import asyncio
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from router import classify_intent  
import sys
import re
import json
import time
import os
from langchain_groq import ChatGroq
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchResults
import operator
import warnings

# Silence the Python REPL warning
warnings.filterwarnings("ignore", category=UserWarning, module='langchain_experimental')

ALL_PERSONAS = ["teacher", "parent", "senior", "friend", "counselor"]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DUAL PROMPT SYSTEM â€” FULL REWRITE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SYSTEM_PREFIX = """
Respond in the same language the user writes in.
Do not begin responses with filler phrases like 'Certainly!', 'Of course!', 'Great question!', or 'Absolutely!'.
Do not hallucinate facts. If you are unsure, say so directly.
Do not repeat the user's question back to them.
Keep responses focused. For technical/explanatory prompts, provide enough depth to be genuinely useful.
"""

NORMAL_PROMPTS = {
    "teacher": """You are a knowledgeable teacher explaining to a curious student.
Be clear, structured, and accurate. Use simple analogies when a concept is abstract.
If the question is explicitly about software or engineering, use this format:
1) One-line definition
2) Real-world workflow (step-by-step)
3) Security/engineering best practices
4) Common mistakes
5) Short practical example (code/config/pseudocode when relevant)
6) Crisp summary

If the question is NOT about software or engineering, ignore the format above and just provide a natural, structured, and helpful explanation without any code snippets or technical checklists.
If the question is ambiguous, answer the most likely interpretation and note your assumption.""",

    "senior": """You are a senior student or young professional giving advice to someone a few steps behind you.
Be direct and practical. Prioritize what actually matters over what sounds good on paper.
Give concrete next actions, realistic trade-offs, and one thing to avoid.
Use examples from real work/study life, not abstract theory.
If the question has no clean answer, say so and explain why.
No motivational fluff or generic lines.""",

    "parent": """You are a caring but firm parent figure responding to someone you genuinely want to see succeed.
Lead with warmth, but do not sugarcoat. If the user is making a bad decision, say so once clearly and kindly.
Focus on immediate practical guidance, routines, and boundaries.
Use short, grounded advice that can be acted on today.
Do not moralize repeatedly. One honest nudge is enough.
Ask a follow-up only if truly needed.""",

    "friend": """You are a close, honest friend who happens to know a lot.
Talk like a real person: casual, direct, emotionally present.
Match the user's energy: if brief, be brief; if deep, go deep.
Give a real opinion and one practical suggestion when helpful.
If the user seems stressed, acknowledge it naturally first.
No corporate tone, no templated pep-talks, no bullet points.""",

    "counselor": """You are an empathetic counselor focused on emotional support.
First, make the user feel heard in specific language.
Then offer one gentle reframe and one small coping action.
Ask one focused follow-up question only when it adds value.
Do not overwhelm with advice.
Avoid clinical jargon. Keep the tone warm, human, and unhurried."""
}
ADVANCED_PROMPTS = {
    "teacher": """You are a rigorous academic authority and subject matter expert.
When explaining concepts:
- Start with the core principle, then build outward with supporting logic
- Use precise terminology and define it when first introduced
- Connect ideas to established theories or frameworks where relevant
- Anticipate the follow-up question and address it preemptively

If the prompt is EXPLICITLY about software engineering or programming, include:
- Terminology clarification (if people commonly confuse terms)
- Real-world system flow (numbered steps)
- Security/reliability checklist
- At least one implementation snippet
- What goes wrong in production pitfalls

CRITICAL: Do NOT include code snippets, Python scripts, or security checklists if the topic is about life advice, general education, or anything unrelated to software.
Tone: intellectually serious, confident, not condescending.
If you are uncertain about something, say so explicitly rather than guessing.""",

    "senior": """You are an experienced industry mentor with years of real-world experience.
Go beyond textbook advice:
- Share the unwritten rules most people learn the hard way
- Give specific, actionable steps, not vague direction
- Call out common mistakes people make at this stage
- Include at least one Pro Tip that is not obvious or generic
- Mention what separates average performers from strong ones
- Give a practical 7-day or 30-day action plan when relevant
Tone: direct, experienced, peer-level. No hand-holding, no fluff.
If the question is too vague to give good advice, ask one clarifying question.""",

    "parent": """You are a wise mentor-parent focused on long-term growth and character.
When responding:
- Connect the immediate question to a bigger-picture life lesson when genuinely relevant
- Focus on resilience, discipline, and self-awareness
- Be specific: name the habit, mindset shift, or action
- Acknowledge the emotional side before giving direction
- Challenge the user to take ownership without shaming them
- End with one concrete routine the user can start today
Tone: warm, firm, deeply invested.
Do not moralize more than once. Trust the user to hear you.""",

    "friend": """You are the user's most trusted, sharp, and loyal best friend.
In advanced mode you go deeper but stay real:
- Use casual language, mild humor, and genuine reactions where natural
- Give your honest, unfiltered take
- Push back if the user is being too hard on themselves or making an obvious mistake
- If the topic is heavy, hold space briefly before advice
- Match the user's energy
- Keep it vivid and specific, not generic encouragement
Tone: warm, honest, zero pretense.
Never use headers or bullet points. Just talk like a person.""",

    "counselor": """You are a clinical psychologist with training in CBT, mindfulness, and trauma-informed care.
Follow this approach:
1. Validate the user's emotion explicitly and specifically
2. Identify likely cognitive distortions gently
3. Offer a gentle cognitive reframe
4. Suggest one concrete coping technique
5. End with an open, non-pressuring reflection question
6. Keep response focused and avoid overload
Tone: calm, professional, deeply human.
Never minimize what the user is feeling.
If the user expresses self-harm risk: acknowledge directly and provide crisis resources first."""
}
import requests

class MwmblSearch:
    def run(self, query: str) -> str:
        try:
            # Added timeout to prevent hanging the whole agent
            resp = requests.get(f"https://mwmbl.org/search?q={query}", timeout=5)
            return resp.text[:2000] # Cap the context
        except:
            return "Search failed or timed out."

    async def arun(self, query: str) -> str:
        return await asyncio.to_thread(self.run, query)

# Tools
python_repl = PythonREPL()
web_search = MwmblSearch() # Switched to Mwmbl



def merge_dicts(a: Dict, b: Optional[Dict]):
    if b is None:
        return a
    a = dict(a)
    a.update(b)
    return a

class State(TypedDict):
    query: str
    history: List[str]
    intent_personas: List[str]
    priority_personas: List[str]
    use_kb: bool
    selected_personas: List[str]
    max_agents: int
    generation_mode: str
    response_style: str
    persona_outputs: Annotated[Dict[str, str], merge_dicts]
    answer: str
    confidence: Annotated[float, lambda a, b: b]
    confidence_label: Annotated[str, lambda a, b: b]
    judge_rationale: str
    can_proceed: Annotated[bool, lambda a, b: a or b] # OR logic: once True, stay True
    gen_latency: Annotated[float, lambda a, b: b]
    judge_latency: Annotated[float, lambda a, b: b]
    node_metrics: Annotated[Dict[str, Any], merge_dicts]
    reviewed_priority_output: str
    focus_persona: Optional[str]

# LLMs
# Production Mode: Check for GROQ_API_KEY for sub-second responses in cloud
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    CURRENT_MODEL_NAME = "llama3-70b-8192"
    print(f"\033[92m[PRODUCTION MODE]\033[0m Using Groq ({CURRENT_MODEL_NAME}) for high-performance agentic synthesis.")
    llm = ChatGroq(model=CURRENT_MODEL_NAME, temperature=0, streaming=True)
    llm_judge = ChatGroq(model=CURRENT_MODEL_NAME, temperature=0, streaming=False)
else:
    CURRENT_MODEL_NAME = "llama3"
    print(f"\033[94m[LOCAL MODE]\033[0m Using Ollama ({CURRENT_MODEL_NAME}) for local processing.")
    llm = ChatOllama(model=CURRENT_MODEL_NAME, temperature=0, streaming=True)
    llm_judge = ChatOllama(model=CURRENT_MODEL_NAME, temperature=0, streaming=False)

def get_streaming_llm():
    if GROQ_API_KEY:
        return ChatGroq(model="llama3-70b-8192", temperature=0, streaming=True)
    return ChatOllama(model="llama3", temperature=0, streaming=True)




# Initiating states
  
async def init_state(state: State):
    return {
        "query": state.get("query", ""),
        "history": state.get("history", []),
        "intent_personas": [],
        "priority_personas": [],
        "use_kb": False,
        "persona_outputs": {},
        "selected_personas": ALL_PERSONAS[:], 
        "answer": "",
        "confidence": 0.0,
        "confidence_label": "Not evaluated",
        "gen_latency": 0.0,
        "judge_latency": 0.0,
        "node_metrics": {},
        "reviewed_priority_output": ""
    }
  
# ROUTER NODE
  
async def router_node(state: State):
    result = await asyncio.to_thread(classify_intent, state["query"])
    personas = result.get("personas", [])
    use_kb = result.get("use_kb", False)
    print("[ROUTER] personas:", personas, "| use_kb:", use_kb)
    return {
        "intent_personas": personas[:2],
        "use_kb": use_kb
    }
  
# PERSONA AGENTS 

# teacher persona

async def persona_teacher(state: State, config: RunnableConfig):
    start_node = time.time()
    q = state["query"]
    priority = state.get("priority_personas", [])
    is_priority = (priority and priority[0] == "teacher")
    
    system_prompt = ADVANCED_PROMPTS["teacher"] if is_priority else NORMAL_PROMPTS["teacher"]
    mode = state.get("generation_mode", "medium")
    length_rules = get_length_instruction(mode)
    
    tool_output = ""
    agent_llm = get_streaming_llm()

    # Agentic Thinking for Priority
    if is_priority:
        print("\033[92m[TEACHER AGENT] Starting research and planning...\033[0m")
        raw_tool = await web_search.arun(q) 
        if "failed or timed out" not in raw_tool:
            tool_output = raw_tool
        else:
            print("\033[93m[TEACHER AGENT] Search failed, relying on internal knowledge.\033[0m")
    else:
        tool_output = ""
        
    link_footer = f"\n\nSources Found:\n{tool_output}" if "http" in tool_output else ""

    # Stream response if possible
    print(f"\033[92m[TEACHER AGENT]\033[0m [{start_node:.3f}] Starting generation for: {q[:50]}...")
    full_ans = ""
    # Get token_queue from config sidecar
    tq = config.get("configurable", {}).get("token_queue")
    
    try:
        async for chunk in agent_llm.astream(f"{SYSTEM_PREFIX}\n{system_prompt}\n{length_rules}\n{tool_output}\nTopic: {q}\nResponse:"):
            content = chunk.content
            full_ans += content
            if tq:
                await tq.put({"type": "token", "id": "teacher", "text": content})
        print(f"\033[92m[TEACHER AGENT]\033[0m Finished ({len(full_ans)} chars).")
    except Exception as e:
        print(f"\033[91m[TEACHER ERROR]\033[0m {e}")
        full_ans = "I'm having trouble connecting to my knowledge base right now. Please try again."

    return {
        "persona_outputs": {"teacher": full_ans + link_footer},
        "node_metrics": {"teacher": {"latency": time.time() - start_node}}
    }

# senior reviewer node
async def senior_reviewer(state: State):
    start = time.time()
    outputs = state.get("persona_outputs", {})
    if "teacher" not in outputs:
        return {}
        
    teacher_draft = outputs["teacher"]
    
    system = ADVANCED_PROMPTS["senior"]
    review = (await llm.ainvoke(
        f"""
{SYSTEM_PREFIX}
{system}

Teacher's Academic Draft: 
{teacher_draft}

Your Task:
1. Validate the facts but REWRITE the introduction to be more 'industry-direct'.
2. Add a specific 'Senior Pro-Tip' section at the end.
3. Ensure the tone is peer-level mentoring.

Revised Agent Output:
"""
    )).content.strip()
    
    return {
        "persona_outputs": {
            "senior": review 
        },
        "node_metrics": {
            "senior_review": {
                "latency": time.time() - start
            }
        }
    }

# Remove old senior_reviewer if it exists as a node
# builder.delete_node("senior_review") # (Pseudo-code, handled by re-building below)

#parent persona

async def persona_parent(state: State, config: RunnableConfig):
    start = time.time()
    q = state["query"]
    priority = state.get("priority_personas", [])
    is_priority = (priority and priority[0] == "parent")
    
    system_prompt = ADVANCED_PROMPTS["parent"] if is_priority else NORMAL_PROMPTS["parent"]
    mode = state.get("generation_mode", "medium")
    length_rules = get_length_instruction(mode)
    
    agent_llm = get_streaming_llm()
    # Agentic Thinking for Priority
    if is_priority:
        print(f"\033[94m[PARENT AGENT]\033[0m [{start:.3f}] Reflecting on wisdom and tone...")
        if any(w in q.lower() for w in ["how to", "advice", "help"]):
            search_context = await web_search.arun(q)
            q = f"{q}\n\n[External Context]: {search_context}"

    full_ans = ""
    tq = config.get("configurable", {}).get("token_queue")
    async for chunk in agent_llm.astream(f"{SYSTEM_PREFIX}\n{system_prompt}\n{length_rules}\nChild's Query: {q}\nParental Advice:"):
        content = chunk.content
        full_ans += content
        if tq:
            await tq.put({"type": "token", "id": "parent", "text": content})

    invite = f"\n\n(I'm always here if you want to talk more about this. We can work through it together.)"

    return {
        "persona_outputs": {"parent": full_ans + invite},
        "node_metrics": {"parent": {"latency": time.time() - start}}
    }

#senior persona

async def persona_senior(state: State, config: RunnableConfig):
    start = time.time()
    q = state["query"]
    priority = state.get("priority_personas", [])
    is_priority = (priority and priority[0] == "senior")
    system_prompt = ADVANCED_PROMPTS["senior"] if is_priority else NORMAL_PROMPTS["senior"]

    agent_llm = get_streaming_llm()
    # Agentic Thinking for Priority
    if is_priority:
        print("\033[93m[SENIOR AGENT] Analyzing industry standards and pitfalls...\033[0m")

    mode = state.get("generation_mode", "medium")
    length_rules = get_length_instruction(mode)
    print(f"\033[93m[SENIOR AGENT]\033[0m [{start:.3f}] Starting generation for: {q[:50]}...")
    full_ans = ""
    tq = config.get("configurable", {}).get("token_queue")
    try:
        async for chunk in agent_llm.astream(f"{SYSTEM_PREFIX}\n{system_prompt}\n{length_rules}\nJunior's Question: {q}\nSenior's Guidance:"):
            content = chunk.content
            full_ans += content
            if tq:
                await tq.put({"type": "token", "id": "senior", "text": content})
        print(f"\033[93m[SENIOR AGENT]\033[0m Finished.")
    except Exception as e:
        print(f"\033[91m[SENIOR ERROR]\033[0m {e}")
        full_ans = "I'm having some trouble right now. Let's try again in a bit."

    invite = f"\n\n(I've seen this before. If you want to talk about the long-term career impact, I'm here.)"

    return {
        "persona_outputs": {"senior": full_ans + invite},
        "node_metrics": {"senior": {"latency": time.time() - start}}
    }

#friend persona

async def persona_friend(state: State, config: RunnableConfig):
    start = time.time()
    q = state["query"]
    priority = state.get("priority_personas", [])
    is_priority = (priority and priority[0] == "friend")
    system_prompt = ADVANCED_PROMPTS["friend"] if is_priority else NORMAL_PROMPTS["friend"]

    agent_llm = get_streaming_llm()
    # Agentic Thinking for Priority
    if is_priority:
        print("\033[96m[FRIEND AGENT] Matching user energy and vibes...\033[0m")

    # Short acknowledgment guard for focused chat turns.
    # Prevents over-replying to messages like "ok thanks".
    q_clean = q.strip().lower()
    ack_phrases = {
        "ok", "okay", "ok thanks", "okay thanks", "thanks", "thank you",
        "got it", "cool", "alright", "kk", "k"
    }
    if q_clean in ack_phrases:
        short_reply = "Anytime. I am here whenever you want to continue."
        return {
            "persona_outputs": {"friend": short_reply},
            "node_metrics": {"friend": {"latency": time.time() - start, "mode": "ack"}}
        }

    print(f"\033[96m[FRIEND AGENT]\033[0m [{start:.3f}] Starting generation...")
    full_ans = ""
    tq = config.get("configurable", {}).get("token_queue")
    try:
        async for chunk in agent_llm.astream(f"{SYSTEM_PREFIX}\n{system_prompt}\nUser: {q}\nReply:"):
            content = chunk.content
            full_ans += content
            if tq:
                await tq.put({"type": "token", "id": "friend", "text": content})
        print(f"\033[96m[FRIEND AGENT]\033[0m Finished.")
    except Exception as e:
        print(f"\033[91m[FRIEND ERROR]\033[0m {e}")
        full_ans = "Sorry, I'm a bit lost right now. Can you say that again?"

    invite = f"\n\n(NGL, I got plenty more to say on this if you're down!)"

    return {
        "persona_outputs": {"friend": full_ans + invite},
        "node_metrics": {"friend": {"latency": time.time() - start}}
    }

#counselor persona

async def persona_counselor(state: State, config: RunnableConfig):
    start = time.time()
    q = state["query"]
    priority = state.get("priority_personas", [])
    is_priority = (priority and priority[0] == "counselor")
    system_prompt = ADVANCED_PROMPTS["counselor"] if is_priority else NORMAL_PROMPTS["counselor"]

    agent_llm = get_streaming_llm()
    # Agentic Thinking for Priority
    if is_priority:
        print(f"\033[95m[COUNSELOR AGENT]\033[0m [{start:.3f}] Identifying emotional patterns...")

    mode = state.get("generation_mode", "short")
    length_rules = get_length_instruction(mode)
    full_ans = ""
    tq = config.get("configurable", {}).get("token_queue")
    async for chunk in agent_llm.astream(f"{SYSTEM_PREFIX}\n{system_prompt}\n{length_rules}\nUser's Concern: {q}\nCounselor's Response:"):
        content = chunk.content
        full_ans += content
        if tq:
            await tq.put({"type": "token", "id": "counselor", "text": content})

    invite = f"\n\n(I'm holding space for you. If you'd like to explore this feeling further, I'm here.)"

    return {
        "persona_outputs": {"counselor": full_ans + invite},
        "node_metrics": {"counselor": {"latency": time.time() - start}}
    }


#We add a small classifier inside the supervisor(detect_task_type)
#Your system will now detect 4 workloads:

def detect_task_type(query: str) -> str:
    q = query.lower()
    # generation 
    generation_patterns = [
        "give", "list", "generate", "make", "create",
        "prepare", "write", "questions", "mcq", "notes",
        "summarize", "summary", "important topics"
    ]
    if any(w in q for w in generation_patterns) and any(x in q for x in ["10", "20", "list", "questions", "topics"]):
        return "generation"
    
    # emotional_support 
    support_patterns = [
        "i feel", "i am stressed", "i'm stressed",
        "i failed", "i am sad", "worried",
        "anxious", "lonely", "scared"
    ]
    if any(w in q for w in support_patterns):
        return "support"

    # advice 
    advice_patterns = [
        "what should i do",
        "which should i choose",
        "career", "future",
        "after graduation",
        "suggest", "guide me",
        "next step", "path"
    ]
    if any(w in q for w in advice_patterns):
        return "advice"

    # default
    return "explanation"


# SUPERVISOR

async def supervisor_planner(state: State):
    start = time.time()
    query = state["query"]
    personas = state["intent_personas"]
    task_type = detect_task_type(query)
    # planning
    if task_type == "generation":
        generation_mode = "long"
        response_style = "detailed"

    elif task_type == "explanation":
        generation_mode = "medium"
        response_style = "normal"

    elif task_type == "advice":
        generation_mode = "medium"
        response_style = "normal"

    elif task_type == "support":
        generation_mode = "short"
        response_style = "concise"

    else:
        generation_mode = "medium"
        response_style = "normal"

    # PRIMARY persona controls advanced-prompt mode.
    # Dispatching is handled as full fan-out (all personas) unless explicitly focused.
    primary = personas[0] if personas else ("teacher" if task_type in ["explanation", "generation"] else "counselor")
    selected = ALL_PERSONAS[:]
    
    print(f"\033[93m[SUPERVISOR]\033[0m Task type: {task_type} | Mode: {generation_mode} | Running agents: {selected}")
    
    # Preserve router-driven priority ordering without artificial caps.
    # The first persona remains the ADVANCED/tool-thinking owner.
    priority_personas = []
    for p in [primary] + personas:
        if p in ALL_PERSONAS and p not in priority_personas:
            priority_personas.append(p)

    # Safety fallback when router gives no valid persona.
    if not priority_personas:
        priority_personas = ["teacher"]
    max_agents = len(selected)

    print("[SUPERVISOR]")
    print("task:", task_type)
    print("mode:", generation_mode)
    print("agents:", max_agents)
    print("priority:", priority_personas)

    return {
        "generation_mode": generation_mode,
        "response_style": response_style,
        "selected_personas": selected,
        "priority_personas": priority_personas,
        "node_metrics": {
            "supervisor": {
                "latency": time.time() - start,
                "task": task_type,
                "mode": generation_mode
            }
        }
    }

#to get LENGHT of the response

def get_length_instruction(mode: str) -> str:

    if mode == "short":
        return """
Response Rules:
- Max 4 sentences
- No paragraphs
- No bullet list
- Be concise and direct
"""

    if mode == "medium":
        return """
Response Rules:
- 8 to 12 sentences
- Use short paragraphs OR small bullet points
- Focus only on key ideas
"""
    if mode == "long":
        return """
Response Rules:
- Structured response
- Use headings or bullet points
- Provide detailed but organized explanation
"""

    return ""


def score_persona_outputs(state: State) -> Dict[str, float]:
    outputs = state.get("persona_outputs", {})
    priority = state.get("priority_personas", [])
    task_type = detect_task_type(state["query"])

    scores: Dict[str, float] = {}
    for persona, text in outputs.items():
        score = 0.2

        if persona in priority:
            # Boost score if this persona was predicted by the router
            score += max(0.3 - (0.1 * priority.index(persona)), 0.1)

        if len(text.split()) >= 25:
            score += 0.15

        if task_type == "support" and persona in {"counselor", "parent", "friend"}:
            score += 0.15
        elif task_type == "advice" and persona in {"senior", "senior_review", "parent"}:
            score += 0.15
        elif task_type in {"generation", "explanation"} and persona in {"teacher", "senior_review"}:
            score += 0.15

        scores[persona] = round(min(score, 1.0), 3)

    return scores


def confidence_from_scores(selected: List[str], priority: List[str], scores: Dict[str, float]) -> tuple[float, str]:
    if not scores:
        return 0.0, "No output"

    avg_score = sum(scores.values()) / len(scores)
    matched_priority = len(set(priority) & set(scores))
    priority_coverage = matched_priority / max(1, len(priority)) if priority else 1.0
    participation = len(scores) / max(1, len(selected))

    confidence = min(0.55 + (0.2 * avg_score) + (0.15 * priority_coverage) + (0.1 * participation), 0.99)

    if confidence >= 0.9:
        label = "Reliable answer"
    elif confidence >= 0.75:
        label = "Good quality answer"
    elif confidence >= 0.6:
        label = "Partial answer"
    else:
        label = "Needs review"

    return round(confidence, 2), label
  
# COMBINE(mainly used for combining multi-agent response only , when single persona responded, combine node is not used therfore)

async def combine_node(state: State, config: RunnableConfig):
    start_gen = time.time()
    outputs = state.get("persona_outputs", {})
    priority = state.get("priority_personas", ["teacher"])
    primary_persona = priority[0]
    reviewed_priority = (state.get("reviewed_priority_output") or "").strip()
    base_answer = reviewed_priority or outputs.get(primary_persona, "")

    # If routing has only one priority persona, synthesis is unnecessary.
    # Return the priority answer directly.
    if len(priority) <= 1:
        return {
            "answer": base_answer or outputs.get(primary_persona, ""),
            "gen_latency": time.time() - start_gen
        }
    
    if len(outputs) == 1:
        persona = list(outputs.keys())[0]
        return {
            "answer": base_answer or outputs[persona],
            "confidence": 1.0,
            "gen_latency": 0.01 
        }

    combined_input = "\n\n".join([f"[{p.upper()}]: {txt}" for p, txt in outputs.items()])
    
    sys = f"""
{SYSTEM_PREFIX}
You are the EXPERT MODERATOR. 
Your goal is to synthesize the following expert perspectives into a single, high-quality response.

CRITICAL RULE:
The query was primarily routed to the {primary_persona} persona. 
You MUST use the PRIORITY_BASE_ANSWER below as your anchor and preserve it almost entirely, including all LINKS, CODE, and specific advice.
Do NOT rewrite their answer from scratch. Instead, ENHANCE it by adding helpful 'Supporting Perspectives' or 'Pro Tips' from the other experts at the end.
The tone must match {primary_persona}.

At the very end of the response, provide a 'NEXT STEPS' section where you list each participating persona and a one-sentence hook for why the user should 'chat more' with them.
"""

    full_ans = base_answer
    tq = config.get("configurable", {}).get("token_queue")
    try:
        response = await asyncio.wait_for(
            llm.ainvoke(f"{sys}\n\nPRIORITY_BASE_ANSWER:\n{base_answer}\n\nInputs:\n{combined_input}"),
            timeout=60
        )
        full_ans = (response.content or "").strip() or full_ans
        if tq and full_ans:
            await tq.put({"type": "token", "id": "combined", "text": full_ans})
    except asyncio.TimeoutError:
        print("[COMBINE] Synthesis timed out after 60s, returning partial answer")
        if not full_ans:
            full_ans = base_answer
    except Exception as e:
        print(f"[COMBINE ERROR] {e}")
        if not full_ans:
            full_ans = base_answer
    
    return {
        "answer": full_ans or base_answer,
        "gen_latency": time.time() - start_gen
    }

async def evaluation_node(state: State):
    start_judge = time.time()
    ans = state.get("answer", "")
    q = state["query"]
    
    sys = "You are a quality judge. Evaluate the following response for accuracy, tone, and helpfulness. Output ONLY a JSON object with 'score' (1-10), 'label' (Poor/Good/Excellent), and 'rationale' (1 sentence)."
    
    try:
        response = (await asyncio.wait_for(llm.ainvoke(f"{sys}\n\nQuery: {q}\nAnswer: {ans}"), timeout=20)).content.strip()
        judge_time = time.time() - start_judge
        # Better JSON extraction
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            eval_data = json.loads(json_match.group())
            return {
                "confidence": eval_data.get("score", 0.0) / 10,
                "confidence_label": eval_data.get("label", "Unknown"),
                "judge_rationale": eval_data.get("rationale", ""),
                "judge_latency": judge_time,
                "node_metrics": {
                    "evaluation": {
                        "latency": judge_time,
                        "label": eval_data.get("label"),
                        "score": eval_data.get("total_score", eval_data.get("score", 0))
                    }
                }
            }
        else:
            scores = score_persona_outputs(state)
            fallback_confidence, fallback_label = confidence_from_scores(
                state.get("selected_personas", []),
                state.get("priority_personas", []),
                scores
            )
            return {
                "confidence": fallback_confidence,
                "confidence_label": fallback_label,
                "judge_rationale": f"Judge failed to output valid JSON. Raw output: {response[:100]}... Falling back to heuristic confidence.",
                "judge_latency": judge_time
            }
    except asyncio.TimeoutError:
        judge_time = time.time() - start_judge
        scores = score_persona_outputs(state)
        fallback_confidence, fallback_label = confidence_from_scores(
            state.get("selected_personas", []),
            state.get("priority_personas", []),
            scores
        )
        return {
            "confidence": fallback_confidence,
            "confidence_label": fallback_label,
            "judge_rationale": "Judge timed out after 20s. Falling back to heuristic confidence.",
            "judge_latency": judge_time
        }
    except Exception as e:
        judge_time = time.time() - start_judge
        print(f"[JUDGE ERROR] {e}")
        scores = score_persona_outputs(state)
        fallback_confidence, fallback_label = confidence_from_scores(
            state.get("selected_personas", []),
            state.get("priority_personas", []),
            scores
        )
        return {
            "confidence": fallback_confidence,
            "confidence_label": fallback_label,
            "judge_rationale": f"Evaluation failed: {str(e)}. Falling back to heuristic confidence.",
            "judge_latency": judge_time
        }

async def fan_in_node(state: State):
    """
    Acts as a synchronization barrier. 
    Only returns a signal to proceed when all selected personas have reported back.
    """
    selected = state.get("selected_personas", [])
    priority = state.get("priority_personas", ["teacher"])
    outputs = state.get("persona_outputs", {})

    # Finalization should not block indefinitely on non-priority laggards.
    # Required set = router-selected priority personas that were dispatched.
    required = [p for p in priority if p in selected] or selected[:1]

    # Proceed once all required personas are available.
    if all(p in outputs for p in required):
        return {"can_proceed": True}
    
    return {"can_proceed": False}

# Universal Quality Reviewer (Dynamic)
async def quality_reviewer(state: State, config: RunnableConfig):
    start = time.time()
    outputs = state.get("persona_outputs", {})
    priority = state.get("priority_personas", ["teacher"])
    leader = priority[0]
    
    if leader not in outputs: return {}
    
    draft = outputs[leader]
    print(f"\033[93m[QUALITY REVIEW]\033[0m Assessing {leader.upper()} output (read-only).")

    review_prompt = f"""
You are a strict quality reviewer. Evaluate this answer for:
1) clarity
2) relevance to user query
3) factual caution
4) tone consistency with persona

Return ONLY JSON with keys:
- clarity_score (1-10)
- relevance_score (1-10)
- factual_caution_score (1-10)
- tone_score (1-10)
- verdict ("pass" or "needs_attention")
- note (one short sentence)

User Query:
{state.get("query", "")}

Persona:
{leader}

Answer:
{draft}
"""

    quality = {}
    try:
        raw = (await asyncio.wait_for(llm_judge.ainvoke(review_prompt), timeout=20)).content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            quality = json.loads(match.group())
        else:
            quality = {"verdict": "needs_attention", "note": "quality review returned non-JSON output"}
    except asyncio.TimeoutError:
        quality = {"verdict": "needs_attention", "note": "quality review timed out after 20s"}
    except Exception as e:
        quality = {"verdict": "needs_attention", "note": f"quality review failed: {str(e)}"}

    return {
        # Preserve answer exactly as generated; review is metadata only.
        "reviewed_priority_output": draft,
        "node_metrics": {
            "reviewer": {
                "latency": time.time() - start,
                "leader": leader,
                "quality": quality
            }
        }
    }

# final decision of supervisor , what persona it selected
# final decision of supervisor , what persona it selected
async def persona_dispatcher(state: State):
    focus_persona = state.get("focus_persona")
    query = state.get("query", "").lower()

    # Explicit user override: "/dispatch" always fans out to all personas.
    if "/dispatch" in query:
        print("\033[95m[SUPERVISOR] /dispatch detected. Forcing full persona fan-out.\033[0m")
        return {"selected_personas": ALL_PERSONAS[:]}
    
    # Collective Deep Dive Logic
    if focus_persona == "combined":
        print("\033[95m[SUPERVISOR] Collective Deep Dive triggered. Boosting all agents to Extensive Mode...\033[0m")
        return {
            "selected_personas": ALL_PERSONAS,
            "priority_personas": state.get("priority_personas", ["teacher"]),
            "generation_mode": "long", 
            "use_kb": True
        }

    if focus_persona:
        selected = [focus_persona]
    else:
        selected = state.get("selected_personas", ALL_PERSONAS[:])

    print("[DISPATCH] ->", selected)
    return {"selected_personas": selected}

# building graph
builder = StateGraph(State)

builder.add_node("init", init_state)
builder.add_node("router", router_node)
builder.add_node("teacher", persona_teacher)
builder.add_node("parent", persona_parent)
builder.add_node("senior", persona_senior)
builder.add_node("friend", persona_friend)
builder.add_node("counselor", persona_counselor)
builder.add_node("supervisor", supervisor_planner)
builder.add_node("dispatcher", persona_dispatcher)
builder.add_node("quality_review", quality_reviewer)
builder.add_node("fan_in", fan_in_node)
builder.add_node("combine", combine_node)
builder.add_node("evaluation", evaluation_node)

# entry point
builder.set_entry_point("init")

# execution flow
builder.add_edge("init", "router")
builder.add_edge("router", "supervisor")
builder.add_edge("supervisor", "dispatcher")

# FAN-OUT: Trigger each selected expert in parallel
# We use a single conditional edge that returns a list of persona nodes to trigger
def route_to_experts(state: State):
    return state.get("selected_personas", [])

builder.add_conditional_edges(
    "dispatcher",
    route_to_experts,
    {p: p for p in ALL_PERSONAS}
)

# All experts lead back to the Fan-In Node
for p in ALL_PERSONAS:
    builder.add_edge(p, "fan_in")

# Fan-In conditionally leads to Quality Reviewer ONLY when all are done
builder.add_conditional_edges(
    "fan_in",
    lambda state: "proceed" if state.get("can_proceed") else "wait",
    {
        "proceed": "quality_review",
        "wait": END
    }
)

# Reviewer leads to final synthesis
builder.add_edge("quality_review", "combine")
builder.add_edge("combine", "evaluation")
builder.add_edge("evaluation", END)


# Graph compile
graph = builder.compile()
