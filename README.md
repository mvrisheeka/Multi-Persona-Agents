# 🧠 Multi-Persona Agent Studio
### *Five Expert Minds. One Unified Answer.*

Welcome to the **Expert Collective**, a high-performance, asynchronous multi-agent studio designed for deep reasoning and synthesized expert perspectives. Built with **LangGraph**, **FastAPI**, and **React**, this studio orchestrates five specialized personas to tackle complex queries in parallel.

---

## 🚀 Key Features

- **⚡ Parallel Surge Orchestration**: Unlike serial agents, our experts (Teacher, Senior, Parent, Friend, Counselor) fire simultaneously using asynchronous graph execution, reducing latency by up to 80%.
- **📽️ The Intelligence Deck**: A premium, carousel-based UI that allows you to flip through the "minds" of each agent as they generate.
- **🌊 Live Hyper-Streaming**: Experience zero-latency token streaming. Watch each agent think and type in real-time.
- **🛡️ Expert Quality Review**: A dedicated Moderator agent revises the primary expert's draft, ensuring professional-grade output before final synthesis.
- **📦 Cloud-Ready Dockerization**: Fully containerized setup with support for local **Ollama** or cloud **Groq** for sub-second responses.

---

## 🛠️ Technical Stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **Backend**: FastAPI (Python 3.11)
- **Frontend**: React + Vite (Vanilla CSS)
- **LLM Engine**: Ollama (Local) / Groq (Production)
- **Deployment**: Docker & Docker Compose

---

## 🏎️ Getting Started

### 1. Local Mode (Ollama)
Ensure you have [Ollama](https://ollama.com) running with the `llama3` model installed.

```bash
# Clone the repository
git clone https://github.com/aryannlol/multi-persona-agent.git
cd multi-persona-agent

# Launch the full stack
docker-compose up --build
```
*Frontend will be at `http://localhost:3000`, Backend at `http://localhost:8000`.*

### 2. Production Mode (Groq)
For sub-second performance without a local GPU, use Groq:

```bash
export GROQ_API_KEY=your_api_key_here
docker-compose up --build
```

### 3. Manual Local Setup (Without Docker)
If you prefer to run the services individually without Docker:

#### Prerequisites
- Python 3.10 or 3.11
- Node.js (v18+)
- Ollama (with `llama3` downloaded via `ollama run llama3`)

#### Setup Environment Variables
Copy the template environment file in the root directory:
```bash
cp .env.example .env
```
Fill in your `GROQ_API_KEY` (if using cloud mode) or `LANGSMITH_API_KEY` (if using LangSmith tracing).

#### Run the Backend
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python server.py
```
*Backend runs at `http://localhost:8000`*

#### Run the Frontend
```bash
cd frontend
npm install
npm run dev
```
*Frontend runs at `http://localhost:5173` (or the port specified by Vite)*

---

## 📂 Project Structure

```
├── backend/
│   ├── data/                 # Training dataset (GroundTruth) and local evaluation logs
│   ├── kb/                   # RAG knowledge base files (PDF sources & Chroma vector store)
│   ├── models/               # Pre-trained classifier models (joblib format)
│   ├── scripts/              # Analytics, dataset splits, and CLI evaluation tools
│   ├── agent.py              # Main LangGraph pipeline definition
│   ├── router.py             # Supervisor intent classification and routing
│   ├── server.py             # FastAPI SSE endpoint
│   └── requirements.txt      # Python dependencies
│
├── frontend/
│   ├── src/                  # React source files (components, styling)
│   ├── Dockerfile
│   └── package.json
│
├── docker-compose.yml        # Full-stack orchestrator
└── .env.example              # Environment variables template
```

---

## 🧠 The Expert Collective

| Persona | Specialty | Tone |
| :--- | :--- | :--- |
| **Teacher** | Academic, Fact-Heavy | Structured & Educational |
| **Senior** | Industry Insights, Pragmatic | Professional Mentor |
| **Parent** | Guidance, Empathy | Warm & Grounded |
| **Counselor** | Emotional Intelligence | Calm & Reflective |
| **Friend** | Vibes, Slang, Real-Talk | Relatable & Energetic |

---

## 📜 Development Logs & Evals
The studio includes an industrial-grade evaluation system. Every query is logged in the `backend/data/evals/` directory with full latency metrics, judge rationales, and persona outputs.

---

## ⚡ Parallel Processing Optimization (Optional)
To enable Ollama to run multiple persona queries concurrently, configure the parallel parameter before launching Ollama:
```powershell
# Windows PowerShell
$env:OLLAMA_NUM_PARALLEL=5
ollama serve
```


Built with ❤️ by the Intelligence Deck Team.




