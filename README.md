# Multi-turn Medical Diagnosis Chatbot

This repository contains a multi-turn medical symptom assessment chatbot (research / demo quality) with both a ML-based backend and a rule-based/stage-1 backend, plus a Vite React frontend. The system demonstrates a confirmation-based conversational flow that extracts and confirms symptoms and (when enough symptoms are collected) returns a possible diagnosis and recommendations.

**Important:** This project is for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

**Contents**
- `medical-chatbot/` — ML-backed chatbot, symptom extractor, disease identifier, and a FastAPI server at `medical-chatbot/server.py`.
- `chatbot/` — Rule-based/stage-1 chatbot and an alternate FastAPI wrapper at `chatbot/server.py`.
- `frontend/` — Vite + React UI that communicates with the backend APIs.
- `models/`, `data/` (under `medical-chatbot/`) — Pretrained models and datasets used by the ML pipeline (may be large).

**Quick facts**
- Backend API (ML): `medical-chatbot/server.py` — exposes endpoints such as `/chats`, `/chats/{chat_id}`, `/chats/{chat_id}/stream`, `/health`.
- Backend API (rule-based): `chatbot/server.py` — alternate FastAPI server wrapping the rule-based `stage1_rulebot_v3` implementation.
- CLI mode: `python medical-chatbot\chatbot.py` (creates and runs a CLI chatbot for local testing).

**Prerequisites**
- Python 3.8+ (recommend 3.10+)
- Node.js (16+) and `npm` for the frontend
- (Optional) GPU & CUDA if you plan to run large transformer models locally

**Recommended Python packages (examples)**
The repository does not include a fully populated `requirements.txt`. Install the exact dependencies required by your use-case or create/refresh `requirements.txt`. Typical packages used in this project include:

```
fastapi uvicorn[standard] spacy torch transformers sentence-transformers scikit-learn pydantic
```

Adjust as needed (some environments may require specific versions). If you already have a `requirements.txt`, use it:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is empty or missing modules, install the packages above manually:

```
pip install fastapi uvicorn[standard] spacy torch transformers sentence-transformers scikit-learn pydantic
```

**Frontend setup (Windows `cmd.exe`)**

```
cd frontend
npm install
npm run dev
```

The Vite dev server typically runs on `http://localhost:5173`. The frontend expects the backend API to be available at `http://localhost:8000` (CORS is configured for `localhost:3000` and `5173` in the backend servers).

**Run the ML-backed backend (recommended)**
This starts the FastAPI server in `medical-chatbot` which initializes the ML components (symptom extractor, disease identifier).

```
cd medical-chatbot
python server.py
```

Server will start on `http://0.0.0.0:8000` by default and provides interactive API docs at `http://0.0.0.0:8000/docs`.

API highlights:
- `GET /` — basic health message
- `GET /health` — health + component readiness
- `POST /chats` — create a new chat session (returns `id`)
- `POST /chats/{chat_id}` — send a message to a session
- `POST /chats/{chat_id}/stream` — streaming response (SSE)
- `GET /chats/{chat_id}` — inspect session state
- `POST /chats/{chat_id}/restart` — reset session

**Run the rule-based backend (alternate)**

```
cd chatbot
python server.py
```

This server provides a lighter rule-based alternative useful for testing and comparisons.

**Run CLI chatbot (local testing)**

```
python medical-chatbot\chatbot.py
```

This runs the `MedicalChatbot` in an interactive CLI mode (loads models from `medical-chatbot/models/*`).

**Models & Data**
- Large model files and pretrained assets are stored under `medical-chatbot/models/` and `medical-chatbot/data/`. These may not be fully included in the repo due to size. Check the `models/` subfolders to confirm availability.
- If a model is missing, follow the original authors' instructions or the project notes to download required checkpoints (BioBERT, Sentence Transformers, spaCy models, etc.).

**Development notes**
- The ML backend initializes heavy components at startup (`MLSymptomExtractor`, `DiseaseIdentifier`) — expect longer cold-start times.
- Session data is kept in-memory (dictionary) for demo purposes. For production use, persist sessions in a DB or Redis.
- The frontend/server URLs and CORS are configured for local development; adjust `allow_origins` if deploying.

**Testing and evaluation**
- There are notebooks and scripts under `stage 2/`, `stage 3/`, `intent-classifier/`, and `reports/` for evaluation and experiments. See those folders for evaluation workflows.

**Contributing**
- Open an issue for bugs or feature requests.
- For code changes, create a branch, add tests (where applicable), and open a PR with a clear description.

**License & Attribution**
- This repository contains research/demo code. Add/confirm a LICENSE file as needed for redistribution.

**Contact / Authors**
- Project owner: repository `Solomon-Chan` (see repo metadata)

---
