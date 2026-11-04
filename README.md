# 🤖 TalentMatch AI – Intelligent Recruitment & Candidate Matching

TalentMatch AI is an AI-powered recruitment support application that helps HR professionals discover and evaluate candidates against job descriptions using semantic search, LLM-based evaluation, and evidence verification.

The application combines AI-driven candidate assessment with human-in-the-loop workflow orchestration to support structured and informed recruitment decisions.

## ✨ Key Features

- 🔎 **Semantic Candidate Search** – Retrieves relevant candidates using vector-based similarity search.
- 🧠 **AI-Based Evaluation** – Evaluates candidate suitability against job requirements using LLMs.
- 📄 **Resume Evidence Retrieval** – Uses relevant resume information to support candidate assessments.
- ✅ **Evidence Verification** – Verifies evaluation claims against source resume content.
- 📊 **Suitability Scoring** – Generates an overall candidate suitability score out of 10.
- 👤 **Human-in-the-Loop** – Keeps HR involved in reviewing and approving recruitment decisions.
- 📧 **Email Communication** – Supports HR-approved candidate communication.
- 📜 **Run History** – Maintains recruitment workflow records.

## 📊 Candidate Evaluation Approach

TalentMatch AI uses an LLM-based evaluation approach rather than predefined weighted scoring.

- Evaluates candidate suitability against the job description.
- Assigns an overall suitability score out of 10.
- Provides an explanation supported by retrieved resume information.
- Prevents unsupported evaluation claims through evidence verification.

**Note:** The score is a decision-support indicator. Final recruitment decisions remain with HR.

## 🏗️ System Architecture

```text
              👤 HR User
                  |
                  v
             Streamlit UI
                app.py
                  |
                  v
          UI & Session Layer
              src/ui/
                  |
                  v
          LangGraph Workflow
             src/graph/
                  |
                  v
        Recruitment Workflow
                  |
        +---------+---------+
        |         |         |
        v         v         v
      SQLite   ChromaDB   Gemini
     Database  Semantic   AI Evaluation
               Search
        |         |         |
        +---------+---------+
                  |
                  v
        Evidence Verification
                  |
                  v
       Suitability Score /10
                  |
                  v
          HR Review & Approval
                  |
                  v
          Email Communication
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | Web application interface |
| LangGraph | Workflow orchestration |
| Google Gemini | AI-based candidate evaluation |
| Groq | Configured fallback LLM provider |
| Sentence Transformers | Text embeddings |
| ChromaDB | Semantic vector search |
| SQLite | Structured data storage |
| Pydantic | Data validation |
| PyMuPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| SMTP | Email communication |

## 📁 Project Structure

```text
AI Recruitment/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   ├── ui/          # Streamlit interface
│   ├── services/    # Application business logic
│   ├── graph/       # LangGraph workflow
│   ├── schemas/     # Pydantic schemas
│   └── prompts/     # LLM evaluation prompts
│
├── data/            # Local sample data
├── storage/         # Local databases and vector store
├── utilities/       # Supporting utilities
└── tests/           # Automated and manual tests
```

## 🚀 Local Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/TalentMatch-AI.git
cd TalentMatch-AI
```

**2. Create and activate a virtual environment**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file using `.env.example` as a reference.

Configure the required credentials:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_SECONDARY_API_KEY`
- `GEMINI_SECONDARY_MODEL`
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_APP_PASSWORD`
- `SMTP_SENDER_NAME`

Never commit API keys, passwords, or the `.env` file to GitHub.

**5. Run the application**

```bash
streamlit run app.py
```

Open the application at `http://localhost:8501`.

## 🔐 Responsible AI & Human Oversight

- Candidate evaluations should be supported by resume evidence.
- Unsupported information must not be treated as verified.
- Sensitive personal information should be protected and anonymized where applicable.
- AI-generated scores should not be the sole basis for recruitment decisions.
- HR retains final decision-making authority.

---

⭐ **TalentMatch AI – Making recruitment more intelligent, structured, and evidence-driven.**
