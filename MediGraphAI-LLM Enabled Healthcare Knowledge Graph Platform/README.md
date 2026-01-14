
# MediGraph AI  
### Intelligent Healthcare Knowledge Graph with LLM Integration

## 👤 Author
**Aravind Balaji**

---

## 📌 Project Overview
MediGraph AI is an intelligent healthcare knowledge graph platform that integrates structured clinical data with graph databases and Large Language Models (LLMs) to enable advanced clinical analytics, semantic querying, and AI-assisted insights.

The system models patients, encounters, conditions, medications, providers, and observations as interconnected graph entities, enabling rich relationship-driven analysis beyond traditional relational systems.

---

## 🏗️ Architecture & Tech Stack
- **Backend / Data**: Snowflake, Neo4j AuraDB
- **Graph Modeling**: Patient–Encounter–Condition–Medication–Observation graph schema
- **AI / LLM**: Graph-aware querying, semantic retrieval, analytics
- **Frontend**: Streamlit
- **Languages**: Python, SQL
- **Standards**: Healthcare-inspired data modeling (FHIR-aligned concepts)

---

## 📊 Data Scale
### Snowflake Views
- Patients: 121  
- Encounters: 7,688  
- Conditions: 4,750  
- Medications: 7,455  
- Providers: 282  
- Observations: 112,177  

### Neo4j AuraDB Graph
- Patient nodes: 122  
- Encounter nodes: 7,689  
- Condition nodes: 202  
- Medication nodes: 144  
- Provider nodes: 282  
- Observation nodes: 69,891  
- **Total relationships**: 170,189  

---

## 🔬 Key Features
- End-to-end ETL from relational to graph systems
- Rich healthcare knowledge graph construction
- Graph analytics on patient journeys
- Observation-level clinical insights
- LLM-assisted semantic querying over graph data
- Qualitative & quantitative graph analytics

---

## 🚀 How to Run Locally

```bash
# Clone repository
git clone https://github.com/<your-username>/MediGraphAI.git
cd MediGraphAI

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📈 Use Cases
- Clinical analytics & patient trajectory analysis
- AI-powered healthcare Q&A
- Knowledge-graph-driven decision support
- Research & experimentation with GraphRAG

---

## 📝 Ownership
This repository is fully maintained and implemented by **Aravind Balaji** as a personal project.
