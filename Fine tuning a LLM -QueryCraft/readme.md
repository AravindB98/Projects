# 🔮 QueryCraft

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **RAG-Enhanced SQL Generation: Fine-Tuning TinyLlama-1.1B for Natural Language to SQL Translation**

---

## 📋 Project Overview

QueryCraft is a **Retrieval-Augmented Generation (RAG)** system that converts natural language questions into executable SQL queries. Built on the Spider benchmark dataset (7,000 examples across 160 database schemas), it combines:

- **Fine-tuned LLM** using QLoRA (4-bit quantization + LoRA adapters)
- **RAG pipeline** for schema-aware context retrieval
- **Interactive Gradio demo** with 3 databases and live SQL execution (dropdown + natural language modes)

### 🎯 Key Results

| Metric | Without RAG | With RAG | Improvement |
|--------|-------------|----------|-------------|
| Component Match | 18.76% | 19.37% | **+3.2%** |
| Valid SQL Rate | 96.00% | 82.00% | — |
| Training Loss (Best) | — | — | **0.7436** |

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Exact Match | 0.0% | 8.0% | +8.0% |
| Has SELECT | 0.0% | 100.0% | **+100.0%** |
| Has FROM | 0.0% | 94.0% | **+94.0%** |
| WHERE Match | 52.0% | 72.0% | +20.0% |

---

## 🏗️ Architecture

```
User Question ──► Embedding ──► Vector Search ──► Schema Context
                     │              │                   │
             [MiniLM-L6-v2]    [ChromaDB/FAISS]        │
                                                       ▼
                                              Combined Prompt
                                              [INST] Schema + Question [/INST]
                                                       │
                                              ┌────────▼────────┐
                                              │  Fine-tuned LLM │
                                              │  (QLoRA 4-bit)  │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              Generated SQL ──► SQLite ──► Results
```

---

## 📁 Repository Structure

```
QueryCraft/
├── 📓 notebooks/
│   ├── QueryCraft_Day1_DataPrep.ipynb         # Dataset loading, preprocessing, RAG setup
│   ├── QueryCraft_Day2_FineTuning.ipynb       # Fine-tuning with 3 HP configs (A, B, C)
│   ├── QueryCraft_Day3_Evaluation.ipynb       # Evaluation metrics, error analysis, visualizations
│   ├── QueryCraft_Day4_Demo.ipynb             # Gradio demo (dropdown + natural language)
│   └── QueryCraft_Day5.ipynb                # ✅ Complete pipeline — RUN THIS
├── 📊 report/
│   └── Technical_Report.docx               # Detailed technical report (5-7 pages)
├── 📈 results/                             # Generated after running Day 3
│   ├── evaluation_report.json              # Baseline vs fine-tuned metrics
│   ├── final_summary.json                  # Complete project summary
│   ├── predictions.csv                     # Model predictions on test set
│   └── visualizations/
│       ├── model_comparison.png            # Baseline vs fine-tuned bar chart
│       ├── error_distribution.png          # Error category breakdown
│       ├── improvement_chart.png           # Metric improvement visualization
│       ├── hyperparameter_comparison.png   # Config A vs B vs C
│       └── training_curves.png             # Loss curves across training
├── 📋 requirements.txt                     # Python dependencies
├── 📖 README.md                            # This file
└── 📜 LICENSE                              # MIT License
```

---

## 🚀 How to Run

### Google Colab (Recommended — Free T4 GPU)

**Run the Day 5 notebook — it contains the complete end-to-end pipeline:**

1. Open **`QueryCraft_Day5.ipynb`** in [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU**
3. Run all cells sequentially from top to bottom

| Section in Day 5 | What It Does | Time on T4 |
|-------------------|-------------|------------|
| Install & Setup | Installs dependencies, loads model with QLoRA | ~5 min |
| Fine-Tuning | Trains Config B (lr=2e-4, rank=16) | ~7 min |
| RAG Setup | Creates ChromaDB/FAISS vector store | ~1 min |
| Gradio Demo | Launches interactive demo with public URL | ~1 min |
| Evaluation & Graphs | Generates 7 visualizations + final results JSON | ~1 min |

**Total runtime: ~15 minutes**

When the Gradio demo launches, you'll see:
```
Running on public URL: https://xxxxx.gradio.live    ← Click this!
```

> **Note:** Days 1-4 notebooks document the development process (data prep, training iterations, evaluation, demo prototyping). They are included for reference but are **not required** to run Day 5. The Day 5 notebook is fully self-contained.

### Notebook Reference (Development History)

| Day | Notebook | Purpose |
|-----|----------|---------|
| Day 1 | `QueryCraft_Day1_DataPrep.ipynb` | Dataset loading, preprocessing, RAG setup |
| Day 2 | `QueryCraft_Day2_FineTuning.ipynb` | Fine-tuning experiments with 3 HP configs |
| Day 3 | `QueryCraft_Day3_Evaluation.ipynb` | Evaluation metrics, error analysis, visualizations |
| Day 4 | `QueryCraft_Day4_Demo.ipynb` | Gradio demo prototyping |
| **Day 5** | **`QueryCraft_Day5.ipynb`** | **✅ Complete pipeline — RUN THIS ONE** |

---

### Local Machine (GPU Required)

```bash
# Clone the repository
git clone https://github.com/yourusername/QueryCraft.git
cd QueryCraft

# Create environment
conda create -n querycraft python=3.10 -y
conda activate querycraft

# Install PyTorch with CUDA
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt

# Run the final notebook
jupyter notebook notebooks/QueryCraft_Day5.ipynb
```

---

## 📊 Dataset

**Spider Dataset** (Yale University) — gold standard for cross-domain text-to-SQL.

| Metric | Value |
|--------|-------|
| Total Examples | 8,034 |
| Training Set | 7,000 (1,000 subset used for Colab feasibility) |
| Validation Set | 1,034 |
| Unique Databases | 160 |
| Unique Tables | 900+ |
| SQL Complexity | Easy → Extra Hard |

### Domain Coverage

| Domain | # Databases | Top Schemas |
|--------|-------------|-------------|
| Education | 15 | college_1, college_2, student_1 |
| Human Resources | 8 | hr_1, employee_hire_evaluation |
| Retail/E-commerce | 12 | store_1, department_store |
| Healthcare | 8 | hospital_1, allergy_1 |
| Sports | 12 | soccer_2, baseball_1, formula_1 |
| Entertainment | 15 | music_1, movie_1, concert_singer |

---

## ⚙️ Model & Training Configuration

### Best Configuration (Config B ⭐)

| Parameter | Value |
|-----------|-------|
| Base Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Learning Rate | 2e-4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.1 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Quantization | 4-bit NF4 + double quantization |
| Effective Batch Size | 16 (4 × 4 gradient accumulation) |
| Optimizer | Paged AdamW 8-bit |
| Training Loss | **0.7436** |

---

## 🎮 Demo Application

The Gradio interface provides two modes of interaction:

### Mode 1: Dropdown Examples (Reliable)
- Select a database → choose from pre-defined queries → execute
- 8 example queries per database, all pre-tested

### Mode 2: Natural Language + RAG (Flexible)
- Type any question in plain English
- RAG retrieves relevant schema context
- Model generates SQL → executes on SQLite → shows results
- Smart fallback system ensures valid results

### Demo Databases

| Database | Tables | Records | Sample Queries |
|----------|--------|---------|----------------|
| **University** | classroom, department, professor, course, student, enrollment | 25+ | "Average GPA?", "Professors earning > $100K?" |
| **HR** | regions, countries, locations, departments, employees, jobs | 30+ | "List all employees", "Salary by department?" |
| **E-commerce** | customers, products, orders, order_items, invoices | 40+ | "Top countries by invoices", "Products under $100?" |

### Example Queries Per Database

**University:**
- Count rooms not in Lamberton building
- Students in CS with GPA > 3.5
- Professors earning more than 100000
- Total credits by department

**HR:**
- Employees with salary > 80000
- Count employees per department
- Average salary by department
- Employees hired after 2021

**E-commerce:**
- Top 5 countries by invoices
- Total revenue by country
- Products under $100
- Top spending customers

---

## 📈 Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | Strict character-level SQL comparison |
| **Component Match** | SQL keyword + structure overlap |
| **Valid SQL Rate** | Syntax validation using `sqlparse` |
| **Has SELECT** | Whether output contains SELECT clause |
| **Has FROM** | Whether output contains FROM clause |
| **WHERE Match** | Correct WHERE clause generation |

### Error Analysis

| Error Type | Count | % | Description |
|------------|-------|---|-------------|
| Missing Clauses | 4 | 40% | Missing FROM, WHERE, ORDER BY |
| Wrong Aggregation | 3 | 30% | Incorrect COUNT, SUM, AVG |
| Syntax Errors | 1 | 10% | Unparseable SQL |
| Other | 2 | 20% | Wrong table/column names |

### Visualizations Generated (Day 3)

```
results/visualizations/
├── model_comparison.png           # Baseline vs fine-tuned side-by-side
├── error_distribution.png         # Error category pie chart
├── improvement_chart.png          # Metric improvement waterfall
├── hyperparameter_comparison.png  # Config A vs B vs C
└── training_curves.png            # Loss over training steps
```

---

## 🎥 Video Walkthrough

5-7 minute demonstration covering:

| Section | Duration | Content |
|---------|----------|---------|
| Intro | 0:30 | Project overview, motivation |
| Problem & Solution | 1:00 | Architecture diagram |
| Dataset | 1:00 | Spider dataset, domain coverage |
| Methodology | 1:30 | QLoRA config, RAG pipeline (show Day 5 notebook code) |
| Results | 1:30 | HP comparison, evaluation metrics (show generated plots) |
| Error Analysis | 0:45 | Error categories, improvements |
| **Live Demo** | 1:30 | Gradio — queries across 3 databases |
| Conclusion | 0:30 | Key achievements, future work |

---

## 🔮 Future Work

| Priority | Enhancement | Expected Impact |
|----------|-------------|-----------------|
| High | Full dataset training (7,000 examples) | +10-15% accuracy |
| High | Multi-epoch training (3 epochs) | Better convergence |
| Medium | Few-shot prompting in RAG context | +5-10% accuracy |
| Medium | SQL post-processing / auto-correction | Reduce syntax errors |
| Low | Execution-based evaluation | More practical metric |

---

## 📚 References

1. Yu, T., et al. (2018). [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887). EMNLP.
2. Hu, E. J., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). arXiv.
3. Dettmers, T., et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). arXiv.
4. Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS.
5. Zhang, P., et al. (2024). [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385). arXiv.
6. Hugging Face. (2024). [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft).
7. ChromaDB. (2024). [ChromaDB Documentation](https://docs.trychroma.com/).

---

## 👤 Author

**Aravind Balaji**
MS in Information Systems | Northeastern University
NUID: 001564773

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers, PEFT, and TRL libraries
- [Yale LILY Lab](https://yale-lily.github.io/) for the Spider dataset
- [TinyLlama](https://github.com/jzhang38/TinyLlama) for the base model
- INFO 7375 — Prompt Engineering & AI course at Northeastern University

---

<p align="center">
  <b>QueryCraft</b> — Crafting SQL from Natural Language
</p>
<p align="center">
  Made with ❤️ for INFO 7375 — Prompt Engineering & AI
</p>
