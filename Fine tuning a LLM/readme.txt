# 🔮 QueryCraft

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **RAG-Enhanced SQL Generation: Fine-Tuning Mistral-7B for Natural Language to SQL Translation**

<p align="center">
  <img src="assets/architecture.png" alt="RAG-SQL Architecture" width="700">
</p>

---

## 📋 Project Overview

QueryCraft is a production-ready **Retrieval-Augmented Generation (RAG)** system that converts natural language questions into executable SQL queries. It combines:

- **Fine-tuned Mistral-7B** using QLoRA (4-bit quantization + LoRA adapters)
- **ChromaDB-based RAG pipeline** with MiniLM-L6-v2 embeddings for schema-aware context retrieval
- **Interactive Gradio demo** with 3 pre-configured databases and live SQL execution

### 🎯 Key Results

| Metric | Without RAG | With RAG | Improvement |
|--------|-------------|----------|-------------|
| Component Match | 18.76% | 19.37% | **+3.2%** |
| Valid SQL Rate | 96.00% | 82.00% | — |
| Training Loss (Best) | — | — | **0.7436** |

---

## 🏗️ Architecture

```
User Question ──► Embedding ──► Vector Search ──► Schema Context
                     │              │                   │
             [MiniLM-L6-v2]    [ChromaDB]              │
                                                       ▼
                                              Combined Prompt
                                                       │
                                              ┌────────▼────────┐
                                              │  Fine-tuned     │
                                              │  Mistral-7B     │
                                              │  (QLoRA 4-bit)  │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              Generated SQL ──► SQLite Execution ──► Results
```

---

## 📁 Repository Structure

```
QueryCraft/
├── 📓 notebooks/
│   ├── QueryCraft_Fine_Tuning_Part1.ipynb    # Sections 1-6: Setup → Evaluation
│   └── QueryCraft_Fine_Tuning_Part2.ipynb    # Sections 7-9: Error Analysis → Demo → Conclusion
├── 📊 report/
│   └── Technical_Report.docx                 # Detailed technical report (5-7 pages)
├── 🎮 demo/
│   └── gradio_demo.py                        # Standalone Gradio demo application
├── 📈 results/                               # Generated after training
│   ├── 01_eda_analysis.png                   # Dataset EDA plots
│   ├── 02_training_progress.png              # Loss curves & LR schedule
│   ├── 03_hp_optimization.png                # Hyperparameter comparison
│   ├── 04_evaluation.png                     # RAG vs No-RAG metrics
│   ├── 05_error_analysis.png                 # Error category breakdown
│   ├── final_results.json                    # All metrics in JSON
│   └── querycraft_final/                     # Saved model weights (LoRA adapters)
├── 🗄️ demo_databases/                        # Generated SQLite databases
│   ├── university.db
│   ├── hr.db
│   └── ecommerce.db
├── 📋 requirements.txt                       # Python dependencies
├── 🎥 video/
│   └── video_script.pdf                      # Video walkthrough script (5-7 min)
├── 📖 README.md                              # This file
└── 📜 LICENSE                                # MIT License
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended — Free T4 GPU)

This is the easiest way to run the entire project, no local setup required.

**Step 1: Upload notebooks to Colab**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload Notebook** → upload `QueryCraft_Fine_Tuning_Part1.ipynb`
3. Repeat for `Part2.ipynb` (or merge them — see [Merging Notebooks](#merging-notebooks))

**Step 2: Enable GPU**
1. **Runtime → Change runtime type → T4 GPU**
2. Verify: run `!nvidia-smi` — you should see "Tesla T4"

**Step 3: Run cells sequentially**

| Section | Notebook | What Happens | Time |
|---------|----------|--------------|------|
| 1. Environment Setup | Part 1 | Installs all dependencies | ~2 min |
| 2. Dataset Preparation | Part 1 | Loads Spider, EDA plots, preprocessing | ~1 min |
| 3. Model Selection | Part 1 | Loads Mistral-7B with 4-bit quantization | ~3 min |
| 4. Fine-Tuning | Part 1 | Trains Config B, saves model, plots loss curves | **~25 min** |
| 5. HP Optimization | Part 1 | Trains 3 configs (A, B, C), comparison plots | **~75 min** |
| 6. Evaluation | Part 1 | Evaluates with/without RAG on 50 test samples | ~10 min |
| 7. Error Analysis | Part 2 | Categorizes errors, visualizations | ~1 min |
| 8. Live Demo | Part 2 | Creates demo DBs, launches Gradio interface | ~1 min |
| 9. Conclusion | Part 2 | Saves final results JSON | ~1 min |

> **⚡ Quick Run (~40 min):** Skip Section 5 (HP search) — Section 4 already trains the best config (B). Section 5 is only needed to show the comparison across all 3 configurations.

**Step 4: Access the Live Demo**

When Section 8 runs, you'll see:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live    ← Click this!
```

The public URL opens an interactive interface where you can:
1. Select a database (University / HR / E-commerce)
2. Type a natural language question
3. Toggle RAG on/off
4. See the generated SQL + live execution results

---

### Option 2: Local Machine (GPU Required)

**Prerequisites:**
- Python 3.10+
- CUDA-compatible GPU with 16GB+ VRAM
- ~10 GB disk space

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/QueryCraft.git
cd QueryCraft

# 2. Create conda environment
conda create -n querycraft python=3.10 -y
conda activate querycraft

# 3. Install PyTorch with CUDA
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Run the notebook
jupyter notebook notebooks/QueryCraft_Fine_Tuning_Part1.ipynb
```

**Or run just the demo** (if you already have saved model weights):
```bash
python demo/gradio_demo.py
```

---

### Merging Notebooks

The notebook is split into 2 parts due to file size. To merge into a single `.ipynb`:

```python
import json

with open('notebooks/QueryCraft_Fine_Tuning_Part1.ipynb') as f:
    nb1 = json.load(f)
with open('notebooks/QueryCraft_Fine_Tuning_Part2.ipynb') as f:
    nb2 = json.load(f)

nb1['cells'].extend(nb2['cells'])

with open('notebooks/QueryCraft_Complete.ipynb', 'w') as f:
    json.dump(nb1, f, indent=2)

print("Merged → QueryCraft_Complete.ipynb")
```

Then upload `QueryCraft_Complete.ipynb` to Colab and run all cells.

---

## 📊 Dataset

**Spider Dataset** (Yale University) — the gold standard benchmark for cross-domain text-to-SQL.

| Metric | Value |
|--------|-------|
| Total Examples | 8,034 |
| Training Set | 7,000 (1,000 used for Colab feasibility) |
| Validation Set | 1,034 |
| Unique Databases | 160 |
| Unique Tables | 900+ |
| Unique Columns | 4,500+ |
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

## ⚙️ Model Configuration

### Best Configuration (Config B ⭐)

| Parameter | Value |
|-----------|-------|
| Base Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Learning Rate | 2e-4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.1 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Quantization | 4-bit NF4 + double quantization |
| Effective Batch Size | 16 (4 × 4 gradient accumulation) |
| Optimizer | Paged AdamW 8-bit |
| LR Scheduler | Cosine with warmup (3%) |
| Training Loss | **0.7436** |

### Hyperparameter Search Results

| Config | LR | LoRA Rank | LoRA α | Training Loss | Status |
|--------|-----|-----------|--------|---------------|--------|
| A | 3e-4 | 8 | 16 | 1.1319 | Higher LR → instability |
| **B** | **2e-4** | **16** | **32** | **0.7436** | **⭐ BEST** |
| C | 1e-4 | 32 | 64 | 1.1827 | Low LR → insufficient learning |

### QLoRA Code Reference

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)
```

---

## 🔍 RAG Pipeline

| Component | Implementation | Details |
|-----------|---------------|---------|
| Embedding Model | all-MiniLM-L6-v2 | 384-dimensional sentence embeddings |
| Vector Store | ChromaDB | In-memory, cosine similarity |
| Documents | 12 total | 8 schema documents + 4 SQL glossary terms |
| Retrieval | Top-3 | Most relevant schemas per query |

**How it works:**
1. User question is embedded using MiniLM-L6-v2
2. ChromaDB retrieves top-3 most similar schema documents
3. Retrieved schemas are injected into the prompt before the question
4. Fine-tuned Mistral-7B generates SQL with schema awareness

---

## 🎮 Demo Databases

The interactive demo includes 3 pre-configured SQLite databases:

| Database | Tables | Records | Sample Queries |
|----------|--------|---------|----------------|
| **University** | department, instructor, student, course | 25+ | "How many students are there?", "Professors earning > $100K?" |
| **HR** | departments, employees, jobs | 30+ | "List all employees", "Average salary by department?" |
| **E-commerce** | customers, products, orders, order_items | 40+ | "Products under $100?", "Which country has most customers?" |

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | Strict character-level SQL comparison (normalized) |
| **Component Match** | SQL keyword + content overlap (Jaccard-style) |
| **Valid SQL Rate** | Syntax validation using `sqlparse` |

### Error Analysis

| Error Type | Count | % | Description |
|------------|-------|---|-------------|
| Missing Clauses | 4 | 40% | Missing FROM, WHERE, ORDER BY |
| Wrong Aggregation | 3 | 30% | Incorrect COUNT, SUM, AVG usage |
| Syntax Errors | 1 | 10% | Unparseable SQL output |
| Other | 2 | 20% | Wrong table/column names |

---

## 📂 Output Files

After running the notebook, the following files are generated:

```
results/
├── 01_eda_analysis.png            # 4-panel dataset EDA (question/SQL lengths, keywords, DBs)
├── 02_training_progress.png       # Loss curve, LR schedule, smoothed convergence
├── 03_hp_optimization.png         # 3-config comparison (loss, time, complexity)
├── 04_evaluation.png              # RAG vs No-RAG bar chart, delta, distribution
├── 05_error_analysis.png          # Error pie chart, component match by category
├── final_results.json             # All metrics, HP results, error counts
├── querycraft_final/              # Saved LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
└── training_log.json              # Step-by-step training losses & LR

demo_databases/
├── university.db                  # SQLite demo database
├── hr.db                          # SQLite demo database
└── ecommerce.db                   # SQLite demo database
```

---

## 🎥 Video Walkthrough

A 5-7 minute video demonstration covering:

| Section | Duration | Content |
|---------|----------|---------|
| Intro | 0:30 | Project overview |
| Problem & Solution | 1:00 | Business problem, architecture |
| Dataset | 1:00 | Spider dataset, domain coverage |
| Methodology | 1:30 | Mistral-7B, QLoRA, RAG pipeline |
| Results | 1:30 | HP comparison, evaluation metrics, RAG improvement |
| Error Analysis | 0:45 | Error categories, improvements |
| Live Demo | 1:30 | Gradio interface across 3 databases |
| Conclusion | 0:30 | Key achievements, future work |

📺 [Watch the Demo Video](https://youtube.com/your-video-link)

---

## 🔮 Future Work

| Priority | Enhancement | Expected Impact |
|----------|-------------|-----------------|
| High | Full dataset training (7,000 examples) | +10-15% accuracy |
| High | Multi-epoch training (3 epochs) | Better convergence |
| Medium | Few-shot prompting in RAG context | +5-10% accuracy |
| Medium | SQL post-processing / auto-correction | Reduce syntax errors |
| Low | Execution-based evaluation | More practical metric |
| Low | Additional demo domains | Broader applicability |

---

## 📚 References

1. Yu, T., et al. (2018). [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887). EMNLP.
2. Hu, E. J., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). arXiv.
3. Dettmers, T., et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). arXiv.
4. Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS.
5. Mistral AI. (2023). [Mistral 7B Technical Report](https://mistral.ai/news/announcing-mistral-7b/).
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
- [Mistral AI](https://mistral.ai/) for the base model
- [ChromaDB](https://www.trychroma.com/) for the vector database
- INFO 7375 — Prompt Engineering & AI course at Northeastern University

---

<p align="center">
  <b>QueryCraft</b> — Crafting SQL from Natural Language
</p>
<p align="center">
  Made with ❤️ for INFO 7375 — Prompt Engineering & AI
</p>
