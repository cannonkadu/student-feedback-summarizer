# Student Feedback Summarizer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/swfdv67bdd/badge.svg)](https://ask.thili.ai/b/swfdv67bdd)

An EdTech tool that fine-tunes a language model from scratch using **Low-Rank Adaptation (LoRA / `lfllm-lflora`)** on student course feedback, evaluates summary quality with automated **ROUGE-1 / ROUGE-2 / ROUGE-L** metrics, merges adapters into base weights for zero-overhead inference, and provides an interactive **Streamlit** studio for instructors.

## Features

- **Synthetic EdTech Dataset (`app/dataset.py`)**: 24 curated course feedback & curriculum-summary pairs across Lectures, Labs, Assignments, Pacing, Grading, Support, and Projects, formatted for `lfllm-lflora` training with train/val splits and JSONL/CSV import & export.
- **LoRA From Scratch (`app/lora.py`)**: Pure PyTorch `LoRALinear` implementation supporting both `nn.Linear` and GPT-2 `Conv1D` layers ($h = W_0 x + \frac{\alpha}{r} B A x$), frozen base weights, configurable rank $r$ & scaling $\alpha$, live causal-LM training, and lightweight `.pt` adapter saving.
- **Interactive Streamlit Studio (`app/main.py`)**:
  - **Summarize Feedback**: Paste comments or upload `.csv` / `.txt` / `.json` files to generate concise summaries, sentiment tags, detected themes, and recommended curriculum actions.
  - **LoRA Fine-Tuning Studio**: Tune rank, alpha, epochs, and learning rate, inspect trainable parameter ratios, watch live loss curves, and save adapter weights.
  - **Dataset Explorer**: Filter feedback pairs, add custom examples, and export JSONL.
  - **Evaluation & ROUGE Lab**: Measure ROUGE-1, ROUGE-2, ROUGE-L, compression ratio, and key-term coverage on the validation split.
  - **Adapter Merging & Deployment**: Fuse $\frac{\alpha}{r} B A$ into $W_0$ in-place, unwrap `LoRALinear` layers, and verify numerical equivalence ($\max |\Delta \text{logits}| < 10^{-4}$).
- **Optional External LLM Support (`app/llm.py`)**: Optionally route summarization to Ollama, Groq, or Gemini via `.env`.

## Milestones

- [x] **1. Prepare a small dataset of synthetic student feedback and summary pairs.**
- [x] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**
- [x] **3. Develop a Streamlit UI to upload feedback and trigger summarization.**
- [x] **4. Evaluate the quality of generated summaries against original feedback.**
- [x] **5. Implement adapter merging for inference.**

## Tech Stack

- **Python 3.12** & **PyTorch** (from-scratch LoRA layers, training loop, adapter merging)
- **Streamlit** (interactive EdTech UI)
- **Pytest** & **GitHub Actions** (automated unit & smoke tests)
- **httpx** (optional OpenAI-compatible LLM integration for Ollama / Groq / Gemini)

## Run It Locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: configure Ollama, Groq, or Gemini
streamlit run app/main.py
```

### Run Tests

```bash
pytest -q
```

## Project Structure

```text
student-feedback-summarizer/
├── app/
│   ├── core.py         # Milestone 1–5 entry points
│   ├── dataset.py      # Synthetic feedback dataset, prompt formatting, CSV/JSON/JSONL I/O
│   ├── lora.py         # From-scratch LoRALinear, injection, training, and adapter merging
│   ├── summarizer.py   # LoRA-guided summarization, sentiment, themes, and action items
│   ├── evaluate.py     # ROUGE-1, ROUGE-2, ROUGE-L, compression & key-term coverage
│   ├── main.py         # 5-tab Streamlit application
│   ├── llm.py          # Optional Ollama / Groq / Gemini chat client
│   └── storage.py      # Lightweight JSON key-value store
├── tests/
│   ├── test_smoke.py   # Starter plumbing checks
│   └── test_core.py    # Unit tests for Milestones 1–5
├── .github/workflows/ci.yml
├── requirements.txt
└── Dockerfile
```

## Settings and Secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.**
