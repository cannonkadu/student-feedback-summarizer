# Student Feedback Summarizer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/swfdv67bdd/badge.svg)](https://ask.thili.ai/b/swfdv67bdd)

This project develops a Student Feedback Summarizer. It leverages the LoRA fine-tuning technique from `lfllm-lflora` to adapt a pre-trained language model, enabling course instructors to generate concise summaries of student comments. The goal is to facilitate rapid curriculum improvement by efficiently processing feedback.

## What you are building

**Goal.** You will build a tool that fine-tunes a language model using LoRA to summarize student feedback, helping course instructors rapidly improve curriculum.

**Done when.** The project is done when the LoRA fine-tuning technique from `lfllm-lflora` is successfully applied to an LLM, and the Streamlit UI can generate relevant summaries of student feedback that are evaluated for quality after training and merging the adapter.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small dataset of synthetic student feedback and summary pairs.**  
      Hint: Structure your dataset for `lfllm-lflora` training.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Use `lfllm-lflora` to inject and train the LoRA adapter.
- [ ] **3. Develop a Streamlit UI to upload feedback and trigger summarization.**  
      Hint: Integrate the `lfllm-lflora` trained model into a Streamlit UI.
- [ ] **4. Evaluate the quality of generated summaries against original feedback.**  
      Hint: Assess the output quality of your `lfllm-lflora` fine-tuned model.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Apply `lfllm-lflora`'s adapter merging for deployment.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Enhance the Streamlit UI with more interactive features or better display of summarization results.
- Experiment with different base LLMs or LoRA hyperparameter configurations to optimize summary quality.
- Implement automated evaluation metrics (e.g., ROUGE scores) to quantitatively assess summary quality.
