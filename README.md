# Benchmarking Long-Form Text Summarisation Approaches: Baseline, Chunked, and RAG

## Project Overview
A core limitation of standard Large Language Models (LLMs) is their context window constraints when processing long-form documents. This project implements, evaluates, and benchmarks three distinct architectural strategies to summarize extensive, public-domain literary texts:

1. **Extractive Baseline:** A statistical TF-IDF ranking model.
2. **Two-Pass Chunked Approach:** A sliding-window pipeline that passes rolling historical context alongside text chunks to preserve structural narrative flow.
3. **Retrieval-Augmented Generation (RAG):** A query-aware pipeline utilizing dense vector embeddings to fetch top-$k$ contextually relevant document fragments.

## Tech Stack & Skills
* **Core Concepts:** Natural Language Processing (NLP), Large Language Models (LLMs), Retrieval-Augmented Generation (RAG)
* **Frameworks & Libraries:** OpenAI API, Scikit-Learn (TF-IDF), NumPy, Pandas, Matplotlib/Seaborn
* **Evaluation Framework:** ROUGE Metrics (ROUGE-1, ROUGE-2, ROUGE-L), Statistical Significance Testing (Wilcoxon Signed-Rank Test)

## Experimental Setup & Evaluation
The architectures were evaluated using three long-form literary texts with multiple human-written reference summaries serving as the ground truth. 

### Key Discoveries:
* **Short vs. Long Documents:** On shorter texts, automatic metrics showed no statistically significant performance gap between extractive and generative models. However, structured pipelines (Chunked and RAG) demonstrated extreme robustness as text length scaled up to full novels.
* **Qualitative Coherence:** The **Two-Pass Chunked approach** generated the most structurally cohesive and chronologically accurate narrative summaries.
* **Query-Task Optimization:** The **RAG pipeline** dominated on target, query-driven retrieval tasks, demonstrating its utility for user-specific search intents.

## Performance Visualizations
*The repository contains the code used to generate comprehensive statistical distribution analyses and performance degradation curves over document lengths.*
