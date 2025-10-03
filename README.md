# User-Centric and Parameter-Efficient LLMs

This project explores how to efficiently align Large Language Models (LLMs) with the unique preferences of individual users.

## Challenge & Approach

* **Challenge**: Standard LLMs are aligned to general population preferences, not individuals. Fine-tuning a unique multi-billion parameter model for every user is computationally impractical.
* **Approach**: This project uses a parameter-efficient method to create lightweight, modular "preference adapters" for each user. These adapters can be easily loaded onto a base LLM to tailor its responses without altering the original model.

## Contributions

* **Created two novel datasets** to capture individual user preferences.
* **Demonstrated effective user-specific alignment** using Direct Preference Optimization (DPO) and Low-Rank Adaptation (LoRA).
* **Analysed adapter merging** as a technique to improve generalisation for new users.
* **Explored data efficiency** for meaningful preference learning on the two datasets.


## Findings

1. The model effectively captures preferences on both datasets, outperforming the baseline.
2. Merging adapters from different users showing commonalities with a tiny sample of an unseen user's preferences improves accuracy compared to baseline.
3. A minimum of 400 training samples is required for meaningful improvements over the baseline (for our two datasets).


## Materials

The two datasets created for this project are available on the Hugging Face Hub:  **[huggingface.co/guoyu-zhang](https://huggingface.co/guoyu-zhang)**

* **Stanford Human Preferences Subset (SHPS)**: A filtered subset of the Stanford Human Preferences dataset which was then annotated by users.
* **User-Specific Preferences (USP)**: LLM-generated data designed to allow for expression of different preferences among users annotating the data.

The trained LoRA adapters for each user are also available on the Hugging Face Hub.
