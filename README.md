# GLASS: A Generative Recommender for Long-sequence Modeling via SID-Tier and Semantic Search

This README gives a concise project overview.

## Overview
Leveraging long-term user behavioral patterns is a key trajectory for enhancing the accuracy of modern recommender systems. While generative recommender systems have emerged as a transformative paradigm, they face hurdles in modeling extensive historical sequences. In this work we propose GLASS, a novel framework that integrates long-term user interests into the generative recommendation process via SID-Tier and Semantic Search.

For detailed architecture and methodology, please refer to [overview.pdf](./overview.pdf).

## Key ideas
- SID-Tier: A module that maps long-term interactions into a unified interest vector to improve prediction of the initial semantic ID (SID) token. Rather than relying on conventional retrieval across massive item spaces, SID-Tier exploits the compactness of a semantic codebook and models cross-features between a user's long-term history and candidate semantic codes.
- Semantic Hard Search: Uses generated coarse-grained semantic IDs as dynamic keys to retrieve relevant historical behaviors. Retrieved behaviors are fused through an adaptive gated fusion module that recalibrates the trajectory of subsequent fine-grained generated tokens.
- Data-sparsity solutions: To mitigate sparsity in semantic hard search, GLASS introduces semantic neighbor augmentation and codebook resizing strategies.

## What we show
Extensive experiments on two large-scale real-world datasets (TAOBAO-MM and KuaiRec) demonstrate that GLASS outperforms state-of-the-art baselines. The method achieves notable gains in recommendation quality while retaining computational efficiency.


## Quick Start

### Data Preparation
1. Download the raw TAOBAO-MM dataset from [taobao-mm.github.io](https://taobao-mm.github.io/) to the `data/TAOBAO_MM/raw` folder.

2. Process the dataset:
   ```bash
   cd data
   python process_TAOBAO.py
   python prepare_rqvae.py
   ```

### Training

3. Train the RQ-VAE model:
   ```bash
   cd ../rqvae
   python main.py
   python generate_code.py
   ```

4. Prepare GLASS data:
   ```bash
   cd ../data
   python process_glass.py
   ```

5. Train the GLASS model:
   ```bash
   cd ../glass
   python main_glass.py
   ```

## Code availability
Our codes are made publicly available to facilitate further research in generative recommendation. The implementation will be published to this repository as soon as it is ready.

---
Thank you for your interest — code and detailed instructions will appear here soon.
