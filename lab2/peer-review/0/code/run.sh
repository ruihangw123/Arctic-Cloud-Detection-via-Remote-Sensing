#!/bin/bash

conda activate env_214

python generate_eda_figs.py
python feature_importance.py
python feature_engineering.py
python run_autoencoder.py configs/bridges2.yaml
python get_embedding.py configs/bridges2.yaml checkpoints/conv-ae-epoch=099.ckpt
python tsne_embeddings.py
python predictive_modeling.py
python diagnostics.py
python posthoc_eda.py
python stability.py

conda deactivate