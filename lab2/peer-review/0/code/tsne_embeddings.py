# t-SNE Visualization of Autoencoder Embeddings
# Run from lab2/code/ directory
# Uses the _with_embeddings.csv files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

###############################################################################
# Load embedding data
###############################################################################

col_ae = [f"ae{i}" for i in range(16)]

df1 = pd.read_csv("../data/image_data/O013257_with_embeddings.csv")
df2 = pd.read_csv("../data/image_data/O013490_with_embeddings.csv")
df3 = pd.read_csv("../data/image_data/O012791_with_embeddings.csv")

df1['image'] = 'O013257'
df2['image'] = 'O013490'
df3['image'] = 'O012791'

all_data = pd.concat([df1, df2, df3], ignore_index=True)

# Filter to labeled only
labeled = all_data[all_data['cloud'] != 0].copy()
labeled['label'] = labeled['cloud'].map({1: 'Cloud', -1: 'Not Cloud'})

print(f"Total labeled pixels: {len(labeled)}")
print(f"Cloud: {(labeled['cloud'] == 1).sum()}, Not Cloud: {(labeled['cloud'] == -1).sum()}")

###############################################################################
# Subsample for t-SNE (it's slow on 200k+ points)
###############################################################################

n_sample = 10000
rng = np.random.RandomState(42)

# Stratified sample: equal from each class and image
samples = []
for img in ['O013257', 'O013490', 'O012791']:
    for lab in [1, -1]:
        subset = labeled[(labeled['image'] == img) & (labeled['cloud'] == lab)]
        n = min(len(subset), n_sample // 6)
        samples.append(subset.sample(n=n, random_state=rng))

sampled = pd.concat(samples, ignore_index=True)
print(f"Sampled {len(sampled)} pixels for t-SNE")

###############################################################################
# Run t-SNE
###############################################################################

print("Running t-SNE (this may take a minute)...")
embeddings = sampled[col_ae].values

tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
tsne_result = tsne.fit_transform(embeddings)

sampled['tsne_1'] = tsne_result[:, 0]
sampled['tsne_2'] = tsne_result[:, 1]

###############################################################################
# FIGURE 1: t-SNE colored by label (all images combined)
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))

cloud = sampled[sampled['cloud'] == 1]
nocloud = sampled[sampled['cloud'] == -1]

ax.scatter(nocloud['tsne_1'], nocloud['tsne_2'], c='lightsalmon',
           alpha=0.4, s=5, label='Not Cloud', edgecolors='none')
ax.scatter(cloud['tsne_1'], cloud['tsne_2'], c='skyblue',
           alpha=0.4, s=5, label='Cloud', edgecolors='none')

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of Autoencoder Embeddings (16-dim → 2-dim)')
ax.legend(markerscale=5)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig("../figs/tsne_embeddings_label.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ../figs/tsne_embeddings_label.png")

###############################################################################
# FIGURE 2: t-SNE colored by label, per image
###############################################################################

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, img in enumerate(['O013257', 'O013490', 'O012791']):
    ax = axes[i]
    img_data = sampled[sampled['image'] == img]
    cloud_img = img_data[img_data['cloud'] == 1]
    nocloud_img = img_data[img_data['cloud'] == -1]

    ax.scatter(nocloud_img['tsne_1'], nocloud_img['tsne_2'], c='lightsalmon',
               alpha=0.4, s=8, label='Not Cloud', edgecolors='none')
    ax.scatter(cloud_img['tsne_1'], cloud_img['tsne_2'], c='skyblue',
               alpha=0.4, s=8, label='Cloud', edgecolors='none')

    ax.set_title(f'{img}')
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.legend(markerscale=4)

plt.suptitle('t-SNE of Autoencoder Embeddings by Image', fontsize=14)
plt.tight_layout()
plt.savefig("../figs/tsne_embeddings_by_image.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ../figs/tsne_embeddings_by_image.png")

###############################################################################
# FIGURE 3: t-SNE colored by image (to see transfer learning effect)
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))

colors = {'O013257': '#e74c3c', 'O013490': '#2ecc71', 'O012791': '#3498db'}
for img in ['O013257', 'O013490', 'O012791']:
    img_data = sampled[sampled['image'] == img]
    ax.scatter(img_data['tsne_1'], img_data['tsne_2'], c=colors[img],
               alpha=0.3, s=5, label=img, edgecolors='none')

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE Colored by Image (Overlap = Good Transfer Learning)')
ax.legend(markerscale=5)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig("../figs/tsne_embeddings_by_source.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ../figs/tsne_embeddings_by_source.png")

print("\nDone! Three figures saved.")
