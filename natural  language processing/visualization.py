import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.manifold import TSNE
import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

words   = np.load(os.path.join(BASE_DIR, 'vocab.npy'),        allow_pickle=True)
weights = np.load(os.path.join(BASE_DIR, 'word_vectors.npy'), allow_pickle=True)

embeddings = weights[()] if weights.ndim == 0 else weights

print(f"Loaded {len(words)} words  |  embedding shape: {embeddings.shape}")

# ── t-SNE ─────────────────────────────────────────────────────────────────────
tsne = TSNE(
    n_components=2,
    perplexity=50,
    random_state=42,
    init='pca',
    learning_rate='auto',
    max_iter=2000
)
coords = tsne.fit_transform(embeddings)
print("t-SNE done")

# ── save coordinates CSV ──────────────────────────────────────────────────────
coords_df = pd.DataFrame({
    'word': [str(w) for w in words],
    'x':    coords[:, 0],
    'y':    coords[:, 1],
})
csv_path = os.path.join(BASE_DIR, 'tsne_coords.csv')
coords_df.to_csv(csv_path, index=False)
print(f"Coordinates saved → {csv_path}")

# ── key characters ────────────────────────────────────────────────────────────
key_words = {
    'harry', 'ron', 'hermione', 'dumbledore', 'voldemort',
    'snape', 'hagrid', 'mcgonagall', 'draco', 'dobby',
    'hogwarts', 'basilisk', 'quidditch', 'gryffindor', 'slytherin',
    'malfoy', 'lockhart', 'potter'
}

colors    = ['#e63946' if str(w) in key_words else '#457b9d' for w in words]
sizes     = [14        if str(w) in key_words else 5         for w in words]
opacities = [1.0       if str(w) in key_words else 0.55      for w in words]

key_mask = [str(w) in key_words for w in words]
key_x    = coords[key_mask, 0]
key_y    = coords[key_mask, 1]
key_text = [str(w) for w in words if str(w) in key_words]

# ── build figure ──────────────────────────────────────────────────────────────
fig = go.Figure()

# all words as hoverable dots
fig.add_trace(go.Scatter(
    x=coords[:, 0],
    y=coords[:, 1],
    mode='markers',
    marker=dict(
        color=colors,
        size=sizes,
        opacity=opacities,
        line=dict(width=0.4, color='white')
    ),
    text=[str(w) for w in words],
    hovertemplate='<b>%{text}</b><extra></extra>',
    name='words'
))

# key characters with labels on top
fig.add_trace(go.Scatter(
    x=key_x,
    y=key_y,
    mode='markers+text',
    marker=dict(color='#e63946', size=14, line=dict(width=1, color='white')),
    text=key_text,
    textposition='top center',
    textfont=dict(size=11, color='#e63946', family='Georgia'),
    hovertemplate='<b>%{text}</b><extra></extra>',
    name='key characters'
))

fig.update_layout(
    title=dict(
        text='Word2Vec Embedding Space — Harry Potter',
        font=dict(size=22, family='Georgia', color='#1d3557'),
        x=0.5
    ),
    width=1800,
    height=1000,
    plot_bgcolor='#f8f9fa',
    paper_bgcolor='#f1faee',
    xaxis=dict(title='t-SNE Dimension 1', showgrid=True,
               gridcolor='#dee2e6', zeroline=False),
    yaxis=dict(title='t-SNE Dimension 2', showgrid=True,
               gridcolor='#dee2e6', zeroline=False),
    hoverlabel=dict(bgcolor='white', font_size=13, font_family='Georgia'),
    showlegend=False,
    margin=dict(l=60, r=60, t=80, b=60)
)

fig.show()
