# NLU Assignment 2

Word embeddings (Problem 1) and character-level name generation using RNNs (Problem 2).

## Setup

```
pip install torch gensim PyPDF2 beautifulsoup4 wordcloud matplotlib numpy
```

Python 3.8+ required.

## Problem 1 - Word Embeddings

```
python P1T1.py        # extract text, preprocess, stats, word cloud
python P1T2.py        # train Word2Vec models (CBOW + Skip-gram)
python P1T3.py        # nearest neighbors and analogy experiments
python P1T4.py        # PCA and t-SNE visualizations
```

Run in order. Each script depends on the output of the previous one.

**Outputs:** `cleaned_corpus.txt`, `wordcloud.png`, `pca_visualization.png`, `tsne_visualization.png`, trained models in `w2v_models/`

## Problem 2 - Name Generation

```
python generate_names.py   # creates TrainingNames.txt (1000 names)
python P2T1.py             # train 3 models (Vanilla RNN, BLSTM, Attention RNN)
python P2T2.py             # quantitative evaluation (novelty, diversity)
python P2T3.py             # qualitative analysis (structure, failures)
```

Again, run in order. P2T2 and P2T3 load saved weights from P2T1.

**Outputs:** `training_loss.png`, `generated_names.txt`, `evaluation_results.txt`, `qualitative_analysis.txt`, model weights in `models/`

## Reports

```
M24CSE014_A2P1_Report.pdf
M24CSE014_A2P2_Report.pdf
```

## File Structure

```
P1T1.py - P1T4.py          Problem 1 scripts
P2T1.py - P2T3.py          Problem 2 scripts
generate_names.py           Training data generator
TrainingNames.txt           1000 Indian names
data/                       Raw source files (PDFs + HTMLs)
w2v_models/                 Saved Word2Vec models
models/                     Saved RNN model weights
```
