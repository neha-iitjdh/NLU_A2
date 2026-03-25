from gensim.models import Word2Vec
import os

# load corpus

def load_corpus(filepath):
    # line = doc, space-separated tokens
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


# train Word2Vec

def train_w2v(sentences, sg, vector_size, window, negative, epochs=30):
    # sg: 0=CBOW, 1=Skip-gram
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=2,       # ignore freq < 2
        sg=sg,
        negative=negative,
        epochs=epochs,
        seed=42,
        workers=1          # reproducible
    )
    return model


if __name__ == '__main__':
    corpus_path = 'e:/NLU2/cleaned_corpus.txt'
    sentences = load_corpus(corpus_path)
    print(f"Loaded {len(sentences)} documents from corpus")

    # hyperparameter grid
    configs = [
        # (dim, window, negative)
        (50,  3, 5),
        (50,  5, 5),
        (100, 3, 5),
        (100, 5, 5),
        (100, 5, 10),
        (200, 5, 5),
    ]

    os.makedirs('e:/NLU2/w2v_models', exist_ok=True)

    # train CBOW + Skip-gram
    print("\n" + "="*70)
    print("TRAINING WORD2VEC MODELS")
    print("="*70)

    results = []

    for sg in [0, 1]:
        model_type = "Skip-gram" if sg == 1 else "CBOW"
        for vec_size, window, neg in configs:
            print(f"\n{model_type}: dim={vec_size}, window={window}, neg_samples={neg}")

            model = train_w2v(sentences, sg, vec_size, window, neg)
            vocab_size = len(model.wv)
            print(f"  Vocabulary: {vocab_size} words")

            # save model
            tag = f"{'sg' if sg else 'cbow'}_d{vec_size}_w{window}_n{neg}"
            model_path = f'e:/NLU2/w2v_models/{tag}.model'
            model.save(model_path)

            # quick check
            test_word = 'student'
            if test_word in model.wv:
                neighbors = model.wv.most_similar(test_word, topn=3)
                neighbor_str = ', '.join([f"{w}({s:.2f})" for w, s in neighbors])
                print(f"  Nearest to '{test_word}': {neighbor_str}")

            results.append({
                'type': model_type, 'dim': vec_size,
                'window': window, 'neg': neg,
                'vocab': vocab_size, 'path': model_path
            })

    # summary table
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"{'Model':<12} {'Dim':<6} {'Window':<8} {'NegSamp':<8} {'Vocab':<8}")
    print("-" * 42)
    for r in results:
        print(f"{r['type']:<12} {r['dim']:<6} {r['window']:<8} {r['neg']:<8} {r['vocab']:<8}")

    # save summary
    with open('e:/NLU2/training_summary.txt', 'w') as f:
        f.write("WORD2VEC TRAINING SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Corpus: {len(sentences)} documents\n")
        f.write(f"Epochs: 30\n")
        f.write(f"Min count: 2\n\n")
        f.write(f"{'Model':<12} {'Dim':<6} {'Window':<8} {'NegSamp':<8} {'Vocab':<8}\n")
        f.write("-" * 42 + "\n")
        for r in results:
            f.write(f"{r['type']:<12} {r['dim']:<6} {r['window']:<8} {r['neg']:<8} {r['vocab']:<8}\n")
    print("\nSummary saved to training_summary.txt")