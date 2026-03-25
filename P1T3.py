from gensim.models import Word2Vec
import os

# semantic analysis

def nearest_neighbors(model, word, topn=5):
    # top-n similar words
    if word not in model.wv:
        print(f"  '{word}' not in vocabulary")
        return []
    neighbors = model.wv.most_similar(word, topn=topn)
    return neighbors

def analogy(model, a, b, c):
    # a:b :: c:?
    # b - a + c
    try:
        results = model.wv.most_similar(positive=[b, c], negative=[a], topn=5)
        return results
    except KeyError as e:
        print(f"  Word not in vocabulary: {e}")
        return []


if __name__ == '__main__':
    # load models
    cbow_path = 'e:/NLU2/w2v_models/cbow_d100_w5_n5.model'
    sg_path = 'e:/NLU2/w2v_models/sg_d100_w5_n5.model'

    if not os.path.exists(cbow_path):
        print("Models not found. Run P1T2.py first.")
        exit(1)

    cbow_model = Word2Vec.load(cbow_path)
    sg_model = Word2Vec.load(sg_path)

    models = {'CBOW': cbow_model, 'Skip-gram': sg_model}

    # nearest neighbors
    # 'exam' missing, use 'examination'
    query_words = ['research', 'student', 'phd', 'examination']

    print("="*60)
    print("PART 1: TOP 5 NEAREST NEIGHBORS")
    print("="*60)

    for model_name, model in models.items():
        print(f"\n--- {model_name} (dim=100, window=5) ---")
        for word in query_words:
            neighbors = nearest_neighbors(model, word, topn=5)
            if neighbors:
                print(f"\n  '{word}':")
                for w, sim in neighbors:
                    print(f"    {w:<20} {sim:.4f}")

    # analogy
    print("\n" + "="*60)
    print("PART 2: ANALOGY EXPERIMENTS")
    print("="*60)

    analogies = [
        # (a, b, c, desc)
        ('ug', 'btech', 'pg', 'UG : BTech :: PG : ?'),
        ('student', 'students', 'course', 'student : students :: course : ?'),
        ('computer', 'science', 'electrical', 'computer : science :: electrical : ?'),
    ]

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        for a, b, c, desc in analogies:
            print(f"\n  {desc}")
            results = analogy(model, a, b, c)
            if results:
                for w, sim in results[:3]:
                    print(f"    {w:<20} {sim:.4f}")
            else:
                print(f"    (could not compute)")

    # save results
    with open('e:/NLU2/semantic_analysis.txt', 'w') as f:
        f.write("SEMANTIC ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")

        f.write("PART 1: TOP 5 NEAREST NEIGHBORS\n")
        f.write("-"*40 + "\n")
        for model_name, model in models.items():
            f.write(f"\n{model_name} (dim=100, window=5):\n")
            for word in query_words:
                neighbors = nearest_neighbors(model, word, topn=5)
                if neighbors:
                    f.write(f"\n  '{word}':\n")
                    for w, sim in neighbors:
                        f.write(f"    {w:<20} {sim:.4f}\n")

        f.write("\n\nPART 2: ANALOGY EXPERIMENTS\n")
        f.write("-"*40 + "\n")
        for model_name, model in models.items():
            f.write(f"\n{model_name}:\n")
            for a, b, c, desc in analogies:
                f.write(f"\n  {desc}\n")
                results = analogy(model, a, b, c)
                if results:
                    for w, sim in results[:3]:
                        f.write(f"    {w:<20} {sim:.4f}\n")

        f.write("\n\nDISCUSSION\n")
        f.write("-"*40 + "\n")
        f.write("The nearest neighbor results show that both models capture\n")
        f.write("domain-specific semantics from the IIT Jodhpur corpus.\n")
        f.write("Words related to academics, courses, and research appear\n")
        f.write("as neighbors for query terms.\n\n")
        f.write("For analogies, the results depend on the corpus coverage.\n")
        f.write("Since the corpus is domain-specific and relatively small,\n")
        f.write("some analogies may not produce ideal results. The models\n")
        f.write("capture relationships that exist in the academic context\n")
        f.write("of IIT Jodhpur data.\n")

    print("\n\nResults saved to semantic_analysis.txt")