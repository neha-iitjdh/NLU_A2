import torch
import os
from P2T1 import (load_names, build_vocab, VanillaRNN, BLSTM, AttentionRNN,
                  generate_name, count_params)

def compute_novelty(generated_names, training_names):
    training_set = set(n.lower() for n in training_names)
    novel = sum(1 for n in generated_names if n.lower() not in training_set)
    return novel / len(generated_names) if generated_names else 0.0

def compute_diversity(generated_names):
    if not generated_names:
        return 0.0
    return len(set(generated_names)) / len(generated_names)


def load_trained_models(vocab_size, device):
    embed_size = 32
    hidden_size = 128

    rnn_model = VanillaRNN(vocab_size, embed_size, hidden_size, 1, dropout=0.1)
    rnn_model.load_state_dict(torch.load('e:/NLU2/models/vanilla_rnn.pt',
                                          map_location=device, weights_only=True))
    rnn_model.to(device)

    blstm_model = BLSTM(vocab_size, embed_size, 64, 1, dropout=0.4)
    blstm_model.load_state_dict(torch.load('e:/NLU2/models/blstm.pt',
                                            map_location=device, weights_only=True))
    blstm_model.to(device)

    attn_model = AttentionRNN(vocab_size, embed_size, hidden_size, 1, dropout=0.3)
    attn_model.load_state_dict(torch.load('e:/NLU2/models/attention_rnn.pt',
                                           map_location=device, weights_only=True))
    attn_model.to(device)

    return rnn_model, blstm_model, attn_model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    names = load_names('e:/NLU2/TrainingNames.txt')
    char_to_idx, idx_to_char = build_vocab(names)
    vocab_size = len(char_to_idx)

    if not os.path.exists('e:/NLU2/models/vanilla_rnn.pt'):
        print("Models not found. Run P2T1.py first.")
        exit(1)

    rnn_model, blstm_model, attn_model = load_trained_models(vocab_size, device)
    print("Loaded models\n")

    models = {
        'Vanilla RNN': rnn_model,
        'BLSTM': blstm_model,
        'RNN + Attention': attn_model
    }

    num_samples = 100

    print("="*60)
    print("QUANTITATIVE EVALUATION")
    print("="*60)
    print(f"Generating {num_samples} names per model...\n")

    results = {}
    for model_name, model in models.items():
        generated = []
        for _ in range(num_samples):
            name = generate_name(model, char_to_idx, idx_to_char, device)
            generated.append(name)

        novelty = compute_novelty(generated, names)
        diversity = compute_diversity(generated)
        results[model_name] = {
            'generated': generated,
            'novelty': novelty,
            'diversity': diversity
        }

    print(f"{'Model':<20} {'Novelty Rate':<15} {'Diversity':<15} {'Parameters':<12}")
    print("-" * 62)
    for model_name in models:
        r = results[model_name]
        p = count_params(models[model_name])
        print(f"{model_name:<20} {r['novelty']:.4f}         {r['diversity']:.4f}         {p}")

    print("\n" + "="*60)
    print("SAMPLE GENERATED NAMES (first 10)")
    print("="*60)
    for model_name in models:
        print(f"\n--- {model_name} ---")
        for name in results[model_name]['generated'][:10]:
            print(f"  {name}")

    with open('e:/NLU2/evaluation_results.txt', 'w') as f:
        f.write("QUANTITATIVE EVALUATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated {num_samples} names per model\n")
        f.write(f"Training set: {len(names)} names\n\n")

        f.write(f"{'Model':<20} {'Novelty':<15} {'Diversity':<15} {'Params':<12}\n")
        f.write("-" * 62 + "\n")
        for model_name in models:
            r = results[model_name]
            p = count_params(models[model_name])
            f.write(f"{model_name:<20} {r['novelty']:.4f}         {r['diversity']:.4f}         {p}\n")

        for model_name in models:
            f.write(f"\n\n--- {model_name} (all {num_samples} names) ---\n")
            for name in results[model_name]['generated']:
                f.write(f"  {name}\n")

    print("\nSaved to evaluation_results.txt")
