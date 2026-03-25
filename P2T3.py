import torch
import os
from P2T1 import (load_names, build_vocab, VanillaRNN, BLSTM, AttentionRNN,
                  generate_name, count_params)
from P2T2 import load_trained_models

def check_name_structure(name):
    parts = name.strip().split()
    if len(parts) == 2:
        first, last = parts
        if (first[0].isupper() and last[0].isupper()
            and first[1:].islower() and last[1:].islower()):
            return "well-formed"
        return "partial"
    elif len(parts) == 1:
        return "single-word"
    else:
        return "multi-word"

def analyze_failures(generated_names):
    failures = {
        'repeated_chars': 0,
        'no_space': 0,
        'too_short': 0,
        'too_long': 0,
        'non_alpha_space': 0,
        'wrong_case': 0,
    }
    for name in generated_names:
        # detect 3 identical chars
        for i in range(len(name) - 2):
            if name[i] == name[i+1] == name[i+2]:
                failures['repeated_chars'] += 1
                break
        if ' ' not in name:
            failures['no_space'] += 1
        if len(name) < 3:
            failures['too_short'] += 1
        if len(name) > 25:
            failures['too_long'] += 1
        if not all(c.isalpha() or c == ' ' for c in name):
            failures['non_alpha_space'] += 1
        if name and not name[0].isupper():
            failures['wrong_case'] += 1
    return failures


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    names = load_names('e:/NLU2/TrainingNames.txt')
    char_to_idx, idx_to_char = build_vocab(names)
    vocab_size = len(char_to_idx)

    if not os.path.exists('e:/NLU2/models/vanilla_rnn.pt'):
        print("Models not found. Run P2T1.py first.")
        exit(1)

    rnn_model, blstm_model, attn_model = load_trained_models(vocab_size, device)
    print("Loaded models\n")

    num_samples = 50
    models_dict = {
        'Vanilla RNN': rnn_model,
        'BLSTM': blstm_model,
        'RNN + Attention': attn_model
    }

    print("="*60)
    print("QUALITATIVE ANALYSIS")
    print("="*60)

    all_results = {}
    for model_name, model in models_dict.items():
        generated = [generate_name(model, char_to_idx, idx_to_char, device)
                     for _ in range(num_samples)]
        all_results[model_name] = generated

    for model_name, generated in all_results.items():
        print(f"\n--- {model_name} ---")
        print("Samples:")
        for name in generated[:10]:
            print(f"  {name}")

        structure_counts = {}
        for name in generated:
            s = check_name_structure(name)
            structure_counts[s] = structure_counts.get(s, 0) + 1
        print(f"\nStructure:")
        for stype, count in sorted(structure_counts.items()):
            print(f"  {stype}: {count}/{num_samples}")

        failures = analyze_failures(generated)
        print(f"\nFailures:")
        has_failures = False
        for ftype, count in failures.items():
            if count > 0:
                print(f"  {ftype}: {count}/{num_samples}")
                has_failures = True
        if not has_failures:
            print("  None")

    with open('e:/NLU2/qualitative_analysis.txt', 'w') as f:
        f.write("QUALITATIVE ANALYSIS\n")
        f.write("="*60 + "\n\n")

        for model_name, generated in all_results.items():
            f.write(f"\n{'='*40}\n")
            f.write(f"{model_name}\n")
            f.write(f"{'='*40}\n\n")

            f.write("Generated samples:\n")
            for i, name in enumerate(generated):
                f.write(f"  {i+1}. {name}\n")

            structure_counts = {}
            for name in generated:
                s = check_name_structure(name)
                structure_counts[s] = structure_counts.get(s, 0) + 1
            f.write(f"\nStructure:\n")
            for stype, count in sorted(structure_counts.items()):
                f.write(f"  {stype}: {count}/{num_samples}\n")

            failures = analyze_failures(generated)
            f.write(f"\nFailures:\n")
            for ftype, count in failures.items():
                if count > 0:
                    f.write(f"  {ftype}: {count}/{num_samples}\n")

    print("\n\nSaved to qualitative_analysis.txt")