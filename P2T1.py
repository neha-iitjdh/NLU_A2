import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

def load_names(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

def build_vocab(names):
    chars = set()
    for name in names:
        for ch in name:
            chars.add(ch)
    chars = sorted(chars)
    char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    for i, ch in enumerate(chars):
        char_to_idx[ch] = i + 3
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char

def name_to_tensor(name, char_to_idx):
    indices = [char_to_idx['<SOS>']]
    for ch in name:
        indices.append(char_to_idx[ch])
    indices.append(char_to_idx['<EOS>'])
    return torch.tensor(indices, dtype=torch.long)

def pad_sequences(batch, pad_idx=0):
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.0):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


class BLSTM(nn.Module):
    # bidirectional LSTM
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.2):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return (h0, c0)


class AttentionRNN(nn.Module):
    # RNN with attention
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.2):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        rnn_out, hidden = self.rnn(emb, hidden)

        batch_size, seq_len, _ = rnn_out.shape
        proj_enc = self.attn_W(rnn_out)

        # causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

        context_list = []
        for t in range(seq_len):
            dec_state = rnn_out[:, t:t+1, :]
            proj_dec = self.attn_U(dec_state)
            scores = self.attn_v(torch.tanh(proj_enc + proj_dec))
            mask = causal_mask[t].unsqueeze(0).unsqueeze(-1)
            scores = scores.masked_fill(~mask, float('-inf'))
            attn_weights = torch.softmax(scores, dim=1)
            context = torch.sum(attn_weights * rnn_out, dim=1, keepdim=True)
            context_list.append(context)

        context_all = torch.cat(context_list, dim=1)
        combined = torch.cat([rnn_out, context_all], dim=2)
        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


def train_model(model, train_data, char_to_idx, epochs=50, batch_size=64,
                lr=0.003, device='cpu'):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    all_tensors = [name_to_tensor(name, char_to_idx) for name in train_data]

    losses = []
    for epoch in range(epochs):
        model.train()
        random.shuffle(all_tensors)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(all_tensors), batch_size):
            batch = all_tensors[i:i+batch_size]
            padded = pad_sequences(batch).to(device)

            inp = padded[:, :-1]
            target = padded[:, 1:]

            hidden = model.init_hidden(inp.size(0), device)
            logits, _ = model(inp, hidden)

            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses


def generate_name(model, char_to_idx, idx_to_char, device='cpu',
                  max_len=25, temperature=0.8):
    model.eval()
    # full context for BLSTM/attention
    needs_full_context = isinstance(model, (BLSTM, AttentionRNN))

    with torch.no_grad():
        generated_indices = [char_to_idx['<SOS>']]
        hidden = None

        for _ in range(max_len):
            if needs_full_context:
                inp = torch.tensor([generated_indices]).to(device)
                h = model.init_hidden(1, device)
                logits, _ = model(inp, h)
                logits = logits[:, -1, :] / temperature
            else:
                inp = torch.tensor([[generated_indices[-1]]]).to(device)
                if hidden is None:
                    hidden = model.init_hidden(1, device)
                logits, hidden = model(inp, hidden)
                logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == char_to_idx['<EOS>']:
                break
            if next_idx == char_to_idx['<PAD>']:
                continue

            generated_indices.append(next_idx)

        name = ''.join(idx_to_char[i] for i in generated_indices[1:])
    return name


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    names = load_names('e:/NLU2/TrainingNames.txt')
    print(f"Loaded {len(names)} names")

    char_to_idx, idx_to_char = build_vocab(names)
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size} characters")

    embed_size = 32
    hidden_size = 128
    num_layers = 1
    epochs = 50
    batch_size = 64
    lr = 0.003

    print(f"\nHyperparameters: embed={embed_size}, hidden={hidden_size}, "
          f"layers={num_layers}, epochs={epochs}, batch={batch_size}, lr={lr}")

    # vanilla RNN
    print("\n" + "="*50)
    print("Training Vanilla RNN")
    print("="*50)
    rnn_model = VanillaRNN(vocab_size, embed_size, hidden_size, num_layers, dropout=0.1)
    print(f"Parameters: {count_params(rnn_model)}")
    rnn_losses = train_model(rnn_model, names, char_to_idx, epochs, batch_size, lr, device)

    print("\nSample generated names:")
    rnn_generated = []
    for i in range(20):
        name = generate_name(rnn_model, char_to_idx, idx_to_char, device)
        rnn_generated.append(name)
        print(f"  {name}")

    # BLSTM (smaller hidden)
    blstm_hidden = 64
    print("\n" + "="*50)
    print("Training BLSTM")
    print("="*50)
    blstm_model = BLSTM(vocab_size, embed_size, blstm_hidden, num_layers, dropout=0.4)
    print(f"Parameters: {count_params(blstm_model)}")
    blstm_losses = train_model(blstm_model, names, char_to_idx, 30, batch_size, lr, device)

    print("\nSample generated names:")
    blstm_generated = []
    for i in range(20):
        name = generate_name(blstm_model, char_to_idx, idx_to_char, device)
        blstm_generated.append(name)
        print(f"  {name}")

    # RNN + attention
    print("\n" + "="*50)
    print("Training RNN with Attention")
    print("="*50)
    attn_model = AttentionRNN(vocab_size, embed_size, hidden_size, num_layers, dropout=0.3)
    print(f"Parameters: {count_params(attn_model)}")
    attn_losses = train_model(attn_model, names, char_to_idx, epochs, batch_size, lr, device)

    print("\nSample generated names:")
    attn_generated = []
    for i in range(20):
        name = generate_name(attn_model, char_to_idx, idx_to_char, device)
        attn_generated.append(name)
        print(f"  {name}")

    # save weights
    os.makedirs('e:/NLU2/models', exist_ok=True)
    torch.save(rnn_model.state_dict(), 'e:/NLU2/models/vanilla_rnn.pt')
    torch.save(blstm_model.state_dict(), 'e:/NLU2/models/blstm.pt')
    torch.save(attn_model.state_dict(), 'e:/NLU2/models/attention_rnn.pt')
    print("\nSaved model weights to models/")

    # save samples
    with open('e:/NLU2/generated_names.txt', 'w') as f:
        for model_name, gen_names in [('vanilla_rnn', rnn_generated),
                                       ('blstm', blstm_generated),
                                       ('attention_rnn', attn_generated)]:
            f.write(f"\n--- {model_name} ---\n")
            for n in gen_names:
                f.write(n + "\n")

    # plot loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rnn_losses, label='Vanilla RNN')
    plt.plot(blstm_losses, label='BLSTM')
    plt.plot(attn_losses, label='RNN + Attention')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('e:/NLU2/training_loss.png', dpi=150)
    plt.close()
    print("Saved training_loss.png")

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"{'Model':<20} {'Params':<15}")
    print("-" * 35)
    print(f"{'Vanilla RNN':<20} {count_params(rnn_model):<15}")
    print(f"{'BLSTM':<20} {count_params(blstm_model):<15}")
    print(f"{'RNN + Attention':<20} {count_params(attn_model):<15}")