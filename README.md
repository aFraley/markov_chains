# From Counting Letters to Modern AI

An educational project that builds the same text generator five different ways, showing the evolution from simple statistics to the architecture behind GPT and Claude.

**The task**: Generate mythology names (Greek, Norse, Celtic) one character at a time.

**The progression**:

| Step | Method | Key Idea | Parameters |
|------|--------|----------|------------|
| 1 | Order-1 Markov Chain | Count letter pairs in training data | ~700 |
| 2 | Higher-Order Markov | Use last N characters as context, not just 1 | ~2,000 |
| 3 | RNN | Learn a hidden state that encodes context | ~7,900 |
| 4 | LSTM | Gated cell state solves vanishing gradients | ~23,580 |
| 5 | Transformer | Attention — every position looks at every earlier position | ~14,590 |

Every implementation is **from scratch with NumPy** — no PyTorch, no TensorFlow, no autograd. Forward pass, backward pass, gradient descent, all written out explicitly so you can see exactly what happens at every step.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy

# Run any model directly
python3 name_generator.py              # Order-1 Markov
python3 name_generator_higher_order.py # Higher-order Markov
python3 rnn_name_generator.py          # Vanilla RNN
python3 lstm_name_generator.py         # LSTM
python3 transformer_name_generator.py  # Transformer
```

## Browser Demos

Start a local server and open in your browser:

```bash
python3 -m http.server 8888
```

Then visit:
- `http://localhost:8888/index.html` — The full explainer site
- `http://localhost:8888/name_compare.html` — All 5 methods side by side
- `http://localhost:8888/rnn_demo.html` — RNN live generation
- `http://localhost:8888/lstm_demo.html` — LSTM live generation
- `http://localhost:8888/transformer_demo.html` — Transformer with attention heatmap

The browser demos run inference entirely client-side using pre-trained weights exported as JSON. No server-side computation needed.

## Project Structure

```
name_generator.py                 # Order-1 Markov chain + training data (NAME_SETS)
name_generator_higher_order.py    # Higher-order Markov with backoff
rnn_name_generator.py             # Vanilla RNN with BPTT
lstm_name_generator.py            # LSTM with gated cell state
transformer_name_generator.py     # Decoder-only transformer with multi-head attention

index.html                        # Main explainer website
name_compare.html                 # Side-by-side 5-method comparison
rnn_demo.html                     # RNN browser demo
lstm_demo.html                    # LSTM browser demo
transformer_demo.html             # Transformer browser demo + attention visualization

markov_chains.json                # Exported Markov chain data
rnn_weights.json                  # Exported RNN weights
lstm_weights.json                 # Exported LSTM weights
transformer_weights.json          # Exported Transformer weights
```

## Training Data

Mythology names from three traditions with distinct phonetic patterns:
- **Greek** (363 names) — Achilles, Persephone, Prometheus, Eurydice...
- **Norse** (169 names) — Aegir, Brunhild, Fenric, Yggdrasil...
- **Celtic** (168 names) — Abandinus, Branwen, Cernunnos, Taliesin...

Source: [repushko/mythology_names_dataset](https://github.com/repushko/mythology_names_dataset) (CC0 public domain), parsed from [Godchecker](https://www.godchecker.com/).

## Inspiration

The original Markov chain name generation concept was inspired by [donjon's Markov Name Generator](https://donjon.bin.sh/code/name/) — a public domain RPG name generator that uses Markov chain analysis to build frequency tables for name length and letter pairs from sample names. The donjon generator and its source code are released to the public domain.

## What You'll Learn

Each step in the progression introduces a fundamental concept:

1. **Markov Chains** — Probability distributions, transition tables, sampling
2. **Higher-Order Markov** — Context windows, state space explosion, backoff
3. **RNN** — Hidden states, backpropagation through time, vanishing gradients
4. **LSTM** — Gating mechanisms, cell state as gradient highway, forget/input/output gates
5. **Transformer** — Attention mechanism, Q/K/V projections, positional encoding, layer normalization

The transformer in step 5 uses the exact same architecture as GPT and Claude — same Q, K, V, same scaled dot-product attention, same residual connections and layer norm. The only difference is scale: our transformer has ~14,590 parameters; GPT-3 has 175 billion.
