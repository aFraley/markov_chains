"""
=============================================================================
LSTM Character-Level Name Generator — From Scratch with NumPy
=============================================================================

THE PROGRESSION SO FAR:
-----------------------
  1. Order-1 Markov (name_generator.py)
     -> Lookup table: given last 1 char, pick the next from recorded frequencies
     -> No memory beyond the current character

  2. Higher-Order Markov (name_generator_higher_order.py)
     -> Lookup table: given last N chars, pick the next (with backoff)
     -> Better results, but state space explodes: 26^N possible contexts
     -> Still just counting — no generalization to unseen contexts

  3. RNN (rnn_name_generator.py)
     -> Replaces the lookup table with a NEURAL NETWORK
     -> Learns a "hidden state" — a compressed summary of EVERYTHING seen so far
     -> Can generalize to unseen contexts
     -> PROBLEM: gradients vanish through repeated W_hh multiplication

  4. LSTM (this file) ★
     -> Adds a CELL STATE with additive updates — the "gradient highway"
     -> Three gates (forget, input, output) learn what to remember and forget
     -> Gradients flow through the cell state without vanishing
     -> Same training data, same task, fundamentally better gradient flow

THE LSTM EQUATIONS:
-------------------
At each timestep t, given input x_t, previous hidden state h_{t-1}, and
previous cell state c_{t-1}:

  f_t = sigmoid(W_xf @ x_t + W_hf @ h_{t-1} + b_f)   # forget gate
  i_t = sigmoid(W_xi @ x_t + W_hi @ h_{t-1} + b_i)   # input gate
  g_t = tanh(W_xg @ x_t + W_hg @ h_{t-1} + b_g)      # candidate values
  c_t = f_t * c_{t-1} + i_t * g_t                      # cell state update
  o_t = sigmoid(W_xo @ x_t + W_ho @ h_{t-1} + b_o)   # output gate
  h_t = o_t * tanh(c_t)                                 # hidden state
  y_t = W_hy @ h_t + b_y                                # output logits
  p_t = softmax(y_t)                                    # probabilities

WHY THIS FIXES VANISHING GRADIENTS:
------------------------------------
In the RNN, gradients flow backward through: dh_next = W_hh.T @ dh_raw
After T steps, the effective multiplication is (W_hh.T)^T — exponential
shrinking if eigenvalues < 1.

In the LSTM, gradients flow through the cell state: dc_prev = dc * f_t
The forget gate f_t is a learned scalar (0 to 1), not a matrix multiplication.
No eigenvalue compounding. The cell state acts as a "gradient highway" —
information and gradients can flow unchanged across many timesteps.

PARAMETER COUNT:
----------------
  RNN:  5 arrays  -> (64×28) + (64×64) + (28×64) + 64 + 28      =  7,900
  LSTM: 13 arrays -> 4×[(64×28) + (64×64) + 64] + (28×64) + 28  = 23,580
  ~3x more parameters for the same hidden size. The cost of gating.

=============================================================================
"""

import time
import random
import json

import numpy as np

from name_generator import NAME_SETS, generate_names
from rnn_name_generator import build_vocab, one_hot, prepare_training_data


# =============================================================================
# HELPER: SIGMOID
# =============================================================================
# The RNN only needed tanh (built into numpy). The LSTM gates use sigmoid,
# which we need to implement ourselves. sigmoid(x) = 1 / (1 + exp(-x)),
# mapping any real number to (0, 1) — perfect for gates that control what
# fraction of information to keep or discard.
# =============================================================================

def sigmoid(x):
    """Numerically stable sigmoid: clip input to prevent overflow in exp."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# THE CHARACTER-LEVEL LSTM
# =============================================================================
# Mirrors the CharRNN class exactly in structure, but replaces the single
# hidden-state recurrence with the gated cell-state mechanism.
#
# The key structural difference: every method that took h_prev now also
# takes c_prev (cell state), and returns c_last alongside h_last.
# =============================================================================

class CharLSTM:
    """
    A character-level LSTM for name generation.

    Architecture:
        input (vocab_size) -> LSTM cell (hidden_size) -> output (vocab_size)

    The LSTM cell maintains TWO pieces of state:
      - h (hidden state): short-term, used for output — same role as RNN's h
      - c (cell state):   long-term, flows with additive updates — THE NEW THING

    The four gates control information flow:
      - Forget gate (f): what to erase from cell state
      - Input gate (i):  what new information to write
      - Candidate (g):   what new values to propose
      - Output gate (o): what to expose as hidden state
    """

    def __init__(self, vocab_size, hidden_size=64):
        """
        Initialize the LSTM with random weights.

        We have 13 learnable parameter arrays:
          4 gates × 3 arrays each (W_x*, W_h*, b_*) = 12 gate arrays
          + W_hy and b_y for the output layer = 2 output arrays
          Total: 14 arrays, but b_y counts with the output layer → 13 unique

        FORGET GATE BIAS = 1.0:
        The forget gate bias is initialized to 1.0 instead of 0.0.
        At the start of training, sigmoid(1.0) ≈ 0.73, so the default
        behavior is "remember most things." With b_f = 0, sigmoid(0) = 0.5,
        meaning the gate starts by discarding half of everything — too
        aggressive before the network has learned what's important.
        This trick is from Jozefowicz et al. (2015).
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # --- Forget gate weights ---
        self.W_xf = np.random.randn(hidden_size, vocab_size) / np.sqrt(vocab_size)
        self.W_hf = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.b_f = np.ones((hidden_size, 1))  # ← initialized to 1.0, not 0!

        # --- Input gate weights ---
        self.W_xi = np.random.randn(hidden_size, vocab_size) / np.sqrt(vocab_size)
        self.W_hi = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.b_i = np.zeros((hidden_size, 1))

        # --- Candidate value weights ---
        self.W_xg = np.random.randn(hidden_size, vocab_size) / np.sqrt(vocab_size)
        self.W_hg = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.b_g = np.zeros((hidden_size, 1))

        # --- Output gate weights ---
        self.W_xo = np.random.randn(hidden_size, vocab_size) / np.sqrt(vocab_size)
        self.W_ho = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.b_o = np.zeros((hidden_size, 1))

        # --- Output layer (same as RNN) ---
        self.W_hy = np.random.randn(vocab_size, hidden_size) / np.sqrt(hidden_size)
        self.b_y = np.zeros((vocab_size, 1))

        # --- Adagrad memory for all 14 parameter arrays ---
        self.mem_W_xf = np.zeros_like(self.W_xf)
        self.mem_W_hf = np.zeros_like(self.W_hf)
        self.mem_b_f = np.zeros_like(self.b_f)
        self.mem_W_xi = np.zeros_like(self.W_xi)
        self.mem_W_hi = np.zeros_like(self.W_hi)
        self.mem_b_i = np.zeros_like(self.b_i)
        self.mem_W_xg = np.zeros_like(self.W_xg)
        self.mem_W_hg = np.zeros_like(self.W_hg)
        self.mem_b_g = np.zeros_like(self.b_g)
        self.mem_W_xo = np.zeros_like(self.W_xo)
        self.mem_W_ho = np.zeros_like(self.W_ho)
        self.mem_b_o = np.zeros_like(self.b_o)
        self.mem_W_hy = np.zeros_like(self.W_hy)
        self.mem_b_y = np.zeros_like(self.b_y)

    def forward(self, inputs, targets, h_prev, c_prev):
        """
        Forward pass: process a sequence through the LSTM and compute loss.

        Same structure as CharRNN.forward(), but we now track:
          - 4 gate activations per timestep (f, i, g, o) for backprop
          - Cell state c alongside hidden state h

        Args:
            inputs:  list of input character indices
            targets: list of target character indices
            h_prev:  initial hidden state (hidden_size, 1)
            c_prev:  initial cell state (hidden_size, 1)

        Returns:
            loss:  scalar cross-entropy loss
            cache: dict of all intermediate values for backprop
        """
        xs, hs, cs = {}, {}, {}
        fs, is_, gs, os_ = {}, {}, {}, {}
        ys, ps = {}, {}

        hs[-1] = h_prev.copy()
        cs[-1] = c_prev.copy()

        loss = 0.0

        for t in range(len(inputs)):
            # Encode input
            xs[t] = one_hot(inputs[t], self.vocab_size)

            # --- THE LSTM EQUATIONS ---
            # These 6 lines replace the RNN's single line:
            #   h = tanh(W_xh @ x + W_hh @ h + b_h)

            # Forget gate: what to erase from cell state
            # f ≈ 1 means "keep everything", f ≈ 0 means "forget everything"
            fs[t] = sigmoid(self.W_xf @ xs[t] + self.W_hf @ hs[t-1] + self.b_f)

            # Input gate: what fraction of new candidate to write
            is_[t] = sigmoid(self.W_xi @ xs[t] + self.W_hi @ hs[t-1] + self.b_i)

            # Candidate values: what new information to propose
            gs[t] = np.tanh(self.W_xg @ xs[t] + self.W_hg @ hs[t-1] + self.b_g)

            # Cell state update: THE KEY DIFFERENCE FROM RNN
            # This is ADDITIVE (f*c_prev + i*g), not multiplicative (W_hh @ h).
            # This is why gradients don't vanish — no matrix multiplication in
            # the gradient path through time, just element-wise multiplication
            # by the forget gate.
            cs[t] = fs[t] * cs[t-1] + is_[t] * gs[t]

            # Output gate: what to expose as hidden state
            os_[t] = sigmoid(self.W_xo @ xs[t] + self.W_ho @ hs[t-1] + self.b_o)

            # Hidden state: gated version of cell state
            hs[t] = os_[t] * np.tanh(cs[t])

            # Output layer (identical to RNN)
            ys[t] = self.W_hy @ hs[t] + self.b_y
            exp_ys = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = exp_ys / np.sum(exp_ys)

            loss += -np.log(ps[t][targets[t], 0] + 1e-8)

        cache = {
            'xs': xs, 'hs': hs, 'cs': cs,
            'fs': fs, 'is_': is_, 'gs': gs, 'os_': os_,
            'ys': ys, 'ps': ps,
            'inputs': inputs, 'targets': targets,
        }

        return loss, cache

    def backward(self, cache):
        """
        Backpropagation Through Time for the LSTM.

        THE KEY DIFFERENCE FROM RNN BACKWARD:
        In the RNN, the gradient through time is:
            dh_next = W_hh.T @ dh_raw     ← matrix multiplication every step

        In the LSTM, the gradient through the cell state is:
            dc_prev = dc * f_t             ← element-wise × forget gate

        The forget gate f_t is a scalar (0 to 1) per hidden unit, not a matrix.
        There is no eigenvalue compounding. If f_t ≈ 1, the gradient flows
        through unchanged — the "gradient highway."

        GATE GRADIENTS:
        Each gate uses either sigmoid or tanh, so we need their derivatives:
          sigmoid: d/dx σ(x) = σ(x) * (1 - σ(x))   → used for f, i, o gates
          tanh:    d/dx tanh(x) = 1 - tanh(x)^2      → used for g (candidate)

        Since we cached the gate OUTPUTS (not pre-activation), we can compute
        these derivatives directly from the cached values.
        """
        xs = cache['xs']
        hs, cs = cache['hs'], cache['cs']
        fs, is_, gs, os_ = cache['fs'], cache['is_'], cache['gs'], cache['os_']
        ps = cache['ps']
        targets = cache['targets']

        T = len(targets)

        # Initialize all gradients to zero
        dW_xf = np.zeros_like(self.W_xf)
        dW_hf = np.zeros_like(self.W_hf)
        db_f = np.zeros_like(self.b_f)
        dW_xi = np.zeros_like(self.W_xi)
        dW_hi = np.zeros_like(self.W_hi)
        db_i = np.zeros_like(self.b_i)
        dW_xg = np.zeros_like(self.W_xg)
        dW_hg = np.zeros_like(self.W_hg)
        db_g = np.zeros_like(self.b_g)
        dW_xo = np.zeros_like(self.W_xo)
        dW_ho = np.zeros_like(self.W_ho)
        db_o = np.zeros_like(self.b_o)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        # Gradient flowing backward from future timesteps
        dh_next = np.zeros_like(hs[0])
        dc_next = np.zeros_like(cs[0])

        for t in reversed(range(T)):
            # --- Output layer gradient (same as RNN) ---
            dy = ps[t].copy()
            dy[targets[t]] -= 1

            dW_hy += dy @ hs[t].T
            db_y += dy

            # --- Hidden state gradient ---
            # Flows from output layer AND from the next timestep
            dh = self.W_hy.T @ dy + dh_next

            # --- Output gate gradient ---
            # h_t = o_t * tanh(c_t), so dh flows to both o and c
            do = dh * np.tanh(cs[t])
            # Backprop through sigmoid: d_raw = d_out * sigmoid * (1 - sigmoid)
            do_raw = do * os_[t] * (1 - os_[t])

            dW_xo += do_raw @ xs[t].T
            dW_ho += do_raw @ hs[t-1].T
            db_o += do_raw

            # --- Cell state gradient ---
            # Gradient from two sources:
            #   1. Through h_t = o_t * tanh(c_t): dh * o_t * (1 - tanh(c_t)^2)
            #   2. From the NEXT timestep's cell state: dc_next
            dc = dh * os_[t] * (1 - np.tanh(cs[t]) ** 2) + dc_next

            # --- Forget gate gradient ---
            # c_t = f_t * c_{t-1} + i_t * g_t, so dc flows to f, i, g
            df = dc * cs[t-1]
            df_raw = df * fs[t] * (1 - fs[t])  # sigmoid derivative

            dW_xf += df_raw @ xs[t].T
            dW_hf += df_raw @ hs[t-1].T
            db_f += df_raw

            # --- Input gate gradient ---
            di = dc * gs[t]
            di_raw = di * is_[t] * (1 - is_[t])  # sigmoid derivative

            dW_xi += di_raw @ xs[t].T
            dW_hi += di_raw @ hs[t-1].T
            db_i += di_raw

            # --- Candidate gradient ---
            dg = dc * is_[t]
            dg_raw = dg * (1 - gs[t] ** 2)  # tanh derivative

            dW_xg += dg_raw @ xs[t].T
            dW_hg += dg_raw @ hs[t-1].T
            db_g += dg_raw

            # --- Propagate gradients backward in time ---

            # Cell state gradient: flows through the forget gate
            # dc_prev = dc * f_t
            # THIS IS THE KEY: no matrix multiplication, just element-wise
            # scaling by the forget gate. If f ≈ 1, gradient flows unchanged.
            dc_next = dc * fs[t]

            # Hidden state gradient: flows through all 4 gates' W_h* matrices
            dh_next = (self.W_hf.T @ df_raw + self.W_hi.T @ di_raw +
                       self.W_hg.T @ dg_raw + self.W_ho.T @ do_raw)

        grads = {
            'W_xf': dW_xf, 'W_hf': dW_hf, 'b_f': db_f,
            'W_xi': dW_xi, 'W_hi': dW_hi, 'b_i': db_i,
            'W_xg': dW_xg, 'W_hg': dW_hg, 'b_g': db_g,
            'W_xo': dW_xo, 'W_ho': dW_ho, 'b_o': db_o,
            'W_hy': dW_hy, 'b_y': db_y,
        }

        return grads

    def clip_gradients(self, grads, max_norm=5):
        """Clip gradients element-wise to [-max_norm, max_norm]. Same as RNN."""
        for key in grads:
            np.clip(grads[key], -max_norm, max_norm, out=grads[key])
        return grads

    def train_step(self, inputs, targets, h_prev, c_prev, learning_rate):
        """
        One complete training step: forward -> backward -> clip -> Adagrad.

        Same as RNN's train_step, but passes cell state alongside hidden state
        and updates 14 parameter arrays instead of 5.

        Returns:
            loss:   scalar loss for this sequence
            h_last: final hidden state
            c_last: final cell state
        """
        loss, cache = self.forward(inputs, targets, h_prev, c_prev)
        grads = self.backward(cache)
        grads = self.clip_gradients(grads)

        # Adagrad update for all 14 parameter arrays
        for param, grad, mem in [
            ('W_xf', grads['W_xf'], 'mem_W_xf'),
            ('W_hf', grads['W_hf'], 'mem_W_hf'),
            ('b_f',  grads['b_f'],  'mem_b_f'),
            ('W_xi', grads['W_xi'], 'mem_W_xi'),
            ('W_hi', grads['W_hi'], 'mem_W_hi'),
            ('b_i',  grads['b_i'],  'mem_b_i'),
            ('W_xg', grads['W_xg'], 'mem_W_xg'),
            ('W_hg', grads['W_hg'], 'mem_W_hg'),
            ('b_g',  grads['b_g'],  'mem_b_g'),
            ('W_xo', grads['W_xo'], 'mem_W_xo'),
            ('W_ho', grads['W_ho'], 'mem_W_ho'),
            ('b_o',  grads['b_o'],  'mem_b_o'),
            ('W_hy', grads['W_hy'], 'mem_W_hy'),
            ('b_y',  grads['b_y'],  'mem_b_y'),
        ]:
            mem_arr = getattr(self, mem)
            mem_arr += grad * grad
            setattr(self, param,
                    getattr(self, param) - learning_rate * grad
                    / (np.sqrt(mem_arr) + 1e-8))

        h_last = cache['hs'][len(inputs) - 1]
        c_last = cache['cs'][len(inputs) - 1]
        return loss, h_last, c_last

    def sample(self, char_to_ix, ix_to_char, temperature=1.0, max_len=20):
        """
        Generate a name autoregressively, same as RNN but tracking cell state.

        The LSTM sample loop is identical to the RNN's, with one addition:
        we maintain both h (hidden state) and c (cell state) across steps.
        """
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        ix = char_to_ix['^']
        name_chars = []

        for _ in range(max_len):
            x = one_hot(ix, self.vocab_size)

            # LSTM forward step (same equations as in forward())
            f = sigmoid(self.W_xf @ x + self.W_hf @ h + self.b_f)
            i = sigmoid(self.W_xi @ x + self.W_hi @ h + self.b_i)
            g = np.tanh(self.W_xg @ x + self.W_hg @ h + self.b_g)
            c = f * c + i * g
            o = sigmoid(self.W_xo @ x + self.W_ho @ h + self.b_o)
            h = o * np.tanh(c)

            # Output + temperature + softmax
            y = self.W_hy @ h + self.b_y
            y = y / temperature
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)

            ix = np.random.choice(self.vocab_size, p=p.ravel())

            if ix_to_char[ix] == '$':
                break

            name_chars.append(ix_to_char[ix])

        return ''.join(name_chars).title()


# =============================================================================
# TRAINING LOOP
# =============================================================================
# Same as the RNN training loop, but passes c_prev alongside h_prev.
# We reset both hidden and cell state for each name (names are independent).
# =============================================================================

def train(model, data, char_to_ix, ix_to_char, epochs=500, lr=0.02,
          print_every=50):
    """Train the LSTM on a set of names."""
    loss_history = []

    for epoch in range(1, epochs + 1):
        random.shuffle(data)

        epoch_loss = 0.0
        total_chars = 0

        for inputs, targets in data:
            h_prev = np.zeros((model.hidden_size, 1))
            c_prev = np.zeros((model.hidden_size, 1))

            loss, _, _ = model.train_step(inputs, targets, h_prev, c_prev, lr)
            epoch_loss += loss
            total_chars += len(targets)

        avg_loss = epoch_loss / total_chars
        loss_history.append(avg_loss)

        if epoch % print_every == 0 or epoch == 1:
            sample_name = model.sample(char_to_ix, ix_to_char, temperature=0.8)
            print(f"  Epoch {epoch:4d}/{epochs}  |  Loss: {avg_loss:.3f}"
                  f"  |  Sample: {sample_name}")

    return loss_history


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("  LSTM Character-Level Name Generator — From Scratch with NumPy")
    print("=" * 70)
    print()
    print("The LSTM adds a cell state with additive updates — the 'gradient")
    print("highway' — and three gates that learn what to remember, write, and")
    print("expose. This solves the vanishing gradient problem that limits the")
    print("vanilla RNN on longer sequences.")
    print()
    print("PROGRESSION:")
    print("  1. Order-1 Markov: lookup table, 1-char memory")
    print("  2. Higher-Order Markov: lookup table, N-char memory (with backoff)")
    print("  3. RNN: neural network, learned memory (gradients vanish)")
    print("  4. LSTM (this file): gated cell state, gradient highway")
    print()

    char_to_ix, ix_to_char, vocab_size = build_vocab(NAME_SETS)
    print(f"Vocabulary: {vocab_size} symbols "
          f"({vocab_size - 2} letters + start + end tokens)")
    print(f"Parameters: ~{4 * ((64 * vocab_size) + (64 * 64) + 64) + (vocab_size * 64) + vocab_size:,} "
          f"(LSTM with hidden_size=64)")
    print()

    temperatures = [0.5, 0.8, 1.2]
    num_samples = 8

    # Import RNN for comparison
    from rnn_name_generator import CharRNN

    for name_type, names in NAME_SETS.items():
        print("=" * 70)
        print(f"  Training on: {name_type.upper()} ({len(names)} names)")
        print("=" * 70)

        data = prepare_training_data(names, char_to_ix)

        # Train LSTM
        model = CharLSTM(vocab_size, hidden_size=64)
        print(f"\n  Training LSTM for 800 epochs (lr=0.02, hidden_size=64)...")
        print(f"  Initial loss should be ~{-np.log(1.0 / vocab_size):.1f} "
              f"(random guessing)\n")

        start_time = time.time()
        loss_history = train(
            model, data, char_to_ix, ix_to_char,
            epochs=800, lr=0.02, print_every=200,
        )
        elapsed = time.time() - start_time

        print(f"\n  Training completed in {elapsed:.1f}s")
        print(f"  Final loss: {loss_history[-1]:.3f}")

        # --- Generate names at different temperatures ---
        print(f"\n  --- LSTM Generated Names (by temperature) ---")
        print(f"  {'t=0.5 (conserv.)':<22}{'t=0.8 (balanced)':<22}"
              f"{'t=1.2 (creative)':<22}")
        print(f"  {'-' * 64}")

        for _ in range(num_samples):
            cols = []
            for temp in temperatures:
                name = model.sample(char_to_ix, ix_to_char, temperature=temp)
                if not name:
                    name = "(empty)"
                cols.append(name[:20].ljust(22))
            print(f"  {''.join(cols)}")

        # --- Side-by-side: LSTM vs RNN vs Markov ---
        print(f"\n  --- Comparison: LSTM vs RNN vs Markov ---")

        # Quick-train an RNN for comparison
        rnn_model = CharRNN(vocab_size, hidden_size=64)
        print(f"  (Training RNN for comparison...)")
        rnn_data = prepare_training_data(names, char_to_ix)
        from rnn_name_generator import train as rnn_train
        rnn_train(rnn_model, rnn_data, char_to_ix, ix_to_char,
                  epochs=500, lr=0.02, print_every=500)

        markov_names = generate_names(name_type, num_samples)
        rnn_names = [rnn_model.sample(char_to_ix, ix_to_char, temperature=0.8)
                     for _ in range(num_samples)]
        lstm_names = [model.sample(char_to_ix, ix_to_char, temperature=0.8)
                      for _ in range(num_samples)]

        print(f"\n  {'LSTM (t=0.8)':<22}{'RNN (t=0.8)':<22}{'Markov (order-1)':<22}")
        print(f"  {'-' * 64}")
        for j in range(num_samples):
            l = (lstm_names[j] or "(empty)")[:20].ljust(22)
            r = (rnn_names[j] or "(empty)")[:20].ljust(22)
            m = (markov_names[j] or "")[:20].ljust(22)
            print(f"  {l}{r}{m}")

        print()

    # --- Final observations ---
    print("=" * 70)
    print("  OBSERVATIONS")
    print("=" * 70)
    print()
    print("  LSTM vs RNN:")
    print("    - Both learn character patterns from the same training data")
    print("    - LSTM has ~3x more parameters (23,580 vs 7,900 for hidden=64)")
    print("    - LSTM's cell state preserves information across longer spans")
    print("    - For short names (5-10 chars), the advantage is subtle")
    print("    - For longer sequences (100+ steps), LSTM dramatically outperforms")
    print()
    print("  THE GRADIENT HIGHWAY:")
    print("    - RNN backward: dh_next = W_hh.T @ dh_raw  (matrix multiply)")
    print("    - LSTM backward: dc_prev = dc * f_t          (element-wise)")
    print("    - No matrix eigenvalues compounding. Gradients flow freely.")
    print()
    print("  WHAT'S NEXT:")
    print("    - Transformer: drop recurrence entirely, use attention")
    print("    - Every position attends directly to every earlier position")
    print("    - No hidden state chain, no sequential bottleneck")
    print("    - The architecture behind GPT, Claude, and modern LLMs")
    print()


# =============================================================================
# EXPORT WEIGHTS
# =============================================================================

def export_weights(epochs=800, lr=0.02, hidden_size=64):
    """Train one model per category and export all weights to lstm_weights.json."""
    char_to_ix, ix_to_char, vocab_size = build_vocab(NAME_SETS)

    models = {}
    for name_type, names in NAME_SETS.items():
        print(f"Training {name_type}...")
        data = prepare_training_data(names, char_to_ix)
        model = CharLSTM(vocab_size, hidden_size=hidden_size)
        train(model, data, char_to_ix, ix_to_char,
              epochs=epochs, lr=lr, print_every=200)

        models[name_type] = {
            "W_xf": model.W_xf.tolist(), "W_hf": model.W_hf.tolist(), "b_f": model.b_f.tolist(),
            "W_xi": model.W_xi.tolist(), "W_hi": model.W_hi.tolist(), "b_i": model.b_i.tolist(),
            "W_xg": model.W_xg.tolist(), "W_hg": model.W_hg.tolist(), "b_g": model.b_g.tolist(),
            "W_xo": model.W_xo.tolist(), "W_ho": model.W_ho.tolist(), "b_o": model.b_o.tolist(),
            "W_hy": model.W_hy.tolist(), "b_y": model.b_y.tolist(),
        }
        sample = model.sample(char_to_ix, ix_to_char, temperature=0.8)
        print(f"  Done. Sample: {sample}")

    payload = {
        "vocab": {
            "char_to_ix": char_to_ix,
            "ix_to_char": {str(k): v for k, v in ix_to_char.items()},
            "vocab_size": vocab_size,
        },
        "hidden_size": hidden_size,
        "models": models,
    }

    with open("lstm_weights.json", "w") as f:
        json.dump(payload, f)

    print(f"\nExported weights to lstm_weights.json ({vocab_size} vocab, "
          f"{hidden_size} hidden, {len(models)} models)")


if __name__ == "__main__":
    demo()
