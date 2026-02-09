"""
=============================================================================
Transformer Character-Level Name Generator — From Scratch with NumPy
=============================================================================

THE PROGRESSION:
----------------
  1. Order-1 Markov (name_generator.py)
     -> Lookup table, 1-char memory, no learning

  2. Higher-Order Markov (name_generator_higher_order.py)
     -> Lookup table, N-char memory, backoff for unseen contexts

  3. RNN (rnn_name_generator.py)
     -> Neural network, learned hidden state, gradients vanish through W_hh

  4. LSTM (lstm_name_generator.py)
     -> Gated cell state, additive gradient highway, still sequential

  5. Transformer (this file) ★
     -> Drop recurrence entirely. Every position attends directly to every
        earlier position through learned attention weights. No hidden state
        chain, no sequential bottleneck, fully parallelizable.

THE TRANSFORMER EQUATIONS:
--------------------------
Given input sequence X (T positions, each a d_model-dimensional vector):

  SELF-ATTENTION (the core new concept):
    Q = X @ W_Q + b_Q                           # queries
    K = X @ W_K + b_K                           # keys
    V = X @ W_V + b_V                           # values
    scores = (Q @ K.T) / sqrt(d_k) + mask       # compatibility + causal mask
    attn = softmax(scores)                       # attention weights
    out = attn @ V                               # weighted combination

  FULL BLOCK (pre-norm, GPT-2/LLaMA style):
    X = X + MultiHeadAttention(LayerNorm(X))     # attend + residual
    X = X + FFN(LayerNorm(X))                    # feed-forward + residual
    logits = LayerNorm(X) @ W_out + b_out        # project to vocab

WHY ATTENTION REPLACES RECURRENCE:
----------------------------------
In an RNN/LSTM, character 8 is 7 sequential steps from character 1.
Information must survive through every intermediate hidden state.

In a transformer, character 8 is ONE attention step from character 1.
Direct connection, no intermediaries, no bottleneck. This is O(1) depth
vs O(T) for recurrent models.

The attention pattern — which positions each character looks at — is
itself LEARNED. The network discovers that, e.g., the ending of a Greek
name should attend strongly to the beginning (to maintain style).

PARAMETER COUNT:
----------------
  RNN:         ~7,900   (hidden=64)
  Transformer: ~14,590  (d_model=32, n_heads=2, d_ff=128)
  LSTM:        ~23,580  (hidden=64)

The transformer sits between RNN and LSTM in parameter count but has
fundamentally different connectivity — every position sees everything.

PRE-NORM vs POST-NORM:
-----------------------
The original "Attention Is All You Need" (2017) uses post-norm: apply
LayerNorm after the residual addition. Modern models (GPT-2, LLaMA)
use pre-norm: apply LayerNorm before the sublayer. Pre-norm is more
stable during training because gradients flow directly through the
residual path without passing through normalization. We use pre-norm.

CAVEAT:
-------
~44 training names is absurdly small for a transformer. We deliberately
overfit to demonstrate the mechanism. In practice, transformers shine
with thousands to billions of training examples.

=============================================================================
"""

import time
import random
import json

import numpy as np

from name_generator import NAME_SETS, generate_names
from rnn_name_generator import build_vocab, prepare_training_data, CharRNN
from lstm_name_generator import CharLSTM


# =============================================================================
# THE CHARACTER-LEVEL TRANSFORMER
# =============================================================================
# A decoder-only transformer — the same architecture class as GPT and Claude.
# Causal (left-to-right) attention, autoregressive generation.
#
# Key structural difference from RNN/LSTM: no hidden state carried between
# timesteps. The entire sequence is processed as a matrix operation.
# All positions are computed simultaneously.
# =============================================================================

class CharTransformer:
    """
    A character-level decoder-only Transformer for name generation.

    Architecture:
        input chars -> embedding + positional encoding
                    -> 1 transformer block:
                         multi-head causal self-attention (2 heads)
                         + residual + LayerNorm
                         feed-forward network (32 -> 128 -> 32)
                         + residual + LayerNorm
                    -> final LayerNorm -> linear -> softmax

    Unlike the RNN/LSTM, there is NO recurrence and NO hidden state.
    Every position attends to every earlier position directly.
    """

    def __init__(self, vocab_size, d_model=32, n_heads=2, d_ff=128, max_len=20):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 16 per head
        self.d_ff = d_ff
        self.max_len = max_len

        # --- Learned character embedding ---
        # Replaces one_hot: instead of (vocab_size,) sparse vector multiplied
        # by a weight matrix, we directly index into an embedding table.
        # E @ one_hot(ix) = E[ix] — same result, much more efficient.
        self.W_emb = np.random.randn(vocab_size, d_model) / np.sqrt(vocab_size)

        # --- Sinusoidal positional encoding (fixed, not learned) ---
        self.PE = self._positional_encoding(max_len, d_model)

        # --- Multi-head self-attention weights ---
        self.W_Q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.b_Q = np.zeros(d_model)
        self.W_K = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.b_K = np.zeros(d_model)
        self.W_V = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.b_V = np.zeros(d_model)
        self.W_O = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.b_O = np.zeros(d_model)

        # --- Feed-forward network weights ---
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

        # --- LayerNorm parameters (3 norms: pre-attn, pre-ffn, final) ---
        self.gamma_1 = np.ones(d_model)
        self.beta_1 = np.zeros(d_model)
        self.gamma_2 = np.ones(d_model)
        self.beta_2 = np.zeros(d_model)
        self.gamma_f = np.ones(d_model)
        self.beta_f = np.zeros(d_model)

        # --- Output projection ---
        self.W_out = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)
        self.b_out = np.zeros(vocab_size)

        # --- Adagrad memory for all learnable parameters ---
        self._param_names = [
            'W_emb',
            'W_Q', 'b_Q', 'W_K', 'b_K', 'W_V', 'b_V', 'W_O', 'b_O',
            'W1', 'b1', 'W2', 'b2',
            'gamma_1', 'beta_1', 'gamma_2', 'beta_2', 'gamma_f', 'beta_f',
            'W_out', 'b_out',
        ]
        for name in self._param_names:
            setattr(self, 'mem_' + name, np.zeros_like(getattr(self, name)))

    def _positional_encoding(self, max_len, d_model):
        """
        Sinusoidal positional encoding from "Attention Is All You Need."

        Since the transformer has no recurrence, it has no notion of position.
        We add a fixed signal that encodes where each character sits:
          PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
          PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

        Different frequencies for different dimensions — the network can learn
        to attend to absolute and relative positions from these signals.
        """
        PE = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                PE[pos, i] = np.sin(angle)
                if i + 1 < d_model:
                    PE[pos, i + 1] = np.cos(angle)
        return PE

    # -----------------------------------------------------------------
    # LayerNorm forward / backward
    # -----------------------------------------------------------------

    def _layer_norm_forward(self, x, gamma, beta, eps=1e-5):
        """
        Layer normalization: normalize each row (position) independently.

        For each position: x_hat = (x - mean) / sqrt(var + eps)
                           out = gamma * x_hat + beta

        Returns (output, cache) where cache is needed for backward.
        """
        mean = x.mean(axis=-1, keepdims=True)       # (T, 1)
        var = x.var(axis=-1, keepdims=True)          # (T, 1)
        x_hat = (x - mean) / np.sqrt(var + eps)     # (T, d_model)
        out = gamma * x_hat + beta                   # (T, d_model)
        cache = (x, x_hat, mean, var, gamma, eps)
        return out, cache

    def _layer_norm_backward(self, dout, cache):
        """
        LayerNorm backward — the standard derivation.

        The variance term creates coupling between all elements in the
        normalization dimension, making this trickier than it looks.
        """
        x, x_hat, mean, var, gamma, eps = cache
        N = x.shape[-1]  # d_model
        dgamma = np.sum(dout * x_hat, axis=0)     # (d_model,)
        dbeta = np.sum(dout, axis=0)               # (d_model,)
        dx_hat = dout * gamma                      # (T, d_model)
        std_inv = 1.0 / np.sqrt(var + eps)         # (T, 1)
        dx = (1.0 / N) * std_inv * (
            N * dx_hat
            - np.sum(dx_hat, axis=-1, keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        )
        return dx, dgamma, dbeta

    # -----------------------------------------------------------------
    # Multi-head causal self-attention forward / backward
    # -----------------------------------------------------------------

    def _attention_forward(self, X):
        """
        Multi-head causal self-attention.

        1. Project X to Q, K, V
        2. Split into n_heads
        3. Scaled dot-product attention with causal mask
        4. Concatenate heads
        5. Project through W_O
        """
        T = X.shape[0]

        # Project to queries, keys, values
        Q = X @ self.W_Q + self.b_Q    # (T, d_model)
        K = X @ self.W_K + self.b_K
        V = X @ self.W_V + self.b_V

        # Reshape for multi-head: (T, d_model) -> (n_heads, T, d_k)
        Q_h = Q.reshape(T, self.n_heads, self.d_k).transpose(1, 0, 2)
        K_h = K.reshape(T, self.n_heads, self.d_k).transpose(1, 0, 2)
        V_h = V.reshape(T, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Scaled dot-product attention
        scores = Q_h @ K_h.transpose(0, 2, 1) / np.sqrt(self.d_k)  # (n_heads, T, T)

        # Causal mask: position i can only attend to positions 0..i
        # Upper triangle (above diagonal) gets -1e9 so softmax gives ~0
        causal_mask = np.triu(np.full((T, T), -1e9), k=1)
        scores = scores + causal_mask  # broadcast over heads

        # Row-wise softmax (over key dimension)
        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # (n_heads, T, T)

        # Weighted sum of values
        attn_out = attn @ V_h  # (n_heads, T, d_k)

        # Concatenate heads: (n_heads, T, d_k) -> (T, d_model)
        concat = attn_out.transpose(1, 0, 2).reshape(T, self.d_model)

        # Output projection
        out = concat @ self.W_O + self.b_O  # (T, d_model)

        cache = (X, Q, K, V, Q_h, K_h, V_h, scores, attn, attn_out, concat)
        return out, cache

    def _attention_backward(self, dout, cache):
        """
        Backward through multi-head causal self-attention.

        Walks backward through: W_O -> concat -> heads -> softmax -> scores -> Q,K,V -> projections
        """
        X, Q, K, V, Q_h, K_h, V_h, scores, attn, attn_out, concat = cache
        T = X.shape[0]

        # Through output projection
        dW_O = concat.T @ dout                       # (d_model, d_model)
        db_O = dout.sum(axis=0)                      # (d_model,)
        dconcat = dout @ self.W_O.T                  # (T, d_model)

        # Un-concatenate heads: (T, d_model) -> (n_heads, T, d_k)
        dattn_out = dconcat.reshape(T, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Through attn @ V:
        #   attn_out = attn @ V_h
        #   d_attn = dattn_out @ V_h.T
        #   dV_h = attn.T @ dattn_out
        d_attn = dattn_out @ V_h.transpose(0, 2, 1)  # (n_heads, T, T)
        dV_h = attn.transpose(0, 2, 1) @ dattn_out   # (n_heads, T, d_k)

        # Through softmax (per row):
        #   ds = attn * (d_attn - sum(d_attn * attn, axis=-1, keepdims=True))
        dscores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))

        # Through scaling
        dscores = dscores / np.sqrt(self.d_k)

        # Through Q @ K.T:
        #   scores = Q_h @ K_h.T
        #   dQ_h = dscores @ K_h
        #   dK_h = dscores.T @ Q_h  (transpose the T,T dimension)
        dQ_h = dscores @ K_h                                     # (n_heads, T, d_k)
        dK_h = dscores.transpose(0, 2, 1) @ Q_h                 # (n_heads, T, d_k)

        # Un-reshape heads: (n_heads, T, d_k) -> (T, d_model)
        dQ = dQ_h.transpose(1, 0, 2).reshape(T, self.d_model)
        dK = dK_h.transpose(1, 0, 2).reshape(T, self.d_model)
        dV = dV_h.transpose(1, 0, 2).reshape(T, self.d_model)

        # Through Q, K, V projections
        dW_Q = X.T @ dQ;  db_Q = dQ.sum(axis=0)
        dW_K = X.T @ dK;  db_K = dK.sum(axis=0)
        dW_V = X.T @ dV;  db_V = dV.sum(axis=0)

        # Gradient to input X from all three projections
        dX = dQ @ self.W_Q.T + dK @ self.W_K.T + dV @ self.W_V.T

        grads = {
            'W_Q': dW_Q, 'b_Q': db_Q, 'W_K': dW_K, 'b_K': db_K,
            'W_V': dW_V, 'b_V': db_V, 'W_O': dW_O, 'b_O': db_O,
        }
        return dX, grads

    # -----------------------------------------------------------------
    # Feed-forward network forward / backward
    # -----------------------------------------------------------------

    def _ffn_forward(self, X):
        """FFN: Linear -> ReLU -> Linear. Expands to d_ff=128, contracts back."""
        hidden = X @ self.W1 + self.b1               # (T, d_ff)
        relu_mask = hidden > 0
        hidden_relu = hidden * relu_mask              # ReLU
        out = hidden_relu @ self.W2 + self.b2         # (T, d_model)
        cache = (X, hidden, relu_mask, hidden_relu)
        return out, cache

    def _ffn_backward(self, dout, cache):
        """Backward through FFN: standard MLP backprop with ReLU mask."""
        X, hidden, relu_mask, hidden_relu = cache

        dW2 = hidden_relu.T @ dout                    # (d_ff, d_model)
        db2 = dout.sum(axis=0)                        # (d_model,)
        dhidden_relu = dout @ self.W2.T               # (T, d_ff)

        dhidden = dhidden_relu * relu_mask            # ReLU backward

        dW1 = X.T @ dhidden                           # (d_model, d_ff)
        db1 = dhidden.sum(axis=0)                     # (d_ff,)
        dX = dhidden @ self.W1.T                      # (T, d_model)

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return dX, grads

    # -----------------------------------------------------------------
    # Full forward pass
    # -----------------------------------------------------------------

    def forward(self, inputs, targets):
        """
        Full transformer forward pass.

        Unlike the RNN/LSTM which loop through timesteps, we process the
        entire sequence as matrix operations. All positions simultaneously.

        Args:
            inputs:  list of input character indices [x_0, ..., x_{T-1}]
            targets: list of target character indices [t_0, ..., t_{T-1}]

        Returns:
            loss:  scalar cross-entropy loss summed over all positions
            cache: all intermediate values for backward pass
        """
        T = len(inputs)
        inputs_arr = np.array(inputs)

        # --- Embedding + positional encoding ---
        X_emb = self.W_emb[inputs_arr]                # (T, d_model) — lookup
        X = X_emb + self.PE[:T]                        # add position signal

        # --- Pre-norm transformer block ---

        # LayerNorm 1 (pre-attention)
        X_ln1, ln1_cache = self._layer_norm_forward(X, self.gamma_1, self.beta_1)

        # Multi-head causal self-attention
        attn_out, attn_cache = self._attention_forward(X_ln1)

        # Residual connection
        X2 = X + attn_out

        # LayerNorm 2 (pre-FFN)
        X_ln2, ln2_cache = self._layer_norm_forward(X2, self.gamma_2, self.beta_2)

        # Feed-forward network
        ffn_out, ffn_cache = self._ffn_forward(X_ln2)

        # Residual connection
        X3 = X2 + ffn_out

        # --- Final LayerNorm + output projection ---
        X_lnf, lnf_cache = self._layer_norm_forward(X3, self.gamma_f, self.beta_f)

        logits = X_lnf @ self.W_out + self.b_out     # (T, vocab_size)

        # --- Softmax + cross-entropy loss ---
        # Row-wise softmax with numerical stability
        logits_stable = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)  # (T, vocab_size)

        # Cross-entropy loss at each position
        targets_arr = np.array(targets)
        loss = 0.0
        for t in range(T):
            loss += -np.log(probs[t, targets_arr[t]] + 1e-8)

        cache = {
            'inputs_arr': inputs_arr, 'targets_arr': targets_arr,
            'X_emb': X_emb, 'X': X,
            'X_ln1': X_ln1, 'ln1_cache': ln1_cache,
            'attn_out': attn_out, 'attn_cache': attn_cache,
            'X2': X2,
            'X_ln2': X_ln2, 'ln2_cache': ln2_cache,
            'ffn_out': ffn_out, 'ffn_cache': ffn_cache,
            'X3': X3,
            'X_lnf': X_lnf, 'lnf_cache': lnf_cache,
            'logits': logits, 'probs': probs,
            'T': T,
        }

        return loss, cache

    # -----------------------------------------------------------------
    # Full backward pass
    # -----------------------------------------------------------------

    def backward(self, cache):
        """
        Full backward pass through the transformer.

        No BPTT loop — backprop is "through layers" not "through time."
        All positions are handled simultaneously via matrix operations.
        """
        inputs_arr = cache['inputs_arr']
        targets_arr = cache['targets_arr']
        probs = cache['probs']
        X_lnf = cache['X_lnf']
        T = cache['T']

        # --- Output layer gradient ---
        # Softmax + cross-entropy shortcut (same as RNN, but for all positions)
        dlogits = probs.copy()                         # (T, vocab_size)
        for t in range(T):
            dlogits[t, targets_arr[t]] -= 1

        # Output projection backward
        dW_out = X_lnf.T @ dlogits                     # (d_model, vocab_size)
        db_out = dlogits.sum(axis=0)                   # (vocab_size,)
        dX_lnf = dlogits @ self.W_out.T                # (T, d_model)

        # --- Final LayerNorm backward ---
        dX3, dgamma_f, dbeta_f = self._layer_norm_backward(dX_lnf, cache['lnf_cache'])

        # --- FFN residual backward ---
        # X3 = X2 + ffn_out, so gradient flows to both
        dX2_from_residual = dX3.copy()
        dffn_out = dX3.copy()

        # --- FFN backward ---
        dX_ln2_from_ffn, ffn_grads = self._ffn_backward(dffn_out, cache['ffn_cache'])

        # --- Pre-FFN LayerNorm backward ---
        dX2_from_ln2, dgamma_2, dbeta_2 = self._layer_norm_backward(
            dX_ln2_from_ffn, cache['ln2_cache'])

        # Combine gradients flowing into X2
        dX2 = dX2_from_residual + dX2_from_ln2

        # --- Attention residual backward ---
        # X2 = X + attn_out
        dX_from_residual = dX2.copy()
        dattn_out = dX2.copy()

        # --- Attention backward ---
        dX_ln1_from_attn, attn_grads = self._attention_backward(
            dattn_out, cache['attn_cache'])

        # --- Pre-attention LayerNorm backward ---
        dX_from_ln1, dgamma_1, dbeta_1 = self._layer_norm_backward(
            dX_ln1_from_attn, cache['ln1_cache'])

        # Combine gradients flowing into X (pre-block input)
        dX = dX_from_residual + dX_from_ln1

        # --- Embedding gradient ---
        # X = W_emb[inputs] + PE[:T]
        # PE is fixed, so gradient only flows to W_emb
        # Multiple positions may use the same character — accumulate with add.at
        dW_emb = np.zeros_like(self.W_emb)
        np.add.at(dW_emb, inputs_arr, dX)

        # Collect all gradients
        grads = {
            'W_emb': dW_emb,
            'W_out': dW_out, 'b_out': db_out,
            'gamma_1': dgamma_1, 'beta_1': dbeta_1,
            'gamma_2': dgamma_2, 'beta_2': dbeta_2,
            'gamma_f': dgamma_f, 'beta_f': dbeta_f,
        }
        grads.update(attn_grads)   # W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O
        grads.update(ffn_grads)    # W1, b1, W2, b2

        return grads

    def clip_gradients(self, grads, max_norm=5):
        """Clip gradients element-wise to [-max_norm, max_norm]."""
        for key in grads:
            np.clip(grads[key], -max_norm, max_norm, out=grads[key])
        return grads

    def train_step(self, inputs, targets, learning_rate):
        """
        One training step: forward -> backward -> clip -> Adagrad update.

        Simpler than RNN/LSTM: no hidden state to pass or return.
        """
        loss, cache = self.forward(inputs, targets)
        grads = self.backward(cache)
        grads = self.clip_gradients(grads)

        # Adagrad update
        for name in self._param_names:
            if name not in grads:
                continue
            grad = grads[name]
            mem = getattr(self, 'mem_' + name)
            mem += grad * grad
            setattr(self, name,
                    getattr(self, name) - learning_rate * grad
                    / (np.sqrt(mem) + 1e-8))

        return loss

    def sample(self, char_to_ix, ix_to_char, temperature=1.0, max_len=20):
        """
        Generate a name autoregressively.

        Start with '^', run the FULL transformer on the sequence so far,
        take the last position's output, sample a character, append, repeat.

        NOTE: This re-runs the entire transformer at every step. In practice,
        KV-caching avoids redundant computation (store K,V from prior positions
        and only compute the new position). We skip that optimization for clarity.
        """
        ix = char_to_ix['^']
        generated = [ix]

        for _ in range(max_len):
            T = len(generated)
            inputs_arr = np.array(generated)

            # Full forward pass on sequence so far
            X = self.W_emb[inputs_arr] + self.PE[:T]

            # LayerNorm 1 + Attention + Residual
            X_ln1, _ = self._layer_norm_forward(X, self.gamma_1, self.beta_1)
            attn_out, _ = self._attention_forward(X_ln1)
            X2 = X + attn_out

            # LayerNorm 2 + FFN + Residual
            X_ln2, _ = self._layer_norm_forward(X2, self.gamma_2, self.beta_2)
            ffn_out, _ = self._ffn_forward(X_ln2)
            X3 = X2 + ffn_out

            # Final LayerNorm + project
            X_lnf, _ = self._layer_norm_forward(X3, self.gamma_f, self.beta_f)
            logits = X_lnf[-1] @ self.W_out + self.b_out  # last position only

            # Temperature + softmax
            logits = logits / temperature
            logits = logits - logits.max()
            exp_logits = np.exp(logits)
            p = exp_logits / exp_logits.sum()

            ix = np.random.choice(self.vocab_size, p=p)

            if ix_to_char[ix] == '$':
                break

            generated.append(ix)

        # Skip the start token, join, title-case
        chars = [ix_to_char[i] for i in generated[1:]]
        return ''.join(chars).title()

    def get_attention_weights(self, inputs):
        """
        Run a forward pass and return the attention weights for visualization.

        Returns: (n_heads, T, T) array of attention weights.
        """
        inputs_arr = np.array(inputs)
        T = len(inputs)
        X = self.W_emb[inputs_arr] + self.PE[:T]
        X_ln1, _ = self._layer_norm_forward(X, self.gamma_1, self.beta_1)
        _, attn_cache = self._attention_forward(X_ln1)
        X, Q, K, V, Q_h, K_h, V_h, scores, attn, attn_out, concat = attn_cache
        return attn  # (n_heads, T, T)


# =============================================================================
# TRAINING LOOP
# =============================================================================
# Simpler than RNN/LSTM: no hidden state to manage.
# =============================================================================

def train(model, data, char_to_ix, ix_to_char, epochs=1000, lr=0.01,
          print_every=200):
    """Train the transformer on a set of names."""
    loss_history = []

    for epoch in range(1, epochs + 1):
        random.shuffle(data)

        epoch_loss = 0.0
        total_chars = 0

        for inputs, targets in data:
            loss = model.train_step(inputs, targets, lr)
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
    print("  Transformer Name Generator — From Scratch with NumPy")
    print("=" * 70)
    print()
    print("The final step: drop recurrence entirely. Every position attends")
    print("directly to every earlier position through learned attention weights.")
    print("No hidden state chain, no sequential bottleneck.")
    print()
    print("PROGRESSION:")
    print("  1. Order-1 Markov: lookup table, 1-char memory")
    print("  2. Higher-Order Markov: lookup table, N-char memory")
    print("  3. RNN: learned hidden state, gradients vanish")
    print("  4. LSTM: gated cell state, gradient highway, still sequential")
    print("  5. Transformer (this file): attention, all positions at once")
    print()

    char_to_ix, ix_to_char, vocab_size = build_vocab(NAME_SETS)
    print(f"Vocabulary: {vocab_size} symbols")

    d_model, n_heads, d_ff = 32, 2, 128
    n_params = (vocab_size * d_model +          # W_emb
                4 * (d_model * d_model + d_model) +  # Q, K, V, O
                d_model * d_ff + d_ff +         # W1, b1
                d_ff * d_model + d_model +      # W2, b2
                3 * (d_model + d_model) +       # 3x LayerNorm
                d_model * vocab_size + vocab_size)  # W_out, b_out
    print(f"Parameters: ~{n_params:,} (Transformer d_model={d_model}, "
          f"heads={n_heads}, d_ff={d_ff})")
    print()

    temperatures = [0.5, 0.8, 1.2]
    num_samples = 8

    for name_type, names in NAME_SETS.items():
        print("=" * 70)
        print(f"  Training on: {name_type.upper()} ({len(names)} names)")
        print("=" * 70)

        data = prepare_training_data(names, char_to_ix)

        # Train Transformer
        model = CharTransformer(vocab_size, d_model=d_model, n_heads=n_heads,
                                d_ff=d_ff)
        print(f"\n  Training Transformer for 1000 epochs (lr=0.01)...")
        print(f"  Initial loss should be ~{-np.log(1.0 / vocab_size):.1f} "
              f"(random guessing)\n")

        start_time = time.time()
        loss_history = train(
            model, data, char_to_ix, ix_to_char,
            epochs=1000, lr=0.01, print_every=200,
        )
        elapsed = time.time() - start_time

        print(f"\n  Training completed in {elapsed:.1f}s")
        print(f"  Final loss: {loss_history[-1]:.3f}")

        # --- Attention visualization ---
        example_name = names[0].lower()
        example_chars = ['^'] + list(example_name) + ['$']
        example_ix = [char_to_ix[ch] for ch in example_chars]

        attn_weights = model.get_attention_weights(example_ix)

        print(f"\n  --- Attention Pattern for \"{names[0]}\" (head 0) ---")
        header = "      " + "".join(f"{ch:>5s}" for ch in example_chars)
        print(header)
        for i, ch in enumerate(example_chars):
            row = f"  {ch:>3s} "
            for j in range(len(example_chars)):
                w = attn_weights[0, i, j]
                if w > 0.01:
                    row += f"{w:5.2f}"
                else:
                    row += "    ."

            print(row)

        # --- Generate names at different temperatures ---
        print(f"\n  --- Transformer Generated Names (by temperature) ---")
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

        # --- Side-by-side comparison: Transformer vs LSTM vs RNN vs Markov ---
        print(f"\n  --- Comparison: Transformer vs LSTM vs RNN vs Markov ---")

        # Quick-train LSTM and RNN
        print(f"  (Training LSTM and RNN for comparison...)")
        lstm_model = CharLSTM(vocab_size, hidden_size=64)
        from lstm_name_generator import train as lstm_train
        lstm_train(lstm_model, data, char_to_ix, ix_to_char,
                   epochs=500, lr=0.02, print_every=500)

        rnn_model = CharRNN(vocab_size, hidden_size=64)
        from rnn_name_generator import train as rnn_train
        rnn_train(rnn_model, data, char_to_ix, ix_to_char,
                  epochs=500, lr=0.02, print_every=500)

        markov_names = generate_names(name_type, num_samples)
        trans_names = [model.sample(char_to_ix, ix_to_char, temperature=0.8)
                       for _ in range(num_samples)]
        lstm_names = [lstm_model.sample(char_to_ix, ix_to_char, temperature=0.8)
                      for _ in range(num_samples)]
        rnn_names = [rnn_model.sample(char_to_ix, ix_to_char, temperature=0.8)
                     for _ in range(num_samples)]

        print(f"\n  {'Transformer':<18}{'LSTM':<18}{'RNN':<18}{'Markov':<18}")
        print(f"  {'-' * 70}")
        for j in range(num_samples):
            t = (trans_names[j] or "(empty)")[:16].ljust(18)
            l = (lstm_names[j] or "(empty)")[:16].ljust(18)
            r = (rnn_names[j] or "(empty)")[:16].ljust(18)
            m = (markov_names[j] or "")[:16].ljust(18)
            print(f"  {t}{l}{r}{m}")

        print()

    # --- Final observations ---
    print("=" * 70)
    print("  OBSERVATIONS")
    print("=" * 70)
    print()
    print("  ATTENTION PATTERNS:")
    print("    - Each position learns which earlier positions to attend to")
    print("    - Head 0 and Head 1 often specialize (e.g., one for recent")
    print("      context, one for start-of-name patterns)")
    print("    - The causal mask ensures no position can 'cheat' by looking ahead")
    print()
    print("  TRANSFORMER vs RNN vs LSTM:")
    print("    - All three produce comparable quality on these short names")
    print("    - The transformer advantage is in SCALE: longer sequences,")
    print("      more data, parallel training on GPUs")
    print("    - For 44 training names, this is overkill — and that's the point.")
    print("      The mechanism is what matters, not the toy results.")
    print()
    print("  THIS IS GPT / CLAUDE:")
    print(f"    - This transformer: {n_params:,} parameters")
    print("    - GPT-3:            175,000,000,000 parameters")
    print("    - Same Q, K, V. Same scaled dot-product. Same softmax.")
    print("    - Same residual + LayerNorm + FFN. Same autoregressive sampling.")
    print("    - The only difference is scale.")
    print()


# =============================================================================
# EXPORT WEIGHTS
# =============================================================================

def export_weights(epochs=1000, lr=0.01, d_model=32, n_heads=2, d_ff=128):
    """Train one model per category and export to transformer_weights.json."""
    char_to_ix, ix_to_char, vocab_size = build_vocab(NAME_SETS)

    models = {}
    for name_type, names in NAME_SETS.items():
        print(f"Training {name_type}...")
        data = prepare_training_data(names, char_to_ix)
        model = CharTransformer(vocab_size, d_model=d_model, n_heads=n_heads,
                                d_ff=d_ff)
        train(model, data, char_to_ix, ix_to_char,
              epochs=epochs, lr=lr, print_every=200)

        model_weights = {}
        for name in model._param_names:
            model_weights[name] = getattr(model, name).tolist()
        model_weights['PE'] = model.PE.tolist()
        models[name_type] = model_weights

        sample = model.sample(char_to_ix, ix_to_char, temperature=0.8)
        print(f"  Done. Sample: {sample}")

    payload = {
        "vocab": {
            "char_to_ix": char_to_ix,
            "ix_to_char": {str(k): v for k, v in ix_to_char.items()},
            "vocab_size": vocab_size,
        },
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "models": models,
    }

    with open("transformer_weights.json", "w") as f:
        json.dump(payload, f)

    print(f"\nExported to transformer_weights.json ({vocab_size} vocab, "
          f"d_model={d_model}, {len(models)} models)")


if __name__ == "__main__":
    demo()
