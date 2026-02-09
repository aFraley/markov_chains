"""
=============================================================================
RNN Character-Level Name Generator — From Scratch with NumPy
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

  3. RNN (this file) ★
     -> Replaces the lookup table with a NEURAL NETWORK
     -> Learns a "hidden state" — a compressed summary of EVERYTHING seen so far
     -> Can generalize: even for character sequences never seen in training,
        the network produces reasonable predictions based on similar patterns
     -> Trained with backpropagation through time (BPTT)

WHY THIS MATTERS (STENN CONNECTION):
-------------------------------------
This RNN implements the same fundamental ideas used in STENN's temporal
component:
  - A hidden state that carries information forward through a sequence
  - Weight matrices that learn to extract useful features
  - Backpropagation through time to learn from sequential data
  - Gradient clipping to stabilize training of recurrent architectures

The only difference is scale: here we process character sequences of length
5-10 to generate names. STENN processes thermal frame sequences of length
100-1000+ to detect anomalies. Same math, different domain.

THE RNN EQUATIONS (the whole thing, really):
---------------------------------------------
At each timestep t, given input x_t and previous hidden state h_{t-1}:

  h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)   # update hidden state
  y_t = W_hy @ h_t + b_y                             # compute output logits
  p_t = softmax(y_t)                                  # convert to probabilities

That's it. Three lines of math. Everything else is:
  - How to LEARN the weight matrices W (backpropagation)
  - How to STABILIZE that learning (gradient clipping)
  - How to SAMPLE from the output distribution (temperature scaling)

=============================================================================
"""

import time
import random

import numpy as np

from name_generator import NAME_SETS, generate_names


# =============================================================================
# SECTION 1: CHARACTER ENCODING / VOCABULARY
# =============================================================================
# Neural networks operate on numbers, not characters. We need a mapping from
# characters to indices (for input) and back (for output).
#
# We also add two special tokens:
#   '^' = START — signals the beginning of a name
#   '$' = END   — signals the network should stop generating
#
# The Markov chain version handles start/end implicitly (pick from "initial",
# stop at target length). The RNN learns when to stop on its own — it outputs
# '$' when it thinks the name is complete. This is more powerful because the
# network can learn that names starting with certain patterns tend to be
# shorter or longer.
# =============================================================================

def build_vocab(name_sets):
    """
    Scan all names across all categories and build character-to-index mappings.

    We combine all categories into one vocabulary so the same encoding works
    for greek, norse, and celtic names. In practice each category trains
    its own model, but they share the same alphabet.

    Returns:
        char_to_ix: dict mapping each character to a unique integer index
        ix_to_char: dict mapping each index back to its character
        vocab_size: total number of symbols (26 letters + start + end = 28)
    """
    # Collect every unique character across all training names
    chars = set()
    for names in name_sets.values():
        for name in names:
            for ch in name.lower():
                chars.add(ch)

    # Sort for deterministic ordering (important for reproducibility)
    chars = sorted(chars)

    # Add special tokens: '^' for start-of-name, '$' for end-of-name
    # These go at the front so they have predictable indices
    vocab = ['^', '$'] + chars

    char_to_ix = {ch: i for i, ch in enumerate(vocab)}
    ix_to_char = {i: ch for i, ch in enumerate(vocab)}

    return char_to_ix, ix_to_char, len(vocab)


def one_hot(ix, vocab_size):
    """
    Create a one-hot column vector: all zeros except a 1.0 at position ix.

    This is how we represent a character as neural network input.
    Character 'a' at index 3 becomes [0, 0, 0, 1, 0, 0, ...].T

    WHY ONE-HOT?
    The network needs numerical input, but characters aren't ordinal —
    'b' isn't "more" than 'a' in any meaningful way. One-hot encoding
    treats each character as an independent dimension, letting the network
    learn its own notion of character similarity through the weight matrix.

    In larger models (GPT, etc.), you'd use learned EMBEDDINGS instead of
    one-hot vectors. The embedding matrix E effectively computes E @ one_hot(ix),
    which is just selecting column ix of E — a "lookup" that's differentiable.

    Args:
        ix: The index to set to 1.0
        vocab_size: Total vocabulary size (length of the vector)

    Returns:
        Column vector of shape (vocab_size, 1) with 1.0 at position ix
    """
    vec = np.zeros((vocab_size, 1))
    vec[ix] = 1.0
    return vec


# =============================================================================
# SECTION 2: THE CHARACTER-LEVEL RNN
# =============================================================================
# This is the core of the file. The CharRNN class implements:
#   - Forward pass: process a sequence, compute loss
#   - Backward pass: BPTT to compute gradients
#   - Gradient clipping: prevent exploding gradients
#   - Training step: forward + backward + clip + Adagrad update
#   - Sampling: generate new names autoregressively
#
# STENN CONNECTION: This is the same architecture pattern used in STENN's
# temporal module. The hidden state here carries character-level context;
# in STENN it carries thermal history. The math is identical.
# =============================================================================

class CharRNN:
    """
    A character-level Recurrent Neural Network for name generation.

    Architecture:
        input (vocab_size) -> hidden (hidden_size) -> output (vocab_size)

    The recurrence comes from W_hh: the hidden state at time t depends on
    the hidden state at time t-1, creating a chain of information flow
    through the entire sequence.

    This is "vanilla RNN" — the simplest possible recurrent architecture.
    LSTM and GRU add gating mechanisms to control information flow, but
    the core idea (carry state forward through time) is the same.
    """

    def __init__(self, vocab_size, hidden_size=64):
        """
        Initialize the RNN with random weights.

        We have 5 learnable parameter matrices:
          W_xh: maps input -> hidden       (how each character influences the state)
          W_hh: maps hidden -> hidden       (how the state evolves over time)
          W_hy: maps hidden -> output       (how the state predicts the next char)
          b_h:  hidden bias                 (offset for the hidden computation)
          b_y:  output bias                 (offset for the output computation)

        INITIALIZATION MATTERS:
        We use Xavier initialization: divide by sqrt(fan_in). Why?
          - All zeros: every neuron computes the same thing (symmetry problem)
          - Too large: tanh saturates at ±1, gradients vanish
          - Too small: signals shrink to zero, gradients are tiny, learning stalls
          - Xavier (1/sqrt(fan_in)): keeps signal variance stable across layers

        The idea: if a layer has N inputs, each with variance v, the output
        variance is N*v*w² where w is the weight scale. Setting w = 1/sqrt(N)
        makes output variance ≈ v, preventing signal from growing or shrinking
        as it passes through the network.

        STENN CONNECTION: Every neural network needs careful initialization.
        The same principles apply whether you're initializing an RNN for
        name generation or a CNN for thermal image analysis.

        Args:
            vocab_size: Number of unique characters (input/output dimension)
            hidden_size: Dimensionality of the hidden state (capacity of memory)
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # --- Weight matrices ---
        # Input to hidden: each character gets projected into hidden space
        # Shape: (hidden_size, vocab_size) because h = W_xh @ x
        # Xavier init: / sqrt(fan_in) keeps signal variance stable
        self.W_xh = np.random.randn(hidden_size, vocab_size) / np.sqrt(vocab_size)

        # Hidden to hidden: THIS IS THE RECURRENCE
        # This matrix controls how the hidden state at time t-1 influences
        # the hidden state at time t. Without this, we'd just have a regular
        # (non-recurrent) neural network with no memory.
        # Shape: (hidden_size, hidden_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)

        # Hidden to output: projects hidden state to character probabilities
        # Shape: (vocab_size, hidden_size) because y = W_hy @ h
        self.W_hy = np.random.randn(vocab_size, hidden_size) / np.sqrt(hidden_size)

        # Biases — initialized to zero (standard practice)
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((vocab_size, 1))

        # --- Adagrad memory ---
        # Adagrad tracks the sum of squared gradients for each parameter.
        # This lets it adapt the learning rate per-parameter: frequently-
        # updated parameters get smaller steps, rarely-updated ones get
        # larger steps. Critical for one-hot inputs where each character's
        # weight row only gets gradient when that character appears.
        self.mem_W_xh = np.zeros_like(self.W_xh)
        self.mem_W_hh = np.zeros_like(self.W_hh)
        self.mem_W_hy = np.zeros_like(self.W_hy)
        self.mem_b_h = np.zeros_like(self.b_h)
        self.mem_b_y = np.zeros_like(self.b_y)

    def forward(self, inputs, targets, h_prev):
        """
        Forward pass: process a sequence and compute the cross-entropy loss.

        This runs the RNN equations at each timestep, building up a cache
        of intermediate values needed for backpropagation.

        THE RNN EQUATIONS (repeated from module docstring for reference):
          h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
          y_t = W_hy @ h_t + b_y
          p_t = softmax(y_t)

        WHERE THE LOSS COMES FROM:
        At each timestep, we know the correct next character (the target).
        The loss measures how surprised the network is by the correct answer:
          loss_t = -log(p_t[target_t])

        If the network assigns probability 0.9 to the correct char:
          -log(0.9) = 0.105  (low loss, good prediction)
        If it assigns 0.01:
          -log(0.01) = 4.605 (high loss, bad prediction)

        Total loss = sum over all timesteps. This is CROSS-ENTROPY LOSS,
        the standard loss function for classification tasks.

        Args:
            inputs:  list of character indices [x_0, x_1, ..., x_{T-1}]
            targets: list of character indices [t_0, t_1, ..., t_{T-1}]
                     (targets[i] = inputs[i+1] in our setup)
            h_prev:  initial hidden state, shape (hidden_size, 1)

        Returns:
            loss:    scalar cross-entropy loss summed over the sequence
            cache:   dict of intermediate values for backpropagation
        """
        # Cache stores everything we need for the backward pass
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev.copy()  # hidden state at t=-1 (before the sequence)

        loss = 0.0

        for t in range(len(inputs)):
            # --- Step 1: Encode the input character as one-hot ---
            xs[t] = one_hot(inputs[t], self.vocab_size)

            # --- Step 2: Compute new hidden state ---
            # h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
            #
            # This is where the magic happens:
            #   W_xh @ x_t:      what the current character says
            #   W_hh @ h_{t-1}:  what everything before says (THE MEMORY)
            #   tanh:            squash to [-1, 1] to prevent unbounded growth
            #
            # STENN CONNECTION: This exact equation appears in STENN's temporal
            # module. There, x_t is a thermal feature vector from frame t, and
            # h_{t-1} carries the thermal history from all prior frames.
            hs[t] = np.tanh(
                self.W_xh @ xs[t] + self.W_hh @ hs[t - 1] + self.b_h
            )

            # --- Step 3: Compute output logits ---
            # y_t = W_hy @ h_t + b_y
            # These are raw, unnormalized scores for each character
            ys[t] = self.W_hy @ hs[t] + self.b_y

            # --- Step 4: Convert logits to probabilities via softmax ---
            # p_t = exp(y_t) / sum(exp(y_t))
            #
            # NUMERICAL STABILITY TRICK: subtract max(y) before exp()
            # Without this, exp(large_number) overflows to infinity.
            # Subtracting max doesn't change the result because:
            #   exp(y - c) / sum(exp(y - c)) = exp(y)/exp(c) / (sum(exp(y))/exp(c))
            #                                = exp(y) / sum(exp(y))
            # The exp(c) cancels out. This is used everywhere — PyTorch, TensorFlow, etc.
            exp_ys = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = exp_ys / np.sum(exp_ys)

            # --- Step 5: Accumulate cross-entropy loss ---
            # -log(probability assigned to the correct character)
            # Adding a tiny epsilon (1e-8) to prevent log(0) = -infinity
            loss += -np.log(ps[t][targets[t], 0] + 1e-8)

        cache = {
            'xs': xs, 'hs': hs, 'ys': ys, 'ps': ps,
            'inputs': inputs, 'targets': targets,
        }

        return loss, cache

    def backward(self, cache):
        """
        Backpropagation Through Time (BPTT) — compute gradients for all weights.

        THIS IS THE KEY EDUCATIONAL SECTION.

        Normal backpropagation computes gradients by walking backward through
        the computation graph. BPTT is the same thing, but the "graph" is
        unrolled through time: the same weights are used at every timestep,
        so we accumulate gradients across all timesteps.

        THE GRADIENT FLOW:
        For each timestep t (walking backward from T-1 to 0):

          1. OUTPUT GRADIENT:
             dy_t = p_t - one_hot(target_t)
             This is the classic softmax + cross-entropy gradient shortcut.
             If the network predicted p=0.9 for the correct char, dy is small.
             If it predicted p=0.01, dy is large. The gradient is proportional
             to the "surprise" — bigger errors produce bigger updates.

          2. WEIGHT GRADIENTS (output layer):
             dW_hy += dy_t @ h_t.T    (outer product — how to adjust output weights)
             db_y  += dy_t             (bias gradient is just the error signal)

          3. HIDDEN STATE GRADIENT:
             dh_t = W_hy.T @ dy_t + dh_next
                    ~~~~~~~~~~~~~~~   ~~~~~~~~
                    from output layer  from FUTURE timestep (the "through time" part!)

             This dh_next term is what makes BPTT different from regular backprop.
             The gradient flows backward not just through layers, but through TIME.

          4. BACKPROP THROUGH TANH:
             dh_raw = dh_t * (1 - h_t^2)
             This is the derivative of tanh: d/dx tanh(x) = 1 - tanh(x)^2
             Since h_t = tanh(raw_t), we just use (1 - h_t^2).

          5. WEIGHT GRADIENTS (hidden layer):
             dW_xh += dh_raw @ x_t.T
             dW_hh += dh_raw @ h_{t-1}.T
             db_h  += dh_raw

          6. PROPAGATE GRADIENT BACKWARD IN TIME:
             dh_next = W_hh.T @ dh_raw
             This becomes the dh_next for timestep t-1.

        THE VANISHING/EXPLODING GRADIENT PROBLEM:
        Notice that dh_next gets multiplied by W_hh.T at every timestep.
        Over T timesteps, the effective multiplication is (W_hh.T)^T.
          - If the largest eigenvalue of W_hh > 1: gradients EXPLODE (grow exponentially)
          - If the largest eigenvalue of W_hh < 1: gradients VANISH (shrink to zero)

        For our short name sequences (5-10 chars), this is manageable with
        gradient clipping. For longer sequences (100+ steps), you need
        LSTM/GRU which add "gates" to control the gradient flow.

        STENN CONNECTION: BPTT is the identical math used when STENN backprops
        through thermal frame sequences. Longer sequences = more severe
        vanishing/exploding gradient problems, which is why STENN uses LSTM
        cells instead of vanilla RNN.

        Args:
            cache: Dict of intermediate values from the forward pass.

        Returns:
            grads: Dict of gradient matrices for each learnable parameter.
        """
        xs = cache['xs']
        hs = cache['hs']
        ps = cache['ps']
        targets = cache['targets']

        # Initialize all gradients to zero
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # This is the gradient flowing backward from FUTURE timesteps.
        # At the last timestep, there is no future, so it starts at zero.
        dh_next = np.zeros_like(hs[0])

        T = len(targets)

        # Walk backward through time (T-1, T-2, ..., 1, 0)
        for t in reversed(range(T)):
            # --- Step 1: Output gradient ---
            # dy = p - one_hot(target)
            # This beautiful shortcut comes from combining the softmax and
            # cross-entropy derivatives. Derivation:
            #   L = -log(p[target])
            #   dL/dy_i = p_i - 1(i == target)
            # where 1(i == target) is 1 for the correct class, 0 otherwise.
            dy = ps[t].copy()
            dy[targets[t]] -= 1  # subtract 1 from the correct class

            # --- Step 2: Output weight gradients ---
            dW_hy += dy @ hs[t].T
            db_y += dy

            # --- Step 3: Hidden state gradient ---
            # Gradient flows from two sources:
            #   1. The output layer at this timestep (W_hy.T @ dy)
            #   2. The hidden layer at the NEXT timestep (dh_next)
            dh = self.W_hy.T @ dy + dh_next

            # --- Step 4: Backprop through tanh ---
            # d/dx tanh(x) = 1 - tanh(x)^2, and hs[t] = tanh(...)
            dh_raw = dh * (1 - hs[t] ** 2)

            # --- Step 5: Hidden weight gradients ---
            dW_xh += dh_raw @ xs[t].T
            dW_hh += dh_raw @ hs[t - 1].T
            db_h += dh_raw

            # --- Step 6: Propagate gradient backward in time ---
            # This becomes dh_next for the previous timestep
            dh_next = self.W_hh.T @ dh_raw

        grads = {
            'W_xh': dW_xh, 'W_hh': dW_hh, 'W_hy': dW_hy,
            'b_h': db_h, 'b_y': db_y,
        }

        return grads

    def clip_gradients(self, grads, max_norm=5):
        """
        Clip gradients element-wise to prevent exploding gradients.

        WHY CLIPPING IS NECESSARY:
        During BPTT, the gradient at timestep 0 has been multiplied by W_hh.T
        once for every timestep. If any eigenvalue of W_hh is > 1, this
        repeated multiplication causes exponential growth:

          timestep 5:  gradient * λ^5
          timestep 10: gradient * λ^10
          timestep 20: gradient * λ^20  (if λ=1.1, this is 6.7x amplification)

        Clipping caps each gradient element at [-max_norm, max_norm], preventing
        any single gradient from dominating the update.

        ALTERNATIVES:
          - Gradient norm scaling: scale entire gradient vector to have max norm
            (used in PyTorch's clip_grad_norm_). More sophisticated but same idea.
          - LSTM/GRU: architectural solution that controls gradient flow with gates
          - Gradient penalty: add a regularization term for large gradients

        STENN CONNECTION: Gradient clipping is a universal necessity for recurrent
        architectures. STENN uses the same technique when training on long thermal
        sequences.

        Args:
            grads: Dict of gradient matrices from backward().
            max_norm: Maximum allowed absolute value for any gradient element.

        Returns:
            grads: The same dict with clipped values (modified in-place).
        """
        for key in grads:
            np.clip(grads[key], -max_norm, max_norm, out=grads[key])
        return grads

    def train_step(self, inputs, targets, h_prev, learning_rate):
        """
        One complete training step: forward → backward → clip → update.

        This is the training loop's inner function. For each name:
          1. Forward pass: compute predictions and loss
          2. Backward pass: compute gradients via BPTT
          3. Clip gradients: prevent exploding gradients
          4. Update weights: Adagrad (adaptive learning rate per parameter)

        WHY ADAGRAD?
        Vanilla SGD (w -= lr * grad) uses the same learning rate for every
        parameter. With one-hot character inputs, this is a problem: each
        character's weight row only gets gradient when that character appears.
        Rare characters learn too slowly, common ones may overshoot.

        Adagrad fixes this with ONE simple idea — track the sum of squared
        gradients per parameter and divide by it:

          mem  += grad^2                     # accumulate squared gradients
          w    -= lr * grad / sqrt(mem + ε)  # adaptive update

        Parameters with large accumulated gradients (frequently updated) get
        smaller effective learning rates. Parameters with small accumulated
        gradients (rarely updated) get larger effective learning rates.

        It's still "gradient × step" — just with an adaptive step size.
        Adam, RMSProp, etc. are all refinements of this same idea.

        STENN CONNECTION: All neural networks learn this way — gradient × step.
        Adagrad, Adam, SGD with momentum — they're all variations on "compute
        the gradient, take a step in that direction." The fundamentals are here.

        Args:
            inputs:        list of input character indices
            targets:       list of target character indices
            h_prev:        initial hidden state
            learning_rate: base step size (Adagrad adapts it per-parameter)

        Returns:
            loss:          scalar loss for this sequence
            h_last:        final hidden state (shape: hidden_size × 1)
        """
        # Forward: compute loss and cache intermediates
        loss, cache = self.forward(inputs, targets, h_prev)

        # Backward: compute gradients via BPTT
        grads = self.backward(cache)

        # Clip: prevent exploding gradients
        grads = self.clip_gradients(grads)

        # Update: Adagrad (adaptive gradient descent)
        # For each parameter: accumulate squared gradient, then step
        for param, grad, mem in [
            ('W_xh', grads['W_xh'], 'mem_W_xh'),
            ('W_hh', grads['W_hh'], 'mem_W_hh'),
            ('W_hy', grads['W_hy'], 'mem_W_hy'),
            ('b_h',  grads['b_h'],  'mem_b_h'),
            ('b_y',  grads['b_y'],  'mem_b_y'),
        ]:
            mem_arr = getattr(self, mem)
            mem_arr += grad * grad                                  # accumulate
            setattr(self, param,
                    getattr(self, param) - learning_rate * grad     # step with
                    / (np.sqrt(mem_arr) + 1e-8))                    # adaptive lr

        # Return the final hidden state from this sequence
        h_last = cache['hs'][len(inputs) - 1]
        return loss, h_last

    def sample(self, char_to_ix, ix_to_char, temperature=1.0, max_len=20):
        """
        Generate a name by running the forward pass autoregressively.

        AUTOREGRESSIVE GENERATION:
        Start with the '^' (start) token, run the forward equations to get
        a probability distribution over next characters, SAMPLE one character
        from that distribution, feed it back as input, repeat until '$' (end)
        or max length.

        This is the exact same loop used in GPT, Claude, and every other
        autoregressive language model. The only differences are:
          - We operate on characters; they operate on tokens (word pieces)
          - We have ~28 possible outputs; they have ~100,000
          - We use a 64-unit hidden state; they use billions of parameters

        TEMPERATURE:
        Temperature controls the "randomness" of sampling by scaling the logits
        before softmax:
          p = softmax(y / temperature)

          temperature < 1.0: sharper distribution, more conservative names
          temperature = 1.0: faithful to learned distribution
          temperature > 1.0: flatter distribution, more creative/random names

        This is DIRECTLY ANALOGOUS to the power scaling in the Markov chain!
        In name_generator.py, we raise counts to power 1.3 (equivalent to
        temperature < 1). Here, we divide logits by temperature. Same effect:
        control the confidence of the model's predictions.

        STENN CONNECTION: Temperature/confidence scaling appears in STENN's
        anomaly scoring — how confident should the model be before flagging
        a thermal reading as anomalous?

        Args:
            char_to_ix: Character to index mapping
            ix_to_char: Index to character mapping
            temperature: Controls randomness (0.5=conservative, 1.2=creative)
            max_len: Maximum name length to prevent infinite loops

        Returns:
            Generated name as a string (title-cased)
        """
        # Start with a fresh hidden state (no prior context)
        h = np.zeros((self.hidden_size, 1))

        # Begin with the start token '^'
        ix = char_to_ix['^']
        name_chars = []

        for _ in range(max_len):
            # Encode current character
            x = one_hot(ix, self.vocab_size)

            # Forward one step (same equations as in forward(), but no loss)
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            y = self.W_hy @ h + self.b_y

            # Apply temperature scaling BEFORE softmax
            y = y / temperature

            # Softmax with numerical stability
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)

            # Sample from the probability distribution
            # This is the numpy equivalent of torch.multinomial
            ix = np.random.choice(self.vocab_size, p=p.ravel())

            # Check for end token
            if ix_to_char[ix] == '$':
                break

            name_chars.append(ix_to_char[ix])

        # Join characters and title-case (lowercase during training, proper on output)
        return ''.join(name_chars).title()


# =============================================================================
# SECTION 3: TRAINING DATA PREPARATION
# =============================================================================
# Each name becomes a sequence of character indices with start/end tokens.
# Example: "Apollo" -> ['^', 'a', 'p', 'o', 'l', 'l', 'o', '$']
#                    -> [0, 3, 18, 17, 14, 14, 17, 1]  (indices)
#
# The input at each timestep is one character, and the target is the NEXT
# character. So inputs = [^, a, p, o, l, l, o] and
#              targets = [g, a, n, d, a, l, f, $].
#
# The network learns: given '^', predict 'g'. Given 'g' (with hidden state
# carrying memory of '^'), predict 'a'. And so on.
# =============================================================================

def prepare_training_data(names, char_to_ix):
    """
    Convert a list of names into input/target index sequences.

    Each name becomes a (inputs, targets) pair where:
      inputs[t]  = index of character at position t
      targets[t] = index of character at position t+1

    The start token '^' is prepended and end token '$' is appended.

    Args:
        names: List of name strings
        char_to_ix: Character to index mapping

    Returns:
        List of (inputs, targets) tuples, one per name
    """
    data = []
    for name in names:
        # Lowercase + wrap with start/end tokens
        chars = ['^'] + list(name.lower()) + ['$']

        # Convert to index sequences
        # Input:  everything except the last char
        # Target: everything except the first char
        inputs = [char_to_ix[ch] for ch in chars[:-1]]
        targets = [char_to_ix[ch] for ch in chars[1:]]

        data.append((inputs, targets))
    return data


# =============================================================================
# SECTION 4: TRAINING LOOP
# =============================================================================
# The training loop iterates over the data multiple times (epochs).
# Each epoch:
#   1. Shuffle the training names (so the network doesn't memorize order)
#   2. For each name, run one training step (forward + backward + update)
#   3. Track the average loss across all names
#   4. Periodically print the loss and a sample name to monitor progress
#
# You should see:
#   - Loss starts around 3.3 (random guessing over ~28 chars: -log(1/28) ≈ 3.3)
#   - Loss drops to 1.0-1.5 (the network is learning character patterns)
#   - Sample names go from gibberish to recognizable patterns
# =============================================================================

def train(model, data, char_to_ix, ix_to_char, epochs=500, lr=0.05,
          print_every=50):
    """
    Train the RNN on a set of names.

    Args:
        model: CharRNN instance
        data: List of (inputs, targets) tuples from prepare_training_data()
        char_to_ix: Character to index mapping (for sampling)
        ix_to_char: Index to character mapping (for sampling)
        epochs: Number of passes through the entire dataset
        lr: Base learning rate for Adagrad
        print_every: Print progress every N epochs

    Returns:
        loss_history: List of average loss values per epoch
    """
    loss_history = []

    for epoch in range(1, epochs + 1):
        # Shuffle training data each epoch
        random.shuffle(data)

        epoch_loss = 0.0
        total_chars = 0

        for inputs, targets in data:
            # Reset hidden state for each name — names are independent sequences.
            # (In text generation, you'd carry the hidden state across sentences
            # to maintain paragraph-level context. But names are independent.)
            h_prev = np.zeros((model.hidden_size, 1))

            loss, _ = model.train_step(inputs, targets, h_prev, lr)
            epoch_loss += loss
            total_chars += len(targets)

        # Per-character loss: total loss / total characters predicted.
        # This gives a number directly comparable to -log(1/vocab_size) ≈ 3.3
        # at random, dropping to ~1.0-1.5 when well-trained.
        avg_loss = epoch_loss / total_chars
        loss_history.append(avg_loss)

        # Print progress: loss + sample name
        if epoch % print_every == 0 or epoch == 1:
            sample_name = model.sample(char_to_ix, ix_to_char, temperature=0.8)
            print(f"  Epoch {epoch:4d}/{epochs}  |  Loss: {avg_loss:.3f}"
                  f"  |  Sample: {sample_name}")

    return loss_history


# =============================================================================
# SECTION 5: DEMO
# =============================================================================
# Train one model per name category, generate names at multiple temperatures,
# and compare side-by-side with the Markov chain output.
# =============================================================================

def demo():
    print("=" * 70)
    print("  RNN Character-Level Name Generator — From Scratch with NumPy")
    print("=" * 70)
    print()
    print("This RNN replaces the Markov chain's lookup table with a neural")
    print("network that learns a hidden state — a compressed summary of")
    print("everything it has seen so far in the sequence.")
    print()
    print("PROGRESSION:")
    print("  1. Order-1 Markov: lookup table, 1-char memory")
    print("  2. Higher-Order Markov: lookup table, N-char memory (with backoff)")
    print("  3. RNN (this file): neural network, learned variable-length memory")
    print()

    # Build shared vocabulary
    char_to_ix, ix_to_char, vocab_size = build_vocab(NAME_SETS)
    print(f"Vocabulary: {vocab_size} symbols "
          f"({vocab_size - 2} letters + start + end tokens)")
    print()

    temperatures = [0.5, 0.8, 1.2]
    num_samples = 8

    for name_type, names in NAME_SETS.items():
        print("=" * 70)
        print(f"  Training on: {name_type.upper()} ({len(names)} names)")
        print("=" * 70)

        # Prepare training data
        data = prepare_training_data(names, char_to_ix)

        # Create and train the model
        model = CharRNN(vocab_size, hidden_size=64)

        print(f"\n  Training for 500 epochs (lr=0.02, hidden_size=64, Adagrad)...")
        print(f"  Initial loss should be ~{-np.log(1.0 / vocab_size):.1f} "
              f"(random guessing)\n")

        start_time = time.time()
        loss_history = train(
            model, data, char_to_ix, ix_to_char,
            epochs=500, lr=0.02, print_every=100,
        )
        elapsed = time.time() - start_time

        print(f"\n  Training completed in {elapsed:.1f}s")
        print(f"  Final loss: {loss_history[-1]:.3f}")

        # --- Generate names at different temperatures ---
        print(f"\n  --- RNN Generated Names (by temperature) ---")
        print(f"  {'t=0.5 (conserv.)':<22}{'t=0.8 (balanced)':<22}"
              f"{'t=1.2 (creative)':<22}")
        print(f"  {'-' * 64}")

        for _ in range(num_samples):
            cols = []
            for temp in temperatures:
                name = model.sample(char_to_ix, ix_to_char, temperature=temp)
                # Ensure we have something to show
                if not name:
                    name = "(empty)"
                cols.append(name[:20].ljust(22))
            print(f"  {''.join(cols)}")

        # --- Side-by-side comparison with Markov ---
        print(f"\n  --- Comparison: RNN (t=0.8) vs Markov (order-1) ---")
        markov_names = generate_names(name_type, 8)
        rnn_names = [
            model.sample(char_to_ix, ix_to_char, temperature=0.8)
            for _ in range(num_samples)
        ]

        print(f"  {'RNN (t=0.8)':<22}{'Markov (order-1)':<22}")
        print(f"  {'-' * 42}")
        for i in range(num_samples):
            rnn_col = (rnn_names[i] or "(empty)")[:20].ljust(22)
            mkv_col = (markov_names[i] or "")[:20].ljust(22)
            print(f"  {rnn_col}{mkv_col}")

        print()

    # --- Final observations ---
    print("=" * 70)
    print("  OBSERVATIONS")
    print("=" * 70)
    print()
    print("  TEMPERATURE EFFECTS:")
    print("    t=0.5: Conservative — sticks to common patterns, less variety")
    print("    t=0.8: Balanced — recognizable style with some novelty")
    print("    t=1.2: Creative — more variety but occasionally incoherent")
    print()
    print("  RNN vs MARKOV:")
    print("    - Markov (order-1) only sees the last character. It cannot")
    print("      learn that 'th' is commonly followed by vowels, because")
    print("      it doesn't know the 't' happened.")
    print("    - The RNN's hidden state carries a compressed summary of ALL")
    print("      prior characters. It can learn multi-character patterns")
    print("      like '-eus' endings in Greek names or consonant clusters in")
    print("      Celtic names, even though it processes one char at a time.")
    print()
    print("  WHAT'S NEXT:")
    print("    - LSTM/GRU: Add gates to control information flow, solving")
    print("      the vanishing gradient problem for longer sequences")
    print("    - Embeddings: Replace one-hot with learned dense vectors")
    print("    - Attention/Transformers: Look at all prior chars at once")
    print("    - These are the building blocks of modern language models")
    print()
    print("  STENN CONNECTION:")
    print("    The hidden state here carries character-level context forward.")
    print("    In STENN, the identical mechanism carries thermal history")
    print("    forward through frame sequences. Same math, different domain.")
    print()


def export_weights(epochs=500, lr=0.02, hidden_size=64):
    """
    Train one model per category and export all weights to rnn_weights.json.

    The JSON file contains everything needed to run the forward pass (sampling)
    in another language (e.g., JavaScript in the browser).
    """
    import json

    char_to_ix, ix_to_char, vocab_size = build_vocab(NAME_SETS)

    models = {}
    for name_type, names in NAME_SETS.items():
        print(f"Training {name_type}...")
        data = prepare_training_data(names, char_to_ix)
        model = CharRNN(vocab_size, hidden_size=hidden_size)
        train(model, data, char_to_ix, ix_to_char,
              epochs=epochs, lr=lr, print_every=100)

        models[name_type] = {
            "W_xh": model.W_xh.tolist(),
            "W_hh": model.W_hh.tolist(),
            "W_hy": model.W_hy.tolist(),
            "b_h":  model.b_h.tolist(),
            "b_y":  model.b_y.tolist(),
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

    with open("rnn_weights.json", "w") as f:
        json.dump(payload, f)

    print(f"\nExported weights to rnn_weights.json ({vocab_size} vocab, "
          f"{hidden_size} hidden, {len(models)} models)")


if __name__ == "__main__":
    demo()
