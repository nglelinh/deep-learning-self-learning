---
layout: post
title: 05-01 Recurrent Neural Networks - Fundamentals
chapter: '05'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter05
lesson_type: required
---

# Recurrent Neural Networks: Processing Sequential Data

![RNN Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/800px-Recurrent_neural_network_unfold.svg.png)
*Hình ảnh: Kiến trúc RNN và quá trình unfolding theo thời gian. Nguồn: Wikimedia Commons*

## 1. Concept Overview

Recurrent Neural Networks represent a fundamentally different approach to neural computation than the feedforward networks we've studied so far. While feedforward networks process each input independently, treating a batch of images or feature vectors as unrelated samples, RNNs are designed specifically for sequential data where the order of elements carries meaning and where context from previous elements should influence how we interpret current ones. This seemingly simple idea—adding memory to neural networks—opens up entire domains that feedforward networks cannot adequately address: natural language, where word order determines meaning; speech, where phonemes unfold over time; financial time series, where past trends inform future predictions; and video, where frames form coherent narratives.

The core insight of RNNs is to maintain a hidden state that serves as the network's memory. At each time step, the network receives a new input and the previous hidden state, processes both through the same set of weights, and produces a new hidden state and output. This recurrence—using the same parameters at every time step while the hidden state evolves—creates a network that can theoretically process sequences of any length while learning temporal patterns through the shared parameters. The elegance lies in parameter sharing across time: just as convolutional networks share weights spatially to detect features regardless of position, RNNs share weights temporally to recognize patterns regardless of when they occur in a sequence.

Understanding why RNNs emerged requires appreciating the fundamental challenge of sequential data: variable length. A sentence might have five words or fifty. An audio clip might last three seconds or three minutes. How do we build neural networks that handle this variability? The naive approach—having separate parameters for each time step—fails immediately because it doesn't generalize to sequences longer than those seen during training and requires enormous numbers of parameters. The elegant solution RNNs provide is to use the same transformation at each step, with the hidden state carrying forward information from all previous steps. This creates an architecture that's both parameter-efficient and theoretically capable of modeling arbitrarily long sequences.

However, RNNs are not without profound limitations. The sequential nature that gives them modeling power also makes them computationally slow—we cannot process time step $$t$$ until we've processed step $$t-1$$, preventing the parallelization that GPUs excel at. The need to propagate gradients backward through many time steps leads to vanishing or exploding gradients, making it difficult to learn long-range dependencies. And the fixed-size hidden state creates an information bottleneck—all information from the past must be compressed into this state, which becomes increasingly difficult as sequences grow longer. These limitations motivated the development of LSTM and GRU architectures, and ultimately, the attention mechanisms and Transformers that now dominate sequence modeling.

Yet RNNs remain important to study, not just for historical reasons but because they embody fundamental principles about sequence processing. They introduce the concept of memory in neural networks. They demonstrate both the power and limitations of sequential processing. They reveal challenges in gradient propagation that inform our understanding of training deep networks more broadly. And in certain applications—particularly where sequences are naturally processed online or where computational efficiency on CPUs matters more than GPU throughput—RNNs and their sophisticated variants still have advantages over Transformers.

## 2. Mathematical Foundation

The mathematical formulation of RNNs reveals both their elegance and their challenges. At each time step $$t$$, an RNN maintains a hidden state $$\mathbf{h}_t$$ that summarizes all information from the sequence up to that point. Given a new input $$\mathbf{x}_t$$, the RNN updates its state according to:

$$\mathbf{h}_t = f(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$

where $$f$$ is a nonlinear activation function (typically $$\tanh$$ for vanilla RNNs), $$\mathbf{W}_{hh}$$ governs how the previous state influences the current state (the recurrent connection), and $$\mathbf{W}_{xh}$$ governs how the current input contributes. The bias $$\mathbf{b}_h$$ provides a baseline activation level.

This formulation deserves careful scrutiny because it encodes several crucial design decisions. First, the same weight matrices $$\mathbf{W}_{hh}$$ and $$\mathbf{W}_{xh}$$ are used at every time step. This parameter sharing across time is what enables RNNs to generalize to sequences of different lengths and to learn temporal patterns independent of their absolute position in the sequence. Just as convolutional filters detect edges anywhere in an image through spatial parameter sharing, recurrent weights detect temporal patterns anywhere in a sequence through temporal parameter sharing.

Second, the hidden state $$\mathbf{h}_t$$ plays a dual role. It serves as both the network's memory (encoding information from all previous inputs) and as input to the next time step. This creates a feedback loop: the current state depends on the previous state, which depends on the state before that, creating a chain of dependencies that theoretically extends back to the beginning of the sequence. Mathematically, we can unroll this recursion:

$$\mathbf{h}_t = f(\mathbf{W}_{hh}f(\mathbf{W}_{hh}\mathbf{h}_{t-2} + \mathbf{W}_{xh}\mathbf{x}_{t-1} + \mathbf{b}_h) + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$

This shows how $$\mathbf{h}_t$$ depends on inputs $$\mathbf{x}_t$$ and $$\mathbf{x}_{t-1}$$, and continuing the expansion, on all previous inputs back to $$\mathbf{x}_1$$. The nested composition of functions is what gives RNNs their power to model complex temporal dependencies, but also what makes them challenging to train—gradients must backpropagate through this entire composition.

The output at each time step is computed from the hidden state:

$$\mathbf{y}_t = g(\mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y)$$

where $$g$$ is an activation function chosen based on the task (softmax for classification, linear for regression, sigmoid for multi-label, etc.). Importantly, $$\mathbf{W}_{hy}$$ is also shared across time steps. This means we're using the same "decoder" to interpret the hidden state at every time step, forcing the hidden state to use a consistent representation scheme throughout the sequence.

The dimensions of these matrices matter for understanding computational and representational tradeoffs. If the input dimension is $$d_x$$, hidden dimension is $$d_h$$, and output dimension is $$d_y$$, then:
- $$\mathbf{W}_{xh} \in \mathbb{R}^{d_h \times d_x}$$: Projects input to hidden dimension
- $$\mathbf{W}_{hh} \in \mathbb{R}^{d_h \times d_h}$$: Recurrent connections (most important!)
- $$\mathbf{W}_{hy} \in \mathbb{R}^{d_y \times d_h}$$: Projects hidden to output

The recurrent weight matrix $$\mathbf{W}_{hh}$$ is particularly important because it determines how information persists over time. If $$\mathbf{W}_{hh}$$ is nearly zero, the hidden state quickly forgets previous inputs. If it's close to identity, the hidden state changes slowly. Its eigenvalues directly control gradient flow during backpropagation through time, as we'll see.

Different tasks require different architectural patterns. For sequence classification (sentiment analysis, video classification), we typically process the entire sequence and use only the final hidden state: $$\mathbf{y} = g(\mathbf{W}_{hy}\mathbf{h}_T + \mathbf{b}_y)$$. The final state $$\mathbf{h}_T$$ must summarize the entire sequence. For sequence labeling (part-of-speech tagging, named entity recognition), we produce outputs at every time step: $$\mathbf{y}_t = g(\mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y)$$ for all $$t$$. For sequence-to-sequence tasks (machine translation), the encoder RNN processes the input sequence into a final hidden state, which initializes a decoder RNN that generates the output sequence.

Training RNNs requires Backpropagation Through Time (BPTT), which is simply backpropagation applied to the unrolled computational graph of the RNN. The loss over a sequence is typically the sum of losses at each time step:

$$\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t(\mathbf{y}_t, \hat{\mathbf{y}}_t)$$

To compute gradients of this loss with respect to $$\mathbf{W}_{hh}$$, we must account for the fact that $$\mathbf{W}_{hh}$$ affects the hidden state at every time step. The gradient is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}$$

Each term $$\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}$$ requires backpropagating through all time steps from 1 to $$t$$ via the chain rule, because $$\mathbf{W}_{hh}$$ influences $$\mathbf{h}_1$$, which influences $$\mathbf{h}_2$$, and so on up to $$\mathbf{h}_t$$. This creates a computational graph that grows with sequence length, leading to both memory challenges (must store all hidden states) and the notorious vanishing/exploding gradient problem.

The gradient flow analysis reveals the core challenge. The gradient of $$\mathbf{h}_t$$ with respect to $$\mathbf{h}_k$$ (for $$k < t$$) involves a product of Jacobian matrices:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}}$$

Each Jacobian $$\frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \text{diag}(f'(\mathbf{z}_i)) \mathbf{W}_{hh}$$ is the product of the activation derivative (which for $$\tanh$$ is bounded by 1) and the recurrent weight matrix. If the largest singular value of $$\mathbf{W}_{hh}$$ is less than 1, these products shrink exponentially with the number of time steps, causing vanishing gradients. If it's greater than 1, they explode exponentially. This is not a minor technical issue but a fundamental challenge in training RNNs to capture long-range dependencies.

## 3. Example / Intuition

To build genuine intuition for how RNNs process sequences, let's trace through a concrete example of processing the sentence "The cat sat on the mat" for sentiment analysis. We'll use a simple RNN with 3-dimensional hidden state (unrealistically small for clarity) to predict positive or negative sentiment.

Initially, before seeing any words, we set $$\mathbf{h}_0 = \mathbf{0}$$, representing no information. The first word "The" arrives. Suppose after embedding it becomes $$\mathbf{x}_1 = [0.2, -0.1, 0.5]$$. With initialized weights (let's use small random values):

$$\mathbf{W}_{hh} = \begin{bmatrix} 0.5 & 0.1 & -0.2 \\ -0.1 & 0.6 & 0.3 \\ 0.2 & -0.3 & 0.4 \end{bmatrix}, \quad \mathbf{W}_{xh} = \begin{bmatrix} 0.3 & 0.2 & -0.1 \\ 0.1 & -0.2 & 0.4 \\ -0.2 & 0.3 & 0.1 \end{bmatrix}$$

The hidden state update computes:

$$\mathbf{z}_1 = \mathbf{W}_{hh}\mathbf{h}_0 + \mathbf{W}_{xh}\mathbf{x}_1 + \mathbf{b}_h = \mathbf{0} + \mathbf{W}_{xh}\mathbf{x}_1 + \mathbf{b}_h$$

Since $$\mathbf{h}_0 = \mathbf{0}$$, the first hidden state depends only on the first input:

$$\mathbf{h}_1 = \tanh(\mathbf{z}_1) = \tanh\begin{pmatrix}\begin{bmatrix} 0.3 & 0.2 & -0.1 \\ 0.1 & -0.2 & 0.4 \\ -0.2 & 0.3 & 0.1 \end{bmatrix} \begin{bmatrix} 0.2 \\ -0.1 \\ 0.5 \end{bmatrix} + \mathbf{b}_h\end{pmatrix}$$

Let's say this gives $$\mathbf{h}_1 = [0.1, 0.05, -0.15]$$ (after $$\tanh$$). This vector now encodes "having seen 'The'"—not very informative, but it's a start. Now "cat" arrives as $$\mathbf{x}_2 = [0.8, 0.3, -0.2]$$. The state update now has TWO contributions:

$$\mathbf{z}_2 = \mathbf{W}_{hh}\mathbf{h}_1 + \mathbf{W}_{xh}\mathbf{x}_2 + \mathbf{b}_h$$

The first term $$\mathbf{W}_{hh}\mathbf{h}_1$$ carries forward information from "The", while $$\mathbf{W}_{xh}\mathbf{x}_2$$ processes "cat". The $$\tanh$$ nonlinearity combines these:

$$\mathbf{h}_2 = \tanh(\mathbf{z}_2)$$

This new state encodes "having seen 'The cat'"—it's been influenced by both words, with the contribution of "The" mediated through $$\mathbf{h}_1$$ and the recurrent weights $$\mathbf{W}_{hh}$$.

As we continue through "sat", "on", "the", "mat", each hidden state incorporates more context. By the time we reach the end, $$\mathbf{h}_6$$ theoretically encodes the entire sentence's meaning. We then use this to predict sentiment:

$$\text{sentiment} = \text{sigmoid}(\mathbf{W}_{hy}\mathbf{h}_6 + \mathbf{b}_y)$$

The key insight is that information from "The" at the beginning must flow through $$\mathbf{h}_1 \to \mathbf{h}_2 \to \cdots \to \mathbf{h}_6$$ to influence the final prediction. Each transition involves multiplication by $$\mathbf{W}_{hh}$$ and passing through $$\tanh$$. If $$\mathbf{W}_{hh}$$ consistently suppresses signals (eigenvalues < 1), information decays exponentially—this is vanishing gradients. If it amplifies signals (eigenvalues > 1), information explodes—exploding gradients. Maintaining stable information flow over many steps is the central challenge of vanilla RNNs.

Consider what happens during backpropagation. When we compute $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}}$$, we must account for $$\mathbf{W}_{hh}$$ appearing at every time step. The gradient involves products like:

$$\frac{\partial \mathbf{h}_6}{\partial \mathbf{h}_1} = \frac{\partial \mathbf{h}_6}{\partial \mathbf{h}_5} \frac{\partial \mathbf{h}_5}{\partial \mathbf{h}_4} \cdots \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}$$

Each Jacobian $$\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} = \text{diag}(\tanh'(\mathbf{z}_{t+1})) \mathbf{W}_{hh}$$ involves the recurrent weight matrix and the activation derivative. 

For $$\tanh$$, the derivative is bounded by 1 and typically much smaller (approaching 0 for saturated activations). This means each Jacobian typically has norm less than $$\|\mathbf{W}_{hh}\|$$, and their product shrinks exponentially unless $$\|\mathbf{W}_{hh}\|$$ is precisely calibrated. In practice, this delicate balancing rarely works for vanilla RNNs, limiting their ability to learn dependencies spanning more than about 10-20 time steps.

The mathematics of different RNN architectures reveals different design philosophies. A many-to-one architecture (sequence → single output) computes $$\mathbf{h}_t$$ for all $$t$$ but only uses $$\mathbf{h}_T$$ for the final prediction, appropriate for classification tasks. A many-to-many architecture with the same length (sequence → sequence of same length) computes outputs $$\mathbf{y}_t = g(\mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y)$$ at every time step, useful for tasks like video frame labeling where we want to classify each frame. A many-to-many architecture with different lengths (sequence-to-sequence) requires an encoder-decoder design: the encoder RNN processes the input into a final hidden state $$\mathbf{h}_T^{enc}$$, which initializes the decoder RNN: $$\mathbf{h}_0^{dec} = \mathbf{h}_T^{enc}$$. The decoder then autoregressively generates output, with each output feeding into the next time step's input until a special end-of-sequence token is generated.

The computational complexity of RNNs is linear in sequence length: $$O(T \cdot d_h^2)$$ where $$T$$ is sequence length and $$d_h$$ is hidden dimension (dominated by the $$\mathbf{W}_{hh}\mathbf{h}_{t-1}$$ multiplication at each step). This seems efficient, but the sequential dependency means we cannot parallelize across time steps. Processing a sequence of length 100 requires 100 sequential matrix multiplications, even with a powerful GPU. Transformers, by contrast, have higher complexity $$O(T^2 d)$$ but can process all positions in parallel, making them much faster in practice on modern hardware when $$T$$ is not extremely large.

## 4. Code Snippet

Let's implement RNNs from first principles to understand every detail, then show modern PyTorch implementations:

```python
import numpy as np
import matplotlib.pyplot as plt

class VanillaRNN:
    """
    Vanilla RNN implementation from scratch using NumPy.
    
    This implementation prioritizes clarity over efficiency. We'll see exactly
    how hidden states evolve over time, how gradients flow backward, and where
    the vanishing gradient problem comes from. Understanding this manual
    implementation is crucial for debugging RNN training issues and for
    appreciating what modern frameworks do automatically.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize RNN with careful weight initialization.
        
        For RNNs, initialization is even more critical than for feedforward networks.
        The recurrent weights Whh determine eigenvalues of the hidden state dynamics.
        If eigenvalues are too large (> 1), hidden states explode. Too small (< 1),
        they vanish. We initialize to small random values hoping to land in a
        stable regime, though this often fails for vanilla RNNs.
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Input to hidden weights: map input space to hidden space
        # Scale by 1/√hidden_size to maintain variance
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        
        # Hidden to hidden weights: THE critical matrix for temporal dynamics
        # Some initialize to identity + noise for better gradient flow initially
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        
        # Hidden to output weights
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs, h_prev):
        """
        Forward pass through entire sequence.
        
        inputs: list of input vectors [x_1, x_2, ..., x_T], each x_t is (input_size, 1)
        h_prev: initial hidden state (hidden_size, 1)
        
        Returns:
            outputs: list of output vectors [y_1, y_2, ..., y_T]
            hidden_states: list of hidden states [h_0, h_1, ..., h_T]
            
        We must store ALL hidden states because backpropagation through time
        needs them. This creates a memory cost linear in sequence length,
        which becomes prohibitive for very long sequences.
        """
        hidden_states = [h_prev]
        outputs = []
        h = h_prev
        
        # Process sequence one step at a time
        # This sequential loop is unavoidable in RNNs - we can't parallelize
        # across time because h_t depends on h_{t-1}
        for x_t in inputs:
            # Hidden state update: combine previous state and current input
            # The tanh bounds activations to [-1, 1], preventing explosion
            # (though it causes saturation and vanishing gradients)
            z = self.Whh @ h + self.Wxh @ x_t + self.bh
            h = np.tanh(z)
            hidden_states.append(h)
            
            # Compute output from hidden state
            # For sequence classification, we'd use only the final output
            # For sequence labeling, we use all outputs
            y = self.Why @ h + self.by
            outputs.append(y)
        
        return outputs, hidden_states
    
    def backward(self, inputs, outputs, hidden_states, targets):
        """
        Backpropagation Through Time (BPTT).
        
        This is where RNNs get challenging. We must backpropagate gradients
        not just through layers (as in feedforward networks) but also backward
        through time. The gradient of the loss with respect to Whh involves
        contributions from all time steps, creating the long product of Jacobians
        that leads to vanishing/exploding gradients.
        """
        T = len(inputs)
        
        # Initialize gradient accumulators
        # These will accumulate gradients from all time steps
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # Gradient flowing back through time
        # Initially zero, will accumulate as we backprop through time
        dh_next = np.zeros((self.hidden_size, 1))
        
        # Backpropagate through time (from T down to 1)
        # This is the "backward" in backpropagation through time
        for t in reversed(range(T)):
            # Output error at this time step
            # For MSE loss: dy = y_pred - y_true
            dy = outputs[t] - targets[t]
            
            # Gradient of output layer parameters
            dWhy += dy @ hidden_states[t+1].T
            dby += dy
            
            # Backprop into hidden state
            # Two sources of gradient: from output at this time step (dy)
            # and from future time steps (dh_next)
            dh = self.Why.T @ dy + dh_next
            
            # Backprop through tanh nonlinearity
            # tanh'(z) = 1 - tanh^2(z) = 1 - h^2
            # This derivative approaches 0 when |h| approaches 1 (saturation)
            # causing vanishing gradients
            dz = (1 - hidden_states[t+1]**2) * dh
            
            # Gradients of hidden layer parameters
            dbh += dz
            dWxh += dz @ inputs[t].T
            dWhh += dz @ hidden_states[t].T
            
            # Pass gradient to previous time step
            # This is where the product of Jacobians occurs
            # dh_next will be used at time step t-1
            dh_next = self.Whh.T @ dz
        
        # Clip gradients to prevent explosion
        # This is a hack, but necessary for vanilla RNNs
        # We're treating the symptom (large gradients) not the cause
        # (fundamental instability of recurrent dynamics)
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        return dWxh, dWhh, dWhy, dbh, dby
    
    def update_parameters(self, dWxh, dWhh, dWhy, dbh, dby):
        """Standard gradient descent update"""
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

# Demonstrate on simple sequence prediction task
print("="*70)
print("Vanilla RNN: Predicting Next Number in Sequence")
print("="*70)

# Task: Given [1, 2, 3], predict [2, 3, 4]
# This tests if RNN can learn simple sequential pattern
def generate_sequence_data(n_sequences=100, seq_length=10):
    """
    Generate training data: sequences of increasing numbers
    Input: [start, start+1, start+2, ...]
    Target: [start+1, start+2, start+3, ...]
    """
    sequences_in = []
    sequences_target = []
    
    for _ in range(n_sequences):
        start = np.random.randint(0, 50)
        seq = np.array([start + i for i in range(seq_length)])
        sequences_in.append([np.array([[x]], dtype=float) for x in seq])
        sequences_target.append([np.array([[x]], dtype=float) for x in seq[1:] + [seq[-1]+1]])
    
    return sequences_in, sequences_target

# Create RNN
rnn = VanillaRNN(input_size=1, hidden_size=10, output_size=1, learning_rate=0.001)

# Generate data
train_inputs, train_targets = generate_sequence_data(n_sequences=50, seq_length=8)

# Training loop
print("Training RNN to predict next number...")
losses = []

for epoch in range(200):
    epoch_loss = 0
    
    for inputs, targets in zip(train_inputs, train_targets):
        # Forward pass
        h_prev = np.zeros((rnn.hidden_size, 1))
        outputs, hidden_states = rnn.forward(inputs, h_prev)
        
        # Compute loss (MSE)
        loss = sum((out - tgt)**2 for out, tgt in zip(outputs, targets))
        epoch_loss += np.sum(loss)
        
        # Backward pass
        grads = rnn.backward(inputs, outputs, hidden_states, targets)
        
        # Update
        rnn.update_parameters(*grads)
    
    losses.append(epoch_loss / len(train_inputs))
    
    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d}: Average Loss = {losses[-1]:.4f}")

# Test on new sequence
print("\n" + "="*70)
print("Testing on new sequence: [10, 11, 12, 13, 14, 15, 16, 17]")
print("="*70)

test_input = [np.array([[x]], dtype=float) for x in [10, 11, 12, 13, 14, 15, 16, 17]]
h_test = np.zeros((rnn.hidden_size, 1))
test_outputs, test_hidden = rnn.forward(test_input, h_test)

print("Predictions:")
for t, (inp, out) in enumerate(zip(test_input, test_outputs)):
    print(f"  After seeing {int(inp[0,0])}, predict next: {out[0,0]:.2f} (true: {int(inp[0,0])+1})")

print("\nNote: Vanilla RNN struggles with this simple task due to vanishing gradients!")
print("This motivates LSTM/GRU architectures we'll cover in the next chapter.")
```

Now let's see a modern PyTorch implementation that handles these details automatically:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RNNSequenceModel(nn.Module):
    """
    Modern RNN using PyTorch's built-in RNN layer.
    
    PyTorch handles the recurrent computation efficiently, unrolling the loop
    in optimized C++ code and managing gradients automatically. This is much
    faster than our NumPy implementation and handles batching elegantly.
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNSequenceModel, self).__init__()
        
        # PyTorch's RNN module handles the recurrent computation
        # num_layers > 1 creates stacked RNNs (output of one feeds into next)
        # batch_first=True means input shape is (batch, seq_len, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                         batch_first=True, nonlinearity='tanh')
        
        # Output layer to map hidden state to predictions
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, input_size)
        hidden: optional initial hidden state (num_layers, batch, hidden_size)
        
        Returns:
            output: (batch, seq_len, output_size)
            hidden: final hidden state
        """
        # If no initial hidden state provided, RNN initializes to zeros
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            if x.is_cuda:
                hidden = hidden.cuda()
        
        # RNN returns:
        # - rnn_out: hidden states at all time steps (batch, seq_len, hidden_size)
        # - hidden: final hidden state (num_layers, batch, hidden_size)
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Apply output layer to each time step
        # Reshape to (batch * seq_len, hidden_size) for efficient computation
        batch_size, seq_len, _ = rnn_out.shape
        rnn_out = rnn_out.reshape(-1, self.hidden_size)
        output = self.fc(rnn_out)
        output = output.reshape(batch_size, seq_len, -1)
        
        return output, hidden

# Character-level language model example
print("\n" + "="*70)
print("Character-Level Language Model with PyTorch RNN")
print("="*70)

# Create simple text dataset
text = "hello world, deep learning is amazing! transformers are powerful."
chars = list(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocabulary: {chars}")
print(f"Vocabulary size: {vocab_size}")

# Prepare sequences: given "hell" predict "ello"
def create_sequences(text, seq_length=10):
    """Create training sequences from text"""
    sequences = []
    targets = []
    
    for i in range(len(text) - seq_length):
        seq = text[i:i+seq_length]
        target = text[i+1:i+seq_length+1]
        
        # Convert to indices
        seq_idx = [char_to_idx[ch] for ch in seq]
        target_idx = [char_to_idx[ch] for ch in target]
        
        sequences.append(seq_idx)
        targets.append(target_idx)
    
    return sequences, targets

seq_length = 15
sequences, targets = create_sequences(text, seq_length)

# Convert to tensors and create one-hot encodings
def to_onehot(sequences, vocab_size):
    """Convert index sequences to one-hot encoded tensors"""
    one_hot = []
    for seq in sequences:
        seq_onehot = torch.zeros(len(seq), vocab_size)
        for t, idx in enumerate(seq):
            seq_onehot[t, idx] = 1
        one_hot.append(seq_onehot)
    return torch.stack(one_hot)

X = to_onehot(sequences, vocab_size)
y = torch.tensor(targets, dtype=torch.long)

print(f"\nDataset: {len(sequences)} sequences of length {seq_length}")
print(f"Input shape: {X.shape}")  # (num_sequences, seq_length, vocab_size)
print(f"Target shape: {y.shape}")  # (num_sequences, seq_length)

# Create model
model = RNNSequenceModel(input_size=vocab_size, hidden_size=32, 
                         output_size=vocab_size, num_layers=2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
print("\nTraining character-level language model...")
model.train()

for epoch in range(500):
    # Forward pass
    outputs, _ = model(X)  # (batch, seq_len, vocab_size)
    
    # Reshape for cross-entropy: (batch * seq_len, vocab_size)
    outputs_flat = outputs.view(-1, vocab_size)
    targets_flat = y.view(-1)
    
    # Compute loss
    loss = criterion(outputs_flat, targets_flat)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()  # BPTT happens here automatically!
    
    # Gradient clipping (essential for RNNs!)
    # Without this, gradients can explode and training diverges
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

# Generate text by sampling from learned distribution
print("\n" + "="*70)
print("Generating text from learned RNN")
print("="*70)

def generate_text(model, start_text, length=50):
    """
    Generate text autoregressively using trained RNN.
    
    We start with a seed text, predict the next character's probability
    distribution, sample from it, append to sequence, and repeat.
    This is autoregressive generation—each prediction conditions on
    all previous predictions.
    """
    model.eval()
    
    # Convert start text to indices
    current_seq = [char_to_idx[ch] for ch in start_text]
    generated = start_text
    
    # Hidden state carries information through generation
    hidden = None
    
    with torch.no_grad():
        for _ in range(length):
            # Prepare input: last seq_length characters (or pad if shorter)
            input_seq = current_seq[-seq_length:] if len(current_seq) >= seq_length else current_seq
            
            # Pad if needed
            while len(input_seq) < seq_length:
                input_seq = [char_to_idx[' ']] + input_seq
            
            # Convert to one-hot
            x = torch.zeros(1, seq_length, vocab_size)
            for t, idx in enumerate(input_seq):
                x[0, t, idx] = 1
            
            # Predict next character
            output, hidden = model(x, hidden)
            
            # Get probabilities for next character (last time step)
            probs = torch.softmax(output[0, -1], dim=0)
            
            # Sample from distribution (more interesting than argmax)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_idx]
            
            generated += next_char
            current_seq.append(next_idx)
    
    return generated

# Generate
seed = "deep "
generated_text = generate_text(model, seed, length=50)
print(f"Seed: '{seed}'")
print(f"Generated: '{generated_text}'")
print("\nThe model learned character-level patterns!")
print("With more data and training, RNNs can generate coherent text.")
```

Let's also demonstrate the vanishing gradient problem empirically:

```python
print("\n" + "="*70)
print("Demonstrating Vanishing Gradients in RNNs")
print("="*70)

def analyze_gradient_flow(sequence_lengths=[5, 10, 20, 50]):
    """
    Show how gradients diminish with sequence length.
    
    We'll create sequences of different lengths, compute gradients, and
    measure their magnitude. This empirically demonstrates why vanilla RNNs
    struggle with long-range dependencies.
    """
    results = []
    
    for seq_len in sequence_lengths:
        # Create simple RNN
        rnn_test = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
        
        # Random input sequence
        x = torch.randn(1, seq_len, 10, requires_grad=True)
        
        # Forward pass
        out, hidden = rnn_test(x)
        
        # Compute loss from FIRST time step output only
        # Gradient must backprop through seq_len-1 steps to reach h_1
        loss = out[:, 0, :].sum()
        
        # Backward pass
        loss.backward()
        
        # Measure gradient magnitude at input
        grad_magnitude = x.grad.abs().mean().item()
        
        results.append((seq_len, grad_magnitude))
        print(f"Sequence length {seq_len:2d}: Gradient magnitude = {grad_magnitude:.6f}")
    
    # Typically see exponential decay in gradient magnitude
    print("\nObservation: Gradients decay exponentially with sequence length!")
    print("This is the vanishing gradient problem that limits vanilla RNNs.")
    
    return results

gradient_analysis = analyze_gradient_flow()
```

## 5. Related Concepts

The relationship between RNNs and feedforward networks illuminates fundamental principles about network architecture design. Feedforward networks assume inputs are independent, identically distributed samples—the order we present images during training doesn't matter because each image is processed in isolation. RNNs, by contrast, explicitly model dependencies between sequential inputs through the hidden state. This difference isn't just about architecture; it reflects different assumptions about data structure. When we choose an RNN over a feedforward network, we're encoding the inductive bias that temporal or sequential order carries information relevant to the task.

The connection to finite state machines and dynamical systems provides deeper theoretical insight. An RNN with discrete hidden states and hard-threshold activations is essentially a finite state machine, transitioning between states based on inputs. With continuous hidden states and smooth activations, RNNs become continuous dynamical systems described by the difference equation $$\mathbf{h}_{t+1} = f(\mathbf{W}_{hh}\mathbf{h}_t + \mathbf{W}_{xh}\mathbf{x}_t)$$. The stability and expressiveness of this dynamical system depend on the spectrum of $$\mathbf{W}_{hh}$$—its eigenvalues determine whether the system is stable, chaotic, or marginally stable. This connection to dynamical systems theory helps explain phenomena like vanishing/exploding gradients and motivates architectures like LSTMs that explicitly manage information flow through gating mechanisms.

The evolution from RNNs to LSTMs to Transformers tells a story about solving fundamental limitations. Vanilla RNNs struggle with long-range dependencies due to vanishing gradients. LSTMs introduce gating mechanisms that create skip connections through time, allowing gradients to flow more easily and information to persist longer. But LSTMs still process sequences sequentially, limiting parallelization. Transformers abandon recurrence entirely, using attention to create direct connections between all time steps, enabling full parallelization at the cost of quadratic complexity in sequence length. Each architecture makes different tradeoffs between expressiveness, trainability, and computational efficiency.

The relationship between RNNs and convolutional networks is subtler but illuminating. Temporal convolution—applying 1D convolution over sequences—can capture some sequential patterns and is fully parallelizable. However, its receptive field grows only linearly with depth (a network with $$L$$ layers of kernel size $$k$$ has receptive field $$1 + L(k-1)$$), whereas RNNs theoretically have infinite receptive field (the hidden state can remember information from arbitrarily far in the past). This tradeoff between parallelizability (favoring convolution) and theoretically unlimited memory (favoring RNNs) has led to hybrid architectures combining both, like WaveNet for audio generation.

Bidirectional RNNs extend the basic architecture by processing sequences in both forward and backward directions, maintaining two hidden states $$\overrightarrow{\mathbf{h}}_t$$ and $$\overleftarrow{\mathbf{h}}_t$$. The output at each time step combines information from both: $$\mathbf{y}_t = g(\mathbf{W}_{hy}[\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] + \mathbf{b}_y)$$. This is powerful for tasks where future context is available (like translating a complete sentence) but impossible for real-time prediction where we must make decisions before seeing the complete sequence. The bidirectional design exemplifies how architecture should match task requirements—using future context when available, processing causally when necessary.

## 6. Fundamental Papers

**["Finding Structure in Time" (1990)](https://doi.org/10.1207/s15516709cog1402_1)**  
*Author*: Jeffrey L. Elman  
This seminal paper introduced the Simple Recurrent Network (SRN), now called Elman network, and demonstrated that recurrent connections enable learning temporal patterns. Elman showed that RNNs could learn to predict the next word in simple sentences, discovering grammatical structure without explicit rules. The key insight was that the hidden state develops internal representations of grammatical categories (noun, verb) and sequential dependencies without being told to do so—purely from the prediction task. The paper established RNNs as viable for sequence modeling and influenced subsequent development of more sophisticated recurrent architectures. Elman's analysis of hidden state dynamics—showing how the state space organizes itself to reflect linguistic structure—demonstrated that neural networks could discover interpretable representations, a theme that continues in modern deep learning research.

**["Learning to Forget: Continual Prediction with LSTM" (2000)](https://doi.org/10.1162/089976600300015015)**  
*Authors*: Felix A. Gers, Jürgen Schmidhuber, Fred Cummins  
While LSTMs were introduced in 1997, this paper made a crucial modification that made them practical: the forget gate. The original LSTM could accumulate information in the cell state but had no mechanism to selectively forget irrelevant information, leading to saturation over long sequences. The forget gate, controlled by $$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f)$$, allows the network to clear its memory when old information becomes irrelevant. This seemingly simple addition—letting the network learn when to forget—dramatically improved LSTM performance on long sequences and became standard in all subsequent LSTM implementations. The paper demonstrates how architectural details that seem minor can have profound practical impacts.

**["On the difficulty of training Recurrent Neural Networks" (2013)](https://arxiv.org/abs/1211.5063)**  
*Authors*: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio  
This paper provided the definitive analysis of vanishing and exploding gradients in RNNs, moving beyond empirical observations to rigorous mathematical treatment. The authors showed that when backpropagating through $$t$$ time steps, gradients involve products of $$t$$ Jacobian matrices, and if the largest eigenvalue of these matrices is less than 1, gradients vanish exponentially; if greater than 1, they explode exponentially. Importantly, they showed this isn't just a training trick issue but a fundamental property of recurrent dynamics. The paper proposed gradient clipping to handle explosions (clip gradient norm to maximum threshold, now standard practice) and analyzed how LSTM's gating mechanisms create effective paths for gradient flow. This work deepened understanding of why vanilla RNNs fail on long sequences and why architectural innovations like LSTMs are necessary, not optional.

**["Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014)](https://arxiv.org/abs/1412.3555)**  
*Authors*: Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio  
This paper systematically compared LSTM and GRU (Gated Recurrent Unit) architectures on multiple sequence modeling tasks, providing empirical evidence about when each architecture excels. GRUs, introduced by Cho et al. in 2014, simplify LSTMs by using only two gates instead of three and no separate cell state, reducing parameters by about 25%. The paper showed that GRUs often match LSTM performance while training faster due to fewer parameters. Importantly, it demonstrated that architectural details matter—carefully engineered recurrent mechanisms consistently outperformed vanilla RNNs on long sequences. The paper's experimental methodology—controlled comparisons on multiple datasets with careful hyperparameter tuning—set a standard for how to evaluate architectural innovations in deep learning.

**["Visualizing and Understanding Recurrent Networks" (2015)](https://arxiv.org/abs/1506.02078)**  
*Authors*: Andrej Karpathy, Justin Johnson, Li Fei-Fei  
This paper investigated what RNNs learn by analyzing their hidden state dynamics on character-level language modeling. By examining which hidden units activate for which input patterns, the authors discovered that RNNs spontaneously develop interpretable internal representations: certain neurons activate for quotes, others for parentheses balancing, others for code indentation. This demonstrated that RNNs don't just memorize but learn meaningful structure. The paper also introduced techniques for visualizing attention-like patterns in RNNs before explicit attention mechanisms were common. Perhaps most influentially, it made accessible the kind of interpretability analysis that helps us understand what neural networks learn, a methodology that has become standard for analyzing all types of models, not just RNNs.

## Common Pitfalls and Tricks

The most common failure mode when training RNNs is gradient explosion, and recognizing its symptoms is crucial for debugging. Training loss suddenly becomes NaN, parameters become infinite, or loss oscillates wildly rather than decreasing smoothly. This happens when the product of gradients through time steps grows exponentially. The standard solution—gradient clipping—is conceptually simple but must be implemented correctly. We compute the global gradient norm across all parameters $$\|\nabla_\theta \mathcal{L}\|_2 = \sqrt{\sum_\theta (\frac{\partial \mathcal{L}}{\partial \theta})^2}$$ and if it exceeds a threshold (typically 5-10), we scale all gradients by $$\frac{\text{threshold}}{\|\nabla_\theta \mathcal{L}\|_2}$$. This preserves gradient direction while preventing explosive updates. It's crucial to clip the global norm, not individual gradient values, because we want to preserve the relative magnitudes of gradients for different parameters.

Vanishing gradients are more insidious because they don't cause obvious training failures—the network trains but simply fails to learn long-range dependencies. Symptoms include the model only using recent context (in language modeling, only considering the last few words) or being unable to learn tasks requiring information from the beginning of long sequences. Detection requires careful analysis: plot gradient magnitudes as a function of backpropagation steps or test specifically on tasks requiring long-range memory. Solutions include switching to LSTM/GRU (which mitigate though don't eliminate vanishing gradients), using smaller sequence lengths during training (truncated BPTT), or adding auxiliary losses at intermediate time steps to provide more direct gradient paths.

Initialization of recurrent weights deserves special attention because it directly affects gradient flow stability. The standard small random initialization $$\mathbf{W}_{hh} \sim \mathcal{N}(0, 0.01^2)$$ often leads to vanishing gradients. A better approach is orthogonal initialization: initialize $$\mathbf{W}_{hh}$$ to a random orthogonal matrix (often generated via QR decomposition of a random matrix). Orthogonal matrices preserve vector norms during multiplication, helping gradients neither vanish nor explode, at least initially. This gives training a better starting point, though as weights update, they drift from orthogonality. Another approach is identity initialization plus small random noise: $$\mathbf{W}_{hh} = I + \mathcal{N}(0, 0.001^2)$$, encouraging the hidden state to change slowly, which can help with gradient flow.

A subtle but important issue is variable-length sequences in batched training. When training on multiple sequences of different lengths simultaneously, we must handle the fact that some sequences end before others. The solution is padding and masking: pad shorter sequences to match the longest sequence in the batch with a special padding token, then mask the loss so padded positions don't contribute to gradients. Without masking, the RNN receives meaningless gradient signals from padding, degrading performance. PyTorch's PackedSequence functionality handles this elegantly, avoiding computation on padded positions entirely.

The choice of hidden state dimension involves important tradeoffs. Larger hidden dimensions provide more capacity to remember complex patterns and longer contexts. However, they increase parameters quadratically ($$\mathbf{W}_{hh}$$ has $$d_h^2$$ elements), slow computation (each time step requires $$O(d_h^2)$$ operations), and can lead to overfitting on small datasets. A common starting point is matching hidden dimension to input dimension or using 128-512 depending on task complexity. For character-level modeling, 128-256 often suffices. For word-level language modeling on large vocabularies, 512-1024 is typical. Always validate on a held-out set and watch for train-test gaps indicating overfitting.

Using teacher forcing during training but autoregressive generation during inference creates train-test mismatch in sequence-to-sequence models. During training with teacher forcing, the decoder receives the true previous token as input, ensuring it sees good inputs even when its predictions are poor. During inference, it must use its own predictions, which may be wrong, leading to compounding errors. This mismatch means the model never learns to recover from its own mistakes during training. Solutions include scheduled sampling (randomly using predicted tokens instead of true tokens during training with increasing probability), or using auxiliary losses that encourage robustness to input perturbations.

## Key Takeaways

Recurrent Neural Networks introduced the fundamental idea of memory in neural networks through hidden states that persist across time steps, enabling modeling of sequential data where order and context matter. The mathematical elegance of parameter sharing across time—using the same weights at every step—allows RNNs to generalize across sequence lengths while learning temporal patterns. However, this same recurrence creates challenges: sequential processing prevents parallelization, making RNNs slow to train on GPUs; the product of Jacobians through time leads to vanishing or exploding gradients, limiting their ability to learn long-range dependencies; and the fixed-size hidden state creates an information bottleneck for long sequences. Despite these limitations, RNNs established principles—that networks can maintain state, that temporal structure should be explicitly modeled, that we can learn to predict future from past—that influence all subsequent sequence modeling architectures. Understanding RNNs deeply means understanding not just how they work but why they're designed this way, where they fail, and how later innovations like LSTMs and Transformers address their limitations while building on their insights.

The journey from feedforward networks to RNNs represents a crucial conceptual leap in deep learning: from processing static inputs independently to modeling dynamic processes with memory and temporal structure. This leap opens up vast new applications but introduces new challenges that have driven decades of research and continue to inspire innovation in sequence modeling architectures today.

