---
layout: post
title: 05-01-02-02 Character LM, Papers, and Pitfalls
chapter: '05'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter05
---

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

