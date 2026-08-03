---
layout: post
title: 10-01-02-02 Optimizer Comparisons, Papers, and Pitfalls
chapter: '10'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter10
---

# Demonstrate optimizer comparison on 2D optimization problem
print("="*70)
print("Comparing Optimizers on Rosenbrock Function")
print("="*70)
print("Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²")
print("Minimum at (1, 1), but narrow curved valley makes optimization hard\n")

def rosenbrock(x, y):
    """Classic optimization test function with narrow curved valley"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    """Gradient of Rosenbrock function"""
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([[dx], [dy]])

# Initialize parameters (start far from minimum)
theta_sgd = np.array([[-0.5], [0.5]])
theta_momentum = np.array([[-0.5], [0.5]])
theta_rmsprop = np.array([[-0.5], [0.5]])
theta_adam = np.array([[-0.5], [0.5]])

# Create optimizers
opt_sgd = type('SGD', (), {'lr': 0.001, 'params': [theta_sgd]})()
opt_momentum = SGDMomentum([theta_momentum], lr=0.001, momentum=0.9)
opt_rmsprop = RMSprop([theta_rmsprop], lr=0.01, beta=0.9)
opt_adam = Adam([theta_adam], lr=0.01, beta1=0.9, beta2=0.999)

# Track trajectories
trajectories = {
    'SGD': [theta_sgd.copy()],
    'Momentum': [theta_momentum.copy()],
    'RMSprop': [theta_rmsprop.copy()],
    'Adam': [theta_adam.copy()]
}

# Optimize for 500 steps
for step in range(500):
    # Vanilla SGD
    grad = rosenbrock_grad(theta_sgd[0,0], theta_sgd[1,0])
    theta_sgd -= opt_sgd.lr * grad
    trajectories['SGD'].append(theta_sgd.copy())
    
    # Momentum
    grad = rosenbrock_grad(theta_momentum[0,0], theta_momentum[1,0])
    opt_momentum.step([grad])
    trajectories['Momentum'].append(theta_momentum.copy())
    
    # RMSprop
    grad = rosenbrock_grad(theta_rmsprop[0,0], theta_rmsprop[1,0])
    opt_rmsprop.step([grad])
    trajectories['RMSprop'].append(theta_rmsprop.copy())
    
    # Adam
    grad = rosenbrock_grad(theta_adam[0,0], theta_adam[1,0])
    opt_adam.step([grad])
    trajectories['Adam'].append(theta_adam.copy())

# Compare final positions
print("Final positions after 500 steps:")
print(f"  SGD:      ({theta_sgd[0,0]:.4f}, {theta_sgd[1,0]:.4f})")
print(f"  Momentum: ({theta_momentum[0,0]:.4f}, {theta_momentum[1,0]:.4f})")
print(f"  RMSprop:  ({theta_rmsprop[0,0]:.4f}, {theta_rmsprop[1,0]:.4f})")
print(f"  Adam:     ({theta_adam[0,0]:.4f}, {theta_adam[1,0]:.4f})")
print(f"  True min: (1.0000, 1.0000)")

print("\nObservations:")
print("- Momentum accelerates along the valley")
print("- RMSprop adapts to different curvatures")
print("- Adam combines benefits of both")
print("- Vanilla SGD is slowest (gets stuck in oscillations)")
```

Now demonstrate on actual neural network training:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create simple classification task
print("\n" + "="*70)
print("Training Neural Network with Different Optimizers")
print("="*70)

# Generate synthetic data: XOR-like problem
np.random.seed(42)
n_samples = 1000

X = np.random.randn(n_samples, 2)
y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(float)  # XOR

X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y).unsqueeze(1)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Train with different optimizers
optimizers_to_test = {
    'SGD': lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
    'SGD+Momentum': lambda model: torch.optim.SGD(model.parameters(), 
                                                   lr=0.1, momentum=0.9),
    'RMSprop': lambda model: torch.optim.RMSprop(model.parameters(), lr=0.01),
    'Adam': lambda model: torch.optim.Adam(model.parameters(), lr=0.01),
    'AdamW': lambda model: torch.optim.AdamW(model.parameters(), lr=0.01, 
                                            weight_decay=0.01)
}

results = {}

for name, optimizer_fn in optimizers_to_test.items():
    print(f"\nTraining with {name}...")
    
    # Create fresh model
    model = SimpleNet()
    optimizer = optimizer_fn(model)
    criterion = nn.BCELoss()
    
    # Train
    losses = []
    for epoch in range(100):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            # Forward
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(dataloader))
        
        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}: Loss = {losses[-1]:.4f}")
    
    # Test accuracy
    model.eval()
    with torch.no_grad():
        pred = model(X_train)
        accuracy = ((pred > 0.5).float() == y_train).float().mean()
    
    results[name] = {
        'losses': losses,
        'final_loss': losses[-1],
        'accuracy': accuracy.item()
    }

# Compare results
print("\n" + "="*70)
print("Optimizer Comparison Results")
print("="*70)
print(f"{'Optimizer':<15} | {'Final Loss':<12} | {'Accuracy':<10}")
print("-" * 45)
for name, res in results.items():
    print(f"{name:<15} | {res['final_loss']:<12.4f} | {res['accuracy']:<10.2%}")

print("\nKey observations:")
print("- Momentum accelerates convergence over vanilla SGD")
print("- Adaptive methods (RMSprop, Adam) converge faster")
print("- AdamW often best generalization with weight decay")
print("- Choice matters: 2-5x speed difference common")
```

Demonstrate learning rate scheduling:

```python
class CosineAnnealingSchedule:
    """
    Cosine annealing learning rate schedule.
    
    Gradually decreases learning rate following cosine curve.
    Often combined with warm restarts for improved performance.
    """
    
    def __init__(self, lr_max, lr_min, T_max):
        """
        lr_max: maximum learning rate
        lr_min: minimum learning rate
        T_max: period of cosine cycle (iterations)
        """
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max
    
    def get_lr(self, t):
        """Get learning rate at iteration t"""
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
               (1 + np.cos(np.pi * (t % self.T_max) / self.T_max))

# Example
schedule = CosineAnnealingSchedule(lr_max=0.1, lr_min=0.001, T_max=100)

print("\n" + "="*70)
print("Learning Rate Scheduling")
print("="*70)

iterations = np.arange(300)
lrs = [schedule.get_lr(t) for t in iterations]

print("Learning rate evolution (first 300 iterations):")
print(f"  Start: {lrs[0]:.6f}")
print(f"  After 50 iters: {lrs[50]:.6f}")
print(f"  After 100 iters: {lrs[100]:.6f} (end of cycle, restarts)")
print(f"  After 150 iters: {lrs[150]:.6f}")
print("\nCosine annealing smoothly reduces LR, enabling fine-tuning near minima")
```

## 5. Related Concepts

The relationship between optimization algorithms and the geometry of loss surfaces illuminates why different optimizers excel in different scenarios. Deep neural network loss surfaces are highly non-convex, featuring local minima, saddle points, and plateaus. Saddle points—where gradients are zero but we're not at a minimum—are particularly common in high dimensions. Momentum helps escape saddle points by building velocity that carries through regions with zero gradient. Adaptive learning rates help when different directions have vastly different curvatures—common in neural networks where some parameters (like biases) receive consistently similar gradients while others (like weights) have highly variable gradient magnitudes.

The connection to second-order optimization methods provides theoretical context. Newton's method uses second derivatives (the Hessian matrix) to account for curvature, enabling faster convergence. However, computing and inverting the Hessian for networks with millions of parameters is computationally prohibitive—$$O(n^2)$$ memory and $$O(n^3)$$ computation. Adaptive learning rate methods like Adam approximate second-order information through gradient statistics (the second moment $$\mathbf{v}_t$$ is related to diagonal Hessian entries) without the prohibitive cost. This approximate curvature information, while cruder than full Newton methods, provides enough benefit to significantly accelerate training while remaining computationally practical.

Optimizers interact intimately with batch normalization and other normalization techniques. Batch normalization changes the loss surface geometry, making it smoother and reducing sensitivity to learning rates. This interaction can be subtle: some optimizers that work well without normalization may be less advantageous with it. Adam with batch normalization sometimes converges to worse minima than SGD with momentum, a phenomenon called "generalization gap." Understanding these interactions guides optimizer choice based on architecture—Transformers (which use layer normalization) often work best with AdamW, while ResNets (with batch normalization) might prefer SGD with momentum for final performance.

The evolution from hand-tuned learning rates to adaptive methods represents a broader trend in deep learning: automating hyperparameter choices. Early neural network training required extensive tuning of learning rates, schedules, and momentum coefficients. Modern adaptive optimizers reduce this burden—Adam's default hyperparameters work reasonably across diverse tasks. This democratization of deep learning made the field more accessible, though it also created a risk: using black-box optimizers without understanding their assumptions can lead to poor performance in edge cases. The best practitioners understand both the algorithms and when their assumptions break down.

Learning rate schedules connect to the exploration-exploitation tradeoff in optimization. Early in training, we want to explore broadly, taking larger steps to find good regions of parameter space. Later, we want to exploit, taking smaller steps to fine-tune parameters near a minimum. Schedules like cosine annealing or step decay formalize this, reducing learning rate as training progresses. Warm-up schedules do the opposite initially—start with very small learning rate and gradually increase—which helps when using very large batches or when parameters are randomly initialized and initial gradients might be misleading. The Transformer paper's warm-up schedule $$\eta_t = d_{\text{model}}^{-0.5} \min(t^{-0.5}, t \cdot \text{warmup}^{-1.5})$$ has become standard for training large models.

## 6. Fundamental Papers

**["On the importance of initialization and momentum in deep learning" (2013)](http://proceedings.mlr.press/v28/sutskever13.html)**  
*Authors*: Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton  
This paper rigorously analyzed momentum's benefits for deep learning, showing it's not just a minor improvement but essential for training deep networks effectively. The authors demonstrated that momentum combined with proper initialization (they used specific schemes for different layer types) enables training much deeper networks than vanilla SGD. They showed momentum helps escape saddle points and reduces the impact of noisy gradients from mini-batch sampling. Importantly, they provided theoretical analysis of momentum's dynamics, connecting it to classical optimization theory while demonstrating its specific advantages for non-convex neural network loss surfaces. The paper established Nesterov momentum as particularly effective, slightly but consistently outperforming standard momentum. This work influenced the field's understanding that optimization algorithms must be tailored to deep learning's unique challenges—high dimensionality, non-convexity, noisy gradients—rather than simply applying classical optimization methods.

**["Adam: A Method for Stochastic Optimization" (2015)](https://arxiv.org/abs/1412.6980)**  
*Authors*: Diederik P. Kingma, Jimmy Ba  
This paper introduced Adam and demonstrated its effectiveness across diverse tasks including image classification, language modeling, and variational inference. The key contribution was combining adaptive learning rates (like RMSprop) with momentum, while including bias correction to ensure good behavior from the first update. Kingma and Ba showed that Adam requires minimal hyperparameter tuning—default values $$\beta_1=0.9, \beta_2=0.999$$ work well across problems—making it accessible to practitioners who can't afford extensive tuning. The paper's empirical comparisons showed Adam consistently matching or exceeding other optimizers while being robust to learning rate choice. Adam became the default optimizer for many applications, particularly in NLP where its adaptation to gradient statistics helps with sparse vocabularies. The paper also introduced AdaMax (a variant using $$L_\infty$$ norm instead of $$L_2$$) and provided regret bound analysis connecting Adam to online convex optimization theory, though these theoretical aspects are less commonly used than the practical algorithm.

**["Decoupled Weight Decay Regularization" (2019)](https://arxiv.org/abs/1711.05101)**  
*Authors*: Ilya Loshchilov, Frank Hutter  
This paper identified a subtle but important flaw in how Adam handles L2 regularization and proposed AdamW as the solution. The authors showed that adding weight decay to gradients (standard practice) and then applying adaptive learning rates (as Adam does) causes the effective weight decay to vary across parameters based on their gradient statistics. This coupling undermines regularization—parameters with large gradients receive less weight decay, opposite of what's desirable. AdamW decouples weight decay from gradient-based updates, applying it directly to parameters after the adaptive update. The paper demonstrated improved generalization across multiple benchmarks, particularly for Transformers where proper regularization is crucial. AdamW has largely replaced Adam for training large language models and other Transformer-based systems. The work exemplifies how understanding the interaction between different training components (optimization + regularization) reveals subtle issues that significantly impact practical performance.

**["On the Variance of the Adaptive Learning Rate and Beyond" (2020)](https://arxiv.org/abs/1908.03265)**  
*Authors*: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Jiawei Han  
This paper analyzed why Adam sometimes generalizes worse than SGD despite converging faster, a phenomenon called the "generalization gap." The authors showed that Adam's adaptive learning rates can lead to sharp minima (low training loss but poor generalization) while SGD with momentum tends to find flatter minima (better generalization). They proposed RAdam (Rectified Adam), which modifies the bias correction to be more conservative early in training when gradient statistics are unreliable. The paper deepened understanding of the optimization-generalization tradeoff: faster convergence doesn't always mean better final performance. It showed that variance in adaptive learning rates can be harmful and proposed variance reduction techniques. This work has influenced how practitioners use Adam—recognizing when its adaptive mechanism helps (sparse gradients, varying scales) versus when simpler methods with better generalization properties (SGD+momentum) are preferable.

**["Lookahead Optimizer: k steps forward, 1 step back" (2019)](https://arxiv.org/abs/1907.08610)**  
*Authors*: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba  
This paper introduced a meta-optimization algorithm that wraps around any base optimizer (SGD, Adam, etc.). Lookahead maintains two sets of weights: fast weights updated by the base optimizer and slow weights that periodically synchronize with fast weights. The algorithm runs the base optimizer for $$k$$ steps (typically 5-10), then updates slow weights toward the fast weights, then resets fast weights to the slow weights. This reduces variance in optimization trajectory and improves convergence. The paper showed that Lookahead improves performance of base optimizers consistently across tasks, providing more stable training and often better generalization. While less commonly used than Adam or SGD+momentum, Lookahead demonstrates that optimization algorithms can be composed—we can build meta-algorithms that enhance existing optimizers. The paper's empirical analysis across vision and language tasks established that optimizer design remains an active research area with room for innovation beyond the classics.

## Common Pitfalls and Tricks

The most common mistake when using adaptive optimizers like Adam is forgetting to adjust hyperparameters when changing batch size. With vanilla SGD, doubling batch size roughly requires doubling learning rate to maintain equivalent parameter updates (since gradients are averaged over batch). But for Adam, the relationship is more complex because adaptive learning rates already account for gradient magnitudes. A practical rule: when increasing batch size, increase learning rate proportionally but less aggressively (perhaps by $$\sqrt{2}$$ instead of $$2$$), and monitor validation performance carefully. Very large batch sizes (thousands) may require learning rate warm-up to prevent early instability.

A subtle issue is optimizer state accumulation when fine-tuning pre-trained models. If you load a pre-trained model and continue training with Adam, the momentum and variance estimates start from zero, not from values appropriate for a nearly-converged model. This can cause instability or prevent fine-tuning from improving the model. The solution: either use a lower learning rate for fine-tuning (allowing gradients to build up optimizer state safely) or reset the optimizer state when loading checkpoints, starting fresh. Understanding that optimizers maintain internal state beyond just parameters helps debug unexpected fine-tuning behavior.

Weight decay in AdamW requires calibration differently than in SGD. For SGD, weight decay around 0.0001-0.001 is typical. For AdamW, values around 0.01-0.1 often work better because the decoupling changes its effective strength. When migrating from Adam to AdamW, don't just enable weight decay with values tuned for SGD—you'll likely over-regularize. Start with 0.01 and tune based on train-test gap. This illustrates a broader principle: hyperparameters are not architecture-agnostic but must be tuned within the context of the full training configuration.

Gradient clipping interacts with optimizers in non-obvious ways. For Adam, clipping gradients before the optimizer sees them affects both momentum and variance estimates. If gradients are clipped to norm 5, the maximum second moment becomes 25, bounding the adaptive scaling. This can be beneficial (prevents extremely small effective learning rates) or harmful (prevents adaptation to true gradient scales). For stability, clip gradients for RNNs and Transformers. For maximum Adam adaptivity on well-behaved networks, skip clipping. Understanding this tradeoff helps choose appropriate configurations.

A powerful technique for hyperparameter tuning is cyclical learning rates—varying learning rate between bounds during training. This allows the model to periodically escape local minima it might settle into, potentially finding better solutions. Combined with snapshot ensembling (saving models at different points in the cycle and ensembling their predictions), this can improve performance beyond single-model training with fixed learning rates. The computational cost is minimal (just scheduling) while benefits can be substantial, making it an underutilized trick in the practitioner's toolkit.

## Key Takeaways

Advanced optimization algorithms improve upon vanilla gradient descent by incorporating momentum to accelerate in consistent directions and dampen oscillations, and by adapting learning rates per parameter based on gradient history. SGD with momentum builds velocity from exponentially weighted gradient averages, helping traverse ravines and escape plateaus. RMSprop adapts learning rates using exponential averages of squared gradients, automatically scaling updates based on typical gradient magnitudes per parameter. Adam combines both mechanisms while including bias correction for proper early-iteration behavior, becoming the de facto standard for many applications due to robust performance with minimal tuning. AdamW improves Adam by decoupling weight decay from gradient-based updates, ensuring regularization strength is independent of adaptive scaling, crucial for training large Transformers. The choice of optimizer involves tradeoffs between convergence speed, final performance, computational cost, and hyperparameter sensitivity, with no single optimizer dominating all scenarios. Understanding each optimizer's assumptions—what loss surface geometry it handles well, what gradient statistics it expects—enables matching algorithms to problems effectively. Modern practice often uses Adam or AdamW for initial experimentation due to robustness, potentially switching to SGD with momentum for final training if better generalization is needed. The sophistication of these algorithms shouldn't obscure the fundamental principle: they're all using gradients computed via backpropagation to iteratively improve parameters, differing only in how they process gradients into parameter updates.

The evolution of optimization algorithms from vanilla gradient descent to modern adaptive methods represents the field learning to automate aspects of training that previously required expert tuning, democratizing deep learning while also introducing new subtleties that practitioners must understand to train models effectively.

