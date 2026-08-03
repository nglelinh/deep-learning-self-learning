---
layout: post
title: 10-01-01 Optimizer Theory
chapter: '10'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter10
---

# Advanced Optimization: Beyond Vanilla Gradient Descent

## 1. Concept Overview

While gradient descent provides the fundamental principle for training neural networks—move parameters in the direction that decreases loss—its vanilla form suffers from several critical limitations that make training deep networks impractical. The learning rate must be carefully tuned: too large causes oscillation or divergence, too small causes painfully slow convergence. The same learning rate is used for all parameters, despite different parameters having different gradient scales and optimal update frequencies. Gradient descent treats all directions in parameter space equally, even though some directions represent ravines (steep in one direction, gentle in another) where we should move carefully. And it has no memory of previous gradients, unable to build momentum to escape shallow local minima or saddle points.

Advanced optimization algorithms address these limitations through various mechanisms: maintaining momentum to accelerate in consistent directions while dampening oscillations; adapting learning rates per parameter based on gradient history, allowing aggressive updates for sparse gradients and conservative updates for frequent large gradients; and incorporating second-order information about the curvature of the loss surface without the prohibitive cost of computing full Hessian matrices. These improvements aren't minor tweaks but essential techniques that have enabled training increasingly large and complex models—modern language models with billions of parameters simply couldn't be trained with vanilla gradient descent.

Understanding these optimizers deeply means recognizing that they're not competing alternatives but tools suited for different scenarios. Stochastic Gradient Descent with momentum excels when the loss surface has clear, consistent gradient directions and is computationally efficient, making it popular for computer vision tasks with large batch sizes. RMSprop adapts learning rates based on recent gradient magnitudes, particularly useful for recurrent networks where gradient scales vary dramatically across time steps. Adam combines momentum and adaptive learning rates, providing good default performance across diverse tasks and becoming the de facto standard for many applications. AdamW improves Adam's weight decay handling, crucial for training large Transformers. Each optimizer embodies different assumptions about the loss surface and gradient dynamics, and choosing appropriately can mean the difference between a model that trains in hours versus days, or that trains successfully versus not at all.

The evolution of optimization algorithms parallels the evolution of neural architectures. As networks became deeper (requiring techniques to handle vanishing/exploding gradients), optimizers evolved to adapt learning rates and build momentum. As networks became larger (requiring training on smaller batches due to memory constraints), optimizers developed to work effectively with noisy gradient estimates. As tasks diversified (from vision to NLP to reinforcement learning), optimizers became more adaptive to different gradient landscapes. This co-evolution of architectures and optimizers is ongoing—new architectures often require optimizer innovations, and new optimizers enable new architectures.

Yet with all these sophisticated algorithms, the fundamentals remain: we're still computing gradients via backpropagation and taking steps opposite to these gradients. The advanced optimizers change how we determine step sizes and directions, leveraging gradient history and statistics, but the core principle—iterative refinement based on loss gradients—stays constant. This means that understanding vanilla gradient descent deeply provides the foundation for understanding all variants, which are best seen as sophisticated modifications addressing specific failure modes rather than entirely different approaches.

## 2. Mathematical Foundation

Let's build up the mathematics of advanced optimizers systematically, understanding each component's purpose and how they combine to improve upon vanilla gradient descent. We'll start with momentum and progress through increasingly sophisticated techniques.

### Momentum: Building Velocity

Vanilla gradient descent updates parameters using only the current gradient:

$$\theta_t = \theta_{t-1} - \eta \nabla_\theta \mathcal{L}(\theta_{t-1})$$

Momentum introduces a velocity term that accumulates gradients over time:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1})$$

$$\theta_t = \theta_{t-1} - \eta \mathbf{v}_t$$

where $$\beta \in [0, 1)$$ is the momentum coefficient (typically 0.9). The velocity $$\mathbf{v}_t$$ is an exponentially weighted moving average of gradients. Expanding the recursion reveals how past gradients influence current updates:

$$\mathbf{v}_t = \nabla_\theta \mathcal{L}(\theta_{t-1}) + \beta \nabla_\theta \mathcal{L}(\theta_{t-2}) + \beta^2 \nabla_\theta \mathcal{L}(\theta_{t-3}) + \ldots$$

Recent gradients have full weight, while older gradients contribute with exponentially decaying weights $$\beta^k$$. This creates several beneficial effects. First, if gradients consistently point in the same direction, the velocity builds up, accelerating progress—like a ball rolling downhill gaining speed. Second, if gradients oscillate (positive then negative), the velocity dampens oscillations—opposing gradients partially cancel. Third, momentum can carry the optimization through shallow local minima or flat regions where current gradients are near zero but past gradients indicated a good direction.

The geometric intuition is that momentum transforms the gradient from a force into a velocity. In physics, force (gradient) causes acceleration, leading to velocity changes. Here, the gradient directly contributes to velocity, which determines position updates. This physics analogy isn't perfect but captures how momentum creates inertia—the optimization continues moving in directions that were previously good even if the current gradient disagrees slightly.

### Nesterov Accelerated Gradient (NAG)

A clever modification of momentum computes gradients not at the current position but at the anticipated future position:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1} - \beta \mathbf{v}_{t-1})$$

$$\theta_t = \theta_{t-1} - \eta \mathbf{v}_t$$

The key difference is $$\nabla_\theta \mathcal{L}(\theta_{t-1} - \beta \mathbf{v}_{t-1})$$ instead of $$\nabla_\theta \mathcal{L}(\theta_{t-1})$$. We're computing the gradient at where momentum would take us, then using that gradient to refine the update. This "look ahead" provides a form of correction: if momentum is carrying us toward a bad region, the gradient at the anticipated position will indicate this, allowing us to slow down or change direction.

The improvement over standard momentum is subtle but consistent across many tasks. NAG typically converges faster and overshoots less at minima. The intuition is that standard momentum is reactive (respond to gradients at current position) while NAG is proactive (anticipate where we're going and plan accordingly). In practice, the difference between momentum and NAG is often small, but NAG is theoretically better motivated and occasionally provides noticeable improvements.

### AdaGrad: Adaptive Learning Rates

AdaGrad adapts learning rates per parameter based on accumulated squared gradients:

$$\mathbf{G}_t = \mathbf{G}_{t-1} + (\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \odot \nabla_\theta \mathcal{L}(\theta_{t-1})$$

where the square and square root are element-wise, $$\mathbf{G}_t$$ accumulates squared gradients, and $$\epsilon$$ (typically $$10^{-8}$$) prevents division by zero. The division by $$\sqrt{\mathbf{G}_t}$$ means parameters with large accumulated gradients receive smaller updates, while parameters with small accumulated gradients receive larger updates.

This adaptive scaling addresses a key limitation of vanilla gradient descent. In sparse features (common in NLP where most words don't appear in most documents), some parameters receive gradient updates rarely. AdaGrad gives these infrequent parameters larger updates when they do receive gradients, while frequently updated parameters (corresponding to common features) receive smaller updates. This is particularly valuable in tasks with sparse data or highly variable feature frequencies.

However, AdaGrad has a fatal flaw for long training runs: $$\mathbf{G}_t$$ only grows, never shrinks. As training progresses, $$\sqrt{\mathbf{G}_t}$$ becomes very large, making effective learning rates approach zero, and learning stops. This aggressive learning rate decay is appropriate for convex optimization where we want to slow down as we approach the minimum, but neural network loss surfaces are non-convex with many local minima, plateaus, and saddle points. Stopping adaptation too early prevents escaping these suboptimal regions.

### RMSprop: Exponential Moving Average

RMSprop fixes AdaGrad's aggressive decay by using an exponentially weighted moving average of squared gradients instead of accumulation:

$$\mathbf{E}[g^2]_t = \beta \mathbf{E}[g^2]_{t-1} + (1-\beta)(\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\mathbf{E}[g^2]_t + \epsilon}} \odot \nabla_\theta \mathcal{L}(\theta_{t-1})$$

Typical $$\beta = 0.9$$ means we consider roughly the last $$1/(1-\beta) = 10$$ gradient updates. This allows the algorithm to forget old gradients, so if the gradient scale changes (as we move through different regions of the loss surface), the learning rate adaptation adjusts. RMSprop became particularly popular for training RNNs where gradient scales vary dramatically, and it remains a solid choice when gradient statistics change over training.

### Adam: Adaptive Moment Estimation

Adam combines momentum and RMSprop's adaptive learning rates, maintaining both first moment (mean) and second moment (uncentered variance) estimates:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}(\theta_{t-1})$$ (momentum term)

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$ (RMSprop term)

These are biased toward zero initially (since $$\mathbf{m}_0 = \mathbf{v}_0 = 0$$). Adam corrects this bias:

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

The update rule combines both:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$

Default hyperparameters $$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$$ work well across many tasks, making Adam popular as a "low-tuning" optimizer. The algorithm adapts to gradient statistics (through $$\mathbf{v}_t$$) while building momentum (through $$\mathbf{m}_t$$), combining benefits of both approaches.

The bias correction deserves careful attention. Early in training, $$\mathbf{m}_t$$ and $$\mathbf{v}_t$$ are dominated by their initialization at zero, making them biased estimates of true moments. For example, $$\mathbf{m}_1 = (1-\beta_1)g_1$$ significantly underestimates $$\mathbb{E}[g]$$ when $$\beta_1 = 0.9$$. Dividing by $$1-\beta_1^t$$ corrects this: $$\hat{\mathbf{m}}_1 = \frac{(1-\beta_1)g_1}{1-\beta_1} = g_1$$. As $$t \to \infty$$, $$\beta_1^t \to 0$$, so the correction factor approaches 1 and has no effect. This ensures good behavior from the first update while asymptotically behaving like uncorrected exponential averages.

### AdamW: Decoupled Weight Decay

A subtle issue with Adam is how it handles L2 regularization (weight decay). Standard practice adds $$\lambda \theta$$ to gradients:

$$\nabla \mathcal{L}_{\text{reg}} = \nabla \mathcal{L} + \lambda \theta$$

But in Adam, this regularized gradient gets processed through adaptive learning rates, which can dilute the regularization effect. AdamW decouples weight decay from gradient-based optimization:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}(\theta_{t-1})$$

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

The weight decay term $$\lambda \theta_{t-1}$$ is added after adaptive scaling, ensuring regularization strength is independent of gradient statistics. This seemingly minor change significantly improves generalization, particularly for Transformers and other large models where proper regularization is crucial.

## 3. Example / Intuition

To understand how different optimizers behave, imagine optimizing a function with a ravine: steep sides and a gentle slope along the bottom toward the minimum. Picture a 2D loss surface where one direction has high curvature (steep) and the perpendicular direction has low curvature (gentle). The minimum lies at the bottom of this ravine.

**Vanilla Gradient Descent**: Steps perpendicular to contours of constant loss. In the ravine, gradients point mostly toward the ravine bottom (steep direction), barely along it (gentle direction). We take large steps toward the sides, bounce between them, and make slow progress along the ravine toward the minimum. It's inefficient—most gradient magnitude is in the wrong direction (perpendicular to the path to minimum) rather than the right direction (along the ravine).

**SGD with Momentum**: Accumulates velocity along the ravine as consistent gradients in that direction build up momentum. When gradients oscillate perpendicular to the ravine (positive then negative as we bounce between sides), the velocity in that direction dampens. We accelerate along the ravine while oscillations perpendicular to it are suppressed. The ball rolling downhill analogy is apt—momentum carries us through flat regions and helps escape shallow bowls.

**AdaGrad/RMSprop**: Notices that gradients in the steep direction are consistently large, while gradients in the gentle direction are small. It reduces learning rate in the steep direction (to prevent bouncing) and maintains it in the gentle direction (to make progress). This automatically does gradient rescaling based on the different curvatures, allowing larger effective steps along the ravine even with smaller steps perpendicular to it.

**Adam**: Combines both mechanisms. Momentum accelerates along the ravine. Adaptive learning rates prevent excessive bouncing. The result is fast, stable progress toward the minimum. Adam also handles the fact that gradient statistics change as we move—early in training, far from the minimum, gradients are large; near the minimum, they shrink. The adaptive scaling adjusts automatically.

Consider a concrete scenario: training a neural network on a dataset with rare but important features. Vanilla SGD updates all parameters equally, so rare features get updated infrequently (only when examples containing them appear). AdaGrad/Adam give these parameters larger effective learning rates (because their $$\mathbf{v}_t$$ is smaller, having accumulated fewer gradient updates), allowing them to learn quickly from the few examples they see. Common features, updated frequently, get smaller effective learning rates, preventing overreaction to individual examples. This adaptivity is why Adam often converges faster than SGD, particularly in NLP where vocabulary sparsity is extreme.

