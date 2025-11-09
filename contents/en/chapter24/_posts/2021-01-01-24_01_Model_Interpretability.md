---
layout: post
title: 24-01 Neural Network Interpretability and Explainability
chapter: '24'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter24
lesson_type: required
---

# Model Interpretability: Understanding the Black Box

## 1. Concept Overview

Neural network interpretability addresses one of deep learning's most significant challenges: understanding why models make particular predictions. While neural networks achieve remarkable performance on diverse tasks, their decision-making process often appears opaque—millions of parameters interacting through nonlinear transformations make it difficult to trace how inputs map to outputs. This "black box" nature creates problems in high-stakes domains like medicine (why did the model diagnose this condition?), law (why was this defendant classified as high-risk?), and autonomous vehicles (why did the car brake suddenly?), where we need not just accurate predictions but justifications we can audit, trust, and debug.

Understanding the distinction between interpretability and explainability clarifies what we're seeking. Interpretability means the model's internal workings are transparent—we can understand the computation from inspection. Simple models like linear regression or decision trees are inherently interpretable; we can see exactly how features combine to produce predictions. Explainability means we can provide post-hoc explanations of why a model made specific predictions, even if the model itself isn't inherently transparent. Deep neural networks are rarely interpretable (understanding millions of parameters is infeasible) but can be made explainable through techniques that highlight relevant inputs, visualize learned features, or approximate decisions with interpretable surrogates.

The motivation for interpretability extends beyond satisfying curiosity. In scientific applications, understanding what features models use can generate new hypotheses—if a medical imaging model identifies a subtle pattern doctors missed, investigating this pattern might reveal new diagnostic markers. In debugging, interpretability reveals when models exploit spurious correlations (detecting huskies by snow in background rather than dog features) or fail to use relevant features. In safety-critical applications, interpretability enables verification that models behave reasonably across diverse scenarios. In regulated industries, explainability may be legally required for decisions affecting individuals. Understanding these diverse motivations helps appreciate that interpretability isn't a single goal but multiple related objectives requiring different techniques.

The landscape of interpretability methods is vast, reflecting the multiple ways we might understand neural networks. Saliency methods highlight which input regions most influenced a prediction, answering "what did the model look at?" Activation visualization shows what patterns activate neurons, revealing "what features has the network learned?" Attribution methods decompose predictions into feature contributions, explaining "how much did each input feature matter?" Concept-based explanations identify high-level concepts the model uses, moving beyond pixel or word-level attributions to semantic understanding. Each approach provides different insights, and comprehensive interpretability often requires multiple complementary techniques.

Yet interpretability has fundamental tensions. More interpretable models are often less accurate (linear models vs deep networks). Faithful explanations (accurately describing model behavior) might be complex and hard to understand. Simple explanations might be understandable but unfaithful to actual model behavior. Perfect interpretability might require understanding millions of parameters—as complex as understanding the phenomenon the model learned. These tensions mean interpretability research involves careful tradeoffs between fidelity, simplicity, and utility, without universal solutions that satisfy all desiderata simultaneously.

## 2. Mathematical Foundation

Interpretability methods often formalize the question "which inputs matter most for this prediction?" through attribution. Given input $$\mathbf{x}$$ and model $$f$$, compute attribution $$\mathbf{a}$$ where $$a_i$$ indicates importance of input dimension $$i$$ for the prediction $$f(\mathbf{x})$$.

### Gradient-Based Saliency

The simplest attribution uses gradients:

$$\mathbf{a} = \left|\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}\right|$$

This measures how much the output would change for small changes in each input dimension. Large gradient indicates high sensitivity—that input dimension strongly influences the output. For images, this produces saliency maps highlighting important pixels.

However, gradient can saturate in ReLU networks (gradient is zero or one, not informative about magnitude of change) and doesn't account for baseline (what are we comparing against?). Improvements address these issues:

**Integrated Gradients** accumulates gradients along path from baseline $$\mathbf{x}'$$ to input $$\mathbf{x}$$:

$$\mathbf{a}_i = (x_i - x_i') \int_{\alpha=0}^1 \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$

This satisfies desirable axioms: sensitivity (if input doesn't affect output, attribution is zero) and implementation invariance (equivalent networks give same attributions).

### SHAP: Shapley Additive Explanations

SHAP uses Shapley values from cooperative game theory. The contribution of feature $$i$$ is:

$$\phi_i = \sum_{S \subseteq \mathcal{F} \backslash \{i\}} \frac{|S|!(|\mathcal{F}|-|S|-1)!}{|\mathcal{F}|!} [f(S \cup \{i\}) - f(S)]$$

where $$\mathcal{F}$$ is all features, $$S$$ are feature subsets, $$f(S)$$ is model output with only features in $$S$$ present (others set to baseline). This computes the average marginal contribution of feature $$i$$ across all possible feature coalitions—a fair allocation of prediction among features.

Computing exact Shapley values requires $$2^{|\mathcal{F}|}$$ model evaluations (exponential in features), so approximations are used. Kernel SHAP approximates through weighted linear regression. For tree-based models, TreeSHAP computes exactly in polynomial time.

### Layer-wise Relevance Propagation (LRP)

LRP backpropagates relevance from output to input:

$$R_i^{(l)} = \sum_j \frac{z_{ij}}{\sum_k z_{kj}} R_j^{(l+1)}$$

where $$z_{ij} = a_i^{(l)} w_{ij}$$ is contribution of neuron $$i$$ in layer $$l$$ to neuron $$j$$ in layer $$l+1$$. Starting with $$R_{\text{out}} = f(\mathbf{x})$$ at output, relevance propagates backward, decomposing prediction into input contributions satisfying $$\sum_i R_i^{(0)} = f(\mathbf{x})$$ (conservation).

## 3. Example / Intuition

Consider a CNN classifying an image as "dog" with 95% confidence. Without interpretability, we don't know why. Was it the dog's face, body shape, background context, or spurious patterns like grass (if all dogs in training had grass backgrounds)?

**Gradient saliency** computes $$\partial p_{\text{dog}}/\partial \text{pixels}$$. Large gradients highlight pixels that, if changed slightly, would most affect the dog probability. Visualized as a heatmap overlay on the image, we might see high values around the dog's face and ears—good, the model uses actual dog features. If high values appear in background, the model might be exploiting spurious correlations.

**Class Activation Mapping (CAM)** for CNNs with global average pooling shows which regions the final convolutional layer found important. For "dog" class, we compute weighted combination of final conv layer's feature maps using the classification weights:

$$\text{CAM} = \sum_k w_k^{\text{dog}} \cdot \text{FeatureMap}_k$$

This produces a heatmap at feature map resolution showing which spatial regions contributed to the "dog" prediction. Upsampling to input resolution and overlaying on the image reveals the model focused on the dog's head and body—interpretable and reassuring.

**SHAP values** for a particular prediction might show:
- Pixel region containing dog face: +0.35 (strong positive contribution)
- Pixel region with dog body: +0.28
- Background grass: +0.08 (small contribution - concerning if high)
- Sky region: -0.02 (slight negative - expected for irrelevant regions)

If grass has high SHAP value, we've discovered the model uses spurious correlation (dogs often photographed on grass). We can then collect more diverse training data or use data augmentation to fix this.

**Adversarial examples** provide another interpretability lens. By finding minimal input perturbations that change predictions, we reveal model vulnerabilities. If adding imperceptible noise to dog image causes "cat" prediction, the model's representation is fragile—it hasn't learned robust features. Studying these adversarial perturbations reveals what features matter: perturbations often add patterns the model strongly associates with target class, revealing learned (but perhaps spurious) class indicators.

## 4. Code Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Visualizes which regions of image are important for prediction
    by computing gradients of class score with respect to final
    convolutional layer activations.
    """
    
    def __init__(self, model, target_layer):
        """
        model: CNN model
        target_layer: name of convolutional layer to visualize
        """
        self.model = model
        self.target_layer = target_layer
        
        # Storage for forward activations and backward gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_image, target_class):
        """
        Generate CAM for target class.
        
        input_image: (1, 3, H, W)
        target_class: index of class to explain
        
        Returns: (H, W) heatmap
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get activations and gradients
        activations = self.activations  # (1, C, H, W)
        gradients = self.gradients  # (1, C, H, W)
        
        # Global average pool gradients to get weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()

# Example usage
print("="*70)
print("Grad-CAM: Visualizing CNN Decisions")
print("="*70)

# Load pre-trained model
from torchvision import models, transforms
from PIL import Image

model = models.resnet50(weights='DEFAULT')
model.eval()

# Create Grad-CAM
gradcam = GradCAM(model, target_layer='layer4')

# Load and preprocess image (simulated here)
print("\nGenerating Grad-CAM for sample image...")
input_tensor = torch.randn(1, 3, 224, 224)

# Get prediction
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    confidence = F.softmax(output, dim=1)[0, predicted_class].item()

print(f"Predicted class: {predicted_class} with confidence {confidence:.3f}")

# Generate CAM for predicted class
cam = gradcam.generate_cam(input_tensor, predicted_class)

print(f"CAM shape: {cam.shape}")
print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
print("\nCAM highlights image regions important for prediction!")
print("High values = important regions, low values = irrelevant")

# SHAP approximation
class SimpleSHAP:
    """Simplified SHAP for neural networks"""
    
    def __init__(self, model, background_data):
        """
        model: neural network
        background_data: reference dataset for baselines
        """
        self.model = model
        self.background = background_data
    
    def explain(self, x, num_samples=100):
        """
        Approximate SHAP values for input x.
        
        Uses sampling to approximate Shapley values:
        repeatedly mask random subsets of features,
        measure prediction changes.
        """
        # Get baseline prediction (average over background)
        with torch.no_grad():
            baseline_output = self.model(self.background).mean(dim=0)
        
        # Generate random feature masks
        n_features = x.numel()
        masks = torch.rand(num_samples, n_features) > 0.5
        
        # Compute predictions with different feature subsets
        attributions = torch.zeros(n_features)
        
        for mask in masks:
            # Mask some features (use background average)
            x_masked = x.clone().view(-1)
            bg_avg = self.background.mean(dim=0).view(-1)
            x_masked[~mask] = bg_avg[~mask]
            x_masked = x_masked.view(x.shape)
            
            # Compute prediction
            with torch.no_grad():
                output = self.model(x_masked.unsqueeze(0))
            
            # Marginal contribution of each feature
            pred_diff = output[0] - baseline_output
            attributions += mask.float() * pred_diff.mean()
        
        attributions /= num_samples
        
        return attributions.view(x.shape)

print("\n" + "="*70)
print("SHAP: Feature Attribution")
print("="*70)

# Simple example with a linear model (for verification)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

simple_model = SimpleModel()
background = torch.randn(100, 10)  # Reference data

shap = SimpleSHAP(simple_model, background)

# Explain a prediction
test_input = torch.randn(1, 10)
attributions = shap.explain(test_input.squeeze(), num_samples=100)

print("Feature attributions (SHAP values):")
print(attributions[:5].numpy())
print("\nPositive = feature increased prediction")
print("Negative = feature decreased prediction")
print("Magnitude = importance")
```

Adversarial examples for probing robustness:

```python
def fgsm_attack(model, image, label, epsilon=0.1):
    """
    Fast Gradient Sign Method: minimal perturbation to fool model.
    
    Reveals model's decision boundaries and vulnerabilities.
    Shows what patterns strongly influence predictions.
    """
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    loss = F.cross_entropy(output, label)
    
    # Backward to get gradients
    model.zero_grad()
    loss.backward()
    
    # Create adversarial example
    # Move in direction that increases loss (fools model)
    perturbation = epsilon * image.grad.sign()
    adversarial = image + perturbation
    
    return adversarial.detach()

print("\n" + "="*70)
print("Adversarial Examples: Probing Model Robustness")
print("="*70)

# Simple classifier
class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.fc = nn.Linear(32 * 6 * 6, 10)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

toy_model = ToyClassifier()
toy_model.eval()

# Original image (random for demo)
original = torch.randn(1, 3, 16, 16)
true_label = torch.tensor([3])

# Get original prediction
with torch.no_grad():
    orig_output = toy_model(original)
    orig_pred = orig_output.argmax(dim=1).item()
    orig_conf = F.softmax(orig_output, dim=1)[0, orig_pred].item()

print(f"Original prediction: class {orig_pred} ({orig_conf:.3f} confidence)")

# Generate adversarial example
adversarial = fgsm_attack(toy_model, original.clone(), true_label, epsilon=0.1)

# Test adversarial
with torch.no_grad():
    adv_output = toy_model(adversarial)
    adv_pred = adv_output.argmax(dim=1).item()
    adv_conf = F.softmax(adv_output, dim=1)[0, adv_pred].item()

print(f"Adversarial prediction: class {adv_pred} ({adv_conf:.3f} confidence)")

# Measure perturbation
perturbation = (adversarial - original).abs().mean().item()
print(f"Average perturbation: {perturbation:.6f}")

if adv_pred != orig_pred:
    print("✗ Model fooled by imperceptible perturbation!")
    print("This reveals model's decision boundary is fragile")
else:
    print("✓ Model robust to this perturbation")
```

## 5. Related Concepts

Interpretability connects to causality through attempts to move beyond correlation to understanding causal mechanisms. Attribution methods identify correlations between inputs and outputs, but correlation doesn't imply causation. Causal interpretability seeks to answer "would changing this input feature actually cause the prediction to change?" requiring interventions and counterfactual reasoning beyond standard attribution.

The relationship to uncertainty quantification provides complementary understanding. Interpretability shows why a prediction was made. Uncertainty quantification shows how confident the model is. Together, they provide comprehensive understanding: "the model predicts class A because of feature X, with confidence Y." Bayesian deep learning, dropout for uncertainty, and ensemble methods complement interpretability by quantifying prediction reliability.

Interpretability relates to fairness and bias detection. If a hiring model uses gender or race features (directly or through proxies), interpretability methods reveal this, enabling auditing for discriminatory behavior. Understanding what features drive predictions is prerequisite for ensuring fairness, though interpretability alone doesn't guarantee fairness—we must also determine whether identified features are legitimate or biased.

## 6. Fundamental Papers

**["Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps" (2014)](https://arxiv.org/abs/1312.6034)**  
*Authors*: Karen Simonyan, Andrea Vedaldi, Andrew Zisserman  
Introduced gradient-based saliency maps for CNNs, showing derivatives reveal which pixels matter for predictions. Established visualization as key interpretability tool.

**["Visualizing and Understanding Convolutional Networks" (2014)](https://arxiv.org/abs/1311.2901)**  
*Authors*: Matthew Zeiler, Rob Fergus  
Deconvolution networks visualized what features CNN layers learn. Showed early layers detect edges/colors, middle layers detect textures/patterns, deep layers detect object parts. Foundational for understanding CNN hierarchical features.

**["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2017)](https://arxiv.org/abs/1610.02391)**  
*Authors*: Ramprasaath Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra  
Grad-CAM generates visual explanations by using gradients to weight feature maps, producing class-discriminative localization without modifying architecture. Became standard for CNN interpretability.

**["A Unified Approach to Interpreting Model Predictions" (2017)](https://arxiv.org/abs/1705.07874)**  
*Authors*: Scott Lundberg, Su-In Lee  
SHAP unified multiple attribution methods under Shapley value framework from game theory, providing theoretically grounded explanations with desirable properties. Became standard for tabular data and model-agnostic explanations.

**["Axiomatic Attribution for Deep Networks" (2017)](https://arxiv.org/abs/1703.01365)**  
*Authors*: Mukund Sundararajan, Ankur Taly, Qiqi Yan  
Integrated Gradients satisfied attribution axioms (sensitivity, implementation invariance) that gradient-based methods violate. Provided principled attribution method with theoretical guarantees.

## Common Pitfalls and Tricks

Saliency maps can be misleading. High gradient doesn't always mean high importance—could be artifact of network architecture (batch norm, ReLU) or optimization. Always verify interpretations with ablation (actually remove features and measure impact) or by checking if explanations align with domain knowledge.

Adversarial examples don't necessarily indicate poor models. All models have adversarial vulnerabilities—it's a fundamental property of high-dimensional spaces. Focus on robustness to natural perturbations (noise, blur) rather than adversarially crafted worst-cases unless security is critical.

## Key Takeaways

Neural network interpretability enables understanding why models make predictions through attribution methods (which inputs mattered), visualization techniques (what features were learned), and explanation frameworks (how decisions decompose). Gradient-based saliency highlights input regions with high sensitivity to prediction changes. Grad-CAM visualizes spatial importance in CNNs through gradient-weighted feature maps. SHAP provides theoretically grounded attributions through Shapley values from game theory. Adversarial examples reveal model robustness by finding minimal perturbations changing predictions. Different interpretability methods provide complementary insights—saliency for input importance, activation visualization for learned features, SHAP for faithful attribution—often requiring multiple techniques for comprehensive understanding. Interpretability enables debugging spurious correlations, building trust in high-stakes applications, satisfying regulatory requirements, and generating scientific insights from learned patterns, making it increasingly important as deep learning deployment expands to critical domains requiring explainability beyond accuracy.

