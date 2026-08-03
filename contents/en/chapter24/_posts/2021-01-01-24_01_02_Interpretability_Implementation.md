---
layout: post
title: 24-01-02 Interpretability Implementation
chapter: '24'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter24
---

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

