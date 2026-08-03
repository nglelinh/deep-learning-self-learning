---
layout: post
title: 24-01-02 Cài đặt Diễn giải
chapter: '24'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter24
---

## 4. Mã minh họa
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

Ví dụ đối kháng để thăm dò độ vững chắc:

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

## 5. Khái niệm liên quan
Diễn giải gắn với nhân quả qua nỗ lực vượt tương quan sang hiểu cơ chế nhân quả. Phương pháp gán công trạng xác định tương quan giữa đầu vào và đầu ra, nhưng tương quan không hàm ý nhân quả. Diễn giải nhân quả tìm cách trả lời “thay đổi đặc trưng đầu vào này có thực sự gây thay đổi dự đoán không?”, đòi hỏi can thiệp và suy luận phản thực vượt gán công trạng chuẩn.

Mối quan hệ với định lượng bất định cung cấp hiểu biết bổ sung. Diễn giải cho thấy vì sao dự đoán được đưa ra. Định lượng bất định cho thấy mô hình tin cậy đến mức nào. Cùng nhau, chúng cung cấp hiểu biết toàn diện: “mô hình dự đoán lớp A vì đặc trưng X, với độ tin cậy Y”. Học sâu Bayes, dropout cho bất định, và phương pháp ensemble bổ sung diễn giải bằng cách định lượng độ tin cậy dự đoán.

Diễn giải liên quan đến công bằng và phát hiện thiên kiến. Nếu mô hình tuyển dụng dùng đặc trưng giới hoặc chủng tộc (trực tiếp hoặc qua proxy), phương pháp diễn giải tiết lộ điều này, cho phép kiểm toán hành vi phân biệt. Hiểu đặc trưng nào lái dự đoán là điều kiện tiên quyết để đảm bảo công bằng, dù diễn giải một mình không đảm bảo công bằng — ta còn phải xác định liệu các đặc trưng đã xác định là chính đáng hay thiên kiến.

## 6. Các Bài báo Nền tảng

**["Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps" (2014)](https://arxiv.org/abs/1312.6034)**  
*Tác giả*: Karen Simonyan, Andrea Vedaldi, Andrew Zisserman  
Giới thiệu bản đồ saliency dựa trên gradient cho CNN, cho thấy đạo hàm tiết lộ pixel nào quan trọng cho dự đoán. Thiết lập trực quan hóa như công cụ diễn giải then chốt.

**["Visualizing and Understanding Convolutional Networks" (2014)](https://arxiv.org/abs/1311.2901)**  
*Tác giả*: Matthew Zeiler, Rob Fergus  
Mạng deconvolution trực quan hóa đặc trưng mà các tầng CNN học. Cho thấy tầng sớm phát hiện cạnh/màu, tầng giữa phát hiện kết cấu/mẫu, tầng sâu phát hiện bộ phận đối tượng. Nền tảng cho hiểu đặc trưng phân cấp của CNN.

**["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2017)](https://arxiv.org/abs/1610.02391)**  
*Tác giả*: Ramprasaath Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra  
Grad-CAM sinh giải thích trực quan bằng cách dùng gradient để cân trọng bản đồ đặc trưng, tạo định vị phân biệt lớp mà không sửa kiến trúc. Trở thành chuẩn cho diễn giải CNN.

**["A Unified Approach to Interpreting Model Predictions" (2017)](https://arxiv.org/abs/1705.07874)**  
*Tác giả*: Scott Lundberg, Su-In Lee  
SHAP thống nhất nhiều phương pháp gán công trạng dưới khung giá trị Shapley từ lý thuyết trò chơi, cung cấp giải thích có nền tảng lý thuyết với các tính chất mong muốn. Trở thành chuẩn cho dữ liệu dạng bảng và giải thích bất chấp mô hình.

**["Axiomatic Attribution for Deep Networks" (2017)](https://arxiv.org/abs/1703.01365)**  
*Tác giả*: Mukund Sundararajan, Ankur Taly, Qiqi Yan  
Integrated Gradients thỏa các tiên đề gán công trạng (độ nhạy, bất biến cài đặt) mà phương pháp dựa trên gradient vi phạm. Cung cấp phương pháp gán công trạng có nguyên tắc với đảm bảo lý thuyết.

## Bẫy Thường gặp và Mẹo

Bản đồ saliency có thể gây hiểu nhầm. Gradient cao không luôn nghĩa là tầm quan trọng cao — có thể là artifact của kiến trúc mạng (batch norm, ReLU) hoặc tối ưu. Luôn kiểm chứng diễn giải bằng ablation (thực sự loại bỏ đặc trưng và đo ảnh hưởng) hoặc kiểm tra liệu giải thích có khớp tri thức miền.

Ví dụ đối kháng không nhất thiết chỉ ra mô hình kém. Mọi mô hình đều có lỗ hổng đối kháng — đó là tính chất cơ bản của không gian chiều cao. Tập trung vào độ vững chắc với nhiễu tự nhiên (nhiễu, mờ) hơn là trường hợp xấu nhất được tạo đối kháng, trừ khi an ninh là then chốt.

## Điểm Chính Cần Nhớ

Diễn giải mạng neuron cho phép hiểu vì sao mô hình đưa ra dự đoán qua phương pháp gán công trạng (đầu vào nào quan trọng), kỹ thuật trực quan hóa (đặc trưng nào được học), và khung giải thích (quyết định phân rã ra sao). Saliency dựa trên gradient làm nổi bật vùng đầu vào có độ nhạy cao với thay đổi dự đoán. Grad-CAM trực quan hóa tầm quan trọng không gian trong CNN qua bản đồ đặc trưng cân trọng gradient. SHAP cung cấp gán công trạng có nền tảng lý thuyết qua giá trị Shapley từ lý thuyết trò chơi. Ví dụ đối kháng tiết lộ độ vững chắc mô hình bằng cách tìm nhiễu tối thiểu thay đổi dự đoán. Các phương pháp diễn giải khác nhau cung cấp góc nhìn bổ sung — saliency cho tầm quan trọng đầu vào, trực quan hóa kích hoạt cho đặc trưng học được, SHAP cho gán công trạng trung thực — thường đòi hỏi nhiều kỹ thuật cho hiểu biết toàn diện. Diễn giải cho phép gỡ lỗi tương quan giả, xây dựng niềm tin trong ứng dụng rủi ro cao, đáp ứng yêu cầu quy định, và sinh hiểu biết khoa học từ mẫu học được, ngày càng quan trọng khi triển khai học sâu mở rộng sang các miền then chốt đòi hỏi khả năng giải thích vượt ngoài độ chính xác.
