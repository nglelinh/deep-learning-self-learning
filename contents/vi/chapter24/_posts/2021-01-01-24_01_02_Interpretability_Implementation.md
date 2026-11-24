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

Muốn nhìn thấy mạng ResNet-50 đang tập trung vào đâu trên bức ảnh mà không phải viết lại hay sửa đổi kiến trúc mạng, vũ khí bí mật của PyTorch chính là cơ chế **Hooks** (`register_forward_hook` và `register_full_backward_hook`). Trong bài thực hành này, bạn sẽ tự tay cài đặt lớp `GradCAM`, cài đặt bản đồ Saliency, Integrated Gradients và kết hợp với thư viện giải thích để soi sáng từng dự đoán của mạng CNN.

## Bản đồ kiến thức nền: Kỹ thuật lập trình Grad-CAM và Integrated Gradients trong PyTorch

- **Cơ chế Forward & Backward Hooks:** Bắt giữ trọn vẹn bản đồ đặc trưng $$A^k$$ (lượt xuôi) và gradient $$\partial y^c / \partial A^k$$ (lượt ngược) tại tầng tích chập cuối (`layer4[-1]`) mà không sửa mã nguồn mô hình.
- **Tính trọng số kênh và phóng to bản đồ nhiệt:** Lấy `mean(dim=(2, 3))` trên gradient, nhân tổ hợp tuyến tính với feature map, áp dụng `F.relu(...)` và nội suy song tuyến (`bilinear`) về đúng kích thước ảnh gốc.
- **Riemann Sum cho Integrated Gradients:** Chia đoạn thẳng từ ảnh đen $$\mathbf{x}'$$ tới ảnh gốc $$\mathbf{x}$$ thành $$m = 50$$ bước rời rạc để xấp xỉ tích phân mượt và kiểm tra sai số hội tụ (completeness delta).
- **Quản lý vòng đời Hook & `zero_grad()`:** Luôn gỡ hook (`handle.remove()`) sau khi dùng xong để tránh rò rỉ bộ nhớ GPU và nhớ xóa gradient trước mỗi lớp mục tiêu.

### Khung giải thích (nên đọc chậm): Tính tay một bản đồ nhiệt Grad-CAM kích thước $$2 \times 2$$ từ 2 kênh đặc trưng

Giả sử tầng tích chập cuối có 2 kênh đặc trưng kích thước $$2 \times 2$$:
$$A^1 = \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix}, \quad A^2 = \begin{pmatrix} 1 & 4 \\ 0 & 1 \end{pmatrix}$$
và gradient của điểm lớp mục tiêu $$y^c$$ theo hai kênh này là:
$$\frac{\partial y^c}{\partial A^1} = \begin{pmatrix} 0{,}6 & 0{,}2 \\ 0{,}4 & 0{,}8 \end{pmatrix}, \quad \frac{\partial y^c}{\partial A^2} = \begin{pmatrix} -0{,}2 & -0{,}6 \\ -0{,}4 & 0{,}0 \end{pmatrix}.$$

1. **Tính trọng số tầm quan trọng từng kênh (Global Average Pooling trên gradient):**
   - $$\alpha_1 = \frac{0{,}6 + 0{,}2 + 0{,}4 + 0{,}8}{4} = +0{,}50$$.
   - $$\alpha_2 = \frac{-0{,}2 - 0{,}6 - 0{,}4 + 0{,}0}{4} = -0{,}30$$.
2. **Tổ hợp tuyến tính $$L = \alpha_1 A^1 + \alpha_2 A^2$$:**
   - Góc trên-trái: $$0{,}5(2) - 0{,}3(1) = +0{,}70$$
   - Góc trên-phải: $$0{,}5(0) - 0{,}3(4) = -1{,}20$$
   - Góc dưới-trái: $$0{,}5(1) - 0{,}3(0) = +0{,}50$$
   - Góc dưới-phải: $$0{,}5(3) - 0{,}3(1) = +1{,}20$$
3. **Áp dụng ReLU** (loại bỏ bằng chứng chống lại lớp $$c$$) và chuẩn hóa về $$[0, 1]$$:
   $$\text{Grad-CAM} = \begin{pmatrix} 0{,}70 & 0 \\ 0{,}50 & 1{,}20 \end{pmatrix} \xrightarrow{\div 1{,}20} \begin{pmatrix} 0{,}58 & 0{,}00 \\ 0{,}42 & 1{,}00 \end{pmatrix}.$$
   Vùng nóng nhất nằm ở góc dưới-phải ($$1{,}00$$), còn góc trên-phải bị cắt về $$0$$!

## 4. Mã minh họa

Ví dụ cài đặt Grad-CAM và xấp xỉ SHAP đơn giản bằng PyTorch:

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

Ví dụ tạo mẫu đối kháng bằng FGSM để kiểm tra độ vững chắc của mô hình:

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
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
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

Diễn giải mô hình liên hệ gần gũi với suy luận nhân quả (*causal inference*). Các phương pháp quy kết thông thường chỉ phản ánh mối tương quan giữa đầu vào và đầu ra, chưa chứng minh quan hệ nhân quả. Diễn giải nhân quả đặt câu hỏi sâu hơn: “nếu can thiệp thay đổi đặc trưng này thì dự đoán có thực sự thay đổi không?”, đòi hỏi can thiệp và phân tích phản thực.

Diễn giải và định lượng độ bất định (*uncertainty quantification*) bổ trợ trực tiếp cho nhau: diễn giải cho biết *vì sao* mô hình dự đoán như vậy, còn định lượng bất định cho biết mô hình *chắc chắn đến mức nào*. Kết hợp cả hai (qua học sâu Bayes, *MC dropout* hoặc *ensemble*) mang lại câu trả lời đầy đủ hơn cho người sử dụng.

Diễn giải cũng là công cụ hỗ trợ kiểm toán tính công bằng và phát hiện thiên kiến. Nếu một mô hình tuyển dụng phụ thuộc trực tiếp hoặc gián tiếp (qua biến đại diện) vào giới tính hay chủng tộc, các phương pháp quy kết sẽ giúp bộc lộ điều đó. Tuy nhiên, diễn giải chỉ là bước phát hiện; việc đánh giá đặc trưng nào hợp lệ hay gây thiên kiến vẫn cần tiêu chuẩn miền và quy định cụ thể.

## 6. Các bài báo nền tảng

**["Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps" (2014)](https://arxiv.org/abs/1312.6034)**  
*Tác giả*: Karen Simonyan, Andrea Vedaldi, Andrew Zisserman  
Giới thiệu bản đồ *saliency* dựa trên gradient cho CNN, đặt nền móng cho nhóm kỹ thuật trực quan hóa tầm quan trọng của từng pixel.

**["Visualizing and Understanding Convolutional Networks" (2014)](https://arxiv.org/abs/1311.2901)**  
*Tác giả*: Matthew Zeiler, Rob Fergus  
Sử dụng mạng *deconvolution* để trực quan hóa đặc trưng qua các tầng CNN, chỉ ra cấu trúc phân cấp từ cạnh/màu ở tầng đầu, kết cấu ở tầng giữa đến bộ phận đối tượng ở tầng sâu.

**["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2017)](https://arxiv.org/abs/1610.02391)**  
*Tác giả*: Ramprasaath Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra  
Grad-CAM dùng gradient để gán trọng số cho các bản đồ đặc trưng, tạo bản đồ định vị theo lớp mà không cần sửa đổi kiến trúc CNN, trở thành công cụ phổ biến cho diễn giải ảnh.

**["A Unified Approach to Interpreting Model Predictions" (2017)](https://arxiv.org/abs/1705.07874)**  
*Tác giả*: Scott Lundberg, Su-In Lee  
SHAP thống nhất nhiều phương pháp quy kết dưới khung giá trị Shapley của lý thuyết trò chơi, được dùng rộng rãi cho dữ liệu dạng bảng và các mô hình tổng quát.

**["Axiomatic Attribution for Deep Networks" (2017)](https://arxiv.org/abs/1703.01365)**  
*Tác giả*: Mukund Sundararajan, Ankur Taly, Qiqi Yan  
Đề xuất Integrated Gradients đáp ứng hai tiên đề quan trọng là tính nhạy và bất biến cài đặt mà gradient thuần túy vi phạm.

## Bẫy thường gặp và mẹo thực hành

Bản đồ *saliency* đôi khi gây hiểu nhầm: gradient lớn không phải lúc nào cũng phản ánh đặc trưng quan trọng về mặt ngữ nghĩa mà có thể do đặc thù kiến trúc (như ReLU hay chuẩn hóa batch). Cần kiểm chứng lại bằng thử nghiệm loại bỏ đặc trưng (*ablation*) và đối chiếu với kiến thức chuyên ngành.

Sự tồn tại của ví dụ đối kháng không đồng nghĩa mô hình hoàn toàn kém; hầu hết mô hình học sâu trong không gian nhiều chiều đều nhạy cảm với nhiễu đối kháng thiết kế riêng. Trừ các bài toán bảo mật đặc thù, cần ưu tiên đánh giá độ vững chắc trước các dạng nhiễu tự nhiên (như mờ ảnh, nhiễu cảm biến).

## Điểm chính cần nhớ

Diễn giải mô hình giúp hiểu cơ chế ra quyết định thông qua quy kết đặc trưng (đầu vào nào quan trọng), trực quan hóa kích hoạt (đặc trưng nào được học) và phân tích độ vững chắc. *Gradient saliency* và Grad-CAM làm nổi bật vùng không gian quan trọng trong CNN, SHAP cung cấp phương pháp quy kết có nền tảng lý thuyết trò chơi, còn ví dụ đối kháng thăm dò độ bền của biên quyết định. Vì mỗi kỹ thuật chỉ phản ánh một khía cạnh, việc kết hợp nhiều phương pháp giúp gỡ lỗi tương quan giả, kiểm toán thiên kiến và tăng độ tin cậy khi triển khai học sâu trong các lĩnh vực rủi ro cao.

## Video tham khảo

- **Xem trước** [Explainable AI explained! | #3 LIME](https://www.youtube.com/watch?v=d6j6bofhj2M) — DeepFindr, tiếng Anh. Minh họa cách cài đặt một phương pháp giải thích cục bộ bằng mô hình có trọng số.
- [SHAP values for beginners | What they mean and their applications](https://www.youtube.com/watch?v=MQ6fFDwjuco) — A Data Odyssey, tiếng Anh. Giúp hiểu rõ đại lượng mà thư viện SHAP tính toán phía sau giao diện gọi hàm.
- [Lecture 12 | Visualizing and Understanding](https://www.youtube.com/watch?v=6wcs6szJWMY) — Stanford University School of Engineering, tiếng Anh. Phân tích các bẫy phổ biến khi trực quan hóa (những hình ảnh trông hợp lý nhưng chưa chắc phản ánh đúng cơ chế của mô hình).
