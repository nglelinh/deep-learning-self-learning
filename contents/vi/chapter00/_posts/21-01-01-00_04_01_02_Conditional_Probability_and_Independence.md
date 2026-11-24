---
layout: post
title: 00-04-01-02 Xác Suất Có Điều Kiện và Tính Độc Lập
chapter: '00'
order: 17
owner: AI Assistant
lang: vi
categories:
- chapter00
---

Nếu một xét nghiệm y khoa AI có độ chính xác 99% báo bạn dương tính với một căn bệnh hiếm gặp (chỉ 1 trên 1{,}000 người mắc), khả năng thực sự bạn bị bệnh có phải là 99% không? Câu trả lời gây sốc cho hầu hết mọi người là **chỉ khoảng 9%!** Nghịch lý nổi tiếng ấy cho thấy trực giác con người rất dễ sụp đổ khi đối mặt với **xác suất có điều kiện** và **định lý Bayes**. Trong Deep Learning, việc hiểu chính xác $$P(y \mid \mathbf{x})$$, giả định độc lập có điều kiện và suy luận Bayes giúp bạn thiết kế bài toán phân loại dữ liệu mất cân bằng đúng đắn ngay từ đầu.

## Bản đồ kiến thức nền: Bốn trụ cột của Xác suất Có điều kiện và Tính Độc lập

1. **Xác suất có điều kiện ($$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$)**:
   Thu hẹp không gian mẫu từ $$\Omega$$ xuống còn đúng vùng biến cố $$B$$ đã xảy ra—mọi mô hình phân biệt (*discriminative model*) đều trực tiếp học $$P(y \mid \mathbf{x})$$.

2. **Định lý Bayes (Bayes' Theorem)**:
   Đảo chiều điều kiện từ $$P(\mathbf{x} \mid y)$$ sang $$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y)P(y)}{P(\mathbf{x})}$$, kết nối xác suất tiên nghiệm (*prior*) với bằng chứng dữ liệu (*likelihood*) để tạo thành hậu nghiệm (*posterior*).

3. **Tính độc lập và Độc lập có điều kiện**:
   Hai biến cố độc lập thỏa $$P(A \cap B) = P(A)P(B)$$. Trong huấn luyện mạng nơ-ron, giả định các mẫu dữ liệu độc lập cho phép tách tích xác suất toàn tập thành tổng các log-loss riêng lẻ.

4. **Định lý Xác suất Toàn phần (Law of Total Probability)**:
   Tính xác suất biên $$P(A) = \sum_i P(A \mid B_i)P(B_i)$$ bằng cách cộng theo từng kịch bản—cốt lõi của cơ chế biên hóa biến ẩn (*latent variables*) trong mô hình sinh.

### Khung giải thích (nên đọc chậm): Giải tay nghịch lý chẩn đoán bệnh hiếm bằng Định lý Bayes

Xét một bộ sàng lọc AI cho căn bệnh hiếm có tỷ lệ mắc trong dân số $$P(D) = 0.001$$ (tức $$P(\neg D) = 0.999$$). Mô hình có độ nhạy $$P(+ \mid D) = 0.99$$ và tỷ lệ dương tính giả $$P(+ \mid \neg D) = 0.01$$:

1. **Tính xác suất ra kết quả dương tính (Xác suất toàn phần)**:
   $$P(+) = P(+ \mid D)P(D) + P(+ \mid \neg D)P(\neg D) = 0.99 \times 0.001 + 0.01 \times 0.999 = 0.00099 + 0.00999 = 0.01098.$$
2. **Áp dụng Định lý Bayes tìm xác suất thực sự mắc bệnh khi có kết quả dương tính**:
   $$P(D \mid +) = \frac{P(+ \mid D)P(D)}{P(+)} = \frac{0.00099}{0.01098} \approx 0.0902 \quad (9.02\%).$$
3. **Bài học thực chiến cho Deep Learning**: Dù độ nhạy và độ đặc hiệu đều đạt 99%, vì lớp âm áp đảo gấp 999 lần, số ca báo động giả ($$0.00999$$) vẫn gấp 10 lần số ca bệnh thật ($$0.00099$$)! Đó là lý do trên tập dữ liệu mất cân bằng, ta không bao giờ chỉ nhìn vào Accuracy mà phải theo dõi Precision, Recall và PR-AUC.

### 4. Xác suất có điều kiện

Xác suất để biến cố $$A$$ xảy ra khi biết biến cố $$B$$ đã xảy ra được định nghĩa là:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Ý nghĩa:** Xác suất có điều kiện cập nhật mức độ tin cậy của ta về biến cố $$A$$ sau khi có thêm thông tin rằng $$B$$ đã xảy ra.

<div id="conditional-prob-demo" style="border: 2px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f9f9f9;">
    <h4 style="text-align: center; color: #333;">Trực quan hóa xác suất có điều kiện</h4>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
        <div style="flex: 1; min-width: 400px;">
            <canvas id="conditionalCanvas" width="400" height="300" style="border: 1px solid #ccc; background: white;"></canvas>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                <strong>Biểu đồ Venn:</strong> Hình tròn xanh biểu diễn biến cố A, hình tròn đỏ biểu diễn biến cố B, phần giao màu tím biểu diễn A ∩ B.
            </p>
        </div>
        
        <div style="flex: 1; min-width: 250px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="margin-top: 0; color: #444;">Điều chỉnh xác suất</h5>
                
                <div style="margin-bottom: 15px;">
                    <label for="prob-a-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">P(A): <span id="prob-a-value">0.4</span></label>
                    <input type="range" id="prob-a-slider" min="0.1" max="0.9" step="0.05" value="0.4" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="prob-b-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">P(B): <span id="prob-b-value">0.5</span></label>
                    <input type="range" id="prob-b-slider" min="0.1" max="0.9" step="0.05" value="0.5" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="overlap-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">Phần giao: <span id="overlap-value">0.2</span></label>
                    <input type="range" id="overlap-slider" min="0" max="0.4" step="0.05" value="0.2" style="width: 100%;">
                </div>
                
                <div id="conditional-results" style="font-family: monospace; font-size: 12px; line-height: 1.4; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    <div><strong>Xác suất:</strong></div>
                    <div>P(A) = <span id="display-prob-a">0.400</span></div>
                    <div>P(B) = <span id="display-prob-b">0.500</span></div>
                    <div>P(A ∩ B) = <span id="display-prob-ab">0.200</span></div>
                    <div>P(A ∪ B) = <span id="display-prob-union">0.700</span></div>
                    <div><strong>Có điều kiện:</strong></div>
                    <div>P(A|B) = <span id="display-prob-a-given-b">0.400</span></div>
                    <div>P(B|A) = <span id="display-prob-b-given-a">0.500</span></div>
                </div>
            </div>
        </div>
    </div>
</div>

### 5. Tính độc lập

Hai biến cố $$A$$ và $$B$$ được gọi là **độc lập** nếu:
$$P(A \cap B) = P(A) \cdot P(B)$$

Tương đương (khi $$P(A) > 0, P(B) > 0$$):
$$P(A|B) = P(A) \quad \text{và} \quad P(B|A) = P(B)$$

**Ý nghĩa:** Việc biết một biến cố đã xảy ra không làm thay đổi xác suất của biến cố còn lại.

<script>
class ConditionalProbDemo {
    constructor() {
        this.canvas = document.getElementById('conditionalCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        this.probA = 0.4;
        this.probB = 0.5;
        this.overlap = 0.2;
        
        this.setupControls();
        this.draw();
    }
    
    setupControls() {
        const probASlider = document.getElementById('prob-a-slider');
        const probBSlider = document.getElementById('prob-b-slider');
        const overlapSlider = document.getElementById('overlap-slider');
        
        probASlider.addEventListener('input', (e) => {
            this.probA = parseFloat(e.target.value);
            document.getElementById('prob-a-value').textContent = this.probA.toFixed(1);
            this.updateCalculations();
            this.draw();
        });
        
        probBSlider.addEventListener('input', (e) => {
            this.probB = parseFloat(e.target.value);
            document.getElementById('prob-b-value').textContent = this.probB.toFixed(1);
            this.updateCalculations();
            this.draw();
        });
        
        overlapSlider.addEventListener('input', (e) => {
            this.overlap = parseFloat(e.target.value);
            document.getElementById('overlap-value').textContent = this.overlap.toFixed(1);
            // Ensure overlap doesn't exceed min(probA, probB)
            const maxOverlap = Math.min(this.probA, this.probB);
            if (this.overlap > maxOverlap) {
                this.overlap = maxOverlap;
                overlapSlider.value = this.overlap;
                document.getElementById('overlap-value').textContent = this.overlap.toFixed(1);
            }
            this.updateCalculations();
            this.draw();
        });
        
        this.updateCalculations();
    }
    
    updateCalculations() {
        const probUnion = this.probA + this.probB - this.overlap;
        const probAGivenB = this.probB > 0 ? this.overlap / this.probB : 0;
        const probBGivenA = this.probA > 0 ? this.overlap / this.probA : 0;
        
        document.getElementById('display-prob-a').textContent = this.probA.toFixed(3);
        document.getElementById('display-prob-b').textContent = this.probB.toFixed(3);
        document.getElementById('display-prob-ab').textContent = this.overlap.toFixed(3);
        document.getElementById('display-prob-union').textContent = probUnion.toFixed(3);
        document.getElementById('display-prob-a-given-b').textContent = probAGivenB.toFixed(3);
        document.getElementById('display-prob-b-given-a').textContent = probBGivenA.toFixed(3);
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // Draw universe rectangle
        this.ctx.strokeStyle = '#000';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(50, 50, 300, 200);
        this.ctx.fillStyle = '#000';
        this.ctx.font = '14px Arial';
        this.ctx.fillText('Ω (Không gian mẫu)', 55, 45);
        
        // Calculate circle parameters
        const centerAX = 150;
        const centerAY = 150;
        const centerBX = 250;
        const centerBY = 150;
        
        // Calculate radii based on probabilities (area proportional to probability)
        const radiusA = Math.sqrt(this.probA * 10000 / Math.PI);
        const radiusB = Math.sqrt(this.probB * 10000 / Math.PI);
        
        // Draw circle A
        this.ctx.globalAlpha = 0.3;
        this.ctx.fillStyle = '#2196f3';
        this.ctx.beginPath();
        this.ctx.arc(centerAX, centerAY, radiusA, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Draw circle B
        this.ctx.fillStyle = '#f44336';
        this.ctx.beginPath();
        this.ctx.arc(centerBX, centerBY, radiusB, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Draw intersection (approximate)
        if (this.overlap > 0) {
            this.ctx.fillStyle = '#9c27b0';
            const overlapRadius = Math.sqrt(this.overlap * 5000 / Math.PI);
            this.ctx.beginPath();
            this.ctx.arc((centerAX + centerBX) / 2, (centerAY + centerBY) / 2, overlapRadius, 0, 2 * Math.PI);
            this.ctx.fill();
        }
        
        this.ctx.globalAlpha = 1.0;
        
        // Draw circle outlines
        this.ctx.strokeStyle = '#2196f3';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(centerAX, centerAY, radiusA, 0, 2 * Math.PI);
        this.ctx.stroke();
        
        this.ctx.strokeStyle = '#f44336';
        this.ctx.beginPath();
        this.ctx.arc(centerBX, centerBY, radiusB, 0, 2 * Math.PI);
        this.ctx.stroke();
        
        // Labels
        this.ctx.fillStyle = '#000';
        this.ctx.font = '16px Arial';
        this.ctx.fillText('A', centerAX - 40, centerAY);
        this.ctx.fillText('B', centerBX + 30, centerBY);
        
        if (this.overlap > 0) {
            this.ctx.fillText('A∩B', (centerAX + centerBX) / 2 - 15, (centerAY + centerBY) / 2 + 5);
        }
    }
}

// MLE Deep Learning Demo

document.addEventListener('DOMContentLoaded', function() {
    new ConditionalProbDemo();
});
</script>

<style>
input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    height: 5px;
    background: #ddd;
    outline: none;
    border-radius: 5px;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 15px;
    height: 15px;
    background: #007bff;
    cursor: pointer;
    border-radius: 50%;
}

input[type="range"]::-moz-range-thumb {
    width: 15px;
    height: 15px;
    background: #007bff;
    cursor: pointer;
    border-radius: 50%;
    border: none;
}

canvas {
    border-radius: 5px;
}

.demo-container {
    margin: 20px 0;
}
</style>

## Video tham khảo

- **Xem trước:** [Conditional Probabilities, Clearly Explained!!!](https://www.youtube.com/watch?v=_IgyaD7vOOA) — StatQuest with Josh Starmer, tiếng Anh. Trình bày rõ định nghĩa $$P(A|B)$$ — khái niệm mở đầu của bài.
- [Independence in Probability](https://www.youtube.com/watch?v=2nDYX8IKrwo) — Steve Brunton, tiếng Anh. Giải thích tính độc lập giữa hai biến cố khi xác suất có điều kiện trùng với xác suất biên.
- [Xác suất có điều kiện (Conditional probability) | Công thức nhân xác suất](https://www.youtube.com/watch?v=EsIW9oLsqiI) — TOÁN HỌC MUÔN NƠI, tiếng Việt. Bài giảng tiếng Việt trình bày xác suất có điều kiện cùng công thức nhân xác suất.
