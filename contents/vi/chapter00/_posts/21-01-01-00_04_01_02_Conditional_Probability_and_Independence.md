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

### 4. Xác Suất Có Điều Kiện

Xác suất của biến cố A khi biết rằng biến cố B đã xảy ra:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Diễn giải**: Xác suất có điều kiện cập nhật niềm tin của chúng ta về A khi có thông tin về B.

<div id="conditional-prob-demo" style="border: 2px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f9f9f9;">
    <h4 style="text-align: center; color: #333;">Trực Quan Hóa Xác Suất Có Điều Kiện</h4>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
        <div style="flex: 1; min-width: 400px;">
            <canvas id="conditionalCanvas" width="400" height="300" style="border: 1px solid #ccc; background: white;"></canvas>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                <strong>Biểu Đồ Venn:</strong> Hình tròn xanh là biến cố A, hình tròn đỏ là biến cố B. Giao màu tím thể hiện A ∩ B.
            </p>
        </div>
        
        <div style="flex: 1; min-width: 250px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="margin-top: 0; color: #444;">Điều Chỉnh Xác Suất</h5>
                
                <div style="margin-bottom: 15px;">
                    <label for="prob-a-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">P(A): <span id="prob-a-value">0.4</span></label>
                    <input type="range" id="prob-a-slider" min="0.1" max="0.9" step="0.05" value="0.4" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="prob-b-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">P(B): <span id="prob-b-value">0.5</span></label>
                    <input type="range" id="prob-b-slider" min="0.1" max="0.9" step="0.05" value="0.5" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="overlap-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">Giao: <span id="overlap-value">0.2</span></label>
                    <input type="range" id="overlap-slider" min="0" max="0.4" step="0.05" value="0.2" style="width: 100%;">
                </div>
                
                <div id="conditional-results" style="font-family: monospace; font-size: 12px; line-height: 1.4; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    <div><strong>Xác Suất:</strong></div>
                    <div>P(A) = <span id="display-prob-a">0.400</span></div>
                    <div>P(B) = <span id="display-prob-b">0.500</span></div>
                    <div>P(A ∩ B) = <span id="display-prob-ab">0.200</span></div>
                    <div>P(A ∪ B) = <span id="display-prob-union">0.700</span></div>
                    <div><strong>Có Điều Kiện:</strong></div>
                    <div>P(A|B) = <span id="display-prob-a-given-b">0.400</span></div>
                    <div>P(B|A) = <span id="display-prob-b-given-a">0.500</span></div>
                </div>
            </div>
        </div>
    </div>
</div>

### 5. Tính Độc Lập

Hai biến cố A và B **độc lập** nếu:
$$P(A \cap B) = P(A) \cdot P(B)$$

Tương đương:
$$P(A|B) = P(A) \quad \text{và} \quad P(B|A) = P(B)$$

**Diễn giải**: Kiến thức về một biến cố không thay đổi xác suất của biến cố kia.

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
