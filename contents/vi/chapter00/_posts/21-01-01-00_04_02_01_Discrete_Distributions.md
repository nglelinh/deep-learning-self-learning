---
layout: post
title: 00-04-02-01 Phân Phối Rời Rạc
chapter: '00'
order: 20
owner: AI Assistant
lang: vi
categories:
- chapter00
---

Từ một lần tung đồng xu xác định bật hay tắt một nơ-ron trong Dropout, cho tới việc một mô hình ngôn ngữ chọn ra 1 từ trong kho 50{,}000 từ vựng, mọi quyết định rời rạc trong Học sâu đều được điều khiển bởi các **phân phối rời rạc**. Nếu không phân biệt rõ Bernoulli, Nhị thức (Binomial), Categorical và Poisson, ta rất dễ dùng nhầm hàm mất mát hoặc hiểu sai bản chất thống kê của độ chính xác phân loại. Bài học này đi sâu vào công thức, kỳ vọng, phương sai và dấu ấn thực tế của từng phân phối rời rạc trong Deep Learning.

## Kiến thức cần nhớ: Bốn phân phối Rời rạc trụ cột

1. **Phân phối Bernoulli ($$p$$)**:
   Mô hình hóa một phép thử Đúng/Sai duy nhất với $$P(X=1)=p$$, $$\mathbb{E}[X]=p$$, $$\text{Var}(X)=p(1-p)$$. Đây là nền tảng của phân loại nhị phân và từng phần tử trong mặt nạ Dropout.

2. **Phân phối Nhị thức — Binomial ($$n, p$$)**:
   Đếm số lần thành công trong $$n$$ phép thử Bernoulli độc lập, với $$\mathbb{E}[X]=np$$ và $$\text{Var}(X)=np(1-p)$$. Dùng để đánh giá độ tin cậy thống kê khi kiểm thử mô hình trên $$n$$ mẫu test.

3. **Phân phối Categorical / Multinomial**:
   Mở rộng Bernoulli từ 2 lớp lên $$K$$ lớp với vector xác suất $$\sum_{k=1}^K p_k = 1$$—đầu ra chuẩn của mọi bài toán phân loại đa lớp và dự đoán token kế tiếp trong Transformer.

4. **Phân phối Poisson ($$\lambda$$)**:
   Mô hình hóa số lần sự kiện xảy ra trong một khoảng thời gian/không gian cố định (ví dụ số lượt nhấp chuột, số lượng phương tiện qua ngã tư mỗi phút), với đặc trưng $$\mathbb{E}[X] = \text{Var}(X) = \lambda$$.

### Giải thích: Tính tay xác suất và phương sai cho Dropout (Bernoulli) và Test Set (Binomial)

**1. Phân tích một lớp Dropout với xác suất giữ lại $$p = 0.8$$ (tỷ lệ tắt 20%):**
- Mỗi nơ-ron nhân với biến $$M \sim \text{Bernoulli}(0.8)$$: $$P(M=1)=0.8, P(M=0)=0.2$$.
- Kỳ vọng: $$\mathbb{E}[M] = 0.8$$; phương sai: $$\text{Var}(M) = 0.8 \times 0.2 = 0.16$$.
- Để kỳ vọng đầu ra khi huấn luyện khớp với lúc suy luận (không tắt nơ-ron), ta chia cho $$0.8$$ (nhân $$1.25$$): $$\mathbb{E}[M / 0.8] = 0.8 / 0.8 = 1.0$$.

**2. Tính tay xác suất Binomial khi chấm điểm 5 mẫu kiểm thử:**
Giả sử mô hình có độ chính xác thật $$p = 0.8$$ trên mỗi mẫu độc lập. Xác suất mô hình đoán đúng **ít nhất 4 trên 5 mẫu** là:
- Đúng đúng 4 mẫu: $$P(X=4) = \binom{5}{4}(0.8)^4(0.2)^1 = 5 \times 0.4096 \times 0.2 = 0.4096$$.
- Đúng cả 5 mẫu: $$P(X=5) = \binom{5}{5}(0.8)^5(0.2)^0 = 1 \times 0.32768 = 0.32768$$.
- Cộng lại: $$P(X \ge 4) = 0.4096 + 0.32768 = 0.73728 \approx 73.73\%$$.
Dù độ chính xác thật là 80%, vẫn có hơn 26% khả năng trên 5 mẫu ngẫu nhiên mô hình chỉ đạt $$\le 60\%$$—nhắc ta luôn cần tập kiểm thử đủ lớn để giảm phương sai đánh giá!

## Các phân phối xác suất thông dụng

Nắm vững các phân phối xác suất cơ bản là yêu cầu thiết yếu cho các bài toán tối ưu hóa trong học máy và thống kê. Các phân phối này thường xuất hiện dưới dạng giả định mô hình dữ liệu, phân phối tiên nghiệm (*prior*) trong phương pháp Bayes và mô hình nhiễu trong hồi quy.

### 1. Phân phối rời rạc

#### Phân phối Bernoulli

Mô hình hóa một phép thử đơn lẻ chỉ có hai kết quả (thành công hoặc thất bại).

**Tham số:** $$p \in [0,1]$$ (xác suất thành công)

**PMF:** $$P(X = k) = p^k (1-p)^{1-k}$$ với $$k \in \{0,1\}$$

**Kỳ vọng:** $$\mathbb{E}[X] = p$$

**Phương sai:** $$\text{Var}(X) = p(1-p)$$

**Ứng dụng:** phân loại nhị phân, tung đồng xu, kiểm định A/B.

#### Phân phối nhị thức (*Binomial*)

Mô hình hóa số lần thành công trong $$n$$ phép thử Bernoulli độc lập.

**Tham số:** $$n \in \mathbb{N}$$ (số phép thử), $$p \in [0,1]$$ (xác suất thành công ở mỗi phép thử)

**PMF:** $$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$ với $$k = 0,1,\ldots,n$$

**Kỳ vọng:** $$\mathbb{E}[X] = np$$

**Phương sai:** $$\text{Var}(X) = np(1-p)$$

#### Phân phối Poisson

Mô hình hóa số sự kiện xảy ra trong một khoảng thời gian (hoặc không gian) cố định khi các sự kiện xuất hiện độc lập với tốc độ trung bình không đổi.

**Tham số:** $$\lambda > 0$$ (tốc độ xuất hiện trung bình)

**PMF:** $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$ với $$k = 0,1,2,\ldots$$

**Kỳ vọng:** $$\mathbb{E}[X] = \lambda$$

**Phương sai:** $$\text{Var}(X) = \lambda$$

**Ứng dụng:** dữ liệu đếm, sự kiện hiếm, lý thuyết hàng đợi.

<div id="discrete-distributions-demo" style="border: 2px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f9f9f9;">
    <h4 style="text-align: center; color: #333;">Phân phối rời rạc tương tác</h4>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
        <div style="flex: 1; min-width: 400px;">
            <canvas id="discreteCanvas" width="400" height="300" style="border: 1px solid #ccc; background: white;"></canvas>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                <strong>Trực quan hóa:</strong> Hàm khối xác suất (PMF) của các phân phối rời rạc.
            </p>
        </div>
        
        <div style="flex: 1; min-width: 250px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="margin-top: 0; color: #444;">Loại phân phối</h5>
                
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="discrete-dist" value="bernoulli" checked> Bernoulli
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="discrete-dist" value="binomial"> Nhị thức
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="discrete-dist" value="poisson"> Poisson
                    </label>
                </div>
                
                <div id="discrete-params">
                    <div id="p-param" style="margin-bottom: 15px;">
                        <label for="p-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">p: <span id="p-value">0.5</span></label>
                        <input type="range" id="p-slider" min="0.1" max="0.9" step="0.05" value="0.5" style="width: 100%;">
                    </div>
                    
                    <div id="n-param" style="margin-bottom: 15px; display: none;">
                        <label for="n-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">n: <span id="n-value">10</span></label>
                        <input type="range" id="n-slider" min="5" max="50" step="1" value="10" style="width: 100%;">
                    </div>
                    
                    <div id="lambda-param" style="margin-bottom: 15px; display: none;">
                        <label for="lambda-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">λ: <span id="lambda-value">3.0</span></label>
                        <input type="range" id="lambda-slider" min="0.5" max="10" step="0.5" value="3.0" style="width: 100%;">
                    </div>
                </div>
                
                <div id="discrete-stats" style="font-family: monospace; font-size: 12px; line-height: 1.4; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    <div><strong>Thống kê:</strong></div>
                    <div>Kỳ vọng: <span id="discrete-mean">0.500</span></div>
                    <div>Phương sai: <span id="discrete-variance">0.250</span></div>
                    <div>Mode: <span id="discrete-mode">0 hoặc 1</span></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
class DiscreteDistributionsDemo {
    constructor() {
        this.canvas = document.getElementById('discreteCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        this.distType = 'bernoulli';
        this.params = { p: 0.5, n: 10, lambda: 3.0 };
        
        this.setupControls();
        this.draw();
    }
    
    setupControls() {
        const radios = document.querySelectorAll('input[name="discrete-dist"]');
        const pSlider = document.getElementById('p-slider');
        const nSlider = document.getElementById('n-slider');
        const lambdaSlider = document.getElementById('lambda-slider');
        
        radios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.distType = e.target.value;
                this.updateParameterVisibility();
                this.updateStats();
                this.draw();
            });
        });
        
        pSlider.addEventListener('input', (e) => {
            this.params.p = parseFloat(e.target.value);
            document.getElementById('p-value').textContent = this.params.p.toFixed(1);
            this.updateStats();
            this.draw();
        });
        
        nSlider.addEventListener('input', (e) => {
            this.params.n = parseInt(e.target.value);
            document.getElementById('n-value').textContent = this.params.n;
            this.updateStats();
            this.draw();
        });
        
        lambdaSlider.addEventListener('input', (e) => {
            this.params.lambda = parseFloat(e.target.value);
            document.getElementById('lambda-value').textContent = this.params.lambda.toFixed(1);
            this.updateStats();
            this.draw();
        });
        
        this.updateParameterVisibility();
        this.updateStats();
    }
    
    updateParameterVisibility() {
        document.getElementById('p-param').style.display = 
            (this.distType === 'bernoulli' || this.distType === 'binomial') ? 'block' : 'none';
        document.getElementById('n-param').style.display = 
            this.distType === 'binomial' ? 'block' : 'none';
        document.getElementById('lambda-param').style.display = 
            this.distType === 'poisson' ? 'block' : 'none';
    }
    
    updateStats() {
        let mean, variance, mode;
        
        switch(this.distType) {
            case 'bernoulli':
                mean = this.params.p;
                variance = this.params.p * (1 - this.params.p);
                mode = this.params.p > 0.5 ? '1' : (this.params.p < 0.5 ? '0' : '0 hoặc 1');
                break;
            case 'binomial':
                mean = this.params.n * this.params.p;
                variance = this.params.n * this.params.p * (1 - this.params.p);
                mode = Math.floor((this.params.n + 1) * this.params.p).toString();
                break;
            case 'poisson':
                mean = this.params.lambda;
                variance = this.params.lambda;
                mode = Math.floor(this.params.lambda).toString();
                break;
        }
        
        document.getElementById('discrete-mean').textContent = mean.toFixed(3);
        document.getElementById('discrete-variance').textContent = variance.toFixed(3);
        document.getElementById('discrete-mode').textContent = mode;
    }
    
    factorial(n) {
        if (n <= 1) return 1;
        return n * this.factorial(n - 1);
    }
    
    binomialCoeff(n, k) {
        if (k > n) return 0;
        return this.factorial(n) / (this.factorial(k) * this.factorial(n - k));
    }
    
    getProbability(k) {
        switch(this.distType) {
            case 'bernoulli':
                return k === 0 ? (1 - this.params.p) : (k === 1 ? this.params.p : 0);
            case 'binomial':
                if (k < 0 || k > this.params.n) return 0;
                return this.binomialCoeff(this.params.n, k) * 
                       Math.pow(this.params.p, k) * 
                       Math.pow(1 - this.params.p, this.params.n - k);
            case 'poisson':
                if (k < 0) return 0;
                return Math.pow(this.params.lambda, k) * Math.exp(-this.params.lambda) / this.factorial(k);
        }
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        const marginX = 50;
        const marginY = 50;
        const plotWidth = this.width - 2 * marginX;
        const plotHeight = this.height - 2 * marginY;
        
        // Draw axes
        this.ctx.strokeStyle = '#ddd';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.moveTo(marginX, this.height - marginY);
        this.ctx.lineTo(this.width - marginX, this.height - marginY);
        this.ctx.moveTo(marginX, marginY);
        this.ctx.lineTo(marginX, this.height - marginY);
        this.ctx.stroke();
        
        // Determine range
        let maxK;
        switch(this.distType) {
            case 'bernoulli': maxK = 1; break;
            case 'binomial': maxK = this.params.n; break;
            case 'poisson': maxK = Math.min(20, this.params.lambda + 3 * Math.sqrt(this.params.lambda)); break;
        }
        
        // Find max probability for scaling
        let maxProb = 0;
        for (let k = 0; k <= maxK; k++) {
            maxProb = Math.max(maxProb, this.getProbability(k));
        }
        
        // Draw bars
        this.ctx.fillStyle = '#2196f3';
        const barWidth = plotWidth / (maxK + 2);
        
        for (let k = 0; k <= maxK; k++) {
            const prob = this.getProbability(k);
            const x = marginX + (k + 0.5) * barWidth;
            const height = (prob / maxProb) * plotHeight * 0.8;
            const y = this.height - marginY - height;
            
            this.ctx.fillRect(x - barWidth * 0.3, y, barWidth * 0.6, height);
            
            // Label
            this.ctx.fillStyle = '#000';
            this.ctx.font = '10px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(k.toString(), x, this.height - marginY + 15);
            this.ctx.fillText(prob.toFixed(3), x, y - 5);
            this.ctx.fillStyle = '#2196f3';
        }
        
        // Labels
        this.ctx.fillStyle = '#000';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('k', this.width / 2, this.height - 10);
        
        this.ctx.save();
        this.ctx.translate(15, this.height / 2);
        this.ctx.rotate(-Math.PI / 2);
        this.ctx.fillText('P(X = k)', 0, 0);
        this.ctx.restore();
    }
}

// Continuous Distributions Demo

document.addEventListener('DOMContentLoaded', function() {
    new DiscreteDistributionsDemo();
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

- **Xem trước:** [The Binomial Distribution and Test, Clearly Explained!!!](https://www.youtube.com/watch?v=J8jNoF-K8E8) — StatQuest with Josh Starmer, tiếng Anh. Giải thích rõ phép thử Bernoulli và phân phối nhị thức — hai mô hình rời rạc trọng tâm của bài.
- [Binomial distributions | Probabilities of probabilities, part 1](https://www.youtube.com/watch?v=8idr1WZ1A7Q) — 3Blue1Brown, tiếng Anh. Trình bày góc nhìn trực quan về phân phối nhị thức khi số lần thành công là biến ngẫu nhiên.
- [An Introduction to the Poisson Distribution](https://www.youtube.com/watch?v=jmqZG6roVqU) — jbstatistics, tiếng Anh. Giới thiệu phân phối Poisson, họ phân phối rời rạc còn lại được khảo sát trong bài.
