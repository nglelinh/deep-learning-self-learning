---
layout: post
title: 00-04-02-02 Phân Phối Liên Tục
chapter: '00'
order: 21
owner: AI Assistant
lang: vi
categories:
- chapter00
---

Tại sao trọng số khởi tạo của hầu hết mạng nơ-ron lại lấy từ phân phối Chuẩn (Gaussian) hoặc Đều (Uniform), và vì sao chỉ một lỗi nhỏ khi chuẩn hóa dữ liệu đầu vào cũng có thể đẩy toàn bộ hàm Sigmoid vào vùng bão hòa? Câu trả lời nằm ở hình học của các **phân phối liên tục**—nơi xác suất không tập trung tại từng điểm rời rạc mà trải đều thành diện tích dưới đường cong mật độ $$p(x)$$. Bài học này giúp bạn nắm vững ba phân phối liên tục quan trọng nhất (Đều, Chuẩn, Mũ) cùng quy tắc chuẩn hóa Z-score áp dụng trực tiếp vào tiền xử lý dữ liệu và Batch Normalization.

## Kiến thức cần nhớ: Bốn trụ cột của Phân phối Liên tục

1. **Hàm mật độ (PDF $$p(x)$$) và Hàm phân phối tích lũy (CDF $$F(x)$$)**:
   Với biến liên tục, xác suất tại đúng một điểm luôn bằng $$0$$ ($$P(X = c) = 0$$); xác suất chỉ có nghĩa trên một khoảng: $$P(a \le X \le b) = \int_a^b p(x)\,dx = F(b) - F(a)$$.

2. **Phân phối Đều (Uniform $$\mathcal{U}(a, b)$$)**:
   Mọi giá trị trong $$[a, b]$$ có mật độ ngang nhau $$p(x) = \frac{1}{b-a}$$, kỳ vọng $$\frac{a+b}{2}$$, phương sai $$\frac{(b-a)^2}{12}$$. Đây là cơ sở của khởi tạo `Kaiming Uniform` mặc định trong `nn.Linear` của PyTorch!

3. **Phân phối Chuẩn (Gaussian $$\mathcal{N}(\mu, \sigma^2)$$)**:
   Hình chuông đối xứng đặc trưng bởi quy tắc 68–95–99.7%, định lý giới hạn trung tâm (tổng nhiều nguồn nhiễu nhỏ hội tụ về Gaussian), và phép chuẩn hóa $$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$.

4. **Phân phối Mũ (Exponential $$\text{Exp}(\lambda)$$)**:
   Mô hình hóa thời gian chờ giữa các sự kiện với tính chất "không nhớ" (*memoryless*), đồng thời xuất hiện trong phân phối Laplace ($$L_1$$ regularization) và phân tích sinh tồn.

### Giải thích: Tính tay biên khởi tạo `Uniform` tương đương phương sai của `Normal`

Trong PyTorch, khi khởi tạo một lớp tuyến tính có $$n_{\text{in}} = 100$$ đầu vào theo chuẩn Xavier, ta muốn trọng số có $$\mathbb{E}[w] = 0$$ và phương sai $$\text{Var}(w) = \frac{1}{100} = 0.01$$:

1. **Nếu dùng phân phối Chuẩn $$\mathcal{N}(0, \sigma^2)$$**:
   Chỉ cần đặt độ lệch chuẩn $$\sigma = \sqrt{0.01} = 0.1$$. Theo quy tắc 68–95–99.7%, khoảng 95% trọng số sẽ nằm trong $$[-0.2, 0.2]$$.
2. **Nếu dùng phân phối Đều đối xứng $$\mathcal{U}(-a, a)$$**:
   Ta biết phương sai của $$\mathcal{U}(-a, a)$$ là $$\text{Var}(w) = \frac{(a - (-a))^2}{12} = \frac{4a^2}{12} = \frac{a^2}{3}$$.
   Để $$\frac{a^2}{3} = 0.01$$, ta giải được:
   $$a^2 = 0.03 \implies a = \sqrt{0.03} \approx 0.1732.$$
3. **Kiểm tra ngược**: Với $$w \sim \mathcal{U}(-0.1732, 0.1732)$$, mật độ $$p(w) = \frac{1}{0.3464} \approx 2.887 > 1$$ (hoàn toàn hợp lệ cho PDF!) và diện tích $$2.887 \times 0.3464 = 1$$. Bạn vừa tự chứng minh xong con số $$\sqrt{3}$$ nổi tiếng trong mã nguồn khởi tạo trọng số của PyTorch!

### 2. Phân phối liên tục

#### Phân phối đều (*Uniform*)

Mọi giá trị trong một đoạn cho trước đều có mật độ xác suất như nhau.

**Tham số:** $$a, b \in \mathbb{R}$$ với $$a < b$$

**PDF:** $$f(x) = \frac{1}{b-a}$$ với $$x \in [a,b]$$, và bằng $$0$$ ngoài đoạn đó.

**Kỳ vọng:** $$\mathbb{E}[X] = \frac{a+b}{2}$$

**Phương sai:** $$\text{Var}(X) = \frac{(b-a)^2}{12}$$

**Ứng dụng:** lấy mẫu ngẫu nhiên, khởi tạo tham số trong thuật toán.

#### Phân phối chuẩn (*Normal / Gaussian*)

Đây là phân phối quan trọng và phổ biến nhất trong thống kê cũng như tối ưu hóa.

**Tham số:** $$\mu \in \mathbb{R}$$ (kỳ vọng), $$\sigma^2 > 0$$ (phương sai)

**PDF:** $$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Kỳ vọng:** $$\mathbb{E}[X] = \mu$$

**Phương sai:** $$\text{Var}(X) = \sigma^2$$

**Tính chất:**
- Đối xứng quanh kỳ vọng $$\mu$$.
- Tuân theo quy tắc 68–95–99,7.
- Gắn trực tiếp với Định lý giới hạn trung tâm (*Central Limit Theorem*).
- Có entropy cực đại trong số mọi phân phối có cùng kỳ vọng và phương sai cho trước.

#### Phân phối mũ (*Exponential*)

Mô hình hóa thời gian chờ giữa hai sự kiện liên tiếp trong một quá trình Poisson.

**Tham số:** $$\lambda > 0$$ (tốc độ xuất hiện)

**PDF:** $$f(x) = \lambda e^{-\lambda x}$$ với $$x \geq 0$$

**Kỳ vọng:** $$\mathbb{E}[X] = \frac{1}{\lambda}$$

**Phương sai:** $$\text{Var}(X) = \frac{1}{\lambda^2}$$

**Tính chất:** có tính không nhớ (*memoryless*).

#### Phân phối Beta

Phân phối liên tục linh hoạt trên đoạn $$[0,1]$$, thường dùng để mô hình hóa các đại lượng mang ý nghĩa xác suất hoặc tỉ lệ.

**Tham số:** $$\alpha, \beta > 0$$ (các tham số hình dạng)

**PDF:** $$f(x) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1}(1-x)^{\beta-1}$$ với $$x \in [0,1]$$

**Kỳ vọng:** $$\mathbb{E}[X] = \frac{\alpha}{\alpha+\beta}$$

**Phương sai:** $$\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

<div id="continuous-distributions-demo" style="border: 2px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f9f9f9;">
    <h4 style="text-align: center; color: #333;">Phân phối liên tục tương tác</h4>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
        <div style="flex: 1; min-width: 400px;">
            <canvas id="continuousCanvas" width="400" height="300" style="border: 1px solid #ccc; background: white;"></canvas>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                <strong>Trực quan hóa:</strong> Hàm mật độ xác suất (PDF) của các phân phối liên tục.
            </p>
        </div>
        
        <div style="flex: 1; min-width: 250px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="margin-top: 0; color: #444;">Loại phân phối</h5>
                
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="continuous-dist" value="uniform" checked> Đều
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="continuous-dist" value="normal"> Chuẩn
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="continuous-dist" value="exponential"> Mũ
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="continuous-dist" value="beta"> Beta
                    </label>
                </div>
                
                <div id="continuous-params">
                    <div id="uniform-params">
                        <div style="margin-bottom: 15px;">
                            <label for="a-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">a: <span id="a-value">0</span></label>
                            <input type="range" id="a-slider" min="-2" max="2" step="0.1" value="0" style="width: 100%;">
                        </div>
                        <div style="margin-bottom: 15px;">
                            <label for="b-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">b: <span id="b-value">1</span></label>
                            <input type="range" id="b-slider" min="0.5" max="4" step="0.1" value="1" style="width: 100%;">
                        </div>
                    </div>
                    
                    <div id="normal-params" style="display: none;">
                        <div style="margin-bottom: 15px;">
                            <label for="mu-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">μ: <span id="mu-value">0</span></label>
                            <input type="range" id="mu-slider" min="-3" max="3" step="0.1" value="0" style="width: 100%;">
                        </div>
                        <div style="margin-bottom: 15px;">
                            <label for="sigma-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">σ: <span id="sigma-value">1.0</span></label>
                            <input type="range" id="sigma-slider" min="0.5" max="3" step="0.1" value="1.0" style="width: 100%;">
                        </div>
                    </div>
                    
                    <div id="exponential-params" style="display: none;">
                        <div style="margin-bottom: 15px;">
                            <label for="exp-lambda-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">λ: <span id="exp-lambda-value">1.0</span></label>
                            <input type="range" id="exp-lambda-slider" min="0.2" max="3" step="0.1" value="1.0" style="width: 100%;">
                        </div>
                    </div>
                    
                    <div id="beta-params" style="display: none;">
                        <div style="margin-bottom: 15px;">
                            <label for="alpha-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">α: <span id="alpha-value">2</span></label>
                            <input type="range" id="alpha-slider" min="0.5" max="5" step="0.1" value="2" style="width: 100%;">
                        </div>
                        <div style="margin-bottom: 15px;">
                            <label for="beta-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">β: <span id="beta-value">2</span></label>
                            <input type="range" id="beta-slider" min="0.5" max="5" step="0.1" value="2" style="width: 100%;">
                        </div>
                    </div>
                </div>
                
                <div id="continuous-stats" style="font-family: monospace; font-size: 12px; line-height: 1.4; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    <div><strong>Thống kê:</strong></div>
                    <div>Kỳ vọng: <span id="continuous-mean">0.500</span></div>
                    <div>Phương sai: <span id="continuous-variance">0.083</span></div>
                    <div>Miền xác định: <span id="continuous-support">[0, 1]</span></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
class ContinuousDistributionsDemo {
    constructor() {
        this.canvas = document.getElementById('continuousCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        this.distType = 'uniform';
        this.params = { a: 0, b: 1, mu: 0, sigma: 1, lambda: 1, alpha: 2, beta: 2 };
        
        this.setupControls();
        this.draw();
    }
    
    setupControls() {
        const radios = document.querySelectorAll('input[name="continuous-dist"]');
        
        radios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.distType = e.target.value;
                this.updateParameterVisibility();
                this.updateStats();
                this.draw();
            });
        });
        
        // Setup all sliders
        const sliders = ['a', 'b', 'mu', 'sigma', 'exp-lambda', 'alpha', 'beta'];
        sliders.forEach(slider => {
            const element = document.getElementById(slider + '-slider');
            if (element) {
                element.addEventListener('input', (e) => {
                    const value = parseFloat(e.target.value);
                    const param = slider === 'exp-lambda' ? 'lambda' : slider;
                    this.params[param] = value;
                    
                    const valueSpan = document.getElementById(slider + '-value');
                    if (valueSpan) {
                        valueSpan.textContent = value.toFixed(1);
                    }
                    
                    this.updateStats();
                    this.draw();
                });
            }
        });
        
        this.updateParameterVisibility();
        this.updateStats();
    }
    
    updateParameterVisibility() {
        document.getElementById('uniform-params').style.display = 
            this.distType === 'uniform' ? 'block' : 'none';
        document.getElementById('normal-params').style.display = 
            this.distType === 'normal' ? 'block' : 'none';
        document.getElementById('exponential-params').style.display = 
            this.distType === 'exponential' ? 'block' : 'none';
        document.getElementById('beta-params').style.display = 
            this.distType === 'beta' ? 'block' : 'none';
    }
    
    updateStats() {
        let mean, variance, support;
        
        switch(this.distType) {
            case 'uniform':
                mean = (this.params.a + this.params.b) / 2;
                variance = Math.pow(this.params.b - this.params.a, 2) / 12;
                support = `[${this.params.a}, ${this.params.b}]`;
                break;
            case 'normal':
                mean = this.params.mu;
                variance = this.params.sigma * this.params.sigma;
                support = '(-∞, ∞)';
                break;
            case 'exponential':
                mean = 1 / this.params.lambda;
                variance = 1 / (this.params.lambda * this.params.lambda);
                support = '[0, ∞)';
                break;
            case 'beta':
                mean = this.params.alpha / (this.params.alpha + this.params.beta);
                variance = (this.params.alpha * this.params.beta) / 
                          (Math.pow(this.params.alpha + this.params.beta, 2) * 
                           (this.params.alpha + this.params.beta + 1));
                support = '[0, 1]';
                break;
        }
        
        document.getElementById('continuous-mean').textContent = mean.toFixed(3);
        document.getElementById('continuous-variance').textContent = variance.toFixed(3);
        document.getElementById('continuous-support').textContent = support;
    }
    
    gamma(z) {
        // Stirling's approximation for gamma function
        if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * this.gamma(1 - z));
        z -= 1;
        let x = 0.99999999999980993;
        const p = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
                  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                  9.9843695780195716e-6, 1.5056327351493116e-7];
        for (let i = 0; i < p.length; i++) {
            x += p[i] / (z + i + 1);
        }
        const t = z + p.length - 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
    }
    
    getPDF(x) {
        switch(this.distType) {
            case 'uniform':
                return (x >= this.params.a && x <= this.params.b) ? 
                       1 / (this.params.b - this.params.a) : 0;
            case 'normal':
                return Math.exp(-0.5 * Math.pow((x - this.params.mu) / this.params.sigma, 2)) / 
                       (this.params.sigma * Math.sqrt(2 * Math.PI));
            case 'exponential':
                return x >= 0 ? this.params.lambda * Math.exp(-this.params.lambda * x) : 0;
            case 'beta':
                if (x < 0 || x > 1) return 0;
                const B = this.gamma(this.params.alpha) * this.gamma(this.params.beta) / 
                         this.gamma(this.params.alpha + this.params.beta);
                return Math.pow(x, this.params.alpha - 1) * Math.pow(1 - x, this.params.beta - 1) / B;
        }
    }
    
    getRange() {
        switch(this.distType) {
            case 'uniform': return [this.params.a - 0.5, this.params.b + 0.5];
            case 'normal': return [this.params.mu - 4 * this.params.sigma, this.params.mu + 4 * this.params.sigma];
            case 'exponential': return [0, 5 / this.params.lambda];
            case 'beta': return [0, 1];
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
        
        const [minX, maxX] = this.getRange();
        
        // Find max PDF for scaling
        let maxPDF = 0;
        for (let i = 0; i <= 200; i++) {
            const x = minX + (maxX - minX) * i / 200;
            maxPDF = Math.max(maxPDF, this.getPDF(x));
        }
        
        // Draw PDF curve
        this.ctx.strokeStyle = '#2196f3';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        for (let i = 0; i <= 200; i++) {
            const x = minX + (maxX - minX) * i / 200;
            const pdf = this.getPDF(x);
            const plotX = marginX + (x - minX) / (maxX - minX) * plotWidth;
            const plotY = this.height - marginY - (pdf / maxPDF) * plotHeight * 0.8;
            
            if (i === 0) {
                this.ctx.moveTo(plotX, plotY);
            } else {
                this.ctx.lineTo(plotX, plotY);
            }
        }
        this.ctx.stroke();
        
        // Labels
        this.ctx.fillStyle = '#000';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('x', this.width / 2, this.height - 10);
        
        this.ctx.save();
        this.ctx.translate(15, this.height / 2);
        this.ctx.rotate(-Math.PI / 2);
        this.ctx.fillText('f(x)', 0, 0);
        this.ctx.restore();
    }
}

// Multivariate Normal Demo

document.addEventListener('DOMContentLoaded', function() {
    new ContinuousDistributionsDemo();
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

- **Xem trước:** [The Normal Distribution, Clearly Explained!!!](https://www.youtube.com/watch?v=rzFX5NWojp0) — StatQuest with Josh Starmer, tiếng Anh. Trình bày phân phối chuẩn — phân phối liên tục trọng tâm của bài.
- [The Main Ideas behind Probability Distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc) — StatQuest with Josh Starmer, tiếng Anh. Làm rõ sự khác biệt giữa hàm mật độ xác suất (PDF) và hàm khối xác suất (PMF) trước khi làm việc với các công thức liên tục.
- [MaDL - Common Probability Distributions](https://www.youtube.com/watch?v=NbzVwvE_syQ) — Tübingen Machine Learning, tiếng Anh. Đặt phân phối Gauss cạnh các phân phối liên tục khác như phân phối mũ và Beta.
