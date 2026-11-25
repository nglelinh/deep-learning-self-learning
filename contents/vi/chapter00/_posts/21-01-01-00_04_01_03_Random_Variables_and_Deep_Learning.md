---
layout: post
title: 00-04-01-03 Biến Ngẫu Nhiên và Deep Learning
chapter: '00'
order: 18
owner: AI Assistant
lang: vi
categories:
- chapter00
---

Máy tính không thể cộng trừ chữ "Mèo" hay "Mưa", nhưng ngay khi ta gán cho mỗi kết cục ngẫu nhiên một con số thực—gọi là **biến ngẫu nhiên** (*random variable*)—toàn bộ kho vũ khí của giải tích và đại số lập tức phát huy tác dụng. Trong một mạng nơ-ron đang huấn luyện, gần như mọi thứ đều là biến ngẫu nhiên: từ mẫu dữ liệu rút ngẫu nhiên trong mini-batch, mặt nạ Dropout bật/tắt 0–1, cho tới chính trọng số khởi tạo ban đầu. Hiểu rõ kỳ vọng, phương sai và hiệp phương sai của biến ngẫu nhiên giúp bạn kiểm soát dòng chảy tín hiệu xuyên qua hàng trăm tầng mạng.

## Kiến thức cần nhớ: Bốn công cụ của Biến Ngẫu nhiên trong Học sâu

1. **Biến ngẫu nhiên Rời rạc (PMF) vs. Liên tục (PDF & CDF)**:
   Biến rời rạc dùng hàm khối xác suất $$P(X = x)$$ và tổng $$\sum$$; biến liên tục dùng hàm mật độ $$p(x)$$ (có thể lớn hơn $$1$$!) và tích phân $$\int p(x)\,dx = 1$$.

2. **Kỳ vọng ($$\mathbb{E}[X]$$) và Tính tuyến tính của Kỳ vọng**:
   $$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$ luôn đúng *kể cả khi $$X, Y$$ phụ thuộc nhau*—đây là cơ sở chứng minh công thức bù tỷ lệ $$\frac{1}{1-p}$$ của Inverted Dropout!

3. **Phương sai ($$\text{Var}(X)$$) và Luật nhân hệ số**:
   $$\text{Var}(aX) = a^2 \text{Var}(X)$$, và với các biến độc lập, $$\text{Var}\!\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i)$$. Công thức này dẫn thẳng tới công thức khởi tạo trọng số **Xavier** ($$\text{Var}(w) = 1/n_{\text{in}}$$) để tín hiệu không phình to khi đi qua lớp rộng.

4. **Hàm mất mát dưới góc nhìn Kỳ vọng (Expected Risk)**:
   Mục tiêu lý tưởng của Deep Learning là cực tiểu hóa rủi ro kỳ vọng $$\mathbb{E}_{(x,y)\sim P_{\text{data}}}[\ell(f_\theta(x), y)]$$, và tập huấn luyện chỉ là trung bình thực nghiệm xấp xỉ kỳ vọng đó.

### Giải thích: Tính tay phương sai qua một nơ-ron tuyến tính để hiểu khởi tạo Xavier

Giả sử một nơ-ron nhận $$n = 100$$ đầu vào độc lập $$x_1, \dots, x_{100}$$ đều có $$\mathbb{E}[x_i] = 0, \text{Var}(x_i) = 1$$, và ta khởi tạo trọng số độc lập $$w_1, \dots, w_{100}$$ với $$\mathbb{E}[w_i] = 0, \text{Var}(w_i) = \sigma_w^2$$. Xét đầu ra $$z = \sum_{i=1}^{100} w_i x_i$$:

1. **Kỳ vọng của $$z$$**: Do $$w_i$$ và $$x_i$$ độc lập, $$\mathbb{E}[w_i x_i] = \mathbb{E}[w_i]\mathbb{E}[x_i] = 0 \implies \mathbb{E}[z] = 0$$.
2. **Phương sai của từng tích $$w_i x_i$$**:
   $$\text{Var}(w_i x_i) = \mathbb{E}[w_i^2 x_i^2] - (\mathbb{E}[w_i x_i])^2 = \mathbb{E}[w_i^2]\mathbb{E}[x_i^2] = \sigma_w^2 \times 1 = \sigma_w^2.$$
3. **Phương sai tổng $$z$$**: Vì 100 số hạng độc lập nhau,
   $$\text{Var}(z) = \sum_{i=1}^{100} \text{Var}(w_i x_i) = 100\,\sigma_w^2.$$
4. **Thử thay số**:
   - Nếu chọn ngây thơ $$\sigma_w = 1$$: $$\text{Var}(z) = 100$$ (độ lệch chuẩn tăng gấp 10 chỉ sau 1 tầng—đi qua 5 tầng sẽ bùng nổ thành $$10^{10}$$!).
   - Nếu chọn $$\sigma_w^2 = \frac{1}{100} \implies \sigma_w = 0.1$$: $$\text{Var}(z) = 100 \times 0.01 = 1$$ (giữ nguyên thang đo tín hiệu!). Đó chính là linh hồn toán học của khởi tạo Xavier/He.

### 6. Biến ngẫu nhiên

**Biến ngẫu nhiên** $$X$$ là một hàm ánh xạ mỗi kết quả trong không gian mẫu sang một số thực:
$$X: \Omega \rightarrow \mathbb{R}$$

#### Phân loại biến ngẫu nhiên

**Biến ngẫu nhiên rời rạc:** nhận tập giá trị hữu hạn hoặc đếm được (ví dụ: số lần xuất hiện mặt sấp).
- Hàm khối xác suất (*PMF*): $$P(X = x)$$

**Biến ngẫu nhiên liên tục:** nhận giá trị trên một khoảng liên tục của trục số thực (ví dụ: chiều cao, cân nặng).
- Hàm mật độ xác suất (*PDF*): $$f_X(x)$$
- $$P(a \leq X \leq b) = \int_a^b f_X(x) dx$$

### 7. Liên hệ với tối ưu hóa

Lý thuyết xác suất gắn liền với tối ưu hóa qua nhiều bài toán quen thuộc:

#### Ước lượng hợp lý cực đại (MLE)
Tìm tham số $$\theta$$ cực đại hóa hàm hợp lý (*likelihood*):
$$\hat{\theta} = \arg\max_\theta P(\text{dữ liệu}|\theta)$$

#### Tối ưu hóa giá trị kỳ vọng
Cực tiểu hóa hàm mất mát kỳ vọng:
$$\min_\theta \mathbb{E}[L(Y, f(X; \theta))]$$

#### Tối ưu hóa Bayes
Sử dụng phân phối xác suất để mô hình hóa sự bất định của hàm mục tiêu và định hướng quá trình tìm kiếm nghiệm tối ưu.

<div id="deep-learning-connection" style="border: 2px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f0f8ff;">
    <h4 style="text-align: center; color: #333;">Ví dụ ứng dụng xác suất trong tối ưu hóa</h4>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
        <div style="flex: 1; min-width: 400px;">
            <canvas id="deep-learningCanvas" width="400" height="300" style="border: 1px solid #ccc; background: white;"></canvas>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                <strong>Ví dụ MLE:</strong> Tìm tham số μ cực đại hóa hàm hợp lý của dữ liệu quan sát từ phân phối chuẩn Normal(μ, 1).
            </p>
        </div>
        
        <div style="flex: 1; min-width: 250px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="margin-top: 0; color: #444;">Minh họa MLE</h5>
                
                <div style="margin-bottom: 15px;">
                    <label for="true-mu-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">μ thực: <span id="true-mu-value">2.0</span></label>
                    <input type="range" id="true-mu-slider" min="-2" max="4" step="0.1" value="2.0" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="sample-size-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">Kích thước mẫu: <span id="sample-size-value">20</span></label>
                    <input type="range" id="sample-size-slider" min="5" max="100" step="5" value="20" style="width: 100%;">
                </div>
                
                <button id="generate-mle-data" style="width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 15px;">Tạo dữ liệu &amp; Tìm MLE</button>
                
                <div id="mle-results" style="font-family: monospace; font-size: 12px; line-height: 1.4; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    <div><strong>Kết quả:</strong></div>
                    <div>μ thực: <span id="display-true-mu">2.000</span></div>
                    <div>Trung bình mẫu: <span id="sample-mean">--</span></div>
                    <div>Ước lượng MLE: <span id="mle-estimate">--</span></div>
                    <div>Sai số: <span id="mle-error">--</span></div>
                </div>
            </div>
        </div>
    </div>
</div>

### Điểm chính cần nhớ

1. **Nền tảng:** các tiên đề xác suất tạo khung toán học chặt chẽ để xử lý sự bất định.
2. **Xác suất có điều kiện:** công cụ cốt lõi để cập nhật niềm tin khi có thông tin mới.
3. **Tính độc lập:** giúp đơn giản hóa phép tính và các giả định mô hình.
4. **Biến ngẫu nhiên:** cầu nối giữa không gian mẫu trừu tượng và các đại lượng số cụ thể.
5. **Liên hệ với tối ưu hóa:** nhiều bài toán tối ưu hóa phát sinh trực tiếp từ mô hình xác suất.

Nắm vững các khái niệm nền tảng này giúp bạn tiếp cận thuận lợi những chủ đề nâng cao hơn như suy luận Bayes, ước lượng hợp lý cực đại và tối ưu hóa ngẫu nhiên — các thành phần cốt lõi của học máy và khoa học dữ liệu hiện đại.

<script>
class MLEDemo {
    constructor() {
        this.canvas = document.getElementById('deep-learningCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        this.trueMu = 2.0;
        this.sampleSize = 20;
        this.data = [];
        
        this.setupControls();
        this.draw();
    }
    
    setupControls() {
        const trueMuSlider = document.getElementById('true-mu-slider');
        const sampleSizeSlider = document.getElementById('sample-size-slider');
        const generateBtn = document.getElementById('generate-mle-data');
        
        trueMuSlider.addEventListener('input', (e) => {
            this.trueMu = parseFloat(e.target.value);
            document.getElementById('true-mu-value').textContent = this.trueMu.toFixed(1);
            document.getElementById('display-true-mu').textContent = this.trueMu.toFixed(3);
        });
        
        sampleSizeSlider.addEventListener('input', (e) => {
            this.sampleSize = parseInt(e.target.value);
            document.getElementById('sample-size-value').textContent = this.sampleSize;
        });
        
        generateBtn.addEventListener('click', () => this.generateDataAndFindMLE());
    }
    
    generateDataAndFindMLE() {
        // Generate data from Normal(trueMu, 1)
        this.data = [];
        for (let i = 0; i < this.sampleSize; i++) {
            // Box-Muller transform for normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            this.data.push(this.trueMu + z); // Normal(trueMu, 1)
        }
        
        // Calculate MLE (sample mean for normal distribution)
        const sampleMean = this.data.reduce((sum, x) => sum + x, 0) / this.data.length;
        const error = Math.abs(sampleMean - this.trueMu);
        
        // Update display
        document.getElementById('sample-mean').textContent = sampleMean.toFixed(3);
        document.getElementById('mle-estimate').textContent = sampleMean.toFixed(3);
        document.getElementById('mle-error').textContent = error.toFixed(3);
        
        this.draw();
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        if (this.data.length === 0) {
            this.ctx.fillStyle = '#666';
            this.ctx.font = '16px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Nhấp "Tạo Dữ Liệu & Tìm MLE" để bắt đầu', this.width / 2, this.height / 2);
            return;
        }
        
        // Draw axes
        this.ctx.strokeStyle = '#ddd';
        this.ctx.lineWidth = 1;
        const marginX = 50;
        const marginY = 50;
        const plotWidth = this.width - 2 * marginX;
        const plotHeight = this.height - 2 * marginY;
        
        // X-axis
        this.ctx.beginPath();
        this.ctx.moveTo(marginX, this.height - marginY);
        this.ctx.lineTo(this.width - marginX, this.height - marginY);
        this.ctx.stroke();
        
        // Y-axis
        this.ctx.beginPath();
        this.ctx.moveTo(marginX, marginY);
        this.ctx.lineTo(marginX, this.height - marginY);
        this.ctx.stroke();
        
        // Find data range
        const minX = Math.min(...this.data) - 1;
        const maxX = Math.max(...this.data) + 1;
        
        // Draw likelihood function
        this.ctx.strokeStyle = '#2196f3';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        for (let i = 0; i <= 100; i++) {
            const mu = minX + (maxX - minX) * i / 100;
            let logLikelihood = 0;
            
            // Calculate log-likelihood
            for (const x of this.data) {
                logLikelihood -= 0.5 * Math.log(2 * Math.PI);
                logLikelihood -= 0.5 * (x - mu) * (x - mu);
            }
            
            const x = marginX + (mu - minX) / (maxX - minX) * plotWidth;
            const y = this.height - marginY - (logLikelihood - (-this.data.length * 2)) / (this.data.length) * plotHeight * 0.8;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        this.ctx.stroke();
        
        // Mark MLE
        const sampleMean = this.data.reduce((sum, x) => sum + x, 0) / this.data.length;
        const mleX = marginX + (sampleMean - minX) / (maxX - minX) * plotWidth;
        
        this.ctx.strokeStyle = '#f44336';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(mleX, marginY);
        this.ctx.lineTo(mleX, this.height - marginY);
        this.ctx.stroke();
        
        // Mark true value
        const trueX = marginX + (this.trueMu - minX) / (maxX - minX) * plotWidth;
        this.ctx.strokeStyle = '#4caf50';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        this.ctx.moveTo(trueX, marginY);
        this.ctx.lineTo(trueX, this.height - marginY);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        // Draw data points
        this.ctx.fillStyle = '#666';
        for (const x of this.data) {
            const pointX = marginX + (x - minX) / (maxX - minX) * plotWidth;
            this.ctx.beginPath();
            this.ctx.arc(pointX, this.height - marginY + 10, 2, 0, 2 * Math.PI);
            this.ctx.fill();
        }
        
        // Labels
        this.ctx.fillStyle = '#000';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('μ', this.width / 2, this.height - 10);
        
        this.ctx.save();
        this.ctx.translate(15, this.height / 2);
        this.ctx.rotate(-Math.PI / 2);
        this.ctx.fillText('Log-Likelihood', 0, 0);
        this.ctx.restore();
        
        // Legend
        this.ctx.textAlign = 'left';
        this.ctx.fillText('— Likelihood', 10, 20);
        this.ctx.fillStyle = '#f44336';
        this.ctx.fillText('— MLE', 10, 35);
        this.ctx.fillStyle = '#4caf50';
        this.ctx.fillText('--- μ Thực', 10, 50);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    new MLEDemo();
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

- **Xem trước:** [MaDL - Random Variables and Probability Distributions](https://www.youtube.com/watch?v=bvBukLjHZh8) — Tübingen Machine Learning, tiếng Anh. Trình bày biến ngẫu nhiên dưới dạng hàm trên không gian mẫu, khớp với cách mô hình hóa dữ liệu và dự đoán trong bài.
- [Random Variables and Probability Distributions](https://www.youtube.com/watch?v=lHCpYeFvTs0) — Dr Nic's Maths and Stats, tiếng Anh. Giới thiệu ngắn gọn về biến ngẫu nhiên rời rạc và liên tục trước khi đi vào các phân phối cụ thể.
- [MaDL - Expectation, Variance and Covariance](https://www.youtube.com/watch?v=kmRm3XIND8M) — Tübingen Machine Learning, tiếng Anh. Giải thích kỳ vọng, phương sai và hiệp phương sai — các đại lượng đặc trưng dùng khi đầu ra mạng nơ-ron mang tính xác suất.
