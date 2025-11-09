---
layout: post
title: 00-04-03 Likelihood và Ước Lượng Hợp Lý Tối Đa (MLE)
chapter: '00'
order: 16
owner: AI Assistant
categories:
- chapter00
lang: vi
lesson_type: required
---

## Likelihood và Ước Lượng Hợp Lý Tối Đa

Trong xác suất thống kê, **likelihood** (tạm dịch: khả năng xảy ra hoặc độ hợp lý) là một khái niệm quan trọng, biểu thị xác suất quan sát được dữ liệu cụ thể khi giả định một mô hình hoặc một tập hợp các tham số nhất định là đúng. Tuy nhiên, likelihood không phải là xác suất thông thường mà là một hàm đo lường mức độ phù hợp của mô hình với dữ liệu.

### Định Nghĩa Likelihood

Likelihood của một tham số $$\theta$$ (hoặc một mô hình) được định nghĩa là xác suất của dữ liệu quan sát được, giả sử tham số $$\theta$$ là đúng. Nó thường được ký hiệu là $$L(\theta \mid x)$$, trong đó $$x$$ là dữ liệu quan sát được.

**Công thức:**
$$L(\theta \mid x) = P(x \mid \theta)$$

Ở đây, $$P(x \mid \theta)$$ là xác suất xảy ra dữ liệu $$x$$ khi tham số $$\theta$$ được cho trước.

### Điểm Khác Biệt với Xác Suất

- **Xác suất** mô tả khả năng xảy ra của một sự kiện trong tương lai, với các tham số đã biết.
- **Likelihood** ngược lại, đo lường mức độ phù hợp của các tham số khác nhau đối với dữ liệu đã quan sát.

<div style="background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0;">
<strong>⚠️ Lưu ý quan trọng:</strong> Likelihood không phải là xác suất của tham số θ, vì nó không được chuẩn hóa như một phân phối xác suất (tổng tích phân của likelihood không nhất thiết bằng 1).
</div>

### Ví Dụ Minh Họa: Tung Đồng Xu

Giả sử bạn tung một đồng xu và quan sát được kết quả là 3 lần "mặt sấp" (S) và 2 lần "mặt ngửa" (N) trong 5 lần tung. Bạn muốn biết xác suất tung được mặt sấp là $$p$$.

Likelihood của $$p$$ là xác suất quan sát được dữ liệu (3 sấp, 2 ngửa) khi $$p$$ được cho trước:

$$L(p \mid \text{dữ liệu}) = P(\text{3 sấp, 2 ngửa} \mid p) = \binom{5}{3} p^3 (1-p)^2$$

Trong đó:
- $$\binom{5}{3}$$ là hệ số nhị thức
- $$p^3$$ là xác suất cho 3 lần sấp
- $$(1-p)^2$$ là xác suất cho 2 lần ngửa

**Thử nghiệm với các giá trị khác nhau:**

Với $$p = 0.6$$:
$$L(0.6 \mid \text{dữ liệu}) = \binom{5}{3} (0.6)^3 (0.4)^2 = 10 \times 0.216 \times 0.16 \approx 0.3456$$

Với $$p = 0.5$$:
$$L(0.5 \mid \text{dữ liệu}) = \binom{5}{3} (0.5)^3 (0.5)^2 = 10 \times 0.125 \times 0.25 \approx 0.3125$$

Likelihood cao hơn với $$p = 0.6$$ cho thấy giá trị này phù hợp hơn với dữ liệu quan sát được so với $$p = 0.5$$.

### Ví Dụ Tương Tác: Tung Đồng Xu

<div id="coin-flip-example" style="border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px;">
    <h4>Thí Nghiệm Tung Đồng Xu</h4>
    
    <div style="margin: 15px 0;">
        <label for="p-slider">Xác suất mặt sấp (p): <span id="p-value">0.5</span></label><br>
        <input type="range" id="p-slider" min="0.01" max="0.99" step="0.01" value="0.5" style="width: 300px;">
    </div>
    
    <div style="margin: 15px 0;">
        <label for="heads-input">Số lần sấp: </label>
        <input type="number" id="heads-input" value="3" min="0" max="20" style="width: 60px;">
        
        <label for="tails-input" style="margin-left: 20px;">Số lần ngửa: </label>
        <input type="number" id="tails-input" value="2" min="0" max="20" style="width: 60px;">
    </div>
    
    <div style="margin: 15px 0;">
        <canvas id="likelihood-plot" width="600" height="300" style="border: 1px solid #ccc;"></canvas>
    </div>
    
    <div id="likelihood-result" style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0;">
        <strong>Likelihood hiện tại:</strong> <span id="current-likelihood">0.3125</span><br>
        <strong>Log-likelihood:</strong> <span id="current-log-likelihood">-1.16</span>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const pSlider = document.getElementById('p-slider');
    const pValue = document.getElementById('p-value');
    const headsInput = document.getElementById('heads-input');
    const tailsInput = document.getElementById('tails-input');
    const canvas = document.getElementById('likelihood-plot');
    const ctx = canvas.getContext('2d');
    const likelihoodResult = document.getElementById('current-likelihood');
    const logLikelihoodResult = document.getElementById('current-log-likelihood');
    
    function binomialCoeff(n, k) {
        if (k > n) return 0;
        if (k === 0 || k === n) return 1;
        
        let result = 1;
        for (let i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }
    
    function likelihood(p, heads, tails) {
        const n = heads + tails;
        const coeff = binomialCoeff(n, heads);
        return coeff * Math.pow(p, heads) * Math.pow(1 - p, tails);
    }
    
    function logLikelihood(p, heads, tails) {
        const n = heads + tails;
        const logCoeff = Math.log(binomialCoeff(n, heads));
        return logCoeff + heads * Math.log(p) + tails * Math.log(1 - p);
    }
    
    function plotLikelihood() {
        const heads = parseInt(headsInput.value);
        const tails = parseInt(tailsInput.value);
        const currentP = parseFloat(pSlider.value);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Vẽ trục
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(50, 250);
        ctx.lineTo(550, 250);
        ctx.moveTo(50, 250);
        ctx.lineTo(50, 50);
        ctx.stroke();
        
        // Nhãn trục
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText('p (xác suất mặt sấp)', 250, 280);
        ctx.save();
        ctx.translate(20, 150);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('Likelihood', 0, 0);
        ctx.restore();
        
        // Vẽ đường likelihood
        ctx.strokeStyle = '#2196F3';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        let maxLikelihood = 0;
        const points = [];
        
        for (let i = 0; i <= 500; i++) {
            const p = 0.01 + (i / 500) * 0.98;
            const l = likelihood(p, heads, tails);
            points.push({p: p, l: l});
            maxLikelihood = Math.max(maxLikelihood, l);
        }
        
        for (let i = 0; i < points.length; i++) {
            const x = 50 + (points[i].p - 0.01) / 0.98 * 500;
            const y = 250 - (points[i].l / maxLikelihood) * 200;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Vẽ điểm hiện tại
        const currentLikelihood = likelihood(currentP, heads, tails);
        const currentX = 50 + (currentP - 0.01) / 0.98 * 500;
        const currentY = 250 - (currentLikelihood / maxLikelihood) * 200;
        
        ctx.fillStyle = '#f44336';
        ctx.beginPath();
        ctx.arc(currentX, currentY, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        // Cập nhật kết quả
        likelihoodResult.textContent = currentLikelihood.toFixed(6);
        logLikelihoodResult.textContent = logLikelihood(currentP, heads, tails).toFixed(4);
        
        // Vẽ các tick marks
        ctx.fillStyle = '#666';
        ctx.font = '10px Arial';
        for (let i = 0; i <= 10; i++) {
            const p = i / 10;
            const x = 50 + (p - 0.01) / 0.98 * 500;
            ctx.fillText(p.toFixed(1), x - 10, 265);
        }
    }
    
    function updateDisplay() {
        pValue.textContent = pSlider.value;
        plotLikelihood();
    }
    
    pSlider.addEventListener('input', updateDisplay);
    headsInput.addEventListener('input', plotLikelihood);
    tailsInput.addEventListener('input', plotLikelihood);
    
    // Khởi tạo
    updateDisplay();
});
</script>

### Maximum Likelihood Estimation (MLE)

**Ước lượng hợp lý tối đa** là phương pháp tìm giá trị của tham số $$\theta$$ sao cho hàm likelihood $$L(\theta \mid x)$$ đạt giá trị lớn nhất.

**Công thức MLE:**
$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta \mid x)$$

Thường thì việc tối đa hóa log-likelihood dễ dàng hơn:
$$\hat{\theta}_{MLE} = \arg\max_{\theta} \log L(\theta \mid x)$$

### Ví Dụ MLE: Phân Phối Chuẩn

Giả sử chúng ta có $$n$$ quan sát độc lập $$x_1, x_2, \ldots, x_n$$ từ phân phối chuẩn $$N(\mu, \sigma^2)$$.

**Hàm likelihood:**
$$L(\mu, \sigma^2 \mid x_1, \ldots, x_n) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

**Log-likelihood:**
$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Ước lượng MLE:**

Để tìm $$\hat{\mu}$$, lấy đạo hàm theo $$\mu$$ và đặt bằng 0:
$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

Giải được: $$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

Tương tự cho $$\sigma^2$$:
$$\hat{\sigma^2}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

### Ví Dụ Tương Tác: MLE cho Phân Phối Chuẩn

<div id="normal-mle-example" style="border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px;">
    <h4>MLE cho Phân Phối Chuẩn</h4>
    
    <div style="margin: 15px 0;">
        <button id="generate-data" style="background: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer;">Tạo Dữ Liệu Mới</button>
        <span style="margin-left: 20px;">Số mẫu: </span>
        <input type="number" id="sample-size" value="50" min="10" max="200" style="width: 60px;">
    </div>
    
    <div style="margin: 15px 0;">
        <label for="mu-guess">Ước lượng μ: <span id="mu-guess-value">0</span></label><br>
        <input type="range" id="mu-guess" min="-3" max="3" step="0.1" value="0" style="width: 300px;">
    </div>
    
    <div style="margin: 15px 0;">
        <label for="sigma-guess">Ước lượng σ: <span id="sigma-guess-value">1</span></label><br>
        <input type="range" id="sigma-guess" min="0.1" max="3" step="0.1" value="1" style="width: 300px;">
    </div>
    
    <div style="margin: 15px 0;">
        <canvas id="normal-plot" width="600" height="300" style="border: 1px solid #ccc;"></canvas>
    </div>
    
    <div id="mle-results" style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0;">
        <div><strong>Dữ liệu thực:</strong> μ = <span id="true-mu">1.0</span>, σ = <span id="true-sigma">1.5</span></div>
        <div><strong>MLE ước lượng:</strong> μ̂ = <span id="mle-mu">0.95</span>, σ̂ = <span id="mle-sigma">1.48</span></div>
        <div><strong>Ước lượng hiện tại:</strong> μ = <span id="current-mu">0</span>, σ = <span id="current-sigma">1</span></div>
        <div><strong>Log-likelihood hiện tại:</strong> <span id="current-ll">-75.2</span></div>
        <div><strong>Log-likelihood tối đa:</strong> <span id="max-ll">-72.1</span></div>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generate-data');
    const sampleSizeInput = document.getElementById('sample-size');
    const muGuess = document.getElementById('mu-guess');
    const sigmaGuess = document.getElementById('sigma-guess');
    const muGuessValue = document.getElementById('mu-guess-value');
    const sigmaGuessValue = document.getElementById('sigma-guess-value');
    const canvas = document.getElementById('normal-plot');
    const ctx = canvas.getContext('2d');
    
    let data = [];
    let trueMu = 1.0;
    let trueSigma = 1.5;
    
    function normalRandom(mu, sigma) {
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * sigma + mu;
    }
    
    function generateData() {
        const n = parseInt(sampleSizeInput.value);
        trueMu = (Math.random() - 0.5) * 4; // -2 to 2
        trueSigma = 0.5 + Math.random() * 2; // 0.5 to 2.5
        
        data = [];
        for (let i = 0; i < n; i++) {
            data.push(normalRandom(trueMu, trueSigma));
        }
        
        // Tính MLE
        const mleMu = data.reduce((sum, x) => sum + x, 0) / data.length;
        const mleSigma = Math.sqrt(data.reduce((sum, x) => sum + (x - mleMu) ** 2, 0) / data.length);
        
        document.getElementById('true-mu').textContent = trueMu.toFixed(2);
        document.getElementById('true-sigma').textContent = trueSigma.toFixed(2);
        document.getElementById('mle-mu').textContent = mleMu.toFixed(2);
        document.getElementById('mle-sigma').textContent = mleSigma.toFixed(2);
        
        plotData();
    }
    
    function logLikelihood(mu, sigma, data) {
        const n = data.length;
        const sumSquares = data.reduce((sum, x) => sum + (x - mu) ** 2, 0);
        return -n/2 * Math.log(2 * Math.PI) - n * Math.log(sigma) - sumSquares / (2 * sigma ** 2);
    }
    
    function normalPDF(x, mu, sigma) {
        return (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mu) / sigma) ** 2);
    }
    
    function plotData() {
        if (data.length === 0) return;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const currentMu = parseFloat(muGuess.value);
        const currentSigma = parseFloat(sigmaGuess.value);
        
        // Tìm phạm vi dữ liệu
        const minX = Math.min(...data) - 1;
        const maxX = Math.max(...data) + 1;
        const range = maxX - minX;
        
        // Vẽ histogram
        const bins = 20;
        const binWidth = range / bins;
        const binCounts = new Array(bins).fill(0);
        
        data.forEach(x => {
            const binIndex = Math.floor((x - minX) / binWidth);
            if (binIndex >= 0 && binIndex < bins) {
                binCounts[binIndex]++;
            }
        });
        
        const maxCount = Math.max(...binCounts);
        
        ctx.fillStyle = 'rgba(33, 150, 243, 0.3)';
        for (let i = 0; i < bins; i++) {
            const x = 50 + (i * binWidth / range) * 500;
            const height = (binCounts[i] / maxCount) * 200;
            const width = (binWidth / range) * 500;
            ctx.fillRect(x, 250 - height, width, height);
        }
        
        // Vẽ đường phân phối thực
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i <= 500; i++) {
            const x = minX + (i / 500) * range;
            const y = normalPDF(x, trueMu, trueSigma);
            const plotX = 50 + (i / 500) * 500;
            const plotY = 250 - (y / (1 / (trueSigma * Math.sqrt(2 * Math.PI)))) * 200;
            
            if (i === 0) ctx.moveTo(plotX, plotY);
            else ctx.lineTo(plotX, plotY);
        }
        ctx.stroke();
        
        // Vẽ đường ước lượng hiện tại
        ctx.strokeStyle = '#f44336';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        for (let i = 0; i <= 500; i++) {
            const x = minX + (i / 500) * range;
            const y = normalPDF(x, currentMu, currentSigma);
            const plotX = 50 + (i / 500) * 500;
            const plotY = 250 - (y / (1 / (currentSigma * Math.sqrt(2 * Math.PI)))) * 200;
            
            if (i === 0) ctx.moveTo(plotX, plotY);
            else ctx.lineTo(plotX, plotY);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Vẽ trục
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(50, 250);
        ctx.lineTo(550, 250);
        ctx.stroke();
        
        // Cập nhật kết quả
        const currentLL = logLikelihood(currentMu, currentSigma, data);
        const mleMu = data.reduce((sum, x) => sum + x, 0) / data.length;
        const mleSigma = Math.sqrt(data.reduce((sum, x) => sum + (x - mleMu) ** 2, 0) / data.length);
        const maxLL = logLikelihood(mleMu, mleSigma, data);
        
        document.getElementById('current-mu').textContent = currentMu.toFixed(2);
        document.getElementById('current-sigma').textContent = currentSigma.toFixed(2);
        document.getElementById('current-ll').textContent = currentLL.toFixed(2);
        document.getElementById('max-ll').textContent = maxLL.toFixed(2);
        
        // Thêm chú thích
        ctx.fillStyle = '#4CAF50';
        ctx.fillRect(450, 60, 15, 3);
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText('Phân phối thực', 470, 65);
        
        ctx.strokeStyle = '#f44336';
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(450, 80);
        ctx.lineTo(465, 80);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillText('Ước lượng hiện tại', 470, 85);
    }
    
    function updateDisplay() {
        muGuessValue.textContent = muGuess.value;
        sigmaGuessValue.textContent = sigmaGuess.value;
        plotData();
    }
    
    generateBtn.addEventListener('click', generateData);
    muGuess.addEventListener('input', updateDisplay);
    sigmaGuess.addEventListener('input', updateDisplay);
    sampleSizeInput.addEventListener('change', generateData);
    
    // Khởi tạo
    generateData();
});
</script>

### Tính Chất của MLE

1. **Tính Nhất Quán (Consistency)**: Khi kích thước mẫu tăng, ước lượng MLE hội tụ về giá trị thực của tham số.

2. **Tính Tiệm Cận Chuẩn (Asymptotic Normality)**: Với mẫu lớn, phân phối của ước lượng MLE xấp xỉ phân phối chuẩn.

3. **Tính Hiệu Quả Tiệm Cận (Asymptotic Efficiency)**: MLE đạt được cận Cramér-Rao, nghĩa là có phương sai nhỏ nhất trong các ước lượng không thiên lệch.

4. **Bất Biến (Invariance)**: Nếu $$\hat{\theta}$$ là MLE của $$\theta$$, thì $$g(\hat{\theta})$$ là MLE của $$g(\theta)$$ với $$g$$ là hàm khả nghịch.

### Ứng Dụng của MLE

#### 1. Ước Lượng Tham Số

MLE được sử dụng rộng rãi để ước lượng tham số trong các mô hình thống kê:
- Phân phối chuẩn: ước lượng trung bình và phương sai
- Phân phối Poisson: ước lượng tham số λ
- Hồi quy tuyến tính: ước lượng hệ số hồi quy

#### 2. So Sánh Mô Hình

Likelihood được dùng để so sánh các mô hình thống kê khác nhau:
- **Likelihood Ratio Test**: So sánh hai mô hình lồng nhau
- **AIC/BIC**: Tiêu chí lựa chọn mô hình dựa trên likelihood

#### 3. Thống Kê Bayes

Trong thống kê Bayes, likelihood kết hợp với phân phối tiên nghiệm (prior) để tính phân phối hậu nghiệm (posterior):

$$P(\theta \mid x) \propto L(\theta \mid x) \cdot P(\theta)$$

### Kết Nối với Tối Ưu Hóa

MLE tạo ra một bài toán tối ưu hóa:

$$\max_{\theta} L(\theta \mid x) \quad \text{hoặc} \quad \max_{\theta} \log L(\theta \mid x)$$

Điều này thường dẫn đến:
- **Tối ưu hóa không ràng buộc**: Khi không có ràng buộc trên tham số
- **Tối ưu hóa có ràng buộc**: Khi tham số phải thỏa mãn điều kiện nhất định
- **Bài toán lồi**: Nhiều hàm log-likelihood là lồi, đảm bảo nghiệm tối ưu toàn cục

<div style="background: #e8f4fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0;">
<strong>💡 Kết Nối với Tối Ưu Hóa Lồi:</strong> Nhiều bài toán MLE dẫn đến tối ưu hóa lồi, đặc biệt trong họ phân phối mũ. Điều này đảm bảo rằng nghiệm tìm được là tối ưu toàn cục và các thuật toán tối ưu hóa sẽ hội tụ đến nghiệm đúng.
</div>

### Ví Dụ Thực Tế: Hồi Quy Logistic

Trong hồi quy logistic, chúng ta mô hình hóa xác suất của biến nhị phân:

$$P(y = 1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

**Log-likelihood:**
$$\ell(\beta_0, \beta_1) = \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

Đây là một bài toán tối ưu hóa lồi, có thể giải bằng gradient descent hoặc Newton's method.

### Tóm Tắt

- **Likelihood** đo lường mức độ phù hợp của tham số với dữ liệu quan sát
- **MLE** tìm tham số tối đa hóa likelihood
- MLE có nhiều tính chất thống kê tốt: nhất quán, hiệu quả, tiệm cận chuẩn
- MLE tạo ra các bài toán tối ưu hóa, nhiều trong số đó là lồi
- Ứng dụng rộng rãi trong thống kê, học máy và khoa học dữ liệu

<div style="background: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0;">
<strong>🎯 Điểm Quan Trọng:</strong> MLE là cầu nối quan trọng giữa thống kê và tối ưu hóa. Hiểu rõ MLE giúp bạn nắm vững cách các mô hình thống kê được ước lượng thông qua các phương pháp tối ưu hóa, đặc biệt là tối ưu hóa lồi.
</div>



