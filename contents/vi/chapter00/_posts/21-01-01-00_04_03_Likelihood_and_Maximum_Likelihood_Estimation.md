---
layout: post
title: 00-04-03 Likelihood và Ước Lượng Hợp Lý Tối Đa (MLE)
chapter: '00'
order: 23
owner: AI Assistant
categories:
- chapter00
lang: vi
---

Một trong những khoảnh khắc "vỡ òa" lớn nhất khi học Deep Learning là nhận ra rằng những hàm mất mát tưởng như được nghĩ ra một cách tùy hứng—MSE cho hồi quy, Binary Cross-Entropy cho phân loại nhị phân, Categorical Cross-Entropy cho đa lớp—thực chất đều là **cùng một nguyên lý duy nhất** đội những chiếc mặt nạ khác nhau: **Ước lượng Hợp lý Cực đại (Maximum Likelihood Estimation — MLE)**. Bài học này mở khóa chiếc cầu nối vĩ đại giữa thống kê và tối ưu hóa, giúp bạn hiểu sâu *tại sao* ta lại huấn luyện mạng nơ-ron bằng chính các hàm mất mát đó.

## Kiến thức cần nhớ: Bốn trụ cột của Hàm Hợp lý và MLE

1. **Xác suất vs. Hàm Hợp lý (Probability vs. Likelihood)**:
   Xác suất cố định tham số $$\theta$$ và hỏi dữ liệu $$x$$ nào dễ xảy ra ($$P(x \mid \theta)$$); ngược lại, hàm hợp lý cố định dữ liệu $$x$$ đã quan sát và xoay tham số $$\theta$$ để xem mô hình nào giải thích dữ liệu tốt nhất ($$\mathcal{L}(\theta \mid x)$$).

2. **Chuyển từ Tích sang Tổng nhờ Log-Likelihood**:
   Với $$N$$ mẫu độc lập, tích $$\prod_{i=1}^N P(x_i \mid \theta)$$ rất nhanh chạm ngưỡng $$0$$ trên máy tính (underflow). Lấy logarit tự nhiên biến tích thành tổng $$\sum_{i=1}^N \ln P(x_i \mid \theta)$$ mà không làm đổi vị trí cực đại, vì $$\ln$$ là hàm đồng biến chặt.

3. **Âm Log-Likelihood (NLL) chính là Hàm Mất mát**:
   Cực đại hóa $$\ln \mathcal{L}(\theta)$$ tương đương cực tiểu hóa $$-\ln \mathcal{L}(\theta)$$:
   - Giả định nhiễu Gaussian $$\implies$$ NLL trở thành **MSE**.
   - Giả định nhãn Bernoulli/Categorical $$\implies$$ NLL trở thành **Cross-Entropy**.

4. **Từ MLE đến MAP (Maximum A Posteriori) và Regularization**:
   Khi nhân thêm phân phối tiên nghiệm $$P(\theta)$$ lên tham số (ví dụ Gaussian $$\mathcal{N}(0, \tau^2)$$), âm log hậu nghiệm tự động sinh ra số hạng phạt $$\lambda \|\theta\|_2^2$$—chính là **L2 Regularization (Weight Decay)**!

### Giải thích: Tự tay suy ra công thức MLE cho 10 lần tung đồng xu và liên hệ Cross-Entropy

Giả sử bạn tung một đồng xu lạ 10 lần và thu được 7 lần Ngửa ($$y=1$$), 3 lần Sấp ($$y=0$$). Ta muốn tìm xác suất ra Ngửa $$p \in (0, 1)$$ hợp lý nhất:

1. **Lập hàm hợp lý**:
   $$\mathcal{L}(p) = p^7 (1 - p)^3.$$
2. **Lấy Log-Likelihood**:
   $$\ell(p) = \ln \mathcal{L}(p) = 7\ln p + 3\ln(1 - p).$$
   (Chú ý rằng $$-\frac{1}{10}\ell(p) = -\big[0.7\ln p + 0.3\ln(1-p)\big]$$ chính là **Binary Cross-Entropy**!)
3. **Đạo hàm theo $$p$$ và cho bằng 0**:
   $$\frac{d\ell}{dp} = \frac{7}{p} - \frac{3}{1 - p} = 0 \iff 7(1 - p) = 3p \iff 7 = 10p \iff \hat{p}_{\text{MLE}} = 0.7.$$
4. **Kiểm chứng số tại hai điểm**:
   - Tại $$p = 0.7$$: $$\ell(0.7) = 7\ln(0.7) + 3\ln(0.3) \approx 7(-0.3567) + 3(-1.2040) = -6.1086$$.
   - Tại $$p = 0.5$$: $$\ell(0.5) = 10\ln(0.5) \approx -6.9315 < -6.1086$$.

Kết quả $$\hat{p}_{\text{MLE}} = 0.7$$ vừa khớp trực giác tần suất, vừa chứng minh việc tối thiểu hóa Cross-Entropy chính là đi tìm tham số làm dữ liệu huấn luyện có khả năng xảy ra cao nhất!

## Hàm hợp lý (*Likelihood*) và Ước lượng hợp lý cực đại (MLE)

Trong xác suất thống kê, **hàm hợp lý** (*likelihood*) đo mức độ phù hợp của một mô hình hoặc một bộ tham số với dữ liệu đã quan sát. Dù được xây dựng từ công thức xác suất, hàm hợp lý không phải là một phân phối xác suất theo tham số, mà là hàm đánh giá xem từng giá trị tham số giải thích dữ liệu quan sát tốt đến mức nào.

### Định nghĩa hàm hợp lý

Hàm hợp lý của tham số $$\theta$$ ứng với dữ liệu quan sát $$x$$ được định nghĩa là xác suất (hoặc mật độ xác suất) thu được dữ liệu $$x$$ khi giả sử tham số $$\theta$$ là đúng, ký hiệu là $$L(\theta \mid x)$$:

$$L(\theta \mid x) = P(x \mid \theta)$$

Trong biểu thức này, dữ liệu $$x$$ được giữ cố định, còn $$\theta$$ là biến số thay đổi.

### Khác biệt giữa xác suất và hàm hợp lý

- **Xác suất** cố định tham số đã biết và mô tả khả năng xuất hiện của các kết quả dữ liệu trong tương lai.
- **Hàm hợp lý** cố định dữ liệu đã quan sát và so sánh mức độ phù hợp giữa các giá trị tham số khác nhau.

<div style="background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0;">
<strong>⚠️ Lưu ý quan trọng:</strong> Hàm hợp lý không phải là phân phối xác suất của tham số θ, vì tổng hoặc tích phân của nó theo θ không nhất thiết bằng 1.
</div>

### Ví dụ minh họa: Tung đồng xu

Giả sử bạn tung một đồng xu 5 lần và thu được 3 lần mặt sấp (S), 2 lần mặt ngửa (N), và muốn ước lượng xác suất ra mặt sấp $$p$$.

Hàm hợp lý của $$p$$ là xác suất quan sát đúng kết quả (3 sấp, 2 ngửa) ứng với tham số $$p$$:

$$L(p \mid \text{dữ liệu}) = P(\text{3 sấp, 2 ngửa} \mid p) = \binom{5}{3} p^3 (1-p)^2$$

trong đó:
- $$\binom{5}{3}$$ là hệ số nhị thức,
- $$p^3$$ là xác suất cho 3 lần sấp,
- $$(1-p)^2$$ là xác suất cho 2 lần ngửa.

**Thử tính với hai giá trị của $$p$$:**

Với $$p = 0.6$$:
$$L(0.6 \mid \text{dữ liệu}) = \binom{5}{3} (0.6)^3 (0.4)^2 = 10 \times 0.216 \times 0.16 = 0.3456$$

Với $$p = 0.5$$:
$$L(0.5 \mid \text{dữ liệu}) = \binom{5}{3} (0.5)^3 (0.5)^2 = 10 \times 0.125 \times 0.25 = 0.3125$$

Giá trị hàm hợp lý tại $$p = 0.6$$ cao hơn tại $$p = 0.5$$, cho thấy $$p = 0.6$$ giải thích dữ liệu quan sát tốt hơn.

### Minh họa tương tác: Tung đồng xu

<div id="coin-flip-example" style="border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px;">
    <h4>Phép thử tung đồng xu</h4>
    
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

### Ước lượng hợp lý cực đại (MLE)

**Ước lượng hợp lý cực đại** (*Maximum Likelihood Estimation — MLE*) là phương pháp chọn giá trị tham số $$\theta$$ sao cho hàm hợp lý $$L(\theta \mid x)$$ đạt cực đại:

$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta \mid x)$$

Trong thực tế, vì hàm logarit đơn điệu tăng, ta thường cực đại hóa **log-likelihood** để chuyển tích thành tổng và giúp việc tính đạo hàm thuận tiện hơn:

$$\hat{\theta}_{MLE} = \arg\max_{\theta} \log L(\theta \mid x)$$

### Ví dụ MLE: Phân phối chuẩn

Giả sử ta có $$n$$ quan sát độc lập $$x_1, x_2, \ldots, x_n$$ lấy từ phân phối chuẩn $$N(\mu, \sigma^2)$$.

**Hàm hợp lý:**
$$L(\mu, \sigma^2 \mid x_1, \ldots, x_n) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

**Log-likelihood:**
$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Tìm ước lượng MLE:**

Lấy đạo hàm riêng theo $$\mu$$ và cho bằng 0:
$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

Giải phương trình trên thu được: $$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$.

Làm tương tự với $$\sigma^2$$:
$$\hat{\sigma^2}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

### Minh họa tương tác: MLE cho phân phối chuẩn

<div id="normal-mle-example" style="border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px;">
    <h4>MLE cho phân phối chuẩn</h4>
    
    <div style="margin: 15px 0;">
        <button id="generate-data" style="background: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer;">Tạo dữ liệu mới</button>
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
        <div><strong>Log-likelihood cực đại:</strong> <span id="max-ll">-72.1</span></div>
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

### Các tính chất của MLE

1. **Tính nhất quán (*Consistency*):** khi kích thước mẫu tăng lên vô hạn, ước lượng MLE hội tụ về giá trị đúng của tham số.
2. **Tính chuẩn tiệm cận (*Asymptotic Normality*):** với kích thước mẫu đủ lớn, phân phối của ước lượng MLE xấp xỉ phân phối chuẩn.
3. **Tính hiệu quả tiệm cận (*Asymptotic Efficiency*):** MLE tiệm cận cận dưới Cramér–Rao, tức đạt phương sai nhỏ nhất trong lớp các ước lượng không chệch.
4. **Tính bất biến (*Invariance*):** nếu $$\hat{\theta}$$ là MLE của $$\theta$$, thì $$g(\hat{\theta})$$ là MLE của $$g(\theta)$$ với mọi hàm $$g$$ khả nghịch.

### Ứng dụng của MLE

#### 1. Ước lượng tham số

MLE được dùng phổ biến để ước lượng tham số cho các mô hình thống kê:
- Phân phối chuẩn: ước lượng kỳ vọng và phương sai.
- Phân phối Poisson: ước lượng tham số tốc độ $$\lambda$$.
- Hồi quy tuyến tính: ước lượng hệ số hồi quy.

#### 2. So sánh mô hình

Hàm hợp lý là cơ sở của nhiều tiêu chí đánh giá và so sánh mô hình:
- **Kiểm định tỉ số hợp lý (*Likelihood Ratio Test*):** so sánh hai mô hình lồng nhau.
- **AIC / BIC:** các tiêu chí lựa chọn mô hình dựa trên log-likelihood và số lượng tham số.

#### 3. Thống kê Bayes

Trong suy luận Bayes, hàm hợp lý kết hợp với phân phối tiên nghiệm (*prior*) để tạo thành phân phối hậu nghiệm (*posterior*):

$$P(\theta \mid x) \propto L(\theta \mid x) \cdot P(\theta)$$

### Liên hệ với tối ưu hóa

MLE phát biểu trực tiếp dưới dạng một bài toán tối ưu hóa:

$$\max_{\theta} L(\theta \mid x) \quad \text{hoặc} \quad \max_{\theta} \log L(\theta \mid x)$$

Bài toán này thường rơi vào các lớp sau:
- **Tối ưu hóa không ràng buộc:** khi tham số không chịu điều kiện phụ.
- **Tối ưu hóa có ràng buộc:** khi tham số phải thỏa mãn các miền điều kiện nhất định.
- **Bài toán lồi:** nhiều hàm log-likelihood là hàm lõm (tương đương âm log-likelihood là hàm lồi), bảo đảm nghiệm tối ưu tìm được là tối ưu toàn cục.

<div style="background: #e8f4fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0;">
<strong>💡 Liên hệ với tối ưu hóa lồi:</strong> Nhiều bài toán MLE dẫn đến bài toán tối ưu hóa lồi, đặc biệt trong họ phân phối mũ. Nhờ đó, cực trị tìm được là tối ưu toàn cục và các thuật toán tối ưu hội tụ đáng tin cậy hơn.
</div>

### Ví dụ thực tế: Hồi quy logistic

Trong hồi quy logistic, xác suất của biến nhị phân được mô hình hóa như sau:

$$P(y = 1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

**Log-likelihood:**
$$\ell(\beta_0, \beta_1) = \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

Đây là bài toán tối ưu hóa lồi (cực đại hóa hàm lõm theo $$\beta_0, \beta_1$$), có thể giải bằng gradient descent hoặc phương pháp Newton.

### Tóm tắt

- **Hàm hợp lý (*likelihood*)** đo mức độ phù hợp của tham số với dữ liệu quan sát.
- **MLE** tìm tham số cực đại hóa hàm hợp lý (hoặc log-likelihood).
- MLE có các tính chất thống kê tốt: nhất quán, hiệu quả tiệm cận và chuẩn tiệm cận.
- MLE dẫn trực tiếp đến các bài toán tối ưu hóa, trong đó nhiều bài toán thuộc lớp tối ưu lồi.
- Phương pháp này được ứng dụng rộng rãi trong thống kê, học máy và khoa học dữ liệu.

<div style="background: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0;">
<strong>🎯 Điểm quan trọng:</strong> MLE là cầu nối trực tiếp giữa thống kê và tối ưu hóa, giúp ta hiểu cách ước lượng tham số mô hình thông qua các phương pháp tối ưu — đặc biệt là tối ưu hóa lồi.
</div>

## Video tham khảo

- **Xem trước:** [In Statistics, Probability is not Likelihood.](https://www.youtube.com/watch?v=pYxNSUDSFH4) — StatQuest with Josh Starmer, tiếng Anh. Làm rõ vì sao hàm hợp lý không phải là phân phối xác suất của tham số, đúng với cảnh báo ở đầu bài.
- [Maximum Likelihood, clearly explained!!!](https://www.youtube.com/watch?v=XepXtl9YKwc) — StatQuest with Josh Starmer, tiếng Anh. Giải thích trực quan ước lượng hợp lý cực đại (argmax của hàm hợp lý), đồng bộ với ví dụ tung đồng xu trong bài.
- [Maximum Likelihood for the Binomial Distribution, Clearly Explained!!!](https://www.youtube.com/watch?v=4KKV9yZCoM4) — StatQuest with Josh Starmer, tiếng Anh. Trình bày từng bước tính hàm hợp lý nhị thức, tương ứng trực tiếp với ví dụ tung đồng xu 5 lần.
