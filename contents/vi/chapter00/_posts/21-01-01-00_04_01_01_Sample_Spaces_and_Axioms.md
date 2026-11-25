---
layout: post
title: 00-04-01-01 Không Gian Mẫu và Tiên Đề Xác Suất
chapter: '00'
order: 16
owner: AI Assistant
lang: vi
categories:
- chapter00
---

Năm 1933, nhà toán học Andrey Kolmogorov đã làm một điều phi thường: chỉ bằng **ba tiên đề ngắn gọn**, ông biến toàn bộ lĩnh vực xác suất—vốn trước đó chỉ xoay quanh các bài toán cờ bạc tung xúc xắc—thành một hệ thống giải tích chặt chẽ như hình học Euclid. Ngày nay, mỗi khi tầng Softmax ở cuối mạng phân loại ảnh ép đầu ra thành các con số nằm trong $$[0, 1]$$ và có tổng đúng bằng $$1$$, nó đang tôn trọng chính xác ba tiên đề Kolmogorov ấy. Bài học này xây dựng nền móng không gian mẫu, biến cố và hệ tiên đề xác suất để bạn hiểu rõ ngôn ngữ gốc của mọi mô hình phân loại.

## Kiến thức cần nhớ: Ba trụ cột của Không gian Mẫu và Tiên đề Kolmogorov

1. **Không gian mẫu ($$\Omega$$) và Biến cố ($$A \subseteq \Omega$$)**:
   Liệt kê toàn bộ kết cục có thể xảy ra (từ 10 lớp chữ số MNIST đến 50{,}000 token từ vựng của LLM); mỗi biến cố là một tập con chứa các kết cục ta quan tâm.

2. **Ba tiên đề Kolmogorov (Không âm, Chuẩn hóa, Cộng tính)**:
   Mọi xác suất đều thỏa $$P(A) \ge 0$$, $$P(\Omega) = 1$$, và với các biến cố đôi một xung khắc ($$A_i \cap A_j = \emptyset$$), $$P(\bigcup_i A_i) = \sum_i P(A_i)$$.

3. **Các hệ quả trực tiếp (Phần bù và Luật cộng tổng quát)**:
   Tính xác suất "không xảy ra" bằng $$P(A^c) = 1 - P(A)$$ và xử lý các biến cố chồng lấn bằng $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$, tránh lỗi đếm trùng cực kỳ phổ biến khi phân tích sai số mô hình.

### Giải thích: Kiểm tra tiên đề Kolmogorov trên đầu ra Softmax và lỗi đếm trùng

Giả sử một bộ phân loại ảnh 3 lớp $$\Omega = \{\text{Mèo}, \text{Chó}, \text{Chim}\}$$ xuất ra vector xác suất $$P(\text{Mèo}) = 0.65$$, $$P(\text{Chó}) = 0.25$$, $$P(\text{Chim}) = 0.10$$:

1. **Kiểm tra ba tiên đề Kolmogorov**:
   - Không âm: cả ba số đều $$\ge 0$$.
   - Chuẩn hóa: $$0.65 + 0.25 + 0.10 = 1.00 = P(\Omega)$$.
   - Cộng tính xung khắc: biến cố $$A = \{\text{Động vật bốn chân}\} = \{\text{Mèo}, \text{Chó}\}$$ có $$P(A) = 0.65 + 0.25 = 0.90$$, và phần bù $$P(A^c) = 1 - 0.90 = 0.10 = P(\text{Chim})$$.
2. **Xử lý hai biến cố *không* xung khắc trên bộ kiểm thử**:
   Trong 100 ảnh test, có 20 ảnh bị mờ ($$P(M) = 0.20$$), 15 ảnh thiếu sáng ($$P(T) = 0.15$$), và 6 ảnh vừa mờ vừa thiếu sáng ($$P(M \cap T) = 0.06$$).
   - Xác suất ảnh bị mờ **hoặc** thiếu sáng là:
     $$P(M \cup T) = 0.20 + 0.15 - 0.06 = 0.29 \quad (\text{không phải } 0.35!).$$
   - Xác suất ảnh sạch (không mờ, không thiếu sáng): $$1 - 0.29 = 0.71$$.

## Lý thuyết xác suất cơ bản

Lý thuyết xác suất cung cấp khung toán học để suy luận về sự bất định — nền tảng của nhiều bài toán tối ưu hóa trong học máy và khoa học dữ liệu.

### 1. Không gian mẫu và biến cố

**Không gian mẫu ($$\Omega$$):** tập hợp tất cả các kết quả có thể xảy ra của một phép thử ngẫu nhiên.

**Biến cố ($$A$$):** một tập con của không gian mẫu, đại diện cho một nhóm kết quả cụ thể.

**Ví dụ:**
- Tung đồng xu: $$\Omega = \{S, N\}$$
- Tung xúc xắc: $$\Omega = \{1, 2, 3, 4, 5, 6\}$$
- Phép thử liên tục: $$\Omega = [0, 1]$$ đối với biến ngẫu nhiên phân phối đều

<div id="sample-space-demo" style="border: 2px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f9f9f9;">
    <h4 style="text-align: center; color: #333;">Trực quan hóa không gian mẫu tương tác</h4>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
        <div style="flex: 1; min-width: 400px;">
            <canvas id="sampleSpaceCanvas" width="400" height="300" style="border: 1px solid #ccc; background: white;"></canvas>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                <strong>Trực quan hóa:</strong> Nhấp để sinh mẫu ngẫu nhiên. Mỗi màu biểu thị một biến cố khác nhau.
            </p>
        </div>
        
        <div style="flex: 1; min-width: 250px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="margin-top: 0; color: #444;">Loại phép thử</h5>
                
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="experiment" value="coin" checked> Tung đồng xu
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="experiment" value="dice"> Tung xúc xắc
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="experiment" value="uniform"> Phân phối đều [0,1]
                    </label>
                </div>
                
                <button id="generate-sample" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 10px;">Tạo mẫu</button>
                <button id="clear-samples" style="width: 100%; padding: 8px; background: #6c757d; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 15px;">Xóa</button>
                
                <div id="sample-stats" style="font-family: monospace; font-size: 12px; line-height: 1.4; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    <div><strong>Thống kê:</strong></div>
                    <div>Tổng mẫu: <span id="total-samples">0</span></div>
                    <div>Biến cố A: <span id="event-a-count">0</span></div>
                    <div>Biến cố B: <span id="event-b-count">0</span></div>
                    <div>P(A) ≈ <span id="prob-a">0.000</span></div>
                    <div>P(B) ≈ <span id="prob-b">0.000</span></div>
                </div>
            </div>
        </div>
    </div>
</div>

### 2. Tiên đề xác suất (Tiên đề Kolmogorov)

Mọi độ đo xác suất $$P$$ đều phải thỏa mãn ba tiên đề sau:

#### Tiên đề 1: Tính không âm
$$P(A) \geq 0 \text{ với mọi biến cố } A$$

#### Tiên đề 2: Chuẩn hóa
$$P(\Omega) = 1$$

#### Tiên đề 3: Tính cộng đếm được
Với dãy các biến cố đôi một xung khắc $$A_1, A_2, \ldots$$:
$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

### 3. Tính chất và quy tắc cơ bản

#### Quy tắc phần bù
$$P(A^c) = 1 - P(A)$$

#### Quy tắc cộng
Với hai biến cố $$A$$ và $$B$$ bất kỳ:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

#### Quy tắc nhân
$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

<script>
class SampleSpaceDemo {
    constructor() {
        this.canvas = document.getElementById('sampleSpaceCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        this.samples = [];
        this.experimentType = 'coin';
        
        this.setupControls();
        this.draw();
    }
    
    setupControls() {
        const radios = document.querySelectorAll('input[name="experiment"]');
        const generateBtn = document.getElementById('generate-sample');
        const clearBtn = document.getElementById('clear-samples');
        
        radios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.experimentType = e.target.value;
                this.samples = [];
                this.updateStats();
                this.draw();
            });
        });
        
        generateBtn.addEventListener('click', () => this.generateSample());
        clearBtn.addEventListener('click', () => {
            this.samples = [];
            this.updateStats();
            this.draw();
        });
        
        this.canvas.addEventListener('click', () => this.generateSample());
    }
    
    generateSample() {
        let sample;
        
        switch(this.experimentType) {
            case 'coin':
                sample = {
                    value: Math.random() < 0.5 ? 'S' : 'N',
                    x: Math.random() * (this.width - 40) + 20,
                    y: Math.random() * (this.height - 40) + 20,
                    eventA: Math.random() < 0.5, // Biến cố A: Sấp
                    eventB: Math.random() < 0.3  // Biến cố B: May mắn
                };
                break;
            case 'dice':
                const diceValue = Math.floor(Math.random() * 6) + 1;
                sample = {
                    value: diceValue,
                    x: Math.random() * (this.width - 40) + 20,
                    y: Math.random() * (this.height - 40) + 20,
                    eventA: diceValue >= 4, // Biến cố A: 4, 5, hoặc 6
                    eventB: diceValue % 2 === 0 // Biến cố B: Chẵn
                };
                break;
            case 'uniform':
                const uniformValue = Math.random();
                sample = {
                    value: uniformValue.toFixed(3),
                    x: uniformValue * (this.width - 40) + 20,
                    y: Math.random() * (this.height - 40) + 20,
                    eventA: uniformValue > 0.5, // Biến cố A: > 0.5
                    eventB: uniformValue < 0.7  // Biến cố B: < 0.7
                };
                break;
        }
        
        this.samples.push(sample);
        this.updateStats();
        this.draw();
    }
    
    updateStats() {
        const total = this.samples.length;
        const eventACount = this.samples.filter(s => s.eventA).length;
        const eventBCount = this.samples.filter(s => s.eventB).length;
        
        document.getElementById('total-samples').textContent = total;
        document.getElementById('event-a-count').textContent = eventACount;
        document.getElementById('event-b-count').textContent = eventBCount;
        document.getElementById('prob-a').textContent = total > 0 ? (eventACount / total).toFixed(3) : '0.000';
        document.getElementById('prob-b').textContent = total > 0 ? (eventBCount / total).toFixed(3) : '0.000';
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // Draw background
        this.ctx.fillStyle = '#f8f9fa';
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // Draw samples
        this.samples.forEach(sample => {
            // Determine color based on events
            let color = '#666';
            if (sample.eventA && sample.eventB) color = '#9c27b0'; // Cả hai biến cố
            else if (sample.eventA) color = '#2196f3'; // Chỉ biến cố A
            else if (sample.eventB) color = '#f44336'; // Chỉ biến cố B
            
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(sample.x, sample.y, 5, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Draw value
            this.ctx.fillStyle = '#000';
            this.ctx.font = '10px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(sample.value, sample.x, sample.y - 8);
        });
        
        // Draw legend
        this.ctx.fillStyle = '#000';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText('Chú thích:', 10, 20);
        
        this.ctx.fillStyle = '#2196f3';
        this.ctx.beginPath();
        this.ctx.arc(20, 35, 4, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.fillStyle = '#000';
        this.ctx.fillText('Chỉ biến cố A', 30, 38);
        
        this.ctx.fillStyle = '#f44336';
        this.ctx.beginPath();
        this.ctx.arc(20, 50, 4, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.fillStyle = '#000';
        this.ctx.fillText('Chỉ biến cố B', 30, 53);
        
        this.ctx.fillStyle = '#9c27b0';
        this.ctx.beginPath();
        this.ctx.arc(20, 65, 4, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.fillStyle = '#000';
        this.ctx.fillText('Cả A và B', 30, 68);
    }
}

// Conditional Probability Visualization

document.addEventListener('DOMContentLoaded', function() {
    new SampleSpaceDemo();
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

- **Xem trước:** [L01.4 Probability Axioms](https://www.youtube.com/watch?v=pA83XtLeVig) — MIT OpenCourseWare, tiếng Anh. Trình bày không gian mẫu, biến cố và ba tiên đề xác suất tương ứng trực tiếp với nội dung bài này.
- [The Main Ideas behind Probability Distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc) — StatQuest with Josh Starmer, tiếng Anh. Giải thích cách độ đo xác suất trên không gian mẫu hình thành nên một phân phối xác suất.
- [Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM) — 3Blue1Brown, tiếng Anh. Minh họa trực quan định lý Bayes — hệ quả trực tiếp của các tiên đề xác suất khi các biến cố phụ thuộc nhau.
