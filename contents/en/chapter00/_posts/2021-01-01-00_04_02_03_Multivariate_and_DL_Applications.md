---
layout: post
title: 00-04-02-03 Multivariate Distributions and Deep Learning
chapter: '00'
order: 22
owner: AI Assistant
lang: en
categories:
- chapter00
---

### 3. Multivariate Distributions

#### Multivariate Normal Distribution

Extension of the normal distribution to multiple dimensions.

**Parameters**: $$\boldsymbol{\mu} \in \mathbb{R}^d$$ (mean vector), $$\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$$ (covariance matrix, positive definite)

**PDF**: $$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**Properties**:
- Marginal distributions are normal
- Linear combinations are normal
- Conditional distributions are normal

<div id="multivariate-demo" style="border: 2px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f0f8ff;">
    <h4 style="text-align: center; color: #333;">Multivariate Normal Distribution</h4>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
        <div style="flex: 1; min-width: 400px;">
            <canvas id="multivariateCanvas" width="400" height="300" style="border: 1px solid #ccc; background: white;"></canvas>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                <strong>2D Visualization:</strong> Contour plot of bivariate normal distribution. Samples shown as dots.
            </p>
        </div>
        
        <div style="flex: 1; min-width: 250px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="margin-top: 0; color: #444;">Parameters</h5>
                
                <div style="margin-bottom: 15px;">
                    <label for="mu1-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">μ₁: <span id="mu1-value">0.0</span></label>
                    <input type="range" id="mu1-slider" min="-2" max="2" step="0.1" value="0" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="mu2-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">μ₂: <span id="mu2-value">0.0</span></label>
                    <input type="range" id="mu2-slider" min="-2" max="2" step="0.1" value="0" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="sigma1-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">σ₁: <span id="sigma1-value">1.0</span></label>
                    <input type="range" id="sigma1-slider" min="0.5" max="2" step="0.1" value="1.0" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="sigma2-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">σ₂: <span id="sigma2-value">1.0</span></label>
                    <input type="range" id="sigma2-slider" min="0.5" max="2" step="0.1" value="1.0" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="rho-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">ρ (correlation): <span id="rho-value">0.0</span></label>
                    <input type="range" id="rho-slider" min="-0.9" max="0.9" step="0.1" value="0" style="width: 100%;">
                </div>
                
                <button id="generate-samples" style="width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 15px;">Generate Samples</button>
                
                <div id="multivariate-stats" style="font-family: monospace; font-size: 12px; line-height: 1.4; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    <div><strong>Covariance Matrix:</strong></div>
                    <div>Σ₁₁: <span id="cov11">1.000</span></div>
                    <div>Σ₁₂: <span id="cov12">0.000</span></div>
                    <div>Σ₂₂: <span id="cov22">1.000</span></div>
                    <div>Det(Σ): <span id="det-cov">1.000</span></div>
                </div>
            </div>
        </div>
    </div>
</div>

### 4. Applications in Deep Learning

#### Maximum Likelihood Estimation
Many deep-learning problems involve finding parameters that maximize the likelihood of observed data under a specific distribution:

$$\hat{\theta} = \arg\max_\theta \prod_{i=1}^n f(x_i; \theta)$$

#### Bayesian Deep Learning
Prior distributions encode beliefs about parameters before seeing data:

$$p(\theta|data) \propto p(data|\theta) \cdot p(\theta)$$

#### Regularization
Distributions can be used as priors to regularize deep-learning problems:
- L2 regularization ↔ Gaussian prior
- L1 regularization ↔ Laplace prior

#### Stochastic Deep Learning
Distributions model noise and uncertainty in objective functions and constraints.

### Key Insights for Deep Learning

1. **Model Selection**: Choose distributions that match your data's characteristics
2. **Parameter Estimation**: Use MLE or Bayesian methods to estimate distribution parameters
3. **Uncertainty Quantification**: Distributions provide natural ways to quantify uncertainty
4. **Regularization**: Prior distributions can prevent overfitting
5. **Computational Efficiency**: Some distributions have closed-form solutions for common operations

Understanding these distributions and their properties is crucial for formulating and solving deep-learning problems in machine learning, statistics, and engineering applications.

<script>
class MultivariateDemo {
    constructor() {
        this.canvas = document.getElementById('multivariateCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        this.params = { mu1: 0, mu2: 0, sigma1: 1, sigma2: 1, rho: 0 };
        this.samples = [];
        
        this.setupControls();
        this.draw();
    }
    
    setupControls() {
        const sliders = ['mu1', 'mu2', 'sigma1', 'sigma2', 'rho'];
        
        sliders.forEach(slider => {
            const element = document.getElementById(slider + '-slider');
            element.addEventListener('input', (e) => {
                this.params[slider] = parseFloat(e.target.value);
                document.getElementById(slider + '-value').textContent = this.params[slider].toFixed(1);
                this.updateStats();
                this.draw();
            });
        });
        
        document.getElementById('generate-samples').addEventListener('click', () => {
            this.generateSamples();
            this.draw();
        });
        
        this.updateStats();
    }
    
    updateStats() {
        const cov11 = this.params.sigma1 * this.params.sigma1;
        const cov12 = this.params.rho * this.params.sigma1 * this.params.sigma2;
        const cov22 = this.params.sigma2 * this.params.sigma2;
        const det = cov11 * cov22 - cov12 * cov12;
        
        document.getElementById('cov11').textContent = cov11.toFixed(3);
        document.getElementById('cov12').textContent = cov12.toFixed(3);
        document.getElementById('cov22').textContent = cov22.toFixed(3);
        document.getElementById('det-cov').textContent = det.toFixed(3);
    }
    
    generateSamples() {
        this.samples = [];
        const n = 100;
        
        for (let i = 0; i < n; i++) {
            // Box-Muller transform
            const u1 = Math.random();
            const u2 = Math.random();
            const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
            
            // Transform to correlated normal
            const x1 = this.params.mu1 + this.params.sigma1 * z1;
            const x2 = this.params.mu2 + this.params.sigma2 * (this.params.rho * z1 + Math.sqrt(1 - this.params.rho * this.params.rho) * z2);
            
            this.samples.push([x1, x2]);
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
        
        // Draw contour ellipses
        const levels = [0.5, 1, 1.5, 2];
        const colors = ['#ff9999', '#ff6666', '#ff3333', '#ff0000'];
        
        levels.forEach((level, idx) => {
            this.ctx.strokeStyle = colors[idx];
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            
            const a = level * this.params.sigma1;
            const b = level * this.params.sigma2;
            const angle = 0.5 * Math.atan2(2 * this.params.rho * this.params.sigma1 * this.params.sigma2,
                                          this.params.sigma1 * this.params.sigma1 - this.params.sigma2 * this.params.sigma2);
            
            for (let i = 0; i <= 100; i++) {
                const t = 2 * Math.PI * i / 100;
                const x = a * Math.cos(t) * Math.cos(angle) - b * Math.sin(t) * Math.sin(angle) + this.params.mu1;
                const y = a * Math.cos(t) * Math.sin(angle) + b * Math.sin(t) * Math.cos(angle) + this.params.mu2;
                
                const plotX = marginX + (x + 4) / 8 * plotWidth;
                const plotY = this.height - marginY - (y + 4) / 8 * plotHeight;
                
                if (i === 0) {
                    this.ctx.moveTo(plotX, plotY);
                } else {
                    this.ctx.lineTo(plotX, plotY);
                }
            }
            this.ctx.stroke();
        });
        
        // Draw samples
        if (this.samples.length > 0) {
            this.ctx.fillStyle = '#2196f3';
            this.samples.forEach(([x1, x2]) => {
                const plotX = marginX + (x1 + 4) / 8 * plotWidth;
                const plotY = this.height - marginY - (x2 + 4) / 8 * plotHeight;
                
                if (plotX >= marginX && plotX <= this.width - marginX &&
                    plotY >= marginY && plotY <= this.height - marginY) {
                    this.ctx.beginPath();
                    this.ctx.arc(plotX, plotY, 2, 0, 2 * Math.PI);
                    this.ctx.fill();
                }
            });
        }
        
        // Draw mean point
        const meanX = marginX + (this.params.mu1 + 4) / 8 * plotWidth;
        const meanY = this.height - marginY - (this.params.mu2 + 4) / 8 * plotHeight;
        this.ctx.fillStyle = '#000';
        this.ctx.beginPath();
        this.ctx.arc(meanX, meanY, 4, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Labels
        this.ctx.fillStyle = '#000';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('X₁', this.width / 2, this.height - 10);
        
        this.ctx.save();
        this.ctx.translate(15, this.height / 2);
        this.ctx.rotate(-Math.PI / 2);
        this.ctx.fillText('X₂', 0, 0);
        this.ctx.restore();
    }
}

// Initialize when DOM is loaded

document.addEventListener('DOMContentLoaded', function() {
    new MultivariateDemo();
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
