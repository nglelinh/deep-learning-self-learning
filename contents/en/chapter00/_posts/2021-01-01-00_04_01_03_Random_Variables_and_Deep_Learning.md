---
layout: post
title: 00-04-01-03 Random Variables and Deep Learning
chapter: '00'
order: 18
owner: AI Assistant
lang: en
categories:
- chapter00
---

> *Every weight initialization, every dropout mask, every mini-batch gradient, and every latent vector in a VAE is a random variable. Understanding expectation, variance, and linearity of expectation lets you predict whether activations will blow up across 100 layers before you even run a single epoch.*

## Background Knowledge Map: Random Variables and Moment Propagation

1. **Discrete vs. Continuous Random Variables**: Describes quantities via probability mass functions (PMFs) for discrete tokens/masks and probability density functions (PDFs) for continuous weights/activations.
2. **Linearity of Expectation (Always Holds)**: Even when variables are correlated, $$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$, which guarantees that mini-batch SGD gradients are unbiased estimators of the full-dataset gradient.
3. **Variance Scaling (The Key to Initialization)**: For independent variables, $$\text{Var}(aX + bY) = a^2\text{Var}(X) + b^2\text{Var}(Y)$$; this quadratic scaling by $$a^2$$ is the exact derivation behind Xavier/Glorot and He initialization.
4. **Law of the Unconscious Statistician (LOTUS)**: Computes $$\mathbb{E}[g(X)]$$ directly from the distribution of $$X$$ without deriving the distribution of $$g(X)$$, enabling Monte Carlo gradient estimation in stochastic networks.

### Step-by-Step Worked Example (Read Slowly): Deriving Variance Preservation for a Linear Layer

Consider a single neuron computing $$z = \sum_{i=1}^{n} w_i x_i$$ with $$n = 100$$ inputs, where inputs $$x_i$$ and weights $$w_i$$ are mutually independent with mean $$0$$, input variance $$\text{Var}(x_i) = 1$$, and weight variance $$\text{Var}(w_i) = \sigma_w^2$$.

1. **Compute the Mean of Each Product $$w_i x_i$$**:
   By independence, $$\mathbb{E}[w_i x_i] = \mathbb{E}[w_i]\mathbb{E}[x_i] = 0 \times 0 = 0$$.
   By linearity of expectation, $$\mathbb{E}[z] = \sum_{i=1}^{100} 0 = 0$$.
2. **Compute the Variance of Each Product $$w_i x_i$$**:
   Since both means are zero:
   $$\text{Var}(w_i x_i) = \mathbb{E}[(w_i x_i)^2] - 0^2 = \mathbb{E}[w_i^2]\mathbb{E}[x_i^2] = \text{Var}(w_i)\text{Var}(x_i) = \sigma_w^2 \cdot 1 = \sigma_w^2$$
3. **Sum Variances Across All $$n = 100$$ Independent Terms**:
   $$\text{Var}(z) = \sum_{i=1}^{100} \text{Var}(w_i x_i) = 100\,\sigma_w^2$$
4. **Solve for Signal-Preserving Weight Scale**:
   - If you naively initialize $$\sigma_w = 1$$, then $$\text{Var}(z) = 100$$ after just one layer (and $$100^{10} = 10^{20}$$ after 10 layers!).
   - Setting $$\text{Var}(z) = 1$$ requires $$100\,\sigma_w^2 = 1 \implies \sigma_w = \frac{1}{\sqrt{100}} = 0.1$$. That exact calculation is Xavier/LeCun initialization!

### 6. Random Variables

A **random variable** X is a function that assigns a real number to each outcome in the sample space:
$$X: \Omega \rightarrow \mathbb{R}$$

#### Types of Random Variables:

**Discrete**: Takes countable values (e.g., number of heads in coin flips)
- Probability Mass Function (PMF): $$P(X = x)$$

**Continuous**: Takes uncountable values (e.g., height, weight)
- Probability Density Function (PDF): $$f_X(x)$$
- $$P(a \leq X \leq b) = \int_a^b f_X(x) dx$$

### 7. Connection to Deep Learning

Probability theory connects to deep-learning in several ways:

#### Maximum Likelihood Estimation
Find parameters θ that maximize the likelihood:
$$\hat{\theta} = \arg\max_\theta P(\text{data}|\theta)$$

#### Expected Value Deep Learning
Minimize expected loss:
$$\min_\theta \mathbb{E}[L(Y, f(X; \theta))]$$

#### Bayesian Deep Learning
Use probability distributions to model uncertainty in objective functions and guide search for optimal solutions.

<div id="deep-learning-connection" style="border: 2px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f0f8ff;">
    <h4 style="text-align: center; color: #333;">Probability in Deep Learning Example</h4>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
        <div style="flex: 1; min-width: 400px;">
            <canvas id="deep-learningCanvas" width="400" height="300" style="border: 1px solid #ccc; background: white;"></canvas>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                <strong>MLE Example:</strong> Finding the parameter μ that maximizes likelihood of observed data from Normal(μ, 1).
            </p>
        </div>
        
        <div style="flex: 1; min-width: 250px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="margin-top: 0; color: #444;">MLE Demo</h5>
                
                <div style="margin-bottom: 15px;">
                    <label for="true-mu-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">True μ: <span id="true-mu-value">2.0</span></label>
                    <input type="range" id="true-mu-slider" min="-2" max="4" step="0.1" value="2.0" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label for="sample-size-slider" style="display: block; margin-bottom: 5px; font-weight: bold;">Sample Size: <span id="sample-size-value">20</span></label>
                    <input type="range" id="sample-size-slider" min="5" max="100" step="5" value="20" style="width: 100%;">
                </div>
                
                <button id="generate-mle-data" style="width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 15px;">Generate Data & Find MLE</button>
                
                <div id="mle-results" style="font-family: monospace; font-size: 12px; line-height: 1.4; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    <div><strong>Results:</strong></div>
                    <div>True μ: <span id="display-true-mu">2.000</span></div>
                    <div>Sample mean: <span id="sample-mean">--</span></div>
                    <div>MLE estimate: <span id="mle-estimate">--</span></div>
                    <div>Error: <span id="mle-error">--</span></div>
                </div>
            </div>
        </div>
    </div>
</div>

### Key Takeaways

1. **Foundation**: Probability axioms provide the mathematical foundation for reasoning about uncertainty
2. **Conditional Probability**: Essential for updating beliefs with new information
3. **Independence**: Simplifies calculations and modeling assumptions
4. **Random Variables**: Bridge between abstract probability and concrete applications
5. **Deep Learning Connection**: Many deep-learning problems arise from probabilistic modeling

Understanding these basics prepares you for more advanced topics like Bayesian inference, maximum likelihood estimation, and stochastic deep-learning that are central to modern machine learning and data science.

<script>
// MLE Deep Learning Demo
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
            this.ctx.fillText('Click "Generate Data & Find MLE" to start', this.width / 2, this.height / 2);
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
        this.ctx.fillText('--- True μ', 10, 50);
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

- **xem trước** [MaDL - Random Variables and Probability Distributions](https://www.youtube.com/watch?v=bvBukLjHZh8) — Tübingen Machine Learning, English. A random variable is a function on the sample space, which is how the lesson treats data and predictions.
- [Random Variables and Probability Distributions](https://www.youtube.com/watch?v=lHCpYeFvTs0) — Dr Nic's Maths and Stats, English. A short concrete pass over discrete and continuous random variables before the distribution catalog.
- [MaDL - Expectation, Variance and Covariance](https://www.youtube.com/watch?v=kmRm3XIND8M) — Tübingen Machine Learning, English. Expectation and variance are the summaries the lesson uses when a network's output is random.
