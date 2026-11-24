---
layout: post
title: 00-04-01-01 Sample Spaces and Probability Axioms
chapter: '00'
order: 16
owner: AI Assistant
lang: en
categories:
- chapter00
---

> *In 1933, Andrey Kolmogorov reduced all of probability theory to just three deceptively simple axioms. Every softmax output layer in deep learning is explicitly engineered so that its vector of class scores obeys those exact three axioms.*

## Background Knowledge Map: Kolmogorov's Axioms and Softmax Validity

1. **Sample Space $$\Omega$$ and Event Space $$\mathcal{F}$$**: Defines the exhaustive universe of mutually exclusive outcomes (e.g., the 10 digit classes $$\{0, \dots, 9\}$$ in MNIST or the 50,000 tokens in a vocabulary).
2. **Non-Negativity & Normalization (Axioms 1 & 2)**: Demands $$P(A) \geq 0$$ and $$P(\Omega) = 1$$; in neural classifiers, exponentiating logits guarantees positivity and dividing by their sum enforces normalization.
3. **Countable Additivity (Axiom 3)**: States that for disjoint events, the probability of their union is the sum of their probabilities—letting us compute top-$$k$$ accuracy or group-level class probabilities by simple summation.
4. **Derived Rules (Complement & Inclusion–Exclusion)**: Immediately gives $$P(A^c) = 1 - P(A)$$ and $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$, preventing double-counting in multi-label classification.

### Step-by-Step Worked Example (Read Slowly): Verifying Kolmogorov's Axioms on a Softmax Output

Suppose a 3-class image classifier outputs raw logits $$\mathbf{z} = [0, \ln 2, \ln 5]$$ for classes $$\{\text{cat}, \text{dog}, \text{bird}\}$$.

1. **Exponentiate Logits (Enforcing Axiom 1: Non-Negativity)**:
   $$e^0 = 1, \quad e^{\ln 2} = 2, \quad e^{\ln 5} = 5$$
   All values are strictly positive.
2. **Normalize by Total Partition Sum $$Z$$ (Enforcing Axiom 2: Normalization)**:
   $$Z = 1 + 2 + 5 = 8$$
   $$P(\text{cat}) = \frac{1}{8} = 0.125, \quad P(\text{dog}) = \frac{2}{8} = 0.250, \quad P(\text{bird}) = \frac{5}{8} = 0.625$$
   Sum check: $$0.125 + 0.250 + 0.625 = 1.000 = P(\Omega)$$.
3. **Apply Axiom 3 (Additivity) and Complement Rule**:
   - Probability of a mammal ($$\{\text{cat}\} \cup \{\text{dog}\}$$, disjoint):
     $$P(\text{mammal}) = P(\text{cat}) + P(\text{dog}) = 0.125 + 0.250 = 0.375$$
   - Complement check:
     $$P(\text{not mammal}) = 1 - 0.375 = 0.625 = P(\text{bird})$$
   Every softmax layer is an axiomatic probability projector.

## Basic Probability Theory

Probability theory provides the mathematical framework for reasoning about uncertainty, which is fundamental to many deep-learning problems in machine learning and data science.

### 1. Sample Space and Events

**Sample Space (Ω)**: The set of all possible outcomes of an experiment.

**Event (A)**: A subset of the sample space representing a collection of outcomes.

**Examples:**
- Coin flip: Ω = {H, T}
- Die roll: Ω = {1, 2, 3, 4, 5, 6}
- Continuous: Ω = [0, 1] for uniform random variable

<div id="sample-space-demo" style="border: 2px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f9f9f9;">
    <h4 style="text-align: center; color: #333;">Interactive Sample Space Visualization</h4>
    
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
        <div style="flex: 1; min-width: 400px;">
            <canvas id="sampleSpaceCanvas" width="400" height="300" style="border: 1px solid #ccc; background: white;"></canvas>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                <strong>Visualization:</strong> Click to generate random samples. Different colors represent different events.
            </p>
        </div>
        
        <div style="flex: 1; min-width: 250px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="margin-top: 0; color: #444;">Experiment Type</h5>
                
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="experiment" value="coin" checked> Coin Flip
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="experiment" value="dice"> Dice Roll
                    </label>
                    <label style="display: block; margin-bottom: 10px;">
                        <input type="radio" name="experiment" value="uniform"> Uniform [0,1]
                    </label>
                </div>
                
                <button id="generate-sample" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 10px;">Generate Sample</button>
                <button id="clear-samples" style="width: 100%; padding: 8px; background: #6c757d; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 15px;">Clear</button>
                
                <div id="sample-stats" style="font-family: monospace; font-size: 12px; line-height: 1.4; background: #f8f9fa; padding: 10px; border-radius: 4px;">
                    <div><strong>Statistics:</strong></div>
                    <div>Total samples: <span id="total-samples">0</span></div>
                    <div>Event A: <span id="event-a-count">0</span></div>
                    <div>Event B: <span id="event-b-count">0</span></div>
                    <div>P(A) ≈ <span id="prob-a">0.000</span></div>
                    <div>P(B) ≈ <span id="prob-b">0.000</span></div>
                </div>
            </div>
        </div>
    </div>
</div>

### 2. Probability Axioms (Kolmogorov Axioms)

For any probability measure P, the following axioms must hold:

#### Axiom 1: Non-negativity
$$P(A) \geq 0 \text{ for all events } A$$

#### Axiom 2: Normalization
$$P(\Omega) = 1$$

#### Axiom 3: Countable Additivity
For mutually exclusive events $$A_1, A_2, \ldots$$:
$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

### 3. Basic Properties and Rules

#### Complement Rule
$$P(A^c) = 1 - P(A)$$

#### Addition Rule
For any two events A and B:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

#### Multiplication Rule
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
                    value: Math.random() < 0.5 ? 'H' : 'T',
                    x: Math.random() * (this.width - 40) + 20,
                    y: Math.random() * (this.height - 40) + 20,
                    eventA: Math.random() < 0.5, // Event A: Heads
                    eventB: Math.random() < 0.3  // Event B: Lucky flip
                };
                break;
            case 'dice':
                const diceValue = Math.floor(Math.random() * 6) + 1;
                sample = {
                    value: diceValue,
                    x: Math.random() * (this.width - 40) + 20,
                    y: Math.random() * (this.height - 40) + 20,
                    eventA: diceValue >= 4, // Event A: 4, 5, or 6
                    eventB: diceValue % 2 === 0 // Event B: Even
                };
                break;
            case 'uniform':
                const uniformValue = Math.random();
                sample = {
                    value: uniformValue.toFixed(3),
                    x: uniformValue * (this.width - 40) + 20,
                    y: Math.random() * (this.height - 40) + 20,
                    eventA: uniformValue > 0.5, // Event A: > 0.5
                    eventB: uniformValue < 0.7  // Event B: < 0.7
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
            if (sample.eventA && sample.eventB) color = '#9c27b0'; // Both events
            else if (sample.eventA) color = '#2196f3'; // Event A only
            else if (sample.eventB) color = '#f44336'; // Event B only
            
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
        this.ctx.fillText('Legend:', 10, 20);
        
        this.ctx.fillStyle = '#2196f3';
        this.ctx.beginPath();
        this.ctx.arc(20, 35, 4, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.fillStyle = '#000';
        this.ctx.fillText('Event A only', 30, 38);
        
        this.ctx.fillStyle = '#f44336';
        this.ctx.beginPath();
        this.ctx.arc(20, 50, 4, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.fillStyle = '#000';
        this.ctx.fillText('Event B only', 30, 53);
        
        this.ctx.fillStyle = '#9c27b0';
        this.ctx.beginPath();
        this.ctx.arc(20, 65, 4, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.fillStyle = '#000';
        this.ctx.fillText('Both A and B', 30, 68);
    }
}

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

- **xem trước** [L01.4 Probability Axioms](https://www.youtube.com/watch?v=pA83XtLeVig) — MIT OpenCourseWare, English. Sample space, events, and the three axioms are exactly this lesson.
- [The Main Ideas behind Probability Distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc) — StatQuest with Josh Starmer, English. Shows how a probability measure on that sample space becomes a distribution.
- [Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM) — 3Blue1Brown, English. Bayes' rule is an identity inside the same axioms, useful once events are no longer independent.
