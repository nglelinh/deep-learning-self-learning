---
layout: post
title: 04-01-02 Convolution Intuition and Examples
chapter: '04'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter04
---

## 3. Example / Intuition

### Example 1: Edge Detection (Hand-Crafted Filters)

Before deep learning, computer vision relied on hand-crafted filters. Understanding these helps build intuition for what CNNs learn automatically.

**Vertical edge detector** (Sobel-like):

$$K_{vertical} = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}$$

**Why it works**: 
- Left column has positive weights → measures brightness on the left
- Right column has negative weights → measures brightness on the right
- If left is bright and right is dark: large positive output (edge!)
- If uniform: positive and negative cancel → output near zero

**Horizontal edge detector**:

$$K_{horizontal} = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}$$

**Worked example**:

**Input image** (simulating a vertical edge):
$$I = \begin{bmatrix} 
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0
\end{bmatrix}$$

**Applying vertical edge kernel at position (1,1)**:

$$\begin{align}
S[1,1] &= (100 \times 1) + (100 \times 0) + (100 \times -1) \\
&+ (100 \times 2) + (100 \times 0) + (100 \times -2) \\
&+ (100 \times 1) + (100 \times 0) + (100 \times -1) \\
&= 100 - 100 + 200 - 200 + 100 - 100 = 0
\end{align}$$

**At position (1,2)** (on the edge):

$$\begin{align}
S[1,2] &= (100 \times 1) + (100 \times 0) + (0 \times -1) \\
&+ (100 \times 2) + (100 \times 0) + (0 \times -2) \\
&+ (100 \times 1) + (100 \times 0) + (0 \times -1) \\
&= 100 + 0 + 0 + 200 + 0 + 0 + 100 + 0 + 0 = 400
\end{align}$$

**Result**: High activation (400) exactly at the vertical edge location!

### Example 2: How CNNs Build Hierarchical Representations

One of the most beautiful aspects of CNNs is how they automatically learn hierarchical features:

**Layer 1 (Early layers - Simple features)**:
- Oriented edges at various angles
- Color blobs and gradients
- Basic textures
- Example filters: $$\begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$$, $$\begin{bmatrix} 1 & 1 & 1 \\ 0 & 0 & 0 \\ -1 & -1 & -1 \end{bmatrix}$$

**Layer 2-3 (Middle layers - Parts)**:
- Corners (combinations of edges meeting)
- Simple shapes (curves, circles, rectangles)
- Textures (repeated patterns of edges)
- Grid patterns, stripes, spots

**Layer 4-5 (Deeper layers - Object parts)**:
- Eyes, noses, ears (for face recognition)
- Wheels, windows, headlights (for car recognition)
- Petals, leaves, stems (for flower recognition)

**Final layers (High-level - Objects)**:
- Complete faces from different angles
- Full car shapes
- Entire animals

**This hierarchy emerges automatically from training** — the network discovers that edges are useful for building corners, corners for building shapes, and shapes for building objects. This mirrors how neuroscientists believe the visual cortex processes information.

### Example 3: Receptive Field Growth

**Receptive field** is the region of the input image that influences a single output neuron.

**Single 3×3 conv layer**: Each output pixel "sees" a 3×3 region of the input.

**Stack of two 3×3 conv layers**: Each output pixel in layer 2 sees a 3×3 region of layer 1's output. But each pixel in that 3×3 region sees a 3×3 region of the input. Combined: the layer 2 output sees a **5×5** region of the original input.

**Formula for receptive field with $$n$$ layers of 3×3 convolutions**:

$$RF = 1 + 2n$$

| Layers | Receptive Field | Equivalent Single Kernel |
|--------|-----------------|-------------------------|
| 1 | 3×3 | 3×3 |
| 2 | 5×5 | 5×5 |
| 3 | 7×7 | 7×7 |
| 5 | 11×11 | 11×11 |
| 10 | 21×21 | 21×21 |

**Why stack small filters instead of using one large filter?**

Compare: Three 3×3 layers vs one 7×7 layer

| Metric | Three 3×3 | One 7×7 |
|--------|-----------|---------|
| Receptive field | 7×7 | 7×7 |
| Parameters per channel | 3×(3×3) = 27 | 7×7 = 49 |
| Non-linearities | 3 ReLUs | 1 ReLU |

Three stacked 3×3 convolutions have:
- **Fewer parameters** (27 vs 49)
- **More non-linearity** (3 ReLUs vs 1)
- **More expressive power** (can represent more complex functions)

This insight from VGGNet (2014) revolutionized CNN architecture design.

---
