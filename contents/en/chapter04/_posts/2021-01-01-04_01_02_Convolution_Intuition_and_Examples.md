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

When Hubel and Wiesel projected bars of light onto a screen in front of an anesthetized cat in 1959, they stumbled on a startling biological fact: individual neurons in the primary visual cortex remained completely silent until a straight edge drifted across a tiny patch of the retina at one exact angle. Decades later, when researchers visualized the first-layer filters of AlexNet trained purely on natural image labels, the exact same oriented edge detectors and color-opponent Gabor blobs emerged spontaneously from gradient descent. Convolution works not by memorizing pixels, but by asking a hierarchy of local geometric questions—and composing small answers into global understanding.

## Background Knowledge Map: From Local Filters to Semantic Hierarchies

- **Directional Derivatives & Edge Detection**: Finite-difference kernels like Sobel $$[-1, 0, +1]$$ approximate the spatial partial derivatives $$\partial I / \partial x$$ and $$\partial I / \partial y$$. Wherever pixel intensity jumps sharply, the inner product spikes; in flat uniform regions, positive and negative weights cancel cleanly to zero.
- **Receptive Field Dynamics**: A single $$3 \times 3$$ filter sees only $$9$$ pixels, which cannot tell a dog's ear from a triangle. However, each stacked convolutional layer expands the **theoretical receptive field** $$r_\ell = r_{\ell-1} + (k_\ell - 1)\prod_{i=1}^{\ell-1} s_i$$, allowing deep neurons to integrate context across the entire image.
- **Effective vs. Theoretical Receptive Field**: Because there are exponentially more paths from the center of a receptive field than from its corners, Luo et al. (2016) showed that the *effective* receptive field follows a Gaussian profile concentrated near the center—meaning depth and stride must be designed generously.
- **Feature Hierarchy & Compositionality**: Combining oriented edges with ReLU non-linearities creates corner and curve detectors in layer 2, texture and part detectors in layers 3–4, and class-discriminative object prototypes near the top of the network.

### Step-by-Step Worked Example (Read Slowly): Receptive Field Growth Across Three Layers

Let us compute the exact receptive field size $$r_\ell$$ and effective stride (jump) $$j_\ell$$ as a signal passes through a three-layer CNN stem:
- Layer 1: $$3 \times 3$$ conv, stride $$s_1 = 1$$
- Layer 2: $$3 \times 3$$ conv, stride $$s_2 = 2$$
- Layer 3: $$3 \times 3$$ conv, stride $$s_3 = 1$$

1. **Initialize at the input image (Layer 0)**:
   $$r_0 = 1, \quad j_0 = 1.$$
   (Each input pixel sees $$1 \times 1$$ pixel, and adjacent pixels are $$1$$ pixel apart.)
2. **After Layer 1 ($$k_1 = 3, s_1 = 1$$)**:
   $$r_1 = r_0 + (k_1 - 1) j_0 = 1 + (3 - 1)(1) = 3, \quad j_1 = j_0 \cdot s_1 = 1 \cdot 1 = 1.$$
3. **After Layer 2 ($$k_2 = 3, s_2 = 2$$)**:
   $$r_2 = r_1 + (k_2 - 1) j_1 = 3 + (3 - 1)(1) = 5, \quad j_2 = j_1 \cdot s_2 = 1 \cdot 2 = 2.$$
   Notice that stride $$2$$ in Layer 2 does **not** enlarge $$r_2$$ immediately—it doubles the jump $$j_2$$ for every subsequent layer!
4. **After Layer 3 ($$k_3 = 3, s_3 = 1$$)**:
   $$r_3 = r_2 + (k_3 - 1) j_2 = 5 + (3 - 1)(2) = 9, \quad j_3 = j_2 \cdot s_3 = 2 \cdot 1 = 2.$$
5. **Key Insight**: A single activation after Layer 3 integrates a $$9 \times 9$$ input patch—nearly matching a giant $$9 \times 9$$ kernel ($$81$$ weights per channel) while using only $$3 \times 9 = 27$$ spatial weights per channel and inserting two non-linearities along the way.

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

## Video tham khảo

- **xem trước** [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown, English. Kernels as detectors (edges, then patterns) are the intuition section.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, English. Shows those detectors on actual image patches.
