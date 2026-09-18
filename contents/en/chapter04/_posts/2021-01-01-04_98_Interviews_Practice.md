---
layout: post
title: 04-98 Interviews Practice (CNNs and early features)
chapter: '04'
order: 15
owner: Deep Learning Course
lang: en
categories:
- chapter04
lesson_type: optional
---

# Optional: interview practice — CNNs and early features

> This lesson is **optional**. It does **not** rewrite convolution math, pooling, or the LeNet–ResNet notes. After those lessons, use this drill, then open *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) for solved items on convolution, CNN architectures, and early feature extraction.

**Book themes for this chapter (study in the PDF, not here):** convolution geometry, CNN architectures, and the first half of the feature-extraction topic. Transfer-learning practice continues in **15-98**.

Hub: **01-98 Deep Learning Interviews practice track**.

## Practice prompts (original)

### Q1. Conv vs dense parameter count

A layer maps a $$32\times32\times3$$ tensor to $$32\times32\times16$$ features. Compare (a) a dense layer that flattens the input and (b) a $$3\times3$$ convolution with 16 output channels, same padding, bias included. Which number should you quote first in an interview?

**Hint.** Conv: $$k_h k_w c_{\mathrm{in}} c_{\mathrm{out}} + c_{\mathrm{out}}$$.

**Discussion.** Dense: input size $$32\cdot32\cdot3=3072$$, output $$32\cdot32\cdot16=16384$$, so about $$3072\cdot16384+16384$$ weights — tens of millions. Conv: $$3\cdot3\cdot3\cdot16+16=448$$. Quote the conv count first, then the *reason*: local connectivity and weight sharing. Interviews fail people who only say “CNNs have fewer parameters” without a formula.

### Q2. Output shape by hand

Input $$H\times W=28\times28$$, kernel $$5\times5$$, stride $$s=1$$, padding $$p=0$$, $$c_{\mathrm{out}}=8$$. What is the output tensor shape? What if $$s=2$$ and $$p=2$$?

**Hint.** $$H'=\bigl\lfloor(H+2p-k)/s\bigr\rfloor+1$$ (same for $$W$$).

**Discussion.** First case: $$H'=28-5+1=24$$, shape $$24\times24\times8$$. Second: $$(28+4-5)/2+1=14$$, shape $$14\times14\times8$$. Always state layout (channels-last vs channels-first) when you answer; frameworks disagree.

### Q3. Two $$3\times3$$ vs one $$5\times5$$

A stack of two $$3\times3$$ convolutions (no pooling, stride 1, padding to keep size) and one $$5\times5$$ convolution have the same theoretical receptive field on the input. Give one parameter-count reason and one expressivity reason to prefer the stack. Assume $$c$$ channels throughout.

**Hint.** Receptive field adds $$(k-1)$$ per layer when stride is 1.

**Discussion.** One $$5\times5$$: $$25c^2$$ weights. Two $$3\times3$$: $$18c^2$$ weights, plus an extra nonlinearity in the middle, so the composed map is no longer a single linear filter. VGG-style stacks use exactly this argument. If the interviewer asks about depthwise-separable or $$1\times1$$ bottlenecks, that is a follow-up, not a requirement for this chapter.

### Q4. Equivariance vs invariance

A vertical-edge filter is applied to an image, then the image is rolled two pixels to the right and the filter is applied again. What should change in the feature map if the implementation is a true convolution? What operation would you add if the *classifier* must not care about that shift?

**Hint.** Convolution commutes with translation; global pooling throws spatial position away.

**Discussion.** The feature map should roll by the same two pixels (translation *equivariance*). A final global average / max pool (or a sufficiently large downsample) makes the *vector* fed to the linear head closer to translation *invariant*. Mixing the two words is a common interview miss.

### Q5. What the identity skip is for

In a residual block $$ \mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x}) $$, write $$\partial\mathbf{y}/\partial\mathbf{x}$$ schematically. Why does this help a 50-layer CNN more than “just add more ReLUs”?

**Hint.** The $$+1$$ from the skip is the story; $$\mathcal{F}$$ can be small at init.

**Discussion.** $$\frac{\partial\mathbf{y}}{\partial\mathbf{x}} = I + \frac{\partial\mathcal{F}}{\partial\mathbf{x}}$$. The identity term keeps a path whose gain does not vanish when $$\mathcal{F}$$ is near zero, so gradients can travel many blocks. Extra ReLUs without skips still multiply many Jacobian factors that can shrink. This is an architecture answer, not a “ResNet always wins on accuracy” claim.

## Attribution

Kashani, S., and Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Download the PDF from arXiv for the full solved Q&A. This page is original course practice, not a reprint.
