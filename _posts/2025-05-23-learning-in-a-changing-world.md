---
layout: post
title:  "Learning in a Changing World"
date:   2025-05-23 8:00:00 -0700
categories: Robotics
usemathjax: true
published: true
---

## Overview

Most machine learning assumes everything stays the same - but it doesn't. Robots work great until the lighting changes or objects move. Language models get outdated as words change meaning and new slang emerges. This is my thoughts on the conditions that learning algorithms must satisfy to handle those problems.

## Preliminaries (with sketchy math)
Start from a mutual-information generalization bound:

$$
G \leq \sqrt{\frac{2}{n} I(D;\theta)}
$$

Differentiating—informally—with respect to changes in the data distribution gives:

$$
\frac{\partial G}{\partial D} \leq \frac{1}{\sqrt{2nI(D;\theta)}} \frac{\partial H(\theta|D)}{\partial D}
$$

where $$H(\theta\mid D)$$ is the conditional entropy of parameters given the data. We desire this sensitivity to be negative, so that dataset shifts tighten the bound.

Yes I know I haven't actually derived this or actually demonstrated that it's possible to take the derivative of this bound like this. Technically, you have to show it's smooth and all that other fun stuff. It's fine, just hand wave it for now. I'll check my math on this later.

## Conditions (important bit)
Parameters $$\theta$$ emerge from a stochastic optimization process (random init, minibatches, dropout, finite floating point, etc.). So I'll model the output of a learning algorithm $$\mathcal{A}$$ with additive noise:

$$
\theta = \mathcal{A}(\mathcal{D}) + \varepsilon(\mathcal{D}), \quad \varepsilon(\mathcal{D})\sim\mathcal{N}(0,\sigma^2(\mathcal{A}(\mathcal{D}))I)
$$

The conditional entropy:

$$
H(\theta|\mathcal{D}) = \tfrac{d}{2} \log(\sigma^2(\mathcal{A}(\mathcal{D}))) + \text{const}
$$

Differentiating in $$\mathcal{D}$$:

$$
\frac{\partial}{\partial\mathcal{D}} H(\theta|\mathcal{D}) = \tfrac{d}{2} \frac{1}{\sigma^2(\mathcal{A}(\mathcal{D}))} \frac{\partial\sigma^2(\mathcal{A}(\mathcal{D}))}{\partial\mathcal{A}} \frac{\partial\mathcal{A}(\mathcal{D})}{\partial\mathcal{D}}
$$

From earlier, we want the product of derivatives must be negative because that means we are still generalizing in changing conditions:

$$
\frac{\partial\sigma^2(\mathcal{A}(\mathcal{D}))}{\partial\mathcal{A}} \frac{\partial\mathcal{A}(\mathcal{D})}{\partial\mathcal{D}} < 0
$$

This means either:
1. $$\tfrac{\partial\sigma^2}{\partial\mathcal{A}} < 0$$ and $$\tfrac{\partial\mathcal{A}}{\partial\mathcal{D}} > 0$$, or
2. $$\tfrac{\partial\sigma^2}{\partial\mathcal{A}} > 0$$ and $$\tfrac{\partial\mathcal{A}}{\partial\mathcal{D}} < 0$$


## Case Studies

It's hard to take my word for this (especially since I hand wave all my math). But I've managed to convince myself this makes sense by looking at two key pieces of evidence through the lens of these conditions. Firstly, why don't our usual bag of pretraining techniques work? Secondly, why does PPO kinda sorta work?

# Why naive stuff doesn't work
The standard toolkit of deep learning optimization techniques, while effective for stationary distributions, can actively prevent adaptation when the data distribution shifts:

- **Variance reduction** through techniques like gradient clipping and moving averages, which make $$\tfrac{\partial\sigma^2}{\partial\mathcal{A}}<0$$.
- **Explicit regularization** (e.g. weight decay) that pulls parameters toward zero, making $$\tfrac{\partial\mathcal{A}}{\partial\mathcal{D}}<0$$.

When both effects are negative, their product in the entropy derivative becomes positive:

$$
\frac{\partial}{\partial\mathcal{D}} H(\theta|\mathcal{D}) \propto \underbrace{\frac{\partial\sigma^2(\mathcal{A}(\mathcal{D}))}{\partial\mathcal{A}}}_{<0} \underbrace{\frac{\partial\mathcal{A}(\mathcal{D})}{\partial\mathcal{D}}}_{<0} > 0
$$

This positive derivative means the generalization bound grows with distribution shifts, preventing effective adaptation.

# Constrained-Update Objective
A general form of a constrained-update surrogate is:

$$
L(\theta) = \mathbb{E}[\min (\Delta(\theta)\hat A, \mathrm{clip}(\Delta(\theta),1-\epsilon,1+\epsilon)\hat A)]
$$

where $$\Delta(\theta)$$ is the ratio of new to old model outputs (or probabilities) and $$\hat A$$ the advantage estimate. Such clipping permits strong updates in promising directions ($$\tfrac{\partial\mathcal{A}}{\partial\mathcal{D}}>0$$) without an explicit weight‐decay pull. Combined with variance‐reduction (e.g. gradient clipping, moving averages), it maintains $$\tfrac{\partial\sigma^2}{\partial\mathcal{A}}<0$$, ensuring that $$\partial G/\partial D$$ stays negative under shifts:

$$
\frac{\partial}{\partial\mathcal{D}} H(\theta|\mathcal{D}) \propto \underbrace{\frac{\partial\sigma^2(\mathcal{A}(\mathcal{D}))}{\partial\mathcal{A}}}_{<0} \underbrace{\frac{\partial\mathcal{A}(\mathcal{D})}{\partial\mathcal{D}}}_{>0} < 0
$$

Constraining parameter updates relative to the previous model state—rather than globally shrinking magnitudes—enables rapid adaptation to non-stationary data while generalizing.