---
layout: post
title:  "Pi0 Codebase Thoughts"
date:   2025-02-05 8:00:00 -0700
categories: Robotics
usemathjax: true
published: true
---

# Overview
Physical Intelligence just open sourced their pi0 codebase - really glad they did, everyone should be really glad they did, it's so awesome. You can check it out [here](https://github.com/Physical-Intelligence/openpi). This post is just going to be a look at the codebase with some semi-structured thoughts.

# Basics
Pi0 is a mixture of transformers model, one expert that handles the visual and language understanding and the other expert that handles the action prediction. To handle the high dimensionality + continuous nature of the action space, they use a flow matching process to generate the actions.

# Code Framework
They use flax. Things to note about flax, it relies heavily on structs to define configurations. It has a super magical (read: annoying) way of using these structs as part of the codebase. It's horrible. Seriously they should learn something from [oryx](https://github.com/jax-ml/oryx).

# pi0.py
The pi0.py base model is at [openpi/models/pi0.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py). We'll start there and then dig into the [gemma.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py) that holds the actual computation part.

These are the four main functions of the Pi0 model:
| Function | Purpose |
|----------|---------|
| embed_prefix | Embeds input data (language + image + robot state) |
| embed_suffix | Embeds input data (language + image + robot state) |
| compute_loss | Computes model loss during training |
| sample_actions | Samples actions from the trained model |

## Embed Prefix
Isn't much to note here other than the masking. We have bidirectional attention amongst image tokens + language tokens. I'm not clear on if it's a good idea to allow for bidirectional attention because that's a distribution shift between paligemma training. Afaik, that was trained only on AR tasks without any masked language modeling. I guess the pretraining on robot datasets, could correct for the shift. Weird.

## Embed Suffix
Note that time here refers to the sampling step, not the actual timestep of the action. This bit will encode the state + noisy actions.

# model.py
This contains the implementation of the mixture of experts over the paligemma implementation. The thing to note here is that the module class gets multiple configs from the higher level class. It then propogates this into everything it touches till the attention. This is because the attention module is the deepest part that combines the multiple experts. Annoying to understand from a codebase perspective but necessary from best practices of not hiding complexity where complexity exists.

# pi0_fast.py
Based on the [FAST blog](https://www.physicalintelligence.company/research/fast), I thought they'd replicated the pi0 action expert approach. I'm an idiot and didn't read the actual paper (let this be a lesson to not skim). Turns out they just used a single paligemma model without an expert. This is also a good reason to release source. It's so much more clear than language. Makes some sense I guess - they don't need to seperate things to different spaces since the FAST tokenizer uses the unused vocab as the paligemma tokenizer.