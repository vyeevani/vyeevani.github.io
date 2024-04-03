---
layout: post
title:  "A slightly different meta learning formulation"
date:   2024-04-02 9:00:00 -0700
categories: Robotics
usemathjax: true
published: false
---

# Overview

I have found that meta learning formulations don't make it clear what's actually been gained by meta learning.

# Traditional Formulation

$$ \mathcal{T} = \{\mathcal{L}(x_1,a_1,...,x_H,a_H),q(x_1),q(x_{t+1}\|x_t,a_t), \mathcal{H}\} $$ consists of a loss function L, a distribution over initial observations $$ q(x_1) $$, a transition distribution $$ q(x_{t+1}\|x_t,a_t) $$, and an episode length $$ \mathcal{H} $$. In i.i.d. supervised learning problems, the length $$ \mathcal{H} = 1 $$. ([MAML](https://arxiv.org/pdf/1703.03400.pdf)).

This formulation of meta learning then says that we'd draw tasks out our task bag, then sample some trajectories from this task. These k trajectories are passed in one shot to the model which learns how to produce those trajectories.

# What's the issue

Let's say your task dataset is exactly two tasks: picking up different colored bananas, and picking up the same colored banana from different locations. In the above formulation, these have two different loss functions. So when you sample trajectories to be learned, you'd sample from those tasks. The question would then, if I give the model a trajectory with a different colored banana in a different location, could it pick it up. I'd expect that the model would be able to cope with this difference and actually be able to generalize.

But my question is, does any of this actually imply that the model has actually learned how to learn. The definition of meta is to nest the following word, i.e meta-learning means learning how to learn. If you simply gave the model the same dataset but instead the few shot meta learning process, you simply gave it trajectories of both varities, I'd expect that it's still able to pick up a different colored banana in a different location. Why is that? Because the model could just be a grey scale model that discards color information making it resilient to this problem. This means that the base multi-position model is able to pick up bananas of different colors without problem. So what have you actually meta-learned here?

# Why tho

My opinion is that the utility of meta learning must be everything that could be learned from learning different tasks that couldn't be learned from learning a single task. Therefore, to be useful, a meta-learning model has to be strictly better than task specific models. In the above formulation, the single task learner's potential generalization isn't taken into account.

Suppose you have a model that's learned how to pick up any color banana and one that learned only to pick up the same banana from different locations, each of those has learned different tasks. However, if you trained the different location model with greyscale, the color of the banana wouldn't actually matter to you. 

The loss function can't take the generalization of the model into account. Therefore, I'd define a task as the following: given dataset $$ \mathcal{D} $$, parameterized model $$ \mathcal{\pi} $$ with params $$ \theta $$, and an algorithm $$ \mathcal{A} $$ that generates new params $$ \theta $$ when given examples from the dataset, a task is all trajectories produced by the final parameters of the algorithm.

# How does this help?

By defining it as a minimum improvement over each of the single tasks, we are able to clearly define what aspect of what's learned is meta learned and what aspect of generalization is inherent to the dataset itself.