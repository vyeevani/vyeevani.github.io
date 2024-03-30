---
layout: post
title:  "Meta learning in robotic manipulation using teleoperation, curriculum learning, and diffusion policies"
date:   2024-03-30 10:54:30 -0700
categories: RL
usemathjax: true
---

# Overview
teleoperation + diffusion policies solve sparse reward tasks required to bootstrap a meta curriculum learning process hopefully achieving human timescale robot manipulation adaptation.

# Why care about this problem?
I've been interested in teleoperation for robotics since seeing a demo of [ALOHA](https://tonyzhaozh.github.io/aloha/) (ty to Tony for showing me). Inspired by this, I got my own trossen arm and have been messing around with it. (wrote a control stack in rust since I hate ros with a burning passion, iphone based teleoperation, and [diffusion policy implementation](https://diffusion-policy.cs.columbia.edu)). The task that I've been driving towards has been to pick up a plastic banana.

Having gotten that task to work for a single position of the banana, I've been contemplating how much of a pain its going to be to collect the samples required generalize to any position of the banana within a fixed environment, then many objects, then many environments, then many manipulation goals (the end goal of robotic manipulation). I did some back of the envelope calculations based on [RTX](https://robotics-transformer-x.github.io) and [Bridge](https://rail-berkeley.github.io/bridgedata/) to figure out how much data I think I'd need. Unfortunately, I did this math on a whiteboard which I since erased, but trust me when I say it was in the millions of dollars of human teleop hours. Sadly I don't have that much money, but I do want this robot. So what to do. The only way I can think is buy more robots (fixed cost) and get them to learn by themselves.

# Sparse Reward RL is hard

RL has always had the problem that there's no guarantees in it. Taking a shitty approach like naive policy gradient, it's clear why RL in complex manipulation is extremely difficult when compared to locomotion or other tasks.

$$
\begin{align*}
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(\tau) R(\tau)]
\end{align*}
$$

In manipulation tasks where reward is extremely sparse, the gradient is going to be zero since reward will mostly be zero. This leads to no learning. Hence teleoperation is really useful because you sidestep the entire problem of from scratch learning.

# Defining teleoperation mathematically

But now, imagine that you have a single teleoperation example of a particular manipulation. For your mental model, you can think of a teleoperation example of a 6-dof arm picking up a banana. The model trained on this teleoperation example would be

$$
\begin{align*}
\theta^{*} = \max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta, \phi_1}} [\nabla_\theta \log \pi_{\theta, \phi_1}(\tau) R_{\phi_1}(\tau)]
\end{align*}
$$

where $$\phi_1$$ is the notation I use to say a single fixed position of the banana. You know

$$
\begin{align*}
\mathbb{E}_{\tau \sim \pi_{\theta^{*}, \phi_1}} [R_{\phi_1}(\tau)] \approx 1
\end{align*}
$$

So we know that teleoperation results in us having a perfect reward on the implicit reward function defined by the human. This doesn't actually get us anything in the way of generalization without more examples in diverse examples as seen by [RTX](https://robotics-transformer-x.github.io) and [Bridge](https://rail-berkeley.github.io/bridgedata/). This of course needs lots of humans to control lots of robots. If you don't have millions of dollars laying around, then you need to get the robot to figure this out on it's own. My belief is that human's learn iteratively, using prior experiences in simple environments to bootstrap more and more knowledge. i.e humanity didn't put on a man on the moon before we created language. I'll have to justify how curriculum learning can work when the reward is really sparse. I like [Terrance Tao's explanation of how research works](https://terrytao.wordpress.com/career-advice/be-sceptical-of-your-own-work/) for how this applies to people in research.

# Explanation of why curriculum learning works

Let's say now that we have a situation that we are investigating (we are trying to generalize the banana over multiple locations). I represent this situation as $$\phi_2$$. But I apply the constraint that trajectories sampled from $$ \pi_{\theta^{*}, \phi_2} $$ would overlap with $$ \pi_{\theta^{*}, \phi_1} $$. In other words one of the positions in the set of positions we are trying for $$ \phi_2 $$ will be the position used in $$ \phi_1 $$. IMO, this is a reasonable constraint. Starting with a few examples of a specific goal, you'd like to generalize to the superset of those goals. Given that you already have a solution to the sub goal $$\phi_1$$ the following should hold $$ R_{\phi_1}(\tau) == R_{\phi_2}(\tau),  \forall \tau \sim \pi_{\theta, \phi_1}(\tau) $$.

Given the above constraint, we know that

$$
\begin{align*}
\mathbb{E}_{\tau \sim \pi_{\theta^{*}, \phi_2}} [R_{\phi_2}(\tau)] > 0
\end{align*}
$$

Therefore, we can say that

$$
\begin{align*}
\nabla_{\theta^*} J(\theta^*) = \mathbb{E}_{\tau \sim \pi_{\theta^*, \phi_2}} [\nabla_\theta \log \pi_{\theta^*, \phi_2}(\tau) R_{\phi_2}(\tau)] \neq 0
\end{align*}
$$

Now with a non-zero gradient, learning can happen. (yes I know it could still be zero if the grad log liklihood is zero but I'm hoping my models don't suck and that's not zero).

This entire approach is task independent. For example, your task could be learning to pick up a single banana, or learning how to learn to pick up the banana (meta learning). I personally find meta learning incredible interesting so I'll focus on that for the remainder of this.

# [ADA](https://sites.google.com/view/adaptive-agent/?pli=1) as a working example of RL driven meta curriculum learning

[ADA](https://sites.google.com/view/adaptive-agent/?pli=1) showed that tabula rasa meta-learning was possible using a mixture of curriculum and reverse distillation technique going from smaller models to larger models (yes they also used transformers but who cares). I'll draw a parallel between the contrived virtual environment of ADA and my task of manipulation.

ADA is able to learn from scratch because the complexity of initial tasks are small enough well understood exploration strategies + rl techniques can solve them . Once you have a "beach head model", the idea that iterative task rewards being non-zero guarantees that there will be a gradient provides some kind of intuition for why ADA is able to continue learning. (The "reverse distillation" of ada handles the non-stationary distributions well and I find that idea rather elegant). In the manipulation setting, given that the task complexity is really high, it's difficult to do from scratch learning. However, once you have the "beach head model", the same thing as ADA should be possible (the intuition of gradients being non-zero should hold).

So how do you meta learn from teleoperation data. In ADA, the model is exploring by itself from the beginning. Which means that it has a set of initial sub-optimal trajectories. It can learn how to reproduce these sub-optimal trajectories + the optimal trajectories that it eventually sees and use those multi-trajectory as a single trajectory which it learns. However, in teleoperation, we don't actually have sub-optimal examples. The whole idea is that whatever happens in teleoperation is technically optimal (ignoring wether or not it actually is optimal). However, diffusion policies allow for a natural sub-optimality to be introduced. Imagine that you want to assemble a training sample for meta-learning in manipulation. You now are in need of several less accurate models. By increasing the stride of the denoising process or increasing the temperature, you can generate less accurate actions that are still somewhat structurally similar to the real actions. In my view, that's somewhat better than truncation or just randomly adding noise to the output because that wouldn't be structurally similar to the real actions. Diffusion has that power of "first let's denoise the coarse grained features, then we will refine", which I feel would work better for this. Who knows though, good science needs experimentation and I haven't done this yet so take it with a grain of salt. This feels like it's somewhat similar to classifier free guidance in the image generation models although I haven't really thought through the connections here.

Now once you have trained a new model on this meta-learning dataset you've collected, you can start varying the environment little by little using the curriculum learning like ADA with confidence from the above justification. 

bing bang boom: meta learning for robotic manipulation using teleoperation, curriculum learning, and diffusion policies.

# Summary

To summarize: teleoperation + diffusion policies solve sparse reward tasks required to bootstrap a meta curriculum learning process hopefully achieving human timescale robot manipulation adaptation.