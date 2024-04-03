---
layout: post
title:  "Increasing robotic manipulation teleop sample efficiency"
date:   2024-03-30 9:00:00 -0700
categories: Robotics
usemathjax: true
---

# What's this about?
Building control algorithms for manipulation is difficult because of two things: reward sparsity and lack of direct data. Here I'll present the sketch of a solution that can solve this problem through the use of autonomous learning AND supervised learning.

DISCLAIMER: I did not do any literature review before writing this. It's just a thing I have been thinking about for a while. If you have a paper that says exactly this, reach out and I'll be happy to cite you!

# Reward is all you need?

Fundamentally, reward sparsity is a problem because it leads to no learning. This is pretty easy to see when exploring the naive policy gradient. 

$$
\begin{align*}
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(\tau) R(\tau)]
\end{align*}
$$

When $$ R(\tau) $$ is zero, $$ \nabla_\theta J(\theta) $$ will be zero. Hence your network will learn absolutely nothing.

# Teleoperation is all you need?

Image that you have a single teleoperation example of a particular manipulation. In a teleoperation example of a 6-dof arm picking up a banana that's in the same position every time, $$ \phi_1 $$ would be the starting position of the banana. Technically speaking, it's a latent controlling the trajectory distribution. The model trained on teleoperation examples would have the following property:

$$
\begin{align*}
\theta^{*} = \max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta, \phi_1}} [\nabla_\theta \log \pi_{\theta, \phi_1}(\tau) R_{\phi_1}(\tau)]
\end{align*}
$$

Obviously:

$$
\begin{align*}
\mathbb{E}_{\tau \sim \pi_{\theta^{*}, \phi_1}} [R_{\phi_1}(\tau)] \approx 1
\end{align*}
$$

Our issue is now if we move the banana a great deal, we likely wouldn't get good results:

$$
\begin{align*}
\mathbb{E}_{\tau \sim \pi_{\theta^{*}, \phi_2}} [R_{\phi_2}(\tau)] \approx 0
\end{align*}
$$

I did some back of the envelope calculations based on [RTX](https://robotics-transformer-x.github.io) and [Bridge](https://rail-berkeley.github.io/bridgedata/) to figure out how much data I think I'd need to overcome this. Unfortunately, I did this math on a whiteboard which I since erased, but trust me when I say it was in the millions of dollars of human teleop hours. Sadly I don't have that much money. Hence, teleop all the way is out of reach for me.

# Teleop + Reward is all you need?

My belief is that human's learn iteratively, using prior experiences in simple environments to build intuition that scales to harder environments.[Terrance Tao's explanation of how research works](https://terrytao.wordpress.com/career-advice/be-sceptical-of-your-own-work/) is a great read on how we are still progressively expanding the limits of our collective knowledge.

Suppose I said now that $$ \phi_i \subset \phi_{i + 1}, \forall i $$

$$
\begin{align*}
\mathbb{E}_{\tau \sim \pi_{\theta^{*}, \phi_2}} [R_{\phi_2}(\tau)] > 0
\end{align*}
$$

because we know that we'd see the set of banana positions defined by $$ \phi_1 $$ would be within the set of banana positions defined by $$ \phi_2 $$.

Now that you have some reward, you can begin to apply RL techniques and have guarantees that they will work. These will eventually get you to

$$
\begin{align*}
\mathbb{E}_{\tau \sim \pi_{\theta^{*}, \phi_2}} [R_{\phi_2}(\tau)] \approx 1
\end{align*}
$$

# This is obvious

You might be saying this is obviously naive curriculum learning and you'd certainly be right. It seems to me that there's two factions: folks that believe humans learn from scratch and therefore a robot must be able to learn from scratch, and folks that believe that the success of chatbots indicates that generalization for robotic manipulation is just a matter of more data.

Robotics has the benefit that language models don't inherently have of being able to autonomously learn without any humans present. At the same time, we don't have infinite robots that can run infinitely long to break through reward sparsity problems. Robots are real things, they cost real money, and no lab/company/gov't has infinite money.

As pointed out in [Terrance Tao's explanation of how research works](https://terrytao.wordpress.com/career-advice/be-sceptical-of-your-own-work/), we should carefully examine the differences in the sub-problems of our field (specifically: large generative models and robotics) to see where they cross and where they don't. 

In my view, the biggest difference is robotic's capacity for autonomous learning. While the knowledge gained from the (partially) solved problem of generative models is that supervised learning has phenomenal generalization power if you throw enough data at big enough models. So let's try and marry the two!

# Summary

RL fails to learn manipulation due to reward sparsity. Teleoperation sidesteps reward sparsity. Curriculum learning bootstrapped with teleoperation can get us good teleop sample efficiency.