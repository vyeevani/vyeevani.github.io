---
layout: post
title:  "Classifier-free guidance as controllable adaptation"
date:   2025-05-27 6:00:00 -0700
categories: Robotics
usemathjax: true
published: true
---

# Overview
Suppose you represent a policy using probability path techniques (flow/diffusion). Then you can use classifier-free guidance to improve how well an agent can adapt to changing environments.

# Bibliography
Yea, it's weird I put this first. What follows mostly just chains some papers together in a row. So it's important to grab the main ideas from the foundations.
1. [Diffusion Guidance Is a Controllable
Policy Improvement Operator](https://arxiv.org/pdf/2505.23458) - Frans et al. Use classifier-free guidance to improve goal following. Leads to better performance policies in offline/online setting
2. [Plasticity as the Mirror of Empowerment](https://arxiv.org/pdf/2505.10361) - Abel et al. If you influence the environment, you don't learn anything new. If the environment influences you, you learn.
3. [General Agents need world models](https://arxiv.org/abs/2506.01622) - Richens et al. All good policies are good world models. Also improving policies improves the implicit/explicit world model. This only applies when you have more than 1 subgoal. For the purpose of this, I'm just gonna go out on a limb and assume we are only talking about cases where we have more than one subgoal - much closer to real systems. 

Side note: these are great papers. Read these instead of the 250th paper on some sub quadratic attention paper or the 10th paper talking about robotics foundation models with behavior cloning.

# How do these papers connect?

During policy iteration, your policy gets better with CFG parameter $$\gamma$$ being $$\gamma > 1$$. Better policies implicitly or explicitly have a better world model + planner. If you are able to control the future states of the environment, you're empowered. Since empowerment and plasticity are mirrors, you're trading off plasticity.

# What can't this connection do?

Well we are giving up plasticity. But why do we care? Why would you ever want to have higher plasticity?

This is an adaptation of the example in Abel et al. to robotics. Suppose you have a robot that's manipulating objects whose friction can vary. So it picks up the same looking object, but sometimes it's slippery, sometimes it's not. Suppose the frictions doesn't vary much, the robot can't learn much about how friction impacts the world. That being said, it'll do really good at picking up the objects. Alternatively, when the friction varies a lot, the robot will learn a lot about how friction impacts the world. However, it's gonna suck at actually picking up the object. Not it's fault, it's hard to pick up objects when you have no idea wtf the friction of the object is gonna be.

In the current approach, we'd only ever be focusing on improving the policy. This means that if the environment changes, we aren't able to adapt to it because we haven't learned shit about it. In real world machine learning, the world is always changing. In order to build systems that work outside the lab, you have to be adapting constantly. For example, if a robot was shown how to pick up a mug during training with a certain friction, it'll do really good at picking up that mug when it has the same friction. But what happens when the mug is slippery? How do we build in the ability for a robotic system to adapt to the world, not stay rigid to the training data.

# Possible patch

So let's try to look at the converse from the original reasoning. Suppose you're willing to trade empowerment for plasticity. During policy iteration, you'd want to not be so rigidly following your world model. If you don't follow your world model, then you have a policy that doesn't follow goals as well. If you want a policy that's not following goals well, set your CFG strength $$\gamma$$ to be $$0 < \gamma < 1$$. I haven't thought through what happens if you set $$\gamma < 0$$. I'm pretty sure this would just send things completely off the known data manifold - but I could be wrong and there could be really interesting methods here.

This has all the benefits of the CFG policy improvement approach
1. Not train time - test time
2. Can be directly optimized

# Conclusion
I think that this concept can tie into recent work [Rethinking the Foundations for Continual Reinforcement Learning](https://arxiv.org/pdf/2504.08161). This paper changes the goal of continual RL. The new goal is to figure out how we could have changed our behavior in the past to do good, i.e should we have deviated from the behavior or not. Using CFG, we can directly optimize the deviation regret with respect to the classifier-free guidance in an online way. I believe this works well because we have a tuneable parameter which keeps us closed under the trajectory distribution but allows us to seemlessly vary the goals. This would be a superset of "constant deviations" mentioned in the paper.