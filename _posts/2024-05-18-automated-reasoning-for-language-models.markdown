---
layout: post
title:  "Automated reasoning for Language Models"
date:   2024-05-18 9:00:00 -0700
categories: Robotics
usemathjax: true
published: true
---

# Overview
I have been thinking about automated learning in robotics for a while. However, dealing with this on real robots is a pain in the ass. Given the hype around language models, I thought I'd sketch out how I think some of my ideas could be tested for cheap on existing language models before I finish flushing them out on robots.

# Problem
There is no way to increase the capabilities of language models automatically right now. This is evident in the ever increasing need for larger and larger datasets to fit more and more capabilities. Learning problem solving would reduce the amount of data needed by allowing agents to autonomously figure things out rather than having to be shown how to do tasks explicitly. 

# Solution
I propose the use of a stupid model, a smart model, and a learning model. The stupid model can't solve some of the problem that the smart model can solve. The learning model is based on the stupid model but is trained through learning the distribution of smart model samples conditioned on stupid model samples. This would in effect implicitly learn an optimizer in the learning model where it's able to extend it's knowledge slowly. If you present the learning model with a problem that the smart model can't do, then the learning model should be able to take the smart models output, and generate the correct solution from it. Note, I never try to find related work before I try things. I find that unless it's already well known solutions to problems, there's value in me pursuing it from scratch. In some sense, I'm trying to do the same thing to myself that I'm doing on the models. If I learn how to generate the edges of our collective knowledge from the mean of our collective knowledge, then maybe I can peer out beyond the edge as well.

# Experimental Method
My requirement here is if I start with a model that can solve problems of a specific difficulty but not those of a higher complexity, then by the end of the algorithm, I'd like a model that can solve problems it once wasn't able to. I need a smart model so I'll pick some version of llama. I'll group samples from somet dataset into samples that we were able to solve and samples we weren't able to solve with the smart version of llama ranking them based on log likelihood of them being generated. Note, smart here is relative I'll pick a small version of llama so it's relatively stupid but still able to do something. I'll put aside the samples we weren't able to solve. Now, I need to come up with a stupid model. I can do this by corrupting the smart model. I take problems we are able to solve and use high temperature llama samples which don't solve the problem along with samples that do solve the problem. Then I train llama on both these samples to corrupt it without completely destroying it's performance. Now I have a smart model and a stupid model. I then apply the procedure listed in the solution. First generate stupid samples, then smart samples. Finally, retrain the stupid model to generate correct solutions when given the problem + incorrect samples. Finally, test this model's performance on problems it wasn't previously able to do. If it's able to do the problems, that's a success, otherwise back to the drawing board!