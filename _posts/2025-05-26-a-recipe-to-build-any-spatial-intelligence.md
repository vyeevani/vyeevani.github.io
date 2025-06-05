---
layout: post
title:  "A Recipe to Vibe Train Spatial Intelligence"
date:   2025-05-27 6:00:00 -0700
categories: Robotics
usemathjax: true
published: false
---

A method to semi-autonomously improve video generation models + spatial intelligence models.


# Table of Contents
- [Overview](#overview)
- [General Solution Sketch](#general-solution-sketch)
- [Vibe Training](#vibe-training)
- [Semi-Autonomously Improve the Spatial Intelligence](#semi-autonomously-improve-the-spatial-intelligence)
- [Semi-Autonomously Improve the Video Generation Model](#semi-autonomously-improve-the-video-generation-model)


# Overview
First, wtf is spatial intelligence? It's the ability to take 4d visual information and answer detailed questions about it. This is a superset of the typical video qa style questions ("does the person pick up the mug on the right or the left?"). It can also be the following hard computer vision questions:
1. track this point throughout the video? (motion tracking)
2. how far is the robot gripper in 3d space from the grape in this 3d picture? (online robotic planning)
3. did we make any loops while moving around? (loop closure detection slam)

Now you may be asking: "isn't this stuff all basically solved?". No not really. We've seen dramatic progress with the adoption of data driven techniques like dust3r for 3d recon. This belief comes from the fact that we already have products that use these techniques - and no one builds products out of unsolved problems, so they must be robust. "we have headsets that can track the real world in real time so slam must be solved". However, we don't note the complexity (leading to fragility) of those solutions - often requiring hundreds of engineers to build (trust me I was one of them). 

So if you have an application like the above and want to improve the sota, then the following is a general purpose method for doing this. Fair warning, we don't have the foundational tools to make this method work yet, but I'm very confident we will get there in a short time horizon. For now, just pretend like it's 2030.

# General Solution Sketch
Once you have a problem you'd like to solve, the classical method to building any spatial intelligence model is composed of the following strategy
1. identify symmetries
2. write an equation that goes from symmetry to your solution

The data driven modern approach is similar but just swaps the identify symmetry + write equation bits with:
1. collect dataset show casing symmetry
2. train conditional model

My pitch here is to replace the collect dataset + train conditional model with
1. prompt 4d vision foundation model for dataset
2. train conditional model

For example, let's say that you wanted to vibe train dust3r. First question, wtf is dust3r. [dust3r](https://github.com/naver/dust3r) is a model that figures out the 3d alignment between two images. It's trained on pairs of images + the 3d alignment of those images. It's derivatives are behind a lot of the really cool demos of 3d recon in static/dynamic scenes.

The simplest way to build a dataset that you could use to build dust3r would be
1. figure out what general environments you'd like your intelligence model to work in: indoor, outdoor, dynamic, static, etc (just fyi dust3r does poorly in dynamic scenes, see [monst3r](https://monst3r-project.github.io))
2. identify transforms across videos or within videos that reflect the spatial intelligence you'd like to extract. In this case, if you ask the model to generate samples of timed camera motion. For example: 60 seconds of video with the camera set back by 1m from an object doing a 360 degree rotation around the object. Now you know the transforms between frames. So you have tuples: (pair of images, transform between images)
3. build conditional model. video -> alignment using generated dataset. The video will likely not be super accurate, so it's best to learn a distribution of possible alignments so that we can reduce the entropy later through refinement

# Vibe Training
You might say, this still doesn't really feel like vibe training. You are still doing a lot of work to algorithmically produce prompts to the video model. The key part of vibe driven development is that all it takes is one prompt + prompt based edits. I think this is still very much possible. Just ask the language model to produce the prompts - I've tried this with examples, it works pretty damn well.

# Semi-Autonomously Improve the Spatial Intelligence
Now you could run online RL on this to iteratively improve dust3r:
1. grab your phone
2. record a video of your room
3. generate image pairs of your room
4. use vibe dust3r to build recons of your room
5. repeat this n times to get n reconstructions
6. direct preference optimization

# Semi-Autonomously Improve the Video Generation Model

Suppose you used [transfusion](https://www.arxiv.org/pdf/2408.11039) + [diffusion forcing](https://arxiv.org/pdf/2407.01392) as your conditional model. i.e. you have a single model that does video diffusion + your task at the same time. So now you have a few things. A video marginal, a output marginal, the output conditioned on the video. Now you can use Bayes to get the video conditioned on the output. This is basically what's proposed in [history guided diffusion](https://arxiv.org/abs/2502.06764). 

The key thing here is that now we can directly improve the video generation model. Suppose you have a 3d reconstruction, you can take the pairwise things from that model to generate a video. Now you can use direct policy optimization over the videos produced from this method to improve the video generation model.

Take [motion prompts](https://motion-prompting.github.io/#causal) as an example for instance. This allows us to generate videos which look pretty different. By applying direct policy optimization over this, we can tune the video model to produce videos that align more to what we are looking at.

# What do we need in video models to get this to work?

So far in the discussion, we'd need multi-turn video editing as a basic thing to help bootstrap. We already have some datasets and some techniques that we can use to help bootstrap: (ego 4d, something something, all the project aria stuff, dust3r, mast3r slam, monst3r, video qa llms, open x embodiment, segment anything v2, video depth anything, nvidia omniverse).

However, it's not clear this is really possible to do well. For example, detailed motion text prompts are hard to get: [motion prompts](https://motion-prompting.github.io/#causal). What if, we can avoid having to build a video model that has this. Can we actually just sidestep text as a requirement, and instead let text <-> video be just another spatial intelligence task, rather than a precondition.

This is where I kinda go completely off the rails. Suppose a few things:
1.  video model has a smooth latent space since it's trained with [smooth diffusion](https://shi-labs.github.io/Smooth-Diffusion/)
2.  video model has been trained on unconditional sequences of videos (i.e it'll produce ten videos at a time where each video is completely unrelated to the prior video)

then you can do this to fit a conditional video model:
1. Generate a set of videos unconditionally
2. non isotropically interpolate between various combinations of their latent spaces
3. pick generated values that align most with the "what's the next logical thing in the sequence of videos for task x?"
4. repeat this n times
5. retrain your model

for example, if you wanted dust3r:
1. record some videos showing known camera motions using a robot
2. interpolate between these videos
3. pick interpolations that linearly transition the camera between different camera motions
4. train sequence model that predicts videos that linearly interpolates between different camera motions

You don't have a grammar of videos over the latent space technically. So you can't say: use the motion from this video with the background from this other video with the objects from this video. This limits you technically. Although, if you have a smooth latent, you can do an isotropic interpolation between the videos to hopefully get the right values.

# Pitch

I think there's a case here for a unified platform that iterates on the fundamentally virtuous cycle between better video generation models leading to better spatial intelligence and vice versa. So every time someone builds something, it improves the base video model for everyone, which improves everyone else's models. Unlike with text, we have access to an oracle (the real world) that allows us to endlessly learn.