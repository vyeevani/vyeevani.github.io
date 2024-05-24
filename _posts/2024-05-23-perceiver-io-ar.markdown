---
layout: post
title:  "Combining Perceiver IO and Perceiver AR"
date:   2024-05-23 20:00:00 -0700
categories: Robotics
usemathjax: true
published: true
---

# Disclaimer
I have not done any research to see if anyone else has already presented these ideas due to me having a job that eats the majority of my time. If anyone believes that I'm stealing credit for their work, lmk and I'll cite you.

# Overview
Given that everyone seems to have gotten onto the transformer bandwagon and seems intent on recreating the same architecture again and again, I'd like present my own take on transformers. It's basically just a Perceiver-IO that also has some bits mixed in their for autoregressive generation. I'll describe in some detail what I think differs from the standard Perceiver-IO and Perceiver-AR architectures. 

But first, a recap. perceivers were meant to decouple the memory consumption of the quadratic transformers from the token length, allowing a more fine-grained trade-off to be made. This may not be needed for language, but there's really no reason to process every pixel as a token in an image. Many papers have been written about numerous algorithmic efficiency improvement techniques based on convolutions, patches, and all manner of whacky shit. Perceivers discard all this crap and instead just say: "give me a byte array" and you can control the number of tokens that are being attended. 

Perceiver-IO is the bit that allows you to plug in large dimensional things. Perceiver-AR presents some ideas on how to preserve autoregressive generation for perceivers.

# Problem
Perceiver-IO and Perceiver-AR are two separate architectures. While they both have the same general principles, to the best of my knowledge they haven't been unified (see disclaimer above). What I want is a sequence to sequence model which can handle millions of input bytes in the same way as the Perceiver-IO.

# Solution
Take the same basic structure as Perceiver-IO. 

There will be two changes that are made, to the latent/input x-attn step and to the latent self attention step. You need the latents to be autoregressively generatable. To this, you can replicate the latents across the number of timesteps and use casual masking at the input and the latent self attention step. 
<script src="https://gist.github.com/vyeevani/aee668ad21b3e4744af26305455790a1.js"></script>
In the above code, I didn't end up doing an casually masked step for the latent self attention. It simply proved to require too much memory for my lone 4090 to handle even with aggressive checkpointing. To do that, I could simply do the following for the mask
```python
casual_input_backbone_mask = einops.rearrange(
    jax.numpy.tril(jax.numpy.ones((self.latent_count, self.latent_count, timesteps, timesteps))),
    "c e t1 t2 -> (t1 c) (t2 e)"
)
```
to add the casual masking for latents.

