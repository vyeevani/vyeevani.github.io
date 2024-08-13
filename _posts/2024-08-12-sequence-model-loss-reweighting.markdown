---
layout: post
title:  "Trajectory Transformer Loss Reweighting"
date:   2024-08-12 16:00:00 -0700
categories: Robotics
usemathjax: true
published: true
---
I've been continuing my explorations in robotics with attention-based sequence models. This is an overview of a really stupid issue that stems from behavior cloning in robotics when using a transformer and a specific solution that I ended up using.

It's widely known that behavior cloning for robotics can suck because it's easy for the model to step off the manifold of known trajectories and catastrophically fail. The new school of thought inspired by recent advancements in language modeling (which suffers from the same issue), is that scale is the problem. Recent work like BC-Z, ACT, Diffusion Policies, and many others have demonstrated the efficacy of this technique. 

Scaling doesn't work for me. I don't got 10K BH100 or whatever tf the latest Nvidia chip is called laying around. Sequence models do have the ability to snap back to a good trajectory if you give them a chance unlike sequential models which have the propensity to diverge exponentially. Therefore, they could address this problem effectively.

My dataset consists of two examples of a 6dof arm picking up a plastic banana, one example with the banana on the left the other with the banana on the right. The goal is to have the model replicate this behavior. My setup is a perceiver model (attention based transformer variant) with input/output normalization computed across all transitions in the trajectory. To put it succinctly, the shape of the mean would be a 3-vector of x, y, z position as opposed to the mean being a 4-vector of t, x, y, z. The robot starts at the idle position which is position (0, 0, 0). 

Initially, all pixels between the two are about the same except for the banana pixels. Because the initial timesteps have poses that are close to zero and I'm normalizing across time as well, the values ended up being very small compared to the later timesteps. This resulted in the model just learning the average between the two first timesteps because the loss was insignificant when compared to the rest of the timesteps. This means the arm didn't actually move left or right. Now at the second timestep, the model has already stepped off the known trajectory manifold. 

I have theories about what happens now rooted in what I've observed. First, the model shows good eval performance. This indicates that in theory the model knows where the banana is. Second, even if the model doesn't pick the correct side where the banana is, it'll pretend like the banana is there and not just flail around randomly. Together this implies that the model just looks for the banana in the first timestep, and then in the later timesteps, it uses the prior position of the arm to guide it or some timing information derived from the transformer. So as long as we can keep the model on the known trajectory manifold by forcing it to pay close attention to the first timesteps, the model will work.

The solution that I ended up using is to apply a loss reweighting across time in a sequence length agnostic manner. 

(This is a snippet I typed up since my code has a lot more junk in this method for padding stuff. Therefore, this prolly won't work as is, there's most definitely a bug in there.)
```python
def compute_time_weighted_loss(predicted_actions, actual_actions, loss_exponent):
    episode_length = predicted_actions.shape[0]
    error = einops.rearrange((predicted_actions - actual_actions) ** 2, "t ... -> t (...)")
    time_averaged_error = jax.numpy.mean(batched_squared_error, axis=-1)
    loss_scaling = episode_length * (1 - loss_exponent) / (1 - (loss_exponent ** episode_length))    
    times = jax.numpy.arange(0, episode_length)
    weight_sequence = jax.vmap(lambda time: loss_exponent ** time)(times) * loss_scaling
    reweighted_time_averaged_error = weight_sequence * loss_sequence
    loss = jax.numpy.mean(reweighted_time_averaged_error)
    return loss
```

This emphasizes the initial timesteps' performance since the model needs to pay more attention to the initial 
timesteps to avoid straying. The loss_exponent is a hyperparameter. I ended up using 0.375 after binary 
searching. Too low and the model doesn't actually ever fit the last timesteps. Too high and there's no benefit to 
this method.