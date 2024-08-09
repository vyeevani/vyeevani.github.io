---
layout: post
title:  "Real Valued Perceiver"
date:   2024-05-23 20:00:00 -0700
categories: Robotics
usemathjax: true
published: true
---

I've been working on using a perceiver for a naive behavior cloing for a robot model. My reasoning is simple here, perceivers present an attractive trade-off between transformers and recurrence. 

However, I've been running into issues with the positional encoding schemes that were used in the original perceiver papers. The input to the model is a sequence of images, the output is a sequence of end effector position. What I observed is that the training keeps getting stuck in a local minima of the average of the end effector position. I'm using an adam optimizer and only huge momentum (0.999) seems to get it out of this rut. 

This is a classic sign of posterior collapse. The model is initially not able to identify the differences between the inputs because of the huge size of the input byte array (12288) input tokens. After thinking about it some more, I came to the realization that the problem is that there's too many similar tokens being fed to the model. In other words, between the two examples that I was feeding to the model, there are few pixels that vary. Because the model has to learn both the relationships between the spatial encoding schemes and the data, it kept getting caught in the posterior collapse local minima. the gradient is being fed through a tight latent bound that's restricting the amount of gradient being fed to each of the individual tokens. Resolving this naively would just be the transformer architecture. But this is exactly what I don't want. Ain't nobody got the compute to train self attention over 12288 tokens.

So I started thinking about how you can distribute information between tokens without losing the actual relationship between them. It hits me that this is basically what a fourier transform does. I wrote up a simple synthetic test to validate that this worked and it did! It dramatically improved the models ability to resolve tiny changes in the inputs. However, it wasn't able to do this completely for longer sequences. This is an interesting problem to analyze theoretically though.

Let's say your input sequence is
$$
x[n] = \begin{cases} 
X_1 & \text{for } n=0 \\
X_2 & \text{for } n=1,2,\ldots,N-1 
\end{cases}
$$
where \( X_1 \) and \( X_2 \) are drawn from normal distributions.

To be more precise, let's assume:
$$
X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)
$$
and
$$
X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)
$$

The dft is:

$$
X[f] = X_1 + X_2 \sum_{n=1}^{N-1} e^{-j2\pi fn/N}
$$


Simplifying the Fourier Transform

The sum of exponentials 
$$
\sum_{n=1}^{N-1} e^{-j2\pi fn/N}
$$
is a geometric series:
$$
\sum_{n=1}^{N-1} e^{-j2\pi fn/N} = e^{-j2\pi f/N} \left( \frac{1 - e^{-j2\pi f (N-1)/N}}{1 - e^{-j2\pi f/N}} \right)
$$

For $f \neq 0$, this expression simplifies to:
$$
X[f] = X_1 + X_2 \cdot e^{-j\pi f (N-1)/N} \cdot \frac{\sin(\pi f (N-1)/N)}{\sin(\pi f / N)}
$$

For $f = 0$:
$$
X[0] = X_1 + (N-1) \cdot X_2
$$

This means that the contribution of $X_2$ would dominate $X_1$ overtime but the impact of $X_1$ being different is still felt for bounded $N$. This implies that if you had two relatively short sequences where one of the values were changed between them, the impact of that changed token would be felt across all frequencies. The implication here is that when you the cross attention in a perceiver, it's easy for the perceiver to see the point wise changes because they are present in all tokens as opposed to a single token. So rather than having to learn a really hard spatial transform, the model can easily observe the global changes in tokens instead. However, if you had a really long sequence, shit would fail because $X_1$ would be lost in $X_2$'s power.

However, we have a simple trick in our toolbox, we can normalize the input data. So all $X_2$ would simply become 0. This would imply that we completely eliminated the dependence on $N$. So as long as we normalize the input sequence, we can get this to work. 

When I tried this approach, things worked beautifully! Now, because this is helping the worst case scenario where the model has to resolve a tiny difference in input, you can ask, are you trading the best case performance away? In short, I have no idea. My belief is this: the fourier transform is a linear transform, this means that there's no information loss that's happening in the transformation. 

So what's the downside? If you actually had periodic data where the frequencies were being shifted in a minute way between two different examples. The fourier transform would then cause the inputs to be impacted. 

There could be an elegant way to handle this though. For a given dataset, you first compute the mean and variance for the inputs and the fourier inputs. Pick whichever has the higher variance between tokens. 