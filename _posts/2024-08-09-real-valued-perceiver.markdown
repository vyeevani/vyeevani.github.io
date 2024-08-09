---
layout: post
title:  "Real Valued Perceiver"
date:   2024-08-09 15:00:00 -0700
categories: Robotics
usemathjax: true
published: true
---
I've been working on using a perceiver for a naive behavior cloning approach for a robot model. My reasoning is simple here: perceivers offer an attractive trade-off between transformers and recurrence.

However, I've been running into issues with the positional encoding schemes that were used in the original perceiver papers. The input to the model is a sequence of images, and the output is a sequence of end-effector positions. What I observed is that the training keeps getting stuck in a local minimum of the average of the end-effector positions. I'm using an Adam optimizer, and only with huge momentum (0.999) does it seem to get out of this rut.

This is a classic sign of posterior collapse. The model is initially not able to identify the differences between the inputs because of the huge size of the input byte array (12288) tokens. After thinking about it some more, I realized the problem is that there are too many similar tokens being fed to the model. In other words, between the two examples that I was feeding to the model, there are few pixels that vary. Because the model has to learn both the relationships between the spatial encoding schemes and the data, it kept getting caught in the posterior collapse local minimum. The gradient is being fed through a tight latent bound that's restricting the amount of gradient being fed to each of the individual tokens. Resolving this naively would just involve using a transformer architecture. But this is exactly what I don't want. Ain't nobody got the compute to train self-attention over 12288 tokens.

So I started thinking about how you can distribute information between tokens without losing the actual relationship between them. It hit me that this is basically what a Fourier transform does. I wrote up a simple synthetic test to validate that this worked, and it did! It dramatically improved the model's ability to resolve tiny changes in the inputs. However, it wasn't able to do this completely for longer sequences. This is an interesting problem to analyze theoretically, though.

Let's say your input sequence is

$$
x[n]=
\begin{cases} 
X_1 & \text{for } n=0 \\
X_2 & \text{for } n=1,2,\ldots,N-1 
\end{cases}
$$

where $$ X_1 $$ and $$ X_2 $$ are drawn from normal distributions.

To be more precise, let's assume:

$$
X_1 \sim N(\mu_1, \sigma_1^2)
$$

and

$$
X_2 \sim N(\mu_2, \sigma_2^2)
$$

The DFT is:

$$
X[f] = X_1 + X_2 \sum_{n=1}^{N-1} e^{-j \frac{2 \pi f n}{N}}
$$

Simplifying the Fourier Transform

The sum of exponentials

$$
\sum_{n=1}^{N-1} e^{-j \frac{2 \pi f n}{N}}
$$

is a geometric series:

$$
\sum_{n=1}^{N-1} e^{-j \frac{2 \pi f n}{N}} = e^{-j \frac{2 \pi f}{N}} \left( \frac{1 - e^{-j \frac{2 \pi f (N-1)}{N}}}{1 - e^{-j \frac{2 \pi f}{N}}} \right)
$$

For $$ f \neq 0 $$, this expression simplifies to:

$$
X[f] = X_1 + X_2 \cdot e^{-j \pi f \frac{N-1}{N}} \cdot \frac{\sin \left( \pi f \frac{N-1}{N} \right)}{\sin \left( \pi f / N \right)}
$$

For $$ f = 0 $$:

$$
X[0] = X_1 + (N-1) \cdot X_2
$$

This means that the contribution of $$ X_2 $$ would dominate $$ X_1 $$ over time, but the impact of $$ X_1 $$ being different is still felt for bounded $$ N $$. This implies that if you had two relatively short sequences where one of the values was changed between them, the impact of that changed token would be felt across all frequencies. The implication here is that when you use the cross-attention in a perceiver, it's easy for the perceiver to see the point-wise changes because they are present in all tokens as opposed to a single token. So rather than having to learn a really hard spatial transform, the model can easily observe the global changes in tokens instead. However, if you had a really long sequence, it would fail because $$ X_1 $$ would be lost in $$ X_2 $$'s power.

However, we have a simple trick in our toolbox: we can normalize the input data. So all $$ X_2 $$ would simply become 0. This would imply that we completely eliminated the dependence on $$ N $$. So as long as we normalize the input sequence, we can get this to work.

When I tried this approach, things worked beautifully! Now, because this is helping the worst-case scenario where the model has to resolve a tiny difference in input, you might ask, are you trading the best-case performance away? In short, I have no idea. My belief is this: the Fourier transform is a linear transform, so there's no information loss happening in the transformation.

So what's the downside? If you actually had periodic data where the frequencies were being shifted in a minute way between two different examples, the Fourier transform would then cause the inputs to be affected. There could be an elegant way to handle this, though. For a given dataset, you could first compute the mean and variance for the inputs and the Fourier inputs. Then, pick whichever has the higher variance between tokens.
