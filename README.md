
# Inference-time scaling for segmentation

## IIS description
Take an interactive image segmentor which is $f(I, x)$ where $I$ is the image and $x$ a pixel. It returns $[l_1, \dots, l_S]  = f(I, x)$ which are logit maps at different scales. We will focus on the simplest case, where there is only one scale $S=1$. During training, there is a mask $m$ sampled from the ground truth mask set $M$ (these are the available ground truth masks corresponding to different objects in image $I$). Then there is a pixel $x$ sampled from $m$ and a pixel $\neg x$ sampled from $\neg m$. The network predicts $l$ and should minimize $L(f_+(I, x), m)$ for a positive click and $L(f_-(I,\neg x), m)$ for a negative click, where $L$ is the combination of Dice, Cross-Entropy, or mIoU losses (for the logit map among $[l_1, \dots, l_S]$ which achieves the min loss if $S>1$).

Therefore the predicted probability for each pixel $y$ (according to one of the logit maps) is 
$$\sigma(f_+(I, x))(y) = P(y\in m | x\in m)$$

In other words, IIS predicts "_the probability that pixel $y$ is contained in the mask that generated the prompt $x$ or the probability that pixel $y$ is contained in the mask that generated the prompt $\neg x$ (for the negative case)_". 

The total loss would be 

$$\sum_{i} \sum_{m_j \in M_i} \sum_{k=1}^{K} L(f_+(I_i, x(m_j)_k), m_j) + L(f_-(I_i, \neg x(m_j)_k), m_j)$$

where $M_i$ is the set of ground truth masks for image $I_i$ and $x(m_j)_k$ is the $k$-th pixel sampled uniformly at random from mask $m_j$, which itself is sampled uniformly at random from $M_i$ (analogously with the negative click $\neg x$).

## Inference-time scaling

Can we improve the performance by taking more time at inference? The idea is to get better performances, albeit at a higher computational cost, which might allow for model distillation later, which is the trend we see in LLMs.
This could be useful for self-training, domain adaptation, or simply getting a better performance if time allows.

In our case, given a positive click (which is the most interesting), we have two ways of doing inference-time scaling: 1. test time augmentation (any image transformation that preserves objectness) and 2. using auxiliary clicks to refine the segmentation. We will describe both in detail later on.


## Test-time augmentation
Recall   
$$\sigma(f_+(I, x))(y) = P(y\in m | x\in m)$$
for a given image $I$ and pixels $x$ and $y$. We can make a transformation $T$ on the image $I$ and the pixels $x$ and $y$ (for instance geometric), and if we assume that the objects don't change under this transformation then we also have that 
$$\sigma(f(I, x))(y) = \sigma(f(T(I), T(x)))(T(y))$$

Therefore we can have many estimates of the same probability distribution. Of course, these estimates might be different, and because we have no reason to prefer one over another under the assumption that the trasnformation preserves objects, we can average them. This should give us a strictly better estimate of the probability distribution, subject to 1. the transformation preserving the objects and 2. the predictor performing similarly under the transformation.
Of course, $T$ might also not be geometrical and simply transform the image. 

## Probabilistic triangulation
Bayes rule states

$$P(y\in m | x\in m) = \frac{P(x\in m | y\in m)P(y\in m)}{P(x\in m)}$$

where for any $z$ we have that $P(z\in m)$ is in fact $\frac{\sum_{m_j\in M} \mathbb{1}_{z\in m_j} }{|M|}$ (because masks are uniformly sampled). Because at inference time we can't access the set $M$, we need to estimate $P(z\in m)$. Methods usually don't do this, and we will not innovate in this regard, therefore we can only assert $P(y\in m | x \in m) \propto P(x\in m | y \in m)$. If we had estimates of $P(x\in m)$ for any $x$ we could enforce consistency, but we don't. 

If we also train the model to compute $\sigma(f_-(I, x))(y)=P(y\in m | x \notin m)$, then we can triangulate using transitivity consistency, which states that for pixels $x,y,z$ we have that

$$P(z\in m | x\in m) = P(z \in m | y \notin m) P(y\notin m | x\in m) +  P(z \in m | y \in m) P(y\in m | x\in m) $$

and of course $P(y \notin m | x \in m) = 1 - P(y \in m | x \in m)$. All the terms can be computed and we can get an estimate of $P(z\in m | x\in m)$ by using an extra point $y$. We might take as an estimate the average estimate over all $y$, as we have no reason to prefer one $y$ over another, which yields

$$P(z\in m | x\in m) = \frac{1}{|\Omega|} \sum_{y\in \Omega} P(z \in m | y \notin m) P(y\notin m | x\in m) +  P(z \in m | y \in m) P(y\in m | x\in m) $$

and any random sample of the pixel space $\Omega$ can also provide a good estimate. This is another form of inference-time scaling, where we use the model to estimate the probability of a pixel being in a mask given that another pixel is in the mask. The result is intuitive, if we take a pixel with high probability to be in the same mask as $x$, then the logit predicted from that pixel largely estimates the probability at $z$. 

## Interactive segmentation

Can we do interactive segmentation given that we trained our models $f_+, f_-$? Consider the simplest interactive segmentation (many-click) case, where we estimate:

$P(z \in m | x \in m, y \in m)$

we then get

$$P(x \in m | y \in m, z \in m) = \frac{P(y \in m, z \in m | x \in m) P(x \in m)}{P(y \in m, z \in m)}$$

or we could also expand

$$P(x\in m | y \in m , z \in m ) = \frac { P(x \in m, y \in m | z \in m)}{P( y \in m| z \in m)}= \frac { P(x \in m, z \in m | y \in m)}{P( z \in m| y \in m)}$$

but in both cases we are missing estimates of joint probabilities, and assuming independence would be wrong. Therefore we can't do interactive segmentation in its probabilistic formulation. 

However we could consider each $P(x\in m| c \in m )$ separately and then average them, but this approach doesn't model the conditional probabilities that are considered in the classical IIS methods, therefore it should be worse.

## Contributions
- revisit segmentation from the theoretical viewpoint
    - segmentation is subjective and defined by the gruond truth
    - AR is mIoU which is fine as a metric for a set maximum number of proposals
    - the max number of proposals is a hyperparameter, but can be set to the number of ground truth masks when they are supposed to represent all possible objects in the image and the model was trained in an object-centric manner
    - iis methods are too object-centered: need for hypersim
- revisit zero-shot segmentation
    - one random click is enough
    - train with positive / negative
    - use convex iou
- inference time scaling 
- distillation to feat space


## comparisons
Start from SegNext (SOTA with code) https://arxiv.org/pdf/2404.00741 , but mention the SOTA without code https://arxiv.org/pdf/2410.12214 .