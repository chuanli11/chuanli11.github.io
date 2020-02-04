---

layout: "post"
title: "Review: Neural Rendering"
date: 2020-02-03 10:00:00
toc: true
---

**Contents**
* TOC
{:toc}

### Introduction

Because rendering is such a massive topic, I’d like to start the conversation with some clarification. 

As far as this article goes, rendering can be both a forward and an inverse process. Forward rendering computes images from some parameters of a scene, such as the camera positions, the shape and the color of the objects, the light sources etc. Forward rendering has been a major research topic in computer graphics. The opposite of this process is called inverse rendering: given some images, inverse rendering reconstructs the scene that was used to produce the input images. Inverse rendering is closely related to many problems in computer vision. For example, shape from X, motion capture etc.

Forward rendering and inverse rendering are intrinsically related because, the higher levels of a vision system should look like the representations used in graphics. We will look into both forward and inverse rendering in this article. In particular, how neural networks can improve the solutions to both these problems.


### A Bit of History

However, before diving into deep learning, let us first pay a quick visit to the conventional approaches. 

This is a toy example of ray tracing, a widely used forward rendering technique. Imagine we are inside a cave. The red bar is the light source, and the grid is the image plane. Ray tracing works by tracing a path from an imaginary eye through each pixel and calculating the color of the objects that are visible through the ray.

More often than not, the path will hit an object surface before reaching the light source. In which case the algorithm will assign the color of surface to the pixel. The color of the surface is computed as the integral of the incident radians at the intersection point. Most of the time these integrals have no analytic solutions and have to be computed via Monte Carlo Sampling: 
some samples are randomly drawn from the integral domain, and the average of their contributions are computed as the integral.


To model different surface materials, reflectance distribution functions is used to guide the sampling process. These functions decide how glossy or rough the surfaces are.

Most of the time, a ray needs multiple bounces before reaching the light source. Such a recursive ray tracing process can be very expensive. Different techniques have been used to speed up the process. To name a few: next event estimation, bidirectional path tracing, Metropolis Light Transportation ... etc. I will skip all of them due to page limit. In general, Monte Carlo ray tracing has high estimator variance and low convergence rate. This means a complex scene will need hundreds of millions, if not billions, rays to render. For this reason, we ask the question whether machine learning can help us speedup Monte Carlo ray tracing. 

As we will see later in this artile, the answer is YES.

Now, let’s switch to inverse rendering for a second. I am going to use shape from stereo as a toy example. The algorithm takes a pair of photos as the input, and produces a reconstruction of the camera poses P0 and P1 and scene geometry. The first step is to find similar features between the photos. With enough matching, we can estimate the camera motion between these two photos, which is often parameterized by a rotation and translation. Then we can compute the 3D location of these features using triangulation. More photos can be used to improve the quality of the results.

Advanced technique can scale up to work with thousands or even hundreds of thousands of photos. These photos were taken under a wide range of lighting conditions and camera devices. This is truly amazing.

However, the output point cloud of such computer vision system is usually very sparse. In contrast, computer graphics applications usually demand razor sharp details. So in film and game industry, human still have to step in to polish the results, or handcraft from scratch.

Everytime when we hear the word “handcrafting”, it is a strong signal for machine learning to step in. In the rest of this article, We will first discuss how neural networks can be used as sub-modules and an end-to-end system to improve forward rendering. We will also discuss neural networks as differentiable renderers which opens the door for many interesting inverse rendering applications.


### Forward Neural Rendering


#### Neural Networks as Sub-modules for MC Ray Tracing

First, let’s see how neural networks can help as a sub-module that speeds up Monte Carlo ray tracing. As mentioned earlier, Monte Carlo ray tracing needs many samples to converge. And here is an example: 

the top left picture is the Monte Carlo ray tracing result using 1 sample per pixel, and it doubles from left to right each square. The computational cost also increases linearly with the number of samples, which motivates us to develop efficient rendering techniques at a reduced sample rate. 

I’d like to make a fuzzy analogy here. You probably have heard of AlphaGO, which uses a value network and a policy network to speed up the MC tree search: the value network takes the board position as its input, and outputs a scalar value that predicts the winning probability. In other words, the value network reduces the depth of the search. The policy network similarly also takes the board position as its input, but it outputs a probability distribution over the best moves to take. In other words, the policy network reduces the breadth of the search.

The analogy I am going to make here is that there is also the value based approach and the policy based approach for speeding up Monte Carlo rendering. 


For example, the value network can be used to denoise renderings with low sample per pixel. We can also use policy networks to make the rendering converge faster.

##### Neural Denoising for Monte Carlo Ray Tracing

Let’s first discuss the value network approach. This is a recent work about denoising MC renderings. On the left is the noisy input image that was rendered with only 4 samples per pixel. In the middle is the output of the denoiser. On the right is the GT rendered with 32k spp. It takes 90 minutes to render with a 12 core intel i9 CPU. 

In contrast, the denoiser only takes a second to finish the rendering with a commodity GPU. So it gives a good trade off between speed and quality.



The denoiser network is based on the auto-encoder architecture. It is trained with L1 loss on VGG features maps, plus adversarial loss for retaining sharp details. And this is the comparison between training with and without the GAN loss. The one with adversarial loss is clearly better in recovering the details.

There are many literature about denoising natural images. However, denoising MC rendering is unique in a few ways:

Frist, one can separate the diffuse and specular components, so they go through different paths of the network, and the outputs are merged together. Studies found this gave better results in practice.

Second, there are inexpensive by-products, such by-product include albedo, normal, and depth maps, that can be used to improve the results. One can feed them into the network as auxiliary features. These features give the network further context where the denoising process should be conditioned on. Finding an effective way to use auxiliary features is an open research question. 

One popular method is called element-wise biasing and scaling: Element-wise biasing transforms the auxiliary features through a sequence of convolutional layers, and add the output to X. And one can proof that it is equivalent to feature concatenation. Element-wise caling runs multiplication between the input X and the auxiliary features.


The argument of having both biasing and scaling is that they capture different relationships between two inputs: intuitively, biasing functions as a “OR” operation that checks if a feature is in either one of the two inputs, 
where scaling functions as a “AND” operation that checks if a feature is in both of the two inputs. Together they allow auxiliary feature to be better utilized in the denoising process.

Here is a comparison between the neural denoising method and alternative methods at 4 SPP. As you can see, the neural denoiser method produces less artifacts and better details.


##### Neural Importance Sampling

So far we have been talking about denoiser as the value based approach. Now let's take a look at the policy based approach.

Neural importance sampling is a technique invented at Disney research. the idea is that, for every location in the scene, we'd like to have a policy that guides the sampling process so the rendering converges faster.

In practice, the best sampling policy we can ask for an arbitary location is the incidence radiance map at that location. Because it tells you where the lights come from.

The question is, how to generate such incidence radiance map for every location in the 3D scene?

The answer can be found in the generative networks literature. Just like how image can be generated from vectors randomly sampled from a normal distribution, incidence radiance map can be generated from a vector of surface properties, include the coordinate of the intersection, the normal of the intersection, the direction of the incoming ray etc. 

The generative network is trained to map the surface properties to the incidence radiance map. 

One unique challenge is the mapping between the surface properties and incidence radiance maps varies from scene to scene. So the learning of the policy network is carried online during the rendering. Meaning the network starts from generating random policies, and incrementally gets better at understanding the scene, and produces more efficient policy accordingly. 

Here is a side-by-side comparison between a regular ray tracing and the ray tracing with neural importance sampling. You can see at low samples per pixel, neural importance sampling is able to achieve much better result.




#### Neural Networks as an End-to-End Forward Rendering Pipeline

So far we have been talking about neural networks as sub-modules for Monte Carlo ray tracing. Next, we will use it as an end-to-end solution. 

Recall ray tracing casts light rays from pixels to object surfaces. This is an “image centric” approach. 

There is a different approach called rasterization, which cast rays from object surfaces to pixels. This is an “object centric” approach. 

There are two main steps in Rasterization: compute visibility and compute shading. To compute visibility, we impose the projected primitives on top of each other based on their distance to the camera, so the front-most objects can be visible. The shading process computes the color of each pixel. It does so by interpolating the color of the vertices. 

Rasterization is in general faster than ray tracing because it only use primary ray. It is also easier for neural networks to learn because it does not use sampling or recursion. All sounds great 



### Inverse Neural Rendering

#### Differentiable Rendering

#### Neural Differentiable Rendering
