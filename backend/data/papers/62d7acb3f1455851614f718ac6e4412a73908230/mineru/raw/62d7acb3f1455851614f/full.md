# EXPOSING VULNERABILITIES IN LATENT-NOISE DIFFUSION WATERMARKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Watermarking techniques are crucial for protecting intellectual property and preventing the fraudulent use of media. Recently, a prominent approach to watermarking diffusion models relies on embedding a secret key in the initial noise. The resulting pattern is often considered hard to forge into unrelated images and remove. In this paper, we make a key observation that there is an inherent manyto-one mapping between images and initial noises. Therefore, there are regions in the clean image latent space pertaining to each watermark that get mapped to the same initial noise when inverted. We expose this as a vulnerability by proposing a black-box adversarial attack that uses only a single watermarked image without presuming access to any diffusion model. Our forgery attack simply adds perturbations to unrelated, potentially harmful images so that they would enter the region of watermarked images and get falsely labeled as watermarked. We show that a similar approach can also be applied to watermark removal by learning perturbations to exit from this region. We report results on multiple watermarking schemes (Tree-Ring, RingID, WIND, and Gaussian Shading). Our results demonstrate the effectiveness of the attack and expose vulnerabilities in current watermarking methods, motivating future research on improving them.

# 1 INTRODUCTION

Latent diffusion models have made significant strides in terms of producing realistic-looking images. However, this advancement comes with its own set of problems, primarily related to image provenance. Image forensics experts are tasked with verifying which generative model generated a particular image, if any, and who is responsible for any harm arising from it. Researchers and policymakers alike have focused on watermarking generative models as a potential solution to the threat from these models. This approach involves embedding an imperceptible pattern into images, allowing them to verify whether or not an image was generated using a specific model.

Image watermarking is a well-studied field (Potdar et al., 2005; Hartung and Kutter, 1999; Podilchuk and Delp, 2001); however, recent advances in generative modeling have transformed both watermarking techniques and attacks against them. To improve robustness against removal attacks, leading watermarking schemes in diffusion models have focused on embedding a secret key into the initial noisy latent vector (Wen et al., 2023; Ci et al., 2024b; Arabi et al., 2024; Yang et al., 2024b). This enables a model owner to generate a realistic image using the standard denoising process without altering the model weights or compromising image quality. During verification, the denoising diffusion implicit model (DDIM) inversion Mokady et al. (2023) is used to invert the image, recovering the initial noise sample and the initial pattern that may have been embedded. These methods have shown promising results in avoiding watermark removal. Yet, as we show in this paper, the process remains vulnerable to adversaries.

Attacks on watermarks fall into two main categories: forgery attacks (Yang et al., 2024a; Kaur et al., 2023; Wang et al., 2021) and removal attacks (Zhao et al., 2025; Lukas et al., 2023; Liu et al., 2024; Yang et al., 2024a; Hu et al., 2024; An et al., 2024). Forgery attacks attempt to steal the watermark and apply it to content unrelated to the model’s owner, raising concerns about false attribution of harmful content. Removal attacks aim to remove the watermark while preserving the image content, raising concerns about intellectual property infringement or harmful use of the generated content to mislead the public. Previous attack methods have shown some success, but their

![](images/95e4f0c7e864aa39111b43dc460bebcd3ee331a7d70bb904e48d077f2e8c8da2.jpg)  
Figure 1: Intuition behind our attack: Within the latent space, an entire region maps to approximately the same key-embedded initial latent noise vector. An attacker only needs to ensure their sample embeds within this region to be falsely classified as watermarked.

success relied on one or more of the following conditions: (i) collecting a large set of watermarked images (and possibly, their non-watermarked counterparts), (ii) access to the entire model weights, or some approximation of it, and (iii) significantly distorting the attacked images.

In this paper, we start by making a key observation about the dependence on the DDIM noise inversion $\mathtt { ( f l o k a d y ~ e t ~ a l . ) } \mathtt { \dot { 2 } 0 2 3 } \mathtt { ) }$ . It is the fact that different prompts can utilize the same initial noise during generation, but the DDIM inversion process to recover the secret key takes place with an empty prompt. This fact implies that there is a many-to-one relationship from the clean denoised latent space to the initial noise latent space. This, in turn, suggests that there might be a non-trivial region in a Variational Auto-Encoder (VAE)’s latent space that corresponds to each watermark. We showcase this idea in Figure 1. We also verify that this hypothesis is true by using a linear support vector machine model to find latent directions corresponding to watermarked and non-watermarked images. Traversing along these directions allows easy forgery and removal of the watermark signal.

Building upon this, we propose an adversarial attack to forge a given watermark, wherein the attacker simply needs to find adversarial perturbations such that a target image gets embedded close enough to a watermarked example in the VAE’s latent representation space. We show that an imperceptible modification to the image, which does not alter its semantic content, is sufficient due to the inherent non-smoothness of the VAE’s representation space (Cemgil et al., 2020) Once this objective is achieved, the DDIM inversion process with an empty prompt will guide both of these latents to a similar initial noise state, successfully fooling the watermark detection system. We show a pictorial representation of our methodology in Figure 2. Moreover, we show that a similar method can be used for a removal attack, removing the watermark from an image while preserving the image content. In this scenario, our attack objective is to ensure that the latent representation for a watermarked image gets as close as possible to a non-watermarked image region, so that it leads to a false negative match when inverted.

Our method has several strengths, which make it easier for an adversary to attack the watermarking system. (1) Our method needs only one watermarked image to forge or remove a watermark. (2) Our method can achieve it without inverting an image to its initial noise state to tamper with the secret key, which would require access to a denoising network approximating the one used in generation. This distinction is key, as the denoising network (U-Net) is often fine-tuned for purposes such as filtering out not-safe-for-work or copyrighted content. (3) Although our method does require access to a VAE, it does not necessarily need access to the same VAE that was used by the watermarked diffusion model. A VAE that was trained on a similar dataset can suffice for this task.

To summarize, we make the following contributions:

• Identifying Watermarked Latent Subspaces – We show that there exist latent directions corresponding to each watermark pattern.   
• Watermark Forgery and Removal Attacks – We propose attacks that rely only on a single watermarked image and a proxy VAE.   
• Impact Assessment – We evaluate and minimize the impact of these attacks on the quality of the target image.

![](images/590cb374840afc71b4433ec119e52169ee8a408ac7ceda368c34e7bca43570b9.jpg)  
Figure 2: Our forgery attack works by finding an adversarial perturbation $\delta$ such that the latent representation of the non-watermarked image $\mathcal { E } _ { \phi } ( \mathbf { x } ^ { ( c ) } + \pmb { \delta } )$ is close to the one corresponding to that of a watermarked image $\mathcal { E } _ { \phi } ( \mathbf { x } ^ { ( w ) } )$ . We do so while ensuring that we only introduce imperceptible changes to the clean image.

# 2 RELATED WORK

Watermarking Schemes for Diffusion Models. Watermarking is a technique that embeds a traceable mark in an image, enabling ownership verification and supporting media authentication and intellectual property protection.(Potdar et al., 2005; Hartung and Kutter, 1999; Podilchuk and Delp, 2001; Wong and Memon, 2001; Cox et al., 2007). Classical watermarking techniques have involved embedding an invisible pattern in an image that can be recovered. Similarly, in the context of generative models, watermarks can be embedded post hoc, after the image is generated (Wong and Memon, 2001; Tancik et al., 2020; Cox et al., 2007; Fernandez et al., 2022; Zhang et al., 2019). Alternatively, watermarking schemes in diffusion models have also focused on fine-tuning VAE decoders to watermark an output image (Ci et al., 2024a; Fernandez et al., 2023; Xiong et al., 2023)

A widely used approach to watermarking images involves embedding a secret key into the initial noise used by the diffusion model. Wen et al. $\underline { { \left( \overline { { 2 0 2 3 } } \right) } }$ , who initially proposed this idea, showed that initial noise-based watermarking is more robust against removal attacks - transformation aiming to render the watermark undetectable (Zhao et al., 2025). Ci et al. (2024b) further improved the watermarking pattern structure to make it more secure. Yang et al. $\textcircled { 1 2 0 2 4 6 } )$ utilized distributionpreserving sampling to ensure that the initial noise follows a Gaussian distribution, preserving the distribution of generated images. Arabi et al. $\textcircled { 1 2 0 2 4 }$ showed that breaking keys into groups and using an initial pattern specific to the groups enables embedding a larger number of secret keys, improving security. $\overline { { \mathrm { G u n n ~ e t ~ a l . } } } ( \overline { { 2 0 2 4 } } )$ used a pseudo-random error correcting code to initialize the initial noise sample. These methods have focused on generating distortion-free images that come from a similar distribution as non-watermarked images. It was often implicitly assumed that such distortion-free watermarks would be more secure against various types of attacks. However, as we will show, this is not always the case.

Forgery and Removal Attacks against Watermarks. Forging and removing watermarking has been an important research area to expose vulnerabilities in watermarking schemes, and therefore lead to their improvement. Zhao et al. $\textcircled { 2 0 2 5 } )$ demonstrated that many watermarks can be removed by simply noising and then denoising a watermarked image using a diffusion model; however, their approach was unsuccessful against the Tree-Ring (Wen et al., $\boxed { 2 0 2 3 }$ watermarking method. Yang et al. $\textcircled { 2 0 2 4 2 }$ showed that methods such as Wen et al. (2023) leave distinct textural patterns in the image, which can be found by averaging multiple watermarked images. Muller et al.¨ $\textcircled { 2 0 2 4 }$ used an auxiliary diffusion model to ensure the inverted initial noises from a watermarked and nonwatermarked image are closely aligned. WAVES (An et al., $\textcircled { 2 0 2 4 }$ benchmarked different watermarking and attack methods to judge their effectiveness. They also proposed a set of attacks to evaluate the robustness of various watermarking schemes. Saberi et al. $\boxed { ( 2 0 2 3 ) }$ proposed training a proxy watermarked image classifier to classify whether an image is watermarked or not. They conducted a progressive gradient-descent-based adversarial attack on the model and showed that

the perturbations were transferable to a black-box detection model. Liu et al. $\mathbb { Q } 0 2 4 )$ proposed regenerating an image similar to watermarked images from clean Gaussian noise so as to remove the watermark. Lukas et al. (2023) proposed using differentiable surrogate keys to learn attack parameters, which enables the removal of traces of watermarked keys from images. This assumes access to not only a surrogate key generator but also a copy of the generative model.

In contrast to Saberi et al. (2023); Yang et al. (2024a), our method does not require access to multiple watermarked images. They assume access to images not only from the same watermarking method but also from the same secret key, making these attacks less practical against some systems (Arabi et al., 2024). We can run our attack using just one watermarked image. Furthermore, unlike Muller ¨ et al. $\textcircled { 1 2 0 2 4 }$ Lukas et al. $\mathbb { Q } 0 2 3 )$ , we do not assume any access to a denoising diffusion model or a proxy version of it. Lastly, in contrast to $\boxed { \mathrm { Z h a o ~ e t ~ a l . } } ( \dot { \mathbb { Z } } 0 2 5 )$ , which was unsuccessful in removing the Tree-Ring watermark, we demonstrate that our method achieves it effectively.

# 3 PRELIMINARIES

Diffusion models (Song et al., 2020) such as Stable Diffusion (SD) (Rombach et al., 2022) and Imagen (Saharia et al., 2022) learn a mapping from an initial random noise state $\overline { { \mathbf { z } _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } ) } }$ to a clean image space $\mathbf { z } _ { 0 } \sim p _ { \mathrm { d a t a } }$ . This is done by iteratively applying a learned denoising network $\epsilon _ { \theta }$ such as U-Net or DiT. Popularly used models (Rombach et al., 2022) compress the image space to a lower-dimensional representation space using a variational autoencoder (an encoder ${ \mathcal { E } } _ { \theta }$ and decoder $\mathcal { D } _ { \theta _ { } }$ ) to reduce the amount of computation required for generating an image.

Using the learned noise estimator network $\epsilon _ { \theta }$ , DDIM’s sampling process (Song et al., 2020) computes the previous state $\mathbf { z } _ { t - 1 }$ from $\mathbf { z } _ { t }$ as follows:

$$
\mathbf {z} _ {t - 1} = \sqrt {\frac {\bar {\alpha} _ {t - 1}}{\bar {\alpha} _ {t}}} \mathbf {z} _ {t} - \left(\sqrt {\frac {1}{\bar {\alpha} _ {t - 1}} - 1} - \sqrt {\frac {1}{\bar {\alpha} _ {t}} - 1}\right) \epsilon_ {\theta} \left(\mathbf {z} _ {t}, t, \mathbf {e} _ {\mathrm {p}}\right), \tag {1}
$$

where $\beta _ { t }$ is defined by the noise scheduler and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } ( 1 - \beta _ { i } ) } \end{array}$ .

DDIM inversion (Mokady et al., 2023; Dhariwal and Nichol, 2021; Song et al., 2020) is a process to invert a clean sample $\mathbf { z } _ { 0 }$ to reconstruct its initial noise state $\mathbf { z } _ { T }$ based on the assumption that ${ \mathbf z } _ { t - 1 } - { \mathbf z } _ { t } \approx { \mathbf z } _ { t + 1 } - { \mathbf z } _ { t }$ . This allows us to estimate $\mathbf { z } _ { t + 1 }$ from $\mathbf { z } _ { t }$ using the formula,

$$
\mathbf {z} _ {t + 1} = \sqrt {\bar {\alpha} _ {t + 1}} \mathbf {z} _ {0} + \sqrt {1 - \bar {\alpha} _ {t + 1}} \boldsymbol {\epsilon} _ {\theta} (\mathbf {z} _ {t}, t). \tag {2}
$$

# 3.1 WATERMARKING SCHEME IN DIFFUSION MODELS

Most watermarking schemes in diffusion models consist of embedding a secret key $k \in \mathcal { K }$ in the initial noise $\mathbf { z } _ { T }$ used to generate an image. This is done in a manner such that the initial noise does not deviate significantly from a standard Gaussian distribution $\mathcal { N } ( 0 , \bf { I } )$ . The standard diffusion denoising process is followed to convert the given initial noise to a clean latent representation $\mathbf { z } _ { 0 }$ , which corresponds to a watermarked image $\mathbf { x } ^ { ( w ) }$ that will be perceptibly indistinguishable from non-watermarked images.

During detection, the DDIM inversion is used to estimate the initial noise sample $\mathbf { z } _ { T } ^ { \prime }$ from the clean sample $\mathbf { z } _ { 0 }$ . Once the initial noise is recovered, the key pattern (if any) is extracted and matched with the set of secret keys that a model owner used. It is important to note here that the inversion process is performed using an empty prompt, as the model owner typically does not keep track of the generated images or the prompts used. This process enables a model owner to watermark an image without requiring any modifications to the diffusion model architecture or weights. This approach has been shown to be robust against image transformations, which can significantly degrade other types of watermarking methods, specifically post-hoc watermarks (Wen et al., 2023).

# 4 WATERMARKING ATTACK TECHNIQUES

In this section, we introduce our watermarking attack techniques. We first define the threat model that we consider in Section $\boxed { 4 . 1 }$ followed by explaining the motivation for our approach in Section 4.2. Lastly, we describe the adversarial attack itself in Section 4.3.

![](images/196f83befcdb13db3b6c3593dd2eb4522806654608d02b01fcb76aae81b82f6b.jpg)  
Figure 3: Motivation for our attack (not explanation of our proposed attack itself). A latent direction exists pertaining to watermarked latents derived from a specific secret key in the clean image latent space. The further we traverse in the relevant direction, the stronger the attack becomes. Our method, proposed in Section 4.3, exploits this vulnerability. See Appendix Figure 12 for more examples.

# 4.1 THREAT MODEL

We consider two parties: a model owner and an attacker. The model owner owns a generative model and controls the generation such that it outputs watermarked images. The attacker is a party with malicious intentions that seeks to tamper with the watermarking system. One type of attacker may wish to forge the watermark into an unrelated image, to falsely claim that a harmful image was generated by the model owner. Another type of attacker may attempt to remove the watermark pattern from a previously watermarked image. This could be done to falsely claim ownership or to spread misinformation by concealing the image’s synthetic origin, making it harder to detect it as AI-generated content (e.g., to create deepfakes). Formally:

• Model Owner (Generation Phase): The model owner owns a diffusion model $\epsilon _ { \theta }$ and uses a random secret key $k$ from a set of secret keys $\kappa$ to generate a latent noise-based watermarked image $\mathbf { x } ^ { ( w ) }$ .   
• Attacker: The attacker wishes to use a single watermarked image $\mathbf { x } ^ { ( w ) }$ to falsely watermark a clean image $\mathbf { x } ^ { ( c ) }$ or to remove the watermark in $\mathbf { x } ^ { ( w ) }$ while preserving their contents. The attacker does not have access to the model $\epsilon _ { \theta }$ used by the model owner or the secret key $k$ embedded by the model owner. We assume the watermarking method is a latent-noise-based one, and the attacker has access to a proxy VAE that was trained on a similar dataset. The attacker can get access to a watermarked sample by sampling the model, but every additional sampling is assumed to lead to an image with a different watermark key.   
• Model Owner (Detection Phase): The model owner is asked to verify whether or not an image provided by the attacker was generated using their diffusion model $\epsilon _ { \theta }$ by matching the extracted key to their set of secret keys $\kappa$ .

# 4.2 MOTIVATION

Our approach is based on the intuition that the mapping from generated images to initial noise is inherently manyto-one, as the same initial noise sample can produce many different images when denoised using different prompts. In the case of watermark detection, the DDIM inversion process is performed using an empty prompt, meaning that all images generated with a specific secret key, regardless of the original text prompt, are expected to be inverted to recover that key. Based on this, we hypothesize that within the clean sample latent space, there exists a region that consistently maps to an initial noise pattern corresponding to the key. Our method focuses on exploiting this vulnerability: i.e., if we can successfully embed our non-watermarked sample in the watermarked region, we will be able to falsely claim that a non-watermarked

![](images/405c72c8044cd94c523174538123ce5ce4078dc33847548b6cece8bb74f592fe.jpg)  
Figure 4: Two-dimensional visualization of the latent space showing the linear separability of watermarked and non-watermarked images. The horizontal axis is obtained by linear discriminant analysis (LDA), while the vertical axis is a random projection.

image is watermarked. Similarly, if we can push a watermarked image away from this region, we will be able to falsely claim that it is not watermarked.

To show that such a region exists, we start by designing a simple experiment under idealized conditions assuming only a single secret key. If there exists a latent region pertaining to a different set of watermarked images from each secret key, then we should be able to find latent directions that can lead randomly sampled latent vectors into these regions and thus be authenticated as being watermarked. To demonstrate this, we sampled 1,000 watermarked images from the Tree-Ring watermark using a specific secret key, along with the same number of non-watermarked images. We trained a linear support vector machine (SVM) model on this dataset to find such a direction in VAE’s latent space (Colbois et al., 2021). We observed that simply traversing the latent space in the direction normal to the learned hyperplane could lead a non-watermarked sample towards being classified as watermarked, as shown in Figure 3. We can also remove the watermark from a watermarked image by traversing the latent space in the opposite direction. We formally define this region in Definition 1.

Definition 1 (Watermark Region) For a watermarking method $\mathcal { W }$ , we define a watermark region as a region in the clean latent space of the latent diffusion model, as,

$$
Z _ {0} ^ {(w)} (\mathcal {W}, k) = \left\{\mathbf {z} _ {0} \in Z _ {0} \mid \mathcal {M} _ {\mathcal {W}} (\mathcal {I} ^ {-} (\mathbf {z} _ {0}), k) <   \tau \right\},
$$

where $Z _ { 0 }$ represents the clean latent space at $t = 0$ , $\mathcal { T } ^ { - }$ is the DDIM inversion process, $\mathcal { M } _ { \mathcal { W } }$ is the matching function used to verify the presence of a particular key, $\tau$ is the operating threshold of the watermarking scheme, and k is the secret key that was embedded.

This represents all the points in the clean latent space that lead to the key k that was embedded in the initial noise latent vector when inverted.

We also visualize high linear separability between watermarked images and non-watermarked ones in Figur e 4 . Yet, this edit in itself is not close to a real-world scenario because we used multiple watermarked samples from the same key and because we could not maintain the semantic content and quality of the corresponding image. This preliminary experiment indicates that such a watermarked region as defined in Definition 1 may exist, and it sets the stage for our novel attack strategy, which we describe in the next subsection. Our strategy aims to introduce imperceptible changes to the image while pushing and pulling latent representations into and out of watermarked regions for forgery and removal, respectively.

# 4.3 IMPERCEPTIBLE AND LIGHTWEIGHT ATTACK AGAINST WATERMARKING METHODS

Forgery Attack. Based on the above intuition, to forge a watermark, the attacker needs to adversarially perturb a non-watermarked image so that it gets embedded in the watermarked region (Definition 1) of the clean latent space. However, finding and defining this region may require multiple generations from the same initial noise vector that has the secret key embedded in it. This is not feasible in scenarios where a model owner can regenerate the key every time (Arabi et al., 2024).

Instead, we propose a method that forges a watermark into the non-watermarked image by guiding it into the watermarked region using only a single watermarked image so that we minimize the distance between the latent representation of a non-watermarked image and that of a watermarked image while regularizing for content preservation. Our method utilizes the encoder of an off-theshelf VAE $\mathcal { E } _ { \phi }$ , which was trained on a similar dataset (which can be different from the VAE of the diffusion model ${ \mathcal { E } } _ { \theta }$ ), to adversarially perturb a non-watermarked image.

The objective for finding the perturbation is defined as:

$$
\min  _ {\boldsymbol {\delta}} \left\| \mathcal {E} _ {\phi} \left(\mathbf {x} ^ {(c)} + \boldsymbol {\delta}\right) - \mathcal {E} _ {\phi} \left(\mathbf {x} ^ {(w)}\right) \right\| _ {2} + \lambda \| \boldsymbol {\delta} \| _ {2}, \tag {3}
$$

where $\pmb { \delta }$ is the adversarial perturbation and $\lambda$ controls the trade-off between the strength of the perturbation and image content preservation. In Appendix $\boxed { \mathrm { A } }$ we present an ablation study on the loss function design, comparing with other possible designs such as progressive gradient descent (PGD) (Kurakin et al., 2016; 2018)

Removal Attack. We adopt a similar approach to remove a watermark by adversarially perturbing a watermarked image so that we can ensure that it gets outside of the vulnerable watermarked region.

![](images/36063ba5075d1e5dbd34ef08bdec76563f022e97a538359a8f6c3c44c1e8a25c.jpg)

![](images/ed8fd0ee0057b6bc0398f1a1015897c41b3d14ceb67c4dc3f636ecb59207c330.jpg)

![](images/894078bc176cb4a3f4979068a1f8867b6a61b645f822b7569b41b5eaab602108.jpg)

![](images/75b137e20e4951649fa3965b8afd9188854c84c809e0ce87630f6571646a8e49.jpg)  
Figure 5: Visual comparison of our forgery attacks on the Tree-Ring watermarking method.

Table 1: Comparison with baselines on the attack success rate (ASR) and imperceptibility of the attack as judged using the $l _ { 2 }$ , $l _ { \infty }$ distances and the LPIPS (Zhang et al., 2018), SSIM (Wang et al., 2004) and PSNR, when falsely watermarking non-watermarked images from the COCO2017 dataset assuming access to SDv1.4’s VAE. The hyperparameter $\lambda$ in Equation $3$ is set to $1 \times 1 0 ^ { 4 }$ .   

<table><tr><td>Method</td><td>Model</td><td>Method</td><td>ASR</td><td>l2</td><td>l∞</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td></tr><tr><td rowspan="6">Tree-Ring</td><td rowspan="3">SDv1.4</td><td>Yang et al. (2024a)</td><td>0.0</td><td>73.69</td><td>0.29</td><td>0.02</td><td>0.94</td><td>27.61</td></tr><tr><td>Müller et al. (2024)</td><td>100.0</td><td>115.22</td><td>1.51</td><td>0.13</td><td>0.70</td><td>23.03</td></tr><tr><td>Ours</td><td>91.06</td><td>63.22</td><td>1.10</td><td>0.33</td><td>0.76</td><td>28.87</td></tr><tr><td rowspan="3">SDv2.0</td><td>Yang et al. (2024a)</td><td>0.0</td><td>68.25</td><td>0.28</td><td>0.04</td><td>0.94</td><td>28.28</td></tr><tr><td>Müller et al. (2024)</td><td>100.0</td><td>114.42</td><td>1.50</td><td>0.13</td><td>0.70</td><td>23.08</td></tr><tr><td>Ours</td><td>93.81</td><td>63.78</td><td>1.08</td><td>0.34</td><td>0.76</td><td>28.78</td></tr><tr><td rowspan="6">Gaussian Shading</td><td rowspan="3">SDv1.4</td><td>Yang et al. (2024a)</td><td>0.0</td><td>107.05</td><td>0.32</td><td>0.04</td><td>0.92</td><td>24.37</td></tr><tr><td>Müller et al. (2024)</td><td>100.0</td><td>116.38</td><td>1.50</td><td>0.13</td><td>0.70</td><td>22.96</td></tr><tr><td>Ours</td><td>96.85</td><td>37.27</td><td>0.70</td><td>0.19</td><td>0.87</td><td>33.48</td></tr><tr><td rowspan="3">SDv2.0</td><td>Yang et al. (2024a)</td><td>0.0</td><td>83.51</td><td>0.33</td><td>0.05</td><td>0.94</td><td>26.52</td></tr><tr><td>Müller et al. (2024)</td><td>100.0</td><td>116.47</td><td>1.49</td><td>0.13</td><td>0.70</td><td>22.95</td></tr><tr><td>Ours</td><td>100.0</td><td>36.78</td><td>0.66</td><td>0.19</td><td>0.87</td><td>33.60</td></tr></table>

Here, instead of using a real camera-captured non-watermarked image, we propose using a plain image whose pixel values are all set to the mean of the watermarked image $\mathbf { x } ^ { ( w ) }$ . We do so because (i) the average image is naturally non-watermarked and does not rely on any specific external image for guidance and (ii) real images contain their own high-frequency information, which can lead to larger perturbations in the optimized adversarial image. We summarize the removal objective as:

$$
\min  _ {\boldsymbol {\delta}} \left\| \mathcal {E} _ {\phi} \left(\mathbf {x} ^ {(w)} + \boldsymbol {\delta}\right) - \mathcal {E} _ {\phi} \left(\boldsymbol {\mu} _ {\mathbf {x} ^ {(w)}}\right) \right\| _ {2} + \lambda \| \boldsymbol {\delta} \| _ {2}, \tag {4}
$$

where ${ \pmb \mu } _ { \mathbf { x } } ( \ b w )$ is the plain image with all pixel values equal to the mean of the watermarked image $\mathbf { x } ^ { ( w ) }$ . In Appendix ${ \dot { \mathbf { C } } } ,$ we present an ablation study that justifies this design choice.

# 5 EXPERIMENTAL RESULTS

In this section, we showcase the effectiveness of our approach. We consider two attack scenarios, where the attacker has access to either (i) the VAE of the watermarked diffusion model or (ii) a proxy VAE that was trained on a similar dataset.

Experimental Setup. We consider two diffusion models, namely Stable Diffusion v1.4 (SDv1.4) and Stable Diffusion v2.0 (SDv2.0), to generate images of size $5 1 2 \times 5 1 2$ . We generate watermarked images using the prompts available in the Gustavosta/Stable-Diffusion-Prompts. When generating reference watermarked images for forgery attacks, we use simpler prompts from the

Table 2: Comparison with baselines on watermark removal on the attack success rate (ASR) and the imperceptibility of the changes using the $l _ { 2 }$ distance and the LPIPS Zhang et al. (2018), SSIM Wang et al. $\textcircled { 2 0 0 4 }$ , and PSNR metrics. The hyperparameter $\lambda$ in Equation 4 is set to $\overline { { 1 \times 1 0 ^ { 4 } } }$ .   

<table><tr><td>Method</td><td>Model</td><td>Method</td><td>ASR</td><td>l2</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td></tr><tr><td rowspan="8">Tree-Ring</td><td rowspan="4">SDv1.4</td><td>Yang et al. (2024a)</td><td>1.15</td><td>115.06</td><td>0.09</td><td>0.90</td><td>19.97</td></tr><tr><td>Zhao et al. (2025)</td><td>5.05</td><td>148.71</td><td>0.18</td><td>0.71</td><td>17.73</td></tr><tr><td>Müller et al. (2024)</td><td>100.0</td><td>391.50</td><td>0.64</td><td>0.46</td><td>12.92</td></tr><tr><td>Ours</td><td>98.84</td><td>62.87</td><td>0.30</td><td>0.78</td><td>28.91</td></tr><tr><td rowspan="4">SDv2.0</td><td>Yang et al. (2024a)</td><td>2.60</td><td>118.28</td><td>0.12</td><td>0.88</td><td>19.42</td></tr><tr><td>Zhao et al. (2025)</td><td>6.0</td><td>142.63</td><td>0.17</td><td>0.69</td><td>18.80</td></tr><tr><td>Müller et al. (2024)</td><td>100.0</td><td>418.21</td><td>0.68</td><td>0.39</td><td>12.39</td></tr><tr><td>Ours</td><td>98.36</td><td>67.62</td><td>0.30</td><td>0.77</td><td>28.02</td></tr><tr><td rowspan="8">Gaussian Shading</td><td rowspan="4">SDv1.4</td><td>Yang et al. (2024a)</td><td>4.0</td><td>131.43</td><td>0.07</td><td>0.89</td><td>19.78</td></tr><tr><td>Zhao et al. (2025)</td><td>3.0</td><td>114.86</td><td>0.13</td><td>0.74</td><td>20.77</td></tr><tr><td>Müller et al. (2024)</td><td>100.0</td><td>499.68</td><td>0.74</td><td>0.31</td><td>10.87</td></tr><tr><td>Ours</td><td>70.10</td><td>74.10</td><td>0.29</td><td>0.77</td><td>24.12</td></tr><tr><td rowspan="4">SDv2.0</td><td>Yang et al. (2024a)</td><td>0.0</td><td>84.78</td><td>0.04</td><td>0.95</td><td>26.33</td></tr><tr><td>Zhao et al. (2025)</td><td>0.0</td><td>113.73</td><td>0.11</td><td>0.70</td><td>23.34</td></tr><tr><td>Müller et al. (2024)</td><td>100.0</td><td>524.10</td><td>0.77</td><td>0.28</td><td>10.44</td></tr><tr><td>Ours</td><td>59.13</td><td>68.73</td><td>0.27</td><td>0.77</td><td>28.13</td></tr></table>

runwayml-stable-diffusion-v1-5-eval-random-prompts dataset, as the resultant images contain more visible watermark patterns/signal due to lower amounts of high-frequency information. We use the COCO2017 dataset (Lin et al., $\textcircled { 2 0 1 4 }$ to obtain images without watermarks. We utilize the VAE from SDv1.4 to forge/remove a watermark generated from both SDv1.4 and SDv2.0, unless otherwise stated. When we use $\mathrm { S D v } 2 . 0$ to generate watermarked images, SDv1.4’s VAE serves as a proxy VAE. We report results on the following publicly available watermarking systems that embed a key in the initial noise space, namely, Tree-Ring (Wen et al., 2023), RingID (Ci et al., 2024b), WIND (Arabi et al., 2024), and Gaussian Shading (Yang et al., 2024b). See Appendix F for further details.

Evaluation Metrics. We consider a forgery attack to be successful if the $p$ -value statistical test comparing the extracted key and the secret key embedded in the reference watermarked image yields a $p$ -value less than 0.05. For removal, we consider it a success if the $p$ -value is greater than 0.05 with respect to the key in the original watermarked image. For Gaussian Shading, which uses a bit sequence, we compute the binary bit accuracy between the embedded and recovered keys. We report the average success rate across 200 examples while randomly drawing new non-watermarked and watermarked samples each time. Each watermarked sample is generated using a new random key.

Even though we test our attack in a black-box scenario, we do not make multiple attempts and consider it unsuccessful if the first attempt fails. We additionally report the $l _ { 2 }$ , $l _ { \infty }$ distances and the Learned Perceptual Image Patch Similarity (LPIPS) $( \mathbb { Z } \mathrm { h a n g ~ e t ~ a l . } ) , \bar { 2 0 1 8 } )$ , Structural Similarity Index (SSIM) (Wang et al., $\boxed { 2 0 0 4 }$ , and Peak Signal to Noise Ratio (PSNR) metrics between the original image and the adversarially perturbed image to assess the extent to which we alter the original image.

Results - Forgery. We summarize quantitative results in Table 1 for Tree-Ring and Gaussian Shading, comparing our attack with baselines. In Appendix Table 3, we present results on the trade-off between imperceptibility and attack success rate as well as results for RingID and WIND watermarks. Our method achieves performance comparable to that of Muller et al.¨ $\underline { { \left. 2 0 2 4 \right. } }$ at half the $l _ { 2 }$ distance. As shown in Figure $\boxed { 5 }$ our method better preserves the image content compared to Muller ¨ et al. (2024), since we perturb the denoised latent space with regularization. We showcase more qualitative examples in the Appendix G. We also show results on using the VAE from FLUX.1-dev in Appendix Table $\bigtriangledown$ Note that the more similar the encoder is to the one used by the model, the less noise is required.

Results - Removal. We report results on watermark removal in Table 2 for Tree-Ring and Gaussian Shading, comparing with baselines, and in Appendix Table $\boxed { 4 }$ for RingID and WIND watermarks. Although our approach achieves success in removing the Tree-Ring watermark, it was harder to remove the watermark signal from RingID and WIND, as these methods embed a watermark into the entire initial latent noise space, which introduces a larger amount of watermark signal into images (see Section 6). We show qualitative results when removing the Tree-Ring watermark in Figure 6.

Results - Computational time. Our attack takes 7.82 minutes per image while Muller et al. (2024) ¨ takes 12.29 minutes on one A100 GPU on average across 100 attacks.

![](images/5092f864f684bdd4ef12a9497df55b15dd2b8c5774f1add8b086c88e65b7903c.jpg)

![](images/09c734091c3467dc5e64b13a3377f265e491d3b852b3d2fa91ff5c2af4f4b079.jpg)  
Original watermarked

![](images/0adabfa015ad7dd2a13de502b0ccfedfd6d637e6d21452b4feab4631a90779d5.jpg)

![](images/4c8b36dd95f740a359f64bb0098fac43cd6f4f369fff15e8f76ee104bcb63c9a.jpg)  
Muller et al. (2024) ¨

![](images/11eb36930b44036759bfdd7f5d0df7aa3e13259a0d5e9eac929c18aeb4f367ea.jpg)

![](images/1e73dcb2a8a2ef37464c48802f6abb1cd041930fda37d8db52b36cd8740754e9.jpg)  
Ours $( \lambda = 5 \times 1 0 ^ { 4 }$

![](images/5d47a1dcddcc96972e7e2c8c40cea54ce24060901e22bb11d5f20e78effab0fa.jpg)

![](images/eb194ac372ff63bd81513f569d7c0682981765306c80346a18309096e2577284.jpg)  
Ours $( \lambda = 1 \times 1 0 ^ { 4 }$ )   
Figure 6: Qualitative comparison of our removal attack on the Tree-Ring watermarking method.

Results - Ablation Studies on Hyperparameter Values We sweep the hyperparameter $\lambda$ to see the trade-off between the attack success rate (ASR) and imperceptibility of watermarking attacks. We report the experimental results in Tables 3 and 4, which show that a larger $\lambda$ value (a stronger regularization) leads to a better preservation of image content at the cost of a lower ASR.

Table 3: Forgery attack performance trade-off between attack success rate (ASR) and imperceptibility of the attack as judged using the $l _ { 2 }$ , $l _ { \infty }$ distances and the Learned Perceptual Image Patch Similarity (LPIPS) $\mathrm { ( } \mathrm { \overline { { Z h a n g ~ e t ~ a l . } } } \mathrm { , \overline { { 2 0 1 8 } } \mathrm { ) } }$ , Structural Similarity Index (SSIM) $\mathtt { ( W a n g e t a l . ) } \mathtt { [ P a n g e t . o n . }$ , and Peak Signal to Noise Ratio (PSNR) metrics. We use non-watermarked images from the COCO2017 dataset and employ the VAE from SDv1.4 for optimization. The hyperparameter $\lambda$ in Equation 3 controls the trade-off between ASR and the amount of perturbation we introduce.

<table><tr><td colspan="2">Method</td><td>Model</td><td>λ</td><td>ASR</td><td>l2</td><td>l∞</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td></tr><tr><td rowspan="6" colspan="2">Tree-Ring (Wen et al., 2023)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>78.65</td><td>33.90</td><td>0.69</td><td>0.17</td><td>0.89</td><td>34.32</td></tr><tr><td>2 × 104</td><td>86.93</td><td>48.42</td><td>0.89</td><td>0.26</td><td>0.82</td><td>31.20</td></tr><tr><td>1 × 104</td><td>91.06</td><td>63.22</td><td>1.10</td><td>0.33</td><td>0.76</td><td>28.87</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>79.89</td><td>34.09</td><td>0.69</td><td>0.17</td><td>0.88</td><td>34.26</td></tr><tr><td>2 × 104</td><td>90.72</td><td>48.83</td><td>0.91</td><td>0.26</td><td>0.82</td><td>31.11</td></tr><tr><td>1 × 104</td><td>93.81</td><td>63.78</td><td>1.08</td><td>0.34</td><td>0.76</td><td>28.78</td></tr><tr><td rowspan="6" colspan="2">RingID (Ci et al., 2024b)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>100.0</td><td>38.45</td><td>0.68</td><td>0.20</td><td>0.87</td><td>33.21</td></tr><tr><td>2 × 104</td><td>100.0</td><td>55.20</td><td>0.86</td><td>0.30</td><td>0.80</td><td>30.06</td></tr><tr><td>1 × 104</td><td>100.0</td><td>73.08</td><td>1.03</td><td>0.38</td><td>0.73</td><td>27.63</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>100.0</td><td>37.31</td><td>0.66</td><td>0.19</td><td>0.87</td><td>33.48</td></tr><tr><td>2 × 104</td><td>100.0</td><td>53.94</td><td>0.84</td><td>0.29</td><td>0.80</td><td>30.27</td></tr><tr><td>1 × 104</td><td>100.0</td><td>71.53</td><td>1.00</td><td>0.37</td><td>0.73</td><td>27.82</td></tr><tr><td rowspan="6" colspan="2">WIND (Arabi et al., 2024)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>97.56</td><td>38.82</td><td>0.70</td><td>0.20</td><td>0.87</td><td>33.11</td></tr><tr><td>2 × 104</td><td>97.56</td><td>56.13</td><td>0.89</td><td>0.29</td><td>0.80</td><td>29.88</td></tr><tr><td>1 × 104</td><td>97.56</td><td>74.66</td><td>1.06</td><td>0.38</td><td>0.73</td><td>27.38</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>100.0</td><td>37.47</td><td>0.67</td><td>0.19</td><td>0.87</td><td>33.45</td></tr><tr><td>2 × 104</td><td>100.0</td><td>54.18</td><td>0.84</td><td>0.28</td><td>0.80</td><td>30.23</td></tr><tr><td>1 × 104</td><td>100.0</td><td>71.86</td><td>0.99</td><td>0.37</td><td>0.74</td><td>27.78</td></tr><tr><td rowspan="6" colspan="2">Gaussian Shading (Yang et al., 2024b)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>96.85</td><td>37.27</td><td>0.70</td><td>0.19</td><td>0.87</td><td>33.48</td></tr><tr><td>2 × 104</td><td>96.96</td><td>54.00</td><td>0.88</td><td>0.29</td><td>0.80</td><td>30.21</td></tr><tr><td>1 × 104</td><td>96.96</td><td>71.97</td><td>1.05</td><td>0.37</td><td>0.73</td><td>27.64</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>100.0</td><td>36.78</td><td>0.66</td><td>0.19</td><td>0.87</td><td>33.60</td></tr><tr><td>2 × 104</td><td>100.0</td><td>52.99</td><td>0.85</td><td>0.29</td><td>0.80</td><td>30.42</td></tr><tr><td>1 × 104</td><td>100.0</td><td>69.83</td><td>1.02</td><td>0.37</td><td>0.74</td><td>28.02</td></tr></table>

# 6 DISCUSSION AND LIMITATION

Multi-Pattern vs. Single-Pattern Watermarks. We found that forging was generally easier for our method than watermark removal across all approaches, particularly for RingID, and WIND watermarks where our attack was unsuccessful in removing the watermark signal. We suggest that this is because these methods do not merely encode a single pattern, like Tree-Ring, but also encode

Table 4: Watermark removal attack performance trade-off between attack success rate (ASR) and imperceptibility of the attack as judged using the $l _ { 2 }$ , $l _ { \infty }$ distances and the Learned Perceptual Image Patch Similarity (LPIPS) $\mathrm { ( } \mathrm { \overline { { Z h a n g ~ e t ~ a l . } } } \mathrm { , \overline { { 2 0 1 8 } } \mathrm { ) } }$ $\boxed { 2 0 1 8 }$ , Structural Similarity Index (SSIM) (Wang et al., $\boxed { 2 0 0 4 }$ , and Peak Signal to Noise Ratio (PSNR) metrics. We use the VAE from SDv1.4 for optimization. The hyperparameter $\lambda$ in Equation 4 controls the trade-off between ASR and the amount of perturbation we introduce.   

<table><tr><td>Method</td><td>Model</td><td>λ</td><td>ASR</td><td>l2</td><td>l∞</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td></tr><tr><td rowspan="6">Tree-Ring (Wen et al., 2023)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>94.21</td><td>34.13</td><td>0.87</td><td>0.15</td><td>0.89</td><td>34.23</td></tr><tr><td>2 × 104</td><td>97.68</td><td>47.93</td><td>1.09</td><td>0.23</td><td>0.83</td><td>31.27</td></tr><tr><td>1 × 104</td><td>98.84</td><td>62.87</td><td>1.31</td><td>0.30</td><td>0.78</td><td>28.91</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>95.08</td><td>40.88</td><td>0.98</td><td>0.15</td><td>0.88</td><td>31.71</td></tr><tr><td>2 × 104</td><td>97.80</td><td>53.56</td><td>1.15</td><td>0.23</td><td>0.82</td><td>29.82</td></tr><tr><td>1 × 104</td><td>98.36</td><td>67.62</td><td>1.37</td><td>0.30</td><td>0.77</td><td>28.02</td></tr><tr><td rowspan="6">RingID (Ci et al., 2024b)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>0.0</td><td>35.62</td><td>0.90</td><td>0.14</td><td>0.88</td><td>33.87</td></tr><tr><td>2 × 104</td><td>0.0</td><td>50.07</td><td>1.13</td><td>0.22</td><td>0.83</td><td>30.90</td></tr><tr><td>1 × 104</td><td>0.0</td><td>65.80</td><td>1.33</td><td>0.29</td><td>0.77</td><td>28.53</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>0.0</td><td>36.46</td><td>0.92</td><td>0.14</td><td>0.88</td><td>33.67</td></tr><tr><td>2 × 104</td><td>0.0</td><td>51.69</td><td>1.15</td><td>0.21</td><td>0.82</td><td>30.63</td></tr><tr><td>1 × 104</td><td>0.0</td><td>68.03</td><td>1.37</td><td>0.28</td><td>0.76</td><td>28.24</td></tr><tr><td rowspan="6">WIND (Arabi et al., 2024)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>0.0</td><td>48.11</td><td>1.06</td><td>0.15</td><td>0.87</td><td>29.59</td></tr><tr><td>2 × 104</td><td>0.0</td><td>60.46</td><td>1.21</td><td>0.23</td><td>0.81</td><td>28.30</td></tr><tr><td>1 × 104</td><td>0.0</td><td>74.47</td><td>1.37</td><td>0.29</td><td>0.75</td><td>26.90</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>0.0</td><td>35.99</td><td>0.93</td><td>0.14</td><td>0.89</td><td>33.79</td></tr><tr><td>2 × 104</td><td>0.0</td><td>50.79</td><td>1.16</td><td>0.21</td><td>0.83</td><td>30.79</td></tr><tr><td>1 × 104</td><td>0.0</td><td>66.86</td><td>1.37</td><td>0.28</td><td>0.77</td><td>28.40</td></tr><tr><td rowspan="6">Gaussian Shading (Yang et al., 2024b)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>11.41</td><td>34.68</td><td>0.86</td><td>0.13</td><td>0.89</td><td>33.95</td></tr><tr><td>2 × 104</td><td>39.79</td><td>63.76</td><td>1.15</td><td>0.23</td><td>0.81</td><td>23.99</td></tr><tr><td>1 × 104</td><td>70.10</td><td>74.10</td><td>1.31</td><td>0.29</td><td>0.77</td><td>24.12</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>12.23</td><td>38.21</td><td>0.98</td><td>0.13</td><td>0.88</td><td>33.17</td></tr><tr><td>2 × 104</td><td>34.73</td><td>53.55</td><td>1.19</td><td>0.21</td><td>0.83</td><td>30.26</td></tr><tr><td>1 × 104</td><td>59.13</td><td>68.73</td><td>1.39</td><td>0.27</td><td>0.77</td><td>28.13</td></tr></table>

additional information, such as the model owner’s identity or other metadata. As the number of possible embedded patterns increases, more encoded information is required to correctly identify the pattern. This, in turn, necessitates a stronger signal-to-noise ratio $\left( \overline { { \mathrm { S h a n n o n } } } , \overline { { \mathrm { 1 9 4 9 } } } \right)$ , where noise refers to patterns in the initial noise that are unrelated to our watermark. When more signal is associated with the embedded watermark, forging at least part of it becomes easier, while completely removing it becomes more difficult. In contrast, Tree-Ring embeds a simpler pattern, affecting only a portion of the initial noise, making it harder to forge under a fixed budget. Using higher detection thresholds will result in the opposite case, where forgery becomes harder and removal easier.

Baseline Distortion in RingID-type Methods. While evaluating image quality, we found that even unperturbed images generated by the RingID and WIND methods exhibit some distortion, as shown in Figure $\perp$ This issue was also discussed in Appendix C of the RingID manuscript $\left( \mathbb { C } \mathrm { i } \ \mathrm { e t } \ \mathrm { a l . } , \lfloor 2 0 2 4 \mathrm { b } \right)$ . The RingID method (and WIND, by incorporating RingID in one of its variants) mitigated these artifacts by embedding a watermark only in a single channel. However, in our evaluation, the distortion is often still noticeable and is more pronounced in

the case of simpler prompts. This further highlights the strength of the watermark signal embedded by these techniques, making it harder to remove.

![](images/bcf3e3e917c987a1e7f9ba5f76a217e053a1e686f6f3992ffaf4f2d0978d716a.jpg)  
Figure 7: Examples of images generated using RingID, where the watermark signal is visible even in the final generated image.

# 7 CONCLUSION

In this paper, we expose a vulnerability of initial noise-based watermarking schemes for diffusion models. We show that when a watermark key is embedded in the initial noise, a latent watermarked region may form in the denoised latent space. This makes it easier for an attacker to forge the watermark by perturbing a sample so that it lies within this region. We show that a similar approach can also be used for removal by pushing a watermarked latent away from this region. We hope this work motivates future research on improving watermarking systems in the face of adversaries.

# ETHICS STATEMENT

Watermarking methods are widely used for ensuring ethical use of media content, as they allow easy authentication of (a) who created the content and (b) whether or not the content was tampered with. The same is applied to generative models to verify whether or not an image was generated and, if so, by which model. This makes it vital to ensure that these systems are robust to adversaries. This paper exposes a vulnerability in such watermarking methods for diffusion models, with the hope that it will allow researchers to understand the current limitations of watermarking approaches and aid them in developing more robust ones.

# REPRODUCIBILITY STATEMENT

To facilitate reproducing the results from the paper, we have included the codebase of the paper in the supplementary materials. Further, we have provided all necessary implementation details, including hyperparameters, in the main paper and the Appendix.

# THE USE OF LARGE LANGUAGE MODELS (LLMS)

We have NOT utilized Large Language Models to (i) help with coming up with ideas for the paper, (ii) help with writing the code, or (iii) help with paper writing or polishing.

# REFERENCES

Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, et al. Waves: Benchmarking the robustness of image watermarks. In Forty-first International Conference on Machine Learning, 2024.   
Kasra Arabi, Benjamin Feuer, R Teal Witter, Chinmay Hegde, and Niv Cohen. Hidden in the noise: Two-stage robust watermarking for images. arXiv preprint arXiv:2412.04653, 2024.   
Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.   
Taylan Cemgil, Sumedh Ghaisas, Krishnamurthy Dj Dvijotham, and Pushmeet Kohli. Adversarially robust representations with smooth encoders. In International Conference on Learning Representations, 2020.   
Hai Ci, Yiren Song, Pei Yang, Jinheng Xie, and Mike Zheng Shou. Wmadapter: Adding watermark control to latent diffusion models. arXiv preprint arXiv:2406.08337, 2024a.   
Hai Ci, Pei Yang, Yiren Song, and Mike Zheng Shou. Ringid: Rethinking tree-ring watermarking for enhanced multi-key identification. In European Conference on Computer Vision, pages 338–354. Springer, 2024b.   
Laurent Colbois, Tiago de Freitas Pereira, and Sebastien Marcel. On the use of automatically generated syn- ´ thetic image datasets for benchmarking face recognition. In 2021 IEEE International Joint Conference on Biometrics (IJCB), pages 1–8. IEEE, 2021.   
Ingemar Cox, Matthew Miller, Jeffrey Bloom, Jessica Fridrich, and Ton Kalker. Digital watermarking and steganography. Morgan kaufmann, 2007.   
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.   
Pierre Fernandez, Alexandre Sablayrolles, Teddy Furon, Herve J ´ egou, and Matthijs Douze. Watermarking ´ images in self-supervised latent spaces. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3054–3058. IEEE, 2022.   
Pierre Fernandez, Guillaume Couairon, Herve J ´ egou, Matthijs Douze, and Teddy Furon. The stable signature: ´ Rooting watermarks in latent diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22466–22477, 2023.   
Sam Gunn, Xuandong Zhao, and Dawn Song. An undetectable watermark for generative image models. arXiv preprint arXiv:2410.07369, 2024.   
Frank Hartung and Martin Kutter. Multimedia watermarking techniques. Proceedings of the IEEE, 87(7): 1079–1107, 1999.

Yuepeng Hu, Zhengyuan Jiang, Moyang Guo, and Neil Gong. A transfer attack to image watermarks. arXiv preprint arXiv:2403.15365, 2024.   
Shuai Jia, Chao Ma, Taiping Yao, Bangjie Yin, Shouhong Ding, and Xiaokang Yang. Exploring frequency adversarial attacks for face forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4103–4112, 2022.   
Gurpreet Kaur, Navdeep Singh, and Munish Kumar. Image forgery techniques: a review. Artificial Intelligence Review, 56(2):1577–1625, 2023.   
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial machine learning at scale. arXiv preprint arXiv:1611.01236, 2016.   
Alexey Kurakin, Ian J Goodfellow, and Samy Bengio. Adversarial examples in the physical world. In Artificial intelligence safety and security, pages 99–112. Chapman and Hall/CRC, 2018.   
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and ´ C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014.   
Yepeng Liu, Yiren Song, Hai Ci, Yu Zhang, Haofan Wang, Mike Zheng Shou, and Yuheng Bu. Image watermarks are removable using controllable regeneration from clean noise. arXiv preprint arXiv:2410.05470, 2024.   
Nils Lukas, Abdulrahman Diaa, Lucas Fenaux, and Florian Kerschbaum. Leveraging optimization for adaptive attacks on image watermarks. arXiv preprint arXiv:2309.16952, 2023.   
Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, and Linlin Shen. Frequency-driven imperceptible adversarial attack on semantic similarity. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15315–15324, 2022.   
Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023.   
Andreas Muller, Denis Lukovnikov, Jonas Thietke, Asja Fischer, and Erwin Quiring. Black-box forgery attacks ¨ on semantic watermarks for diffusion models. arXiv preprint arXiv:2412.03283, 2024.   
Christine I Podilchuk and Edward J Delp. Digital watermarking: algorithms and applications. IEEE signal processing Magazine, 18(4):33–46, 2001.   
Vidyasagar M Potdar, Song Han, and Elizabeth Chang. A survey of digital image watermarking techniques. In INDIN’05. 2005 3rd IEEE International Conference on Industrial Informatics, 2005., pages 709–716. IEEE, 2005.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution ¨ image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
Mehrdad Saberi, Vinu Sankar Sadasivan, Keivan Rezaei, Aounon Kumar, Atoosa Chegini, Wenxiao Wang, and Soheil Feizi. Robustness of ai-image detectors: Fundamental limits and practical attacks. arXiv preprint arXiv:2310.00076, 2023.   
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479– 36494, 2022.   
Claude Elwood Shannon. Communication in the presence of noise. Proceedings of the IRE, 37(1):10–21, 1949.   
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.   
Matthew Tancik, Ben Mildenhall, and Ren Ng. Stegastamp: Invisible hyperlinks in physical photographs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2117–2126, 2020.

Ruowei Wang, Chenguo Lin, Qijun Zhao, and Feiyu Zhu. Watermark faker: towards forgery of digital image watermarking. In 2021 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2021.   
Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.   
Yuxin Wen, John Kirchenbauer, Jonas Geiping, and Tom Goldstein. Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust. arXiv preprint arXiv:2305.20030, 2023.   
Ping Wah Wong and Nasir Memon. Secret and public key image watermarking schemes for image authentication and ownership verification. IEEE transactions on image processing, 10(10):1593–1601, 2001.   
Cheng Xiong, Chuan Qin, Guorui Feng, and Xinpeng Zhang. Flexible and secure watermarking for latent diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1668– 1676, 2023.   
Pei Yang, Hai Ci, Yiren Song, and Mike Zheng Shou. Steganalysis on digital watermarking: Is your defense truly impervious? arXiv preprint arXiv:2406.09026, 2024a.   
Zijin Yang, Kai Zeng, Kejiang Chen, Han Fang, Weiming Zhang, and Nenghai Yu. Gaussian shading: Provable performance-lossless image watermarking for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12162–12171, 2024b.   
Kevin Alex Zhang, Lei Xu, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. Robust invisible video watermarking with attention. arXiv preprint arXiv:1909.01285, 2019.   
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.   
Xuandong Zhao, Kexun Zhang, Zihao Su, Saastha Vasan, Ilya Grishchenko, Christopher Kruegel, Giovanni Vigna, Yu-Xiang Wang, and Lei Li. Invisible image watermarks are provably removable using generative ai. Advances in Neural Information Processing Systems, 37:8643–8672, 2025.

# A Experimental Results using Alternative Loss Formulations 1 5

A. Alternative Loss Formulations 15   
A.2 Experimental Results 15

# B Watermark Removal using Images with Fixed Pixel Values 1 6

# C Watermark Removal using Real Images 1 6

# D Experimental Results on using a Significantly Different VAE 1 7

# E Discussion: Alternative Watermarking Approaches 1 9

# F Detailed Experimental Setup 1 9

F.1 Prompt Datasets 19   
F.2 Watermarking Methods 19   
F.3 Hyperparameters 19

# G More Visual Examples 2 0

G. Existence of Latent Directions 20   
G.2 Qualitative Results on Watermark Forgery and Removal 21

# A EXPERIMENTAL RESULTS USING ALTERNATIVE LOSS FORMULATIONS

In this section, we discuss an alternative design of the adversarial loss function to forge a watermark signal.

# A.1 ALTERNATIVE LOSS FORMULATIONS

Using Progressive Gradient Descent In our main experiments, we control the amount of perturbations by adding a term to our loss function to minimize the perturbations while attacking the watermarking method (Equation 3). In this section, we discuss an alternative optimization objective by utilizing progressive gradient descent (Kurakin et al., 2016; 2018), through which we can set an explicit perturbation budget to control the amount of perturbations we introduce.

The objective for the forgery attack in this case becomes,

$$
\min  _ {\delta} \left\| \mathcal {E} _ {\phi} \left(\mathbf {x} ^ {(c)} + \boldsymbol {\delta}\right) - \mathcal {E} _ {\phi} \left(\mathbf {x} ^ {(w)}\right) \right\| _ {2} \quad \text {s . t .} \quad \| \boldsymbol {\delta} \| _ {\infty} \leq \epsilon , \tag {5}
$$

where $\epsilon$ is the perturbation budget.

Perturbations in the Frequency Domain Furthermore, we can reduce the perceptible impact by, instead of directly perturbing in the RGB space, converting the image to its frequency domain using the discrete cosine transform (DCT) and perturbing only the high-frequency regions in the frequency domain (Jia et al., 2022; Luo et al., $\boxed { 2 0 2 2 }$ . To control the frequency regions we want to alter, we introduce a binary mask $\mathbf { \bar { m } } \in \{ 0 , 1 \} ^ { N \times N }$ , which consists of zeros in the upper-left triangle that corresponds to the low-frequency region ending at indices $\lfloor ( \bar { 1 } - \alpha ) \times N \rfloor$ and ones everywhere else. Here, $\alpha \in [ 0 , 1 ]$ controls the frequency regions we want to perturb (see Figure $\textcircled { 8 }$ for reference on how the mask looks).

We can write this objective as,

![](images/100e25e5572a22d7f0962562362bfe5a1b39aa9612c0ea8667cf72438b730b71.jpg)  
Figure 8: Visualization of the mask for controlling the adversarial perturbations. Noise is added only in the higher frequency region of an image.

$$
\min  _ {\boldsymbol {\delta}} \left\| \mathcal {E} _ {\phi} (\mathrm {I D C T} (\mathrm {D C T} (\mathbf {x} ^ {(c)}) + \mathbf {m} \odot \boldsymbol {\delta})) - \mathcal {E} _ {\phi} (\mathbf {x} ^ {(w)}) \right\| _ {2} \quad \text {s . t .} \quad \| \boldsymbol {\delta} \| _ {\infty} \leq \epsilon . \tag {6}
$$

# A.2 EXPERIMENTAL RESULTS

We summarize the results in Table $\boxed { 5 }$ where we show that using the proposed loss formulation in Equation $\bigtriangledown$ achieves better imperceptibility results with a similar attack success rate. We also show visual examples in Figure 9. Although the optimization using Equation $\boxed { 6 }$ yields lower imperceptibility scores, it can better preserve lower-frequency regions, resulting in improved visual results.

Table 5: Comparing trade-off between attack success rate (ASR) and imperceptibility of the attack for different optimization objectives when they achieve similar ASR. We have set the hyperparameters $\epsilon$ in Eq. 5 to 0.1, $( \epsilon , \alpha N )$ in Eq. 6 to (3, 300) and $\lambda$ in Eq. $3$ to $2 . 5 \times 1 0 ^ { 4 }$ .

<table><tr><td>Method</td><td>ASR</td><td>l2</td><td>l∞</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td></tr><tr><td>Eq. 5</td><td>83.24</td><td>68.04</td><td>0.10</td><td>0.35</td><td>0.66</td><td>28.31</td></tr><tr><td>Eq. 6</td><td>83.33</td><td>67.77</td><td>0.93</td><td>0.32</td><td>0.66</td><td>28.35</td></tr><tr><td>Eq. 3</td><td>84.91</td><td>44.34</td><td>0.85</td><td>0.23</td><td>0.84</td><td>31.97</td></tr></table>

![](images/6ef52b373d279f383b368981e73e5c68e2be73020358caabadc14a71395065ab.jpg)  
Original

![](images/e5d1cc045b182afea303570ffce2f2a0a65b929df25712fa234568423af93396.jpg)  
Eq. 5

![](images/8c2f969bf9e787b48bfbafc33d6e967207c3ae785190bc537426e0be60cef3f4.jpg)  
Eq. 6

![](images/d557f9a06e6b0c24c0c64f322ba4cadc46b3b1ab07dcd25138c350a8ccbec289.jpg)  
Eq. 3   
Figure 9: Comparing trade-off between attack success rate (ASR) and imperceptibility of the attack for different optimization objectives when they achieve similar ASR.

# B WATERMARK REMOVAL USING IMAGES WITH FIXED PIXEL VALUES

In Section $^ { 4 . 3 }$ for watermark removal, we use images with all pixel values equal to the mean value of the given watermarked image for guidance. In this section, we demonstrate that this approach works better in practice compared to using a fixed pixel value of 127.5. We report results in Table 6. As shown, using the mean value performs better using a fixed pixel value of 127.5.

Table 6: Comparing trade-off between attack success rate (ASR) and imperceptibility of the watermark removal attack when using either images with all pixel values equal to 127.5 or the mean of the watermarked image for guidance. These experiments are run on SDv1.4 for the Gaussian Shading watermarking scheme. The hyperparameter $\lambda$ in Equation 4 is set to $1 \times 1 0 ^ { 4 }$ .

<table><tr><td>Method</td><td>ASR</td><td>l2</td><td>l∞</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td></tr><tr><td>Mean</td><td>70.10</td><td>74.10</td><td>1.31</td><td>0.29</td><td>0.77</td><td>24.12</td></tr><tr><td>127.5</td><td>64.58</td><td>78.44</td><td>1.36</td><td>0.31</td><td>0.74</td><td>23.18</td></tr></table>

# C WATERMARK REMOVAL USING REAL IMAGES

Another option is to directly use a camera-captured non-watermarked image for guidance. In this section, we show that our approach works better in practice without requiring a non-watermarked image. For the study, we use non-watermarked images from the COCO (Lin et al., 2014) dataset for guidance, where we minimize the distance between their respective representations while perturbing the watermarked image. The optimization objective remains the same as Equation 4, i.e.,

$$
\min  _ {\boldsymbol {\delta}} \left\| \mathcal {E} _ {\phi} \left(\mathbf {x} ^ {(w)} + \boldsymbol {\delta}\right) - \mathcal {E} _ {\phi} \left(\mathbf {x} ^ {(c)}\right) \right\| _ {2} + \lambda \| \boldsymbol {\delta} \| _ {2}, \tag {7}
$$

where $\mathbf { x } ^ { ( c ) }$ is a randomly selected real image. We show results when using a real image in Table 7. We also show qualitative comparisons in Figure 10 and Figure 11.

![](images/f6d87588ad3a1adce8fa1031e2e4380e7cdef3881a6249b59fe91a1506a203fc.jpg)  
Original

![](images/4b0c41e853df57d1b41fc7906e9dbafd8567bbac515428432dc6fba9e0da72fe.jpg)

![](images/16e0d0095dc6d02536db8140fef80d5e54cf204d4b85dc0d981013a2a5447b51.jpg)

![](images/372ad57b2567856bf8c291812bc440f8861528f6aee83e4fb80970d043bb7ed1.jpg)

![](images/c8596c03a63dbbe6047a9dc73ff543171c2c0ce3ace020c2c4e0d9141c54ce78.jpg)  
Figure 10: Examples showing successful watermark removal attacks on the Tree-Ring watermarking method with different hyperparameter $\lambda$ values when using an image with all pixels equal to the mean of the watermarked image for guidance.   
Original

![](images/7b75baf9931e91d9fb3bba3520ec76c226e41f13467e3d2300c5b1fb95fcb051.jpg)

![](images/761db5cc2840e94dcfd316dd13bf28fb975e3e6afda38c97074ebef771d05d1c.jpg)

![](images/83ca1f59278525ec8c8d6644bf063223dccab80f0715318b836b96e3355f2a85.jpg)  
  
Figure 11: Examples showing successful watermark removal attacks on the Tree-Ring watermarking method with different hyperparameter $\lambda$ values when using real images for guidance.

# D EXPERIMENTAL RESULTS ON USING A SIGNIFICANTLY DIFFERENT VAE

We conducted a study to assess whether our forgery attack is still successful if the VAE learns a different representation. For this experiment, we have chosen the VAE from FLUX.1-dev (Black Forest Labs, 2024), which uses a different compression ratio as compared to SDv1.4. The FLUX.1-dev VAE compresses the latent representation to 16 channels as compared to 4 by SDv1.4. This implies that there are differences in the latent representations learned by these models. We present results in Table 8, which shows that for RingID and WIND, we can preserve image quality (PSNR⇡35) while achieving a high attack success rate. Tree-Ring, which embeds a smaller signal into the initial latent noise space, was harder to forge using a FLUX.1-dev VAE. This observation can be interesting to inform future research on developing robust watermarking methods. A potential defense based on this observation could be for a model owner to train a dissimilar VAE from publicly available ones so as not to allow such attacks to be successful.

Table 7: Results on watermark removal using a real image for guidance (Equation 7)   

<table><tr><td>Method</td><td>Model</td><td>λ</td><td>ASR</td><td>l2</td><td>l∞</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td></tr><tr><td rowspan="6">Tree-Ring (Wen et al., 2023)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>68.23</td><td>41.19</td><td>0.79</td><td>0.21</td><td>0.86</td><td>31.95</td></tr><tr><td>2 × 104</td><td>78.94</td><td>61.04</td><td>0.97</td><td>0.31</td><td>0.78</td><td>28.86</td></tr><tr><td>1 × 104</td><td>83.62</td><td>80.81</td><td>1.09</td><td>0.41</td><td>0.70</td><td>26.73</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>61.62</td><td>44.40</td><td>0.88</td><td>0.21</td><td>0.86</td><td>31.60</td></tr><tr><td>2 × 104</td><td>70.81</td><td>63.13</td><td>1.04</td><td>0.32</td><td>0.77</td><td>28.68</td></tr><tr><td>1 × 104</td><td>74.59</td><td>84.01</td><td>1.15</td><td>0.42</td><td>0.69</td><td>26.35</td></tr><tr><td rowspan="6">RingID (Ci et al., 2024b)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>0.0</td><td>38.25</td><td>0.74</td><td>0.18</td><td>0.87</td><td>33.27</td></tr><tr><td>2 × 104</td><td>1.10</td><td>56.98</td><td>0.95</td><td>0.28</td><td>0.79</td><td>29.79</td></tr><tr><td>1 × 104</td><td>1.65</td><td>78.01</td><td>1.13</td><td>0.37</td><td>0.72</td><td>27.05</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>0.0</td><td>38.90</td><td>0.73</td><td>0.18</td><td>0.87</td><td>33.13</td></tr><tr><td>2 × 104</td><td>0.54</td><td>58.26</td><td>0.97</td><td>0.28</td><td>0.79</td><td>29.60</td></tr><tr><td>1 × 104</td><td>0.54</td><td>79.85</td><td>1.15</td><td>0.36</td><td>0.72</td><td>26.85</td></tr><tr><td rowspan="6">WIND (Arabi et al., 2024)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>0.0</td><td>50.64</td><td>0.98</td><td>0.19</td><td>0.85</td><td>29.33</td></tr><tr><td>2 × 104</td><td>1.09</td><td>66.79</td><td>1.07</td><td>0.29</td><td>0.78</td><td>27.66</td></tr><tr><td>1 × 104</td><td>1.09</td><td>85.81</td><td>1.23</td><td>0.38</td><td>0.70</td><td>25.85</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>0.0</td><td>38.44</td><td>0.74</td><td>0.18</td><td>0.87</td><td>33.23</td></tr><tr><td>2 × 104</td><td>0.54</td><td>57.55</td><td>0.97</td><td>0.28</td><td>0.80</td><td>29.71</td></tr><tr><td>1 × 104</td><td>0.54</td><td>78.91</td><td>1.13</td><td>0.36</td><td>0.72</td><td>26.96</td></tr><tr><td rowspan="6">Gaussian Shading (Yang et al., 2024b)</td><td rowspan="3">SDv1.4</td><td>5 × 104</td><td>28.0</td><td>46.65</td><td>0.84</td><td>0.21</td><td>0.84</td><td>28.36</td></tr><tr><td>2 × 104</td><td>65.0</td><td>73.49</td><td>1.08</td><td>0.32</td><td>0.75</td><td>23.87</td></tr><tr><td>1 × 104</td><td>81.0</td><td>92.20</td><td>1.21</td><td>0.42</td><td>0.67</td><td>23.70</td></tr><tr><td rowspan="3">SDv2.0</td><td>5 × 104</td><td>25.0</td><td>44.31</td><td>0.84</td><td>0.18</td><td>0.86</td><td>31.58</td></tr><tr><td>2 × 104</td><td>49.0</td><td>65.25</td><td>1.06</td><td>0.29</td><td>0.78</td><td>28.44</td></tr><tr><td>1 × 104</td><td>74.0</td><td>87.61</td><td>1.20</td><td>0.39</td><td>0.70</td><td>25.96</td></tr></table>

Table 8: Comparing forgery performance when using a significantly different VAE. We use the VAEs from SDv1.4 and FLUX.1-dev to attack watermarks in SDv1.4.   

<table><tr><td>Method</td><td>VAE</td><td>λ</td><td>ASR</td><td>l2</td><td>l∞</td><td>LPIPS</td><td>SSIM</td><td>PSNR</td></tr><tr><td rowspan="4">Tree-Ring</td><td rowspan="2">FLUX.1-dev</td><td>1 × 103</td><td>2.63</td><td>46.51</td><td>1.42</td><td>0.26</td><td>0.84</td><td>31.53</td></tr><tr><td>5 × 102</td><td>2.94</td><td>52.46</td><td>1.51</td><td>0.29</td><td>0.82</td><td>30.50</td></tr><tr><td rowspan="2">SDv1.4</td><td>5 × 104</td><td>78.65</td><td>33.90</td><td>0.69</td><td>0.17</td><td>0.89</td><td>34.32</td></tr><tr><td>1 × 104</td><td>91.06</td><td>63.22</td><td>1.10</td><td>0.33</td><td>0.76</td><td>28.87</td></tr><tr><td rowspan="4">RingID</td><td rowspan="2">FLUX.1-dev</td><td>1 × 104</td><td>61.20</td><td>22.70</td><td>1.08</td><td>0.12</td><td>0.93</td><td>37.69</td></tr><tr><td>5 × 103</td><td>82.51</td><td>29.22</td><td>1.08</td><td>0.17</td><td>0.91</td><td>35.58</td></tr><tr><td rowspan="2">SDv1.4</td><td>5 × 104</td><td>100.0</td><td>38.45</td><td>0.68</td><td>0.20</td><td>0.87</td><td>33.21</td></tr><tr><td>1 × 104</td><td>100.0</td><td>73.08</td><td>1.03</td><td>0.38</td><td>0.73</td><td>27.63</td></tr><tr><td rowspan="4">WIND</td><td rowspan="2">FLUX.1-dev</td><td>1 × 104</td><td>55.72</td><td>22.83</td><td>1.04</td><td>0.12</td><td>0.94</td><td>37.51</td></tr><tr><td>5 × 103</td><td>84.44</td><td>30.83</td><td>1.14</td><td>0.17</td><td>0.91</td><td>34.76</td></tr><tr><td rowspan="2">SDv1.4</td><td>5 × 104</td><td>97.56</td><td>38.82</td><td>0.70</td><td>0.20</td><td>0.87</td><td>33.11</td></tr><tr><td>1 × 104</td><td>97.56</td><td>74.66</td><td>1.06</td><td>0.38</td><td>0.73</td><td>27.38</td></tr><tr><td rowspan="4">Gaussian Shading</td><td rowspan="2">FLUX.1-dev</td><td>1 × 103</td><td>19.69</td><td>47.43</td><td>1.37</td><td>0.27</td><td>0.83</td><td>31.27</td></tr><tr><td>5 × 102</td><td>20.20</td><td>54.14</td><td>1.50</td><td>0.31</td><td>0.80</td><td>30.13</td></tr><tr><td rowspan="2">SDv1.4</td><td>5 × 104</td><td>96.85</td><td>37.27</td><td>0.70</td><td>0.19</td><td>0.87</td><td>33.48</td></tr><tr><td>1 × 104</td><td>96.96</td><td>71.97</td><td>1.05</td><td>0.37</td><td>0.73</td><td>27.64</td></tr></table>

# E DISCUSSION: ALTERNATIVE WATERMARKING APPROACHES

Other watermarking approaches that add a pattern during the latent decoding phase are less susceptible to attacks such as ours (Ci et al., 2024a; Fernandez et al., 2023). As these methods fine-tune the decoder, the adversary would require access to a similar decoder to attack the system. There is an inherent trade-off here, i.e., these methods are less resistant to image transformations, which was a major advantage of initial noise-based watermarking schemes.

Other alternatives for building a more secure watermarking scheme could be to embed a secret message that encodes information pertaining to the contents of the watermarked image. This would make it harder for an attacker to forge a watermark, as they would need to embed a new message, which can only be done if they have access to both the entire diffusion model and the secret message generation method. It would allow a model owner to quickly verify that the content of the image and the recovered secret message match.

# F DETAILED EXPERIMENTAL SETUP

# F.1 PROMPT DATASETS

We utilize two datasets namely, the Gustavosta/Stable-Diffusion-Prompts1 dataset and the runwayml-stable-diffusion-v1-5-eval-random-prompts2 dataset. The former contains around 80,000 prompts extracted from the image finder for Stable Diffusion: ”Lexica.art”. These are generally longer and more detailed prompts. The latter, on the other hand, contains 200 short and simple prompts that contain less information/ details.

# F.2 WATERMARKING METHODS

We consider four latent-noise based watermarking schemes - Tree-Rings (Wen et al., 2023), RingID (Ci et al., 2024b), Gaussian Shading (Yang et al., 2024b) and WIND (Arabi et al., 2024). We utilized their publicly available implementations from the following GitHub repositories,

• Tree-Ring: https://github.com/YuxinWenRick/tree-ring-watermark.   
• RingID: https://github.com/showlab/RingID.   
• Gaussian Shading: https://github.com/bsmhmmlf/Gaussian-Shading/.   
• WIND: https://github.com/anonymousiclr2025submission/Hidden-in-the-Noise.

# F.3 HYPERPARAMETERS

We adopt the following hyperparameter values in our optimization:

• Number of iterations: 15,000   
• Learning rate ↵: 0.020   
• Image size: $5 1 2 \times 5 1 2$   
• Stable Diffusion versions: CompVis/stable-diffusion-v1-4 and stabilityai/stable-diffusion-2

# G MORE VISUAL EXAMPLES

# G.1 EXISTENCE OF LATENT DIRECTIONS

We present additional visual examples showcasing the effectiveness of latent directions in forging and removing watermarks in Figure 12.

![](images/e2f02afd847a4afd69c5f01bbf9bfd98a8f30374127cce59315642862291b022.jpg)  
Towards Watermarked Image Latents   
p-value:0.72

![](images/b385f85b9ed3ccb6b964da1c3c7433d9cb6103590836502353971a21dd13d2ef.jpg)  
p-value:0.14

![](images/d9c687cc0a720c1168e371b4bbd125258e820baa5984b76ca3fdeabb950398b7.jpg)  
p-value:0.02

![](images/ed5cda9bc893358d9fade750636fbcf322e0db5eec91b23ad112d4a5e3bab313.jpg)  
p-value:0.005

![](images/50abf6bd3565a76ff40111de1f43859e4133b3fcb5b70a4c13f17bc46d2c0bc7.jpg)  
p-value:0.70

![](images/fe56a1d14e751b62fe123ac7f628be4e43f1d31e7a216b5b0fd531ad4e10570a.jpg)  
p-value:0.10

![](images/b3e229cfc63c69670b8acaea91b5e6e7e4aee955834dd3fa1d682d2824336b47.jpg)  
p-value: 0.01

![](images/b5f2f3ce8761ba9def8b3fedb71d21291d27e1bc22c6d62be3be1f6c5e635ded.jpg)  
p-value:0.002

![](images/ff24bfcb44f27da6a8421be7b4fc9c8ad2ce0f4aa19f3be865f1db9553c4a051.jpg)  
p-value:0.64

![](images/0366b79f3e3b73d0d93db3836fef1a682a200b3c487738c63409967609af5ff6.jpg)  
p-value:0.02

![](images/b0d0078094b164a1591f956a1341b1487b838b31ac2abe3b293e6ff074998348.jpg)  
p-value:5.39x10-4

![](images/588a276909ed5838c791cbb22f896a0f0c7b5055b82f18e841da0c3ec2961935.jpg)  
p-value:4.27×10-5

![](images/c35052f942be2f605a884c1cbe8dad09db7b50573a3440f0b9a19770207dc856.jpg)  
Non-Watermarked Image Late

p-value:0.48

![](images/77410be6802502825b49965f00f32e0cd4839a04bfce08b58b3b01de274eb498.jpg)  
p-value: 0.20

![](images/ee94371ed3c381351af6699188e37ec43882fcf5566c03cb7db4a40b4f68951f.jpg)  
p-value:0.29

![](images/3a6d6bd9a78e28e0968b2f3a657c8ffb52e64d41d60251e2530fc50041704923.jpg)  
p-value:0.14

![](images/84f9c45041ed6e81ccef39b0d380892cafb36f8a3332f1da9c04296ae01d9a16.jpg)  
p-value:3.47×10-5

![](images/a35739647d9922837aedab38cf670b074ea1a189a3da823f2e75d18656b4bd31.jpg)  
p-value: 0.43   
p-value: 0.24

![](images/55956bcea173aeee0141eb125b6219f530d0360f09e15e8a6c2ea2e1b3701ccf.jpg)  
p-value: 0.19

![](images/ff68e516890364824454e4fec8fa450f4d3f907b08db94d11d147ca7c37b1dce.jpg)  
p-value: 0.01

![](images/9e4b6dba9940b7a5b49380d3faa2228d65a8081dccec581f21c2c096c4c34cc1.jpg)

![](images/17f8bb8311cc7b7635c9b27fae6d1a6fbbf000bd2d106efb1efe2d6ff43c6326.jpg)  
p-value: 0.32

![](images/5398b4f3b2804199ba57eeb91533ab67988a0f90e2631a6667898d68d8030683.jpg)  
p-value:0.24

![](images/8cd6f22cf7913596723b48d9bbbc0c0354db675edd7b0b3cb6db573dab47c1aa.jpg)  
p-value:9.36x10-5   
Figure 12: Examples showcasing the idea that there exist latent directions which pertain to watermarking and removal. We learn these directions using a linear SVM, and they are the normal to the learned hyperplane. Traversing further along them increases the strength of the attack.

# G.2 QUALITATIVE RESULTS ON WATERMARK FORGERY AND REMOVAL

We provide additional visual examples to show that successful watermark forgery and removal attacks do not harm the semantics or quality of the resultant image. We show examples of forgery attacks on Tree-Ring in Figure 13, RingID in Figure 14, WIND in Figure $\boxed { 1 5 }$ and Gaussian Shading in Figure 16. We show examples of the removal attack in Figure 10 for the Tree-Ring system and in Figure 17 for the Gaussian Shading system.

![](images/0fa06a0c375905f38889bd4db5b79f394b5cd3e6f80c7503ec58dc0feb9dc710.jpg)  
Original

![](images/dfc244aa31d72160ecb71f89120344812e82efcfde591ea4050a14545826ad5e.jpg)

![](images/c99d67ae1033bc88cb3b149f294d796cfd239161196a43e8e7397a4b3091f16f.jpg)

![](images/47068a33edb75345e893939f42772b1313642c185c92aff8b0ea9cd0d44948d5.jpg)

![](images/467f239c342f99f69ecddae307b16d4c2f51ef41ea9579f332d4c65725a2651b.jpg)  
Figure 13: Examples showing successful watermark forgery attacks on the Tree-Ring watermarking method with different hyperparameter $\lambda$ values.

![](images/279ddb79b5de6281c360e659f5819c40db692b309f7da711a8984d84f57dd173.jpg)

![](images/88909fbcef5092a956f782e38be60c6156b234928b9141a3b2dd493818af5d65.jpg)

![](images/f5ed2328b4129f4b1a6fd11d33a9d820aae36c843a0f1ee38dc048806697ffd8.jpg)  
Figure 14: Examples showing successful watermark forgery attacks on the RingID watermarking method with different hyperparameter $\lambda$ values.

![](images/02beb93b90cfd351ffb146dc014203e3e2d4dd9c919b6ea7e7466e248389252c.jpg)

![](images/e2adfc6afca29f2c75479772bbf3ba565acd4316d0a3b6e64117098248d016e8.jpg)  
Original

![](images/7eba75e7e480cc721fa15daab73e4435170eece0b46344741faba47ebead0e77.jpg)

![](images/be6d4d7ebd7d41640fdf0af1e2b47f0072a2f8ed1695da2f26055220e8ceed8e.jpg)

![](images/9de05fd5bccfc4c377886b99d2dc95e5fc20d5e66ac1865e370b3e8dfa627265.jpg)

![](images/194797aed72785bb6faa7dc22697ec33b3bec14fc37771fb5764ec332d43284a.jpg)

![](images/c2a669c76d47cc45ff6238d28fb9bef6514a2257b4c5153b61b31eb2e5a46ac4.jpg)

![](images/99036d7d190d419af9f26cd8b15075cf065f7f8296c9fcf764527c6ab4deaded.jpg)  
  
Figure 15: Examples showing successful watermark forgery attacks on the WIND watermarking method with different hyperparameter $\lambda$ values.

![](images/5e594e93996492d1332d35e1ade91bb96e289de478470181d01d910d202dae0c.jpg)

![](images/f0305d467cfbb5a79e4bc802e3111fc7e8f0bb4019b0b9e915876ea2fe813567.jpg)  
Original

![](images/8b10c058d0080975b6db219490776bfcec853dc17cbbd5c04f11894b8096d2f7.jpg)

![](images/2f5e48ffea70a3d43b41a097f0f2c05353425659de789a343e787e983ce27420.jpg)

![](images/f93e95b5c2c6625d7e651932c1fcbbcaa979066128628e5bb30f93b8f4ff12d0.jpg)

![](images/9892c0f5a712897b2b596d75504468e959a4d308bad0ce6fe8677e42563cba73.jpg)

![](images/175e21c060362b3ef7757b0293f86d1a2c5040c6db0a9109b3f61e4fce542d93.jpg)

![](images/fb98c3c42933cecaa68c4f311c5f9bcd61b7b05ac32b68876f25aaa075b94bd4.jpg)  
  
Figure 16: Examples showing successful watermark forgery attacks on the Gaussian Shading watermarking method with different hyperparameter $\lambda$ values.

![](images/bbac12d61aa6abab1b752eb925a6ae07bc1cc1f150ef016b75bc795656ef3788.jpg)

![](images/64cf19ec63a15bdd1bc759abc7171f9c7b1fced6112b9fc28f994c08629be67b.jpg)  
Original

![](images/114edcdaa491c5e5af176ef8f33f163aede1b168dfebd5c9bd18642bdbdf91e0.jpg)

![](images/2f152aba0ac42b42068a0fd1d6fd6fbe5388a74c225808ce32092c9f476fc2df.jpg)

![](images/19840baf1af8ab91f7216d573c6a89d433f9e04912564ffe0648bca9801438ff.jpg)

![](images/98eb2e4c036ba5f65a04532cc69ba0e2f888fcd6f6b18a9ab991627df3cbcbc7.jpg)

![](images/9fcb752d02a66a302e4fc40bf8c92ca9c50b77ed139efecc1dab5f0fb27afb51.jpg)

![](images/60efa6cb08d093e08f32e3190044cb3288dbeb7ea4509675898f2c6a9b3450c9.jpg)  
  
Figure 17: Examples showing successful watermark removal attacks on the Gaussian Shading watermarking method with hyperparameter $\lambda$ values.