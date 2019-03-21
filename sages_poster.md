**Unsupervised Detection of Tool Presence in Endoscopic Video Frames**

David Z. Li<sup>1</sup>, Masaru Ishii, M.D., Ph.D.<sup>2</sup>, Russell H. Taylor, Ph.D.<sup>1</sup>, Gregory D. Hager, Ph.D.<sup>1</sup>, Ayushi Sinha, Ph.D.<sup>1</sup> 

<sup>1</sup>The Johns Hopkins University, <sup>2</sup>Johns Hopkins Medical Instutitions

**Introduction**

There is an abundance of medical imaging data but it is mostly unlabeled

* Surgical tools enter and leave the endoscopic field of view during procedures

* Detecting these events can enable:

  * Detection of tool presence in video frames

  * Tool segmentation

* Goal: classify video frames into two classes -- with or without tools

**Methods**

Use Variational Autoencoder<sup>2</sup> to learn latent representation of surgical video

* Encoder and decoder are convolutional neural networks (CNNs)

  * Encoder learns probability distribution of latent variables from images

  * Decoder reconstructs input image from random sample from latent space

* Use dimensionality reduction algorithms to analyze latent space encodings

  * Interpolation between latent vectors qualitatively shows understanding of underlying structure

  * Latent vector sums and differences isolate tool-encoding feature

**Model**

* We train a 2-layer CNN encoder and 2-layer deconvolutional NN decoder

* Learn mean and standard deviation vectors that model each latent space dimension as a Gaussian

* Randomly select 1241 images from our dataset for training and remaining 310 as held-out test set

* Trained network for 50 epochs using sum of KL-divergence and cross entropy loss, using the Adam optimizer<sup>4</sup> with default values except weight decay set to 10<sup>-3</sup>

**Experiments & Results**

**Conclusions & Future Work**

Conclusions

* Reconstruction and generation of tool and anatomical features is promising

* Interpolation smoothly translates tool in direction of expected motion

* Latent vector arithmetic we can isolate tool encoding and transfer to new scenes

Future Work

* Currently working on expanding our dataset with new endoscopy data

* Working on how to best manipulate the encoding to obtain a clearer representation of the tool

* Explore more complex network architectures to encode input data such as incorporating temporal dependencies in our model

 

**References**

[1] "Acute Sinusitis in HD" _YouTube_, uploaded by Dr. Moshe Ephrat, 15 Feb 2013, _www.youtube.com/watch?v=6niL7Poc_qQ_

[2] Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_. 

[3] Frans, K. (2016). "Variational Autoencoders Explained". _http://kvfrans.com/variational-autoencoders-explained/_.

[4] Kingma, D. P. and Ba, J (2014). Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_.

Email: dzli@jhu.edu

Code: www.github.com/zdavidli/tool-presence
