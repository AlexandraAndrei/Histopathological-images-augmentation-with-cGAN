Privacy preserving histopathological image augmentation with Conditional Generative Adversarial Networks

Pattern Recognition Letters, Volume 188, 2025, Pages 185-192, ISSN 0167-8655, https://doi.org/10.1016/j.patrec.2024.12.014.
(https://www.sciencedirect.com/science/article/pii/S0167865524003696)


Abstract: Deep learning approaches for histopathology image processing and analysis are gaining increasing interest in the research field, and this comes with a demand to extract more information from images. Pathological datasets are relatively small mainly due to confidentiality of medical data and legal questions, data complexity and labeling costs. Typically, a large number of annotated images for different tissue subtypes are required as training samples to automate the learning algorithms. In this paper, we present a latent-to-image approach for generating synthetic images by applying a Conditional Deep Convolutional Generative Adversarial Network for generating images of human colorectal cancer and healthy tissue. We generate high-quality images of various tissue types that preserve the general structure and features of the source classes, and we investigate an important yet overlooked aspect of data generation: ensuring privacy-preserving capabilities. The quality of these images is evaluated through perceptual experiments with pathologists and the Fréchet Inception Distance (FID) metric. Using the generated data to train classifiers improved MobileNet’s accuracy by 35.36%, and also enhanced the accuracies of DenseNet, ResNet, and EfficientNet. We further validated the robustness and versatility of our model on a different dataset, yielding promising results. Additionally, we make a novel contribution by addressing security and privacy concerns in personal medical image data, ensuring that training medical images “fingerprints” are not contained in the synthetic images generated with the model we propose.

Keywords: Histopathology; Dataset augmentation; Synthetic medical images; Privacy-preserving; Fingerprints; Generative adversarial networks

![cGAN](https://user-images.githubusercontent.com/92714719/200405358-39a5e0fd-d14c-465c-bb47-7167b9180167.png)


