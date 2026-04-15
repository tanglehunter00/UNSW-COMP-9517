# 中英文对照

## 中文

U-Net 是一种经典的卷积神经网络结构，由 University of Freiburg 的研究团队在 2015 年提出，最初用于生物医学图像分割。其结构简单、效果稳定、在小样本数据上表现尤为优秀。其核心结构呈「U」形，由两部分组成：

编码器通过多层卷积和下采样，例如 max pooling，逐步提取高层语义特征，同时降低空间分辨率。

解码器通过上采样，例如转置卷积，逐步恢复空间分辨率，并输出像素级预测。

然而，在本次实验中，可用训练素材较少，模型容易产生过拟合。为提高效果，先在无标签外部图像上做掩码自编码器 MAE 自监督预训练：采用 Vision Transformer 式的编码器与解码器，并在自注意力中融入 DFormer v2 的几何先验。预训练得到的编码器权重再作为下游 U-Net 式分割网络中对应支路的初始化，以缓解小样本过拟合。

在预处理阶段，将统一为 350px×350px 像素并且进行过亮度优化的图片统一处理为 5px×5px 的 patch，每个 patch 的 25 个像素生成为一个 token，以计算 patch 之间的相关性。采用这一 patch 大小是因为 Transformer 类模型在 patch 内部的细节上效果不好，而更小的 patch size 要求超过 40 GB 的显存，超出设备允许范围。

MAE 自监督预训练按一定比例随机遮盖部分 patch token，编码器只处理未被遮盖的 token，解码器在还原后的全序列上运行，预测被遮盖 patch 内的像素；训练目标为仅在遮盖位置上的重建误差，采用均方误差形式的损失。本实验将遮盖比例设为 50%。在更早的测试中，遮盖 75% 时可见内容过少，模型更易退化为用大面积单色块填图以降低损失，重建质量差，对后续微调不利。

几何先验部分借鉴 DFormer v2，在标准多头自注意力得分上加入与 patch 空间距离相关的偏置。同一图像中绿色植物往往成片出现，最左下角与最右上角 patch 关联弱、与邻近 patch 关联强，但是在典型ViT中会将其视作有相同的影响权重，这是不合适的。实现上依据 patch 间距离配合按注意力头递减的衰减，使远处 patch 对注意力的影响减弱，从而强化对空间邻域的建模，有助于边界与局部结构。

预训练得到的权重载入下游分割网络，该网络由 MAE 编码路径与卷积 U-Net 式解码及融合部分组成；下游使用相同的 patch 设定，并在 EWS 数据集训练集的标注数据上微调。由于在预训练中已从大量无标签数据学习重建与空间结构，编码器初始化应优于从零随机初始化的对应分割网络，从而有利于分割任务。

## English

U-Net is a classic convolutional neural network architecture proposed in 2015 by a research team at the University of Freiburg and originally developed for biomedical image segmentation. It is structurally simple, empirically stable, and performs particularly well on small-sample data. Its backbone has a U-shaped layout with two main parts:

The encoder applies stacked convolutional layers and downsampling, such as max pooling, to gradually extract high-level semantic features while reducing spatial resolution.

The decoder restores spatial resolution step by step via upsampling, for example transposed convolutions, and produces pixel-wise predictions.

In this experiment, however, the amount of usable training material is limited and the model tends to overfit. To improve results, we first perform self-supervised pre-training with a masked autoencoder, MAE, on unlabeled external images: we use a Vision Transformer–style encoder and decoder and inject the geometric prior from DFormer v2 into self-attention. The resulting encoder weights then initialize the corresponding branch of the downstream U-Net-style segmentation network, mitigating overfitting under small data.

During preprocessing, images are standardized to 350×350 pixels with brightness optimization, then divided into 5×5-pixel patches; twenty-five pixels from each patch form one token so that correlations between patches can be modeled. This patch size is chosen because Transformer-style models capture fine detail inside a patch less effectively, while a smaller patch size would demand more than 40 GB of GPU memory, which exceeds our hardware limit.

In MAE self-supervised pre-training, a fraction of patch tokens is masked at random; the encoder processes only visible tokens, and the decoder runs on the restored full sequence to predict pixels inside masked patches. The objective is reconstruction error, in the form of mean squared error, evaluated only at masked locations. In this work the mask ratio is set to 50%. In earlier trials, masking 75% left too little visible context, and the model more often collapsed to filling the image with large uniform color blocks to minimize loss, yielding poor reconstructions that hurt later fine-tuning.

The geometric prior draws on DFormer v2: we add a bias tied to spatial distance between patches to the scores of standard multi-head self-attention. In the same image, green vegetation often forms contiguous regions; the bottom-left and top-right patches are weakly related, whereas neighboring patches are strongly related, yet a typical ViT would treat their influence equally, which is undesirable. In implementation, distances between patches are combined with head-wise decay so that distant patches contribute less to attention, strengthening modeling of local neighborhoods and helping boundaries and fine structure.

Pre-trained weights are loaded into the downstream segmentation network, which consists of the MAE encoder pathway plus a convolutional U-Net-style decoder with fusion blocks; downstream training keeps the same patch configuration and fine-tunes on labeled data from the training split of the EWS dataset. Because pre-training already learns reconstruction and spatial structure from large-scale unlabeled data, encoder initialization should outperform that of an otherwise identical segmentation network trained from random initialization, which benefits the segmentation task.
