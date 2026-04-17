对于预训练取最优得到的权重与完全随机初始化的权重，在下游微调中分别训练20 个 epoch，得到如下结果：

图 1

图 2

在测试集上的结果如下：

图 3

从 EWS 数据集中随机抽取一例做分割，可视化如下：

图 4

可以看出，载入预训练权重的模型在 Dice、准确率与 AUC 等指标上均优于无预训练、随机初始化的同构 U-Net 式分割网络，其中前景区域重叠率，即 Dice，改善尤为明显，实际分割效果也更稳定。实验表明无标签阶段学到的表征对 EWS 分割有稳定帮助。

---

## English version

We fine-tuned the downstream segmentation model for 20 epochs in two settings: one starting from the best pre-trained weights, and one from a fully random initialization. The outcomes are summarized as follows:

Figure 1

Figure 2

Results on the test set are as follows:

Figure 3

We randomly pick one example from the EWS dataset, run segmentation, and visualize the output as follows:

Figure 4

The pre-trained initialization consistently improves Dice, pixel accuracy, and ROC-AUC over the same U-Net-style architecture trained from random weights, with the largest gain in overlap on the foreground, i.e. Dice, and more reliable qualitative masks. This supports the claim that representations learned in the label-free pre-training stage help EWS segmentation.
