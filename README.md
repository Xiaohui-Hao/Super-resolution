# Super-Resolution
A collection of state-of-the-art video or single-image super-resolution architectures.

## Network list and reference (Updating)
The hyperlink directs to paper site, follows the official codes if the authors open sources.

|Model |Published |Code* |Keywords|
|:-----|:---------|:-----|:-------|
|(SRNTT) Image Super-Resolution by Neural Texture Transfer |[CVPR19](http://web.eecs.utk.edu/~zzhang61/project_page/SRNTT/cvpr2019_final.pdf)|[TensorFlow](https://github.com/ZZUTK/SRNTT)| Texture Transfer |
|(TecoGAN) Temporally Coherent GANs for Video Super-Resolution |[arXiv](http://arxiv.org/abs/1811.09393)|[Tensorflow](https://github.com/thunil/TecoGAN)| VideoSR GAN|
|(CrossNet) CrossNet: An End-to-end Reference-based Super Resolution Network using Cross-scale Warping |[ECCV18](http://www.liuyebin.com/CrossNet/CrossNet.pdf)| [PyTorch](https://github.com/htzheng/ECCV2018_CrossNet_RefSR) | Cross |
|(DBPN) Deep Back-Projection Networks For Super-Resolution |[CVPR18](https://arxiv.org/abs/1803.02735)|[PyTorch](https://github.com/alterzero/DBPN-Pytorch)| NTIRE18 Champion |
|(RDN) Residual Dense Network for Image Super-Resolution|[CVPR18](https://arxiv.org/abs/1802.08797)|[Torch](https://github.com/yulunzhang/RDN)| Deep, BI-BD-DN |
|(SRDenseNet) Image Super-Resolution Using Dense Skip Connections |[ICCV17](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)|[PyTorch](https://github.com/wxywhu/SRDenseNet-pytorch)| Dense |
|(SRGAN) Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network |[CVPR17](https://arxiv.org/abs/1609.04802)|[Tensorflow](https://github.com/jiaming-wang/SRGAN-tensorflow)| 1st proposed GAN |
|(EDSR) Enhanced Deep Residual Networks for Single Image Super-Resolution |[CVPR17](https://arxiv.org/abs/1707.02921)|[PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)| NTIRE17 Champion |
|(VDSR) Accurate Image Super-Resolution Using Very Deep Convolutional Networks |[CVPR16](https://arxiv.org/abs/1511.04587)|[Caffe](https://github.com/huangzehao/caffe-vdsr)| Deep, Residual |
|(ESPCN) Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network |[CVPR16](https://arxiv.org/abs/1609.05158)|[Keras](https://github.com/qobilidop/srcnn)| Real time |
|(SRCNN) Image Super-Resolution Using Deep Convolutional Networks|[ECCV14](https://arxiv.org/abs/1501.00092)|[Keras](https://github.com/qobilidop/srcnn)| 1st DL SR |

\*The 1st repo is by paper author.

## Link of datasets
*(please contact me if any of links offend you or any one disabled)*

|Name|Usage|#|Site|Comments|
|:---|:----|:----|:---|:-----|
|SET5|Test|5|[download](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|SET14|Test|14|[download](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|SunHay80|Test|80|[download](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|Urban100|Test|100|[download](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|VID4|Test|4|[download](https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip)|4 videos|
|BSD100|Train|300|[download](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|BSD300|Train/Val|300|[download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz)|-|
|BSD500|Train/Val|500|[download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)|-|
|91-Image|Train|91|[download](http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar)|Yang|
|DIV2K|Train/Val|900|[website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)|NTIRE17|
|Waterloo|Train|4741|[website](https://ece.uwaterloo.ca/~k29ma/exploration/)|-|
|MCL-V|Train|12|[website](http://mcl.usc.edu/mcl-v-database/)|12 videos|
|GOPRO|Train/Val|33|[website](https://github.com/SeungjunNah/DeepDeblur_release)|33 videos, deblur|
|CelebA|Train|202599|[website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)|Human faces|
|Sintel|Train/Val|35|[website](http://sintel.is.tue.mpg.de/downloads)|Optical flow|
|FlyingChairs|Train|22872|[website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)|Optical flow|
|DND|Test|50|[website](https://noise.visinf.tu-darmstadt.de/)|Real noisy photos|
|RENOIR|Train|120|[website](http://ani.stat.fsu.edu/~abarbu/Renoir.html)|Real noisy photos|
|NC|Test|60|[website](http://demo.ipol.im/demo/125/)|Noisy photos|
|SIDD(M)|Train/Val|200|[website](https://www.eecs.yorku.ca/~kamel/sidd/)|NTIRE 2019 Real Denoise|
|RSR|Train/Val|80|[download]()|NTIRE 2019 Real SR|
|Vimeo-90k|Train/Test|89800|[website](http://toflow.csail.mit.edu/)|90k HQ videos|
|General-100|Test|100|[website](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html/)|Chao Dong|

Other open datasets:
[Kaggle](https://www.kaggle.com/datasets), [ImageNet](http://www.image-net.org/), [COCO](http://cocodataset.org/)
