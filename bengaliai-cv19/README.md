# bengaliai-cv19

## コンペの全体

- まずはベースラインを作ることに徹する
  - シングルモデルでスコアを向上させることに徹したい
  - データの意味も理解できていないので、ベースができたら理解する
    - https://bengali.ai/wp-content/uploads/CV19-COCO-Grapheme.pdf
- コンペのポイント
  - grapheme roots をとにかく精度良く当てること
    - クラスがかなり多い (160 程度)
    - 不均衡なデータ
    - スコアにも重みが付けられている
  - 問題はどうやって、imbalance や misslabeling に対処するか...?
    - Balance Sampler
    - Custom Loss
    - under/over sampling

## 実験

- ベース
  - 画像
    - 224x224 の方が精度は良さそう
    - 3 channel にする
      - imagenet の重みを使うと入力が 3channel しか受け付けないからだった
    - Resize の際には、文字が中心になるように変換すべき (こういう細かな配慮は大事だと思う)
  - Model
    - resnet34 (12 min / epoch)
    - パラメータ数と精度の観点から
  - Optmizer
    - Adam with ReduceLROnPlateau
  - Loss
    - cross entropy (= log loss)
  - Augment
    - ShiftScaleRotate + CutOut
  - CV
    - MultilabelStratifiedKFold (one hold)

### 過去コンペの Tips

細かいところは以下を参照するのが良い(下手に Discussion を見るよりは、ここのやつを参考にするのが良い)

- https://www.kaggle.com/c/bengaliai-cv19/discussion/12797
- https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065
  - generalized mean pooling (GeM)
    - pool の一般化 (p=1 で mean, p=∞ で max と等しい)
    - 論文では p=3 を推奨
    - https://arxiv.org/pdf/1711.02512.pdf
    - https://www.slideshare.net/xavigiro/d1l5-contentbased-image-retrieval-upc-2018-deep-learning-for-computer-vision
- https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926
  - pseudo-labeling (Test データが公開されないので難しい気がする...)
- https://www.kaggle.com/c/understanding_cloud_organization/discussion/118080
  - AdamW OneCycle scheduler
    - Wramup の理解：https://qiita.com/koshian2/items/c3e37f026e8db0c5a398
- https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110543
  - ArcFaceLoss
    - 深層距離学習の損失は足した方が確かに精度出そう
    - https://qiita.com/yu4u/items/078054dfb5592cbb80cc
- catalyst の classfication tutorial
  - https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb
  - focal loss, custom callback, BalanceClassSampler などかなり実践的で参考になる

### 実験

ベース設定  
optmizer : AdamW + OneCycle  
epoch : 25

この論文の内容をやる方針にする  
Bag of Tricks for Image Classification with Convolutional Neural Networks  
https://arxiv.org/pdf/1812.01187.pdf

- Model
  - GeM
  - ResNet50-D (論文参照)
  - Manifold Mixup
  - efficientnet-b3
  - se_resnext50_32x4d
- Augmentation
  - Cutout
  - Mixup
  - CutMix
- Loss
  - OHEM
    - 簡単なタスクとデータの少ない難しいタスクの両方が存在するデータに対して作られた
    - 難しいデータを Loss から判断して、それらについて優先的に勾配を更新する(?)
      - mining top 70% gradient for Backpropagation
    - https://qiita.com/woody_egg/items/28a9656aafcb4cd9cebd
    - https://www.slideshare.net/DeepLearningJP2016/dlfocal-loss-for-dense-object-detection
  - label smoothing
    - https://www.kaggle.com/c/bengaliai-cv19/discussion/128115
    - https://www.slideshare.net/DeepLearningJP2016/dlwhen-does-label-smoothing-help
  - metric learning loss
    - https://qiita.com/yu4u/items/078054dfb5592cbb80cc
    - ArcFaceLoss or Center Loss
    - https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
- Scheduler & Optimizer
  - AdamW with OneCycle scheduler
  - Cosine Annealing
