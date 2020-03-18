# bengaliai-cv19

## 反省 (コンペ後)

### よかったところ

- ベースライン作った
  - 一通り画像コンペのやり方が理解できた
  - 前処理 + データ拡張 + fine tuning の流れの理解
- 最新の深層学習の手法の理解
  - data augmentation
    - (manifold)mixup, cutout, random erasing, ....
  - metric learning
  - snapshot ensemble (cosine annealing with warmup)

### よくなかったところ

- sub を出せなかった
  - kernel error で出せなかった...
  - やはりコンペには早めに参加するべき
  - 今回もかなり shake したので、出しとくことに意味はある
    - Public LB は鵜呑みにしすぎない
- 実験をもう少し高速に回せるとよかった
  - Catalyst に固執しすぎた
    - 細かなところの修正が必要になったりするので自分で組むのがやっぱり良い
  - 画像サイズ 3 channel + 224×224 もいらない
    - kernel only だと特に submit のときも困るので minimum で必ずやる
  - colab... 今度やるときは colab pro も検討
  - 実験条件の設定ミスをなくす
- MultilabelStratifiedKFold は悪手
  - unseen のラベルが当てることが今回のコンペのポイント
  - seen / unseen でちゃんと検証することが大事だった (CV の切り方大事)

### 今回の学習した手法(実装)の資料

- CNN による分類問題の精度向上のトリック
  - https://arxiv.org/pdf/1812.01187.pdf
- data augmentation
  - データ拡張のまとめ : https://blog.shikoan.com/manual-augmentation/
  - mixup, cutout, random erasing, augmix : http://nonbiri-tereka.hatenablog.com/entry/2020/01/06/082921
  - manifold mixup : https://medium.com/@akichan_f/%E6%9C%80%E7%B5%82%E5%B1%A4%E3%81%A7mixup%E3%81%97%E3%81%9F%E3%82%89%E8%89%AF%E3%81%95%E3%81%92%E3%81%A0%E3%81%A3%E3%81%9F%E4%BB%B6-bd2ff167c388
- generalized mean pooling (GeM)
  - pool の一般化 (p=1 で mean, p=∞ で max と等しい), 論文では p=3 を推奨
  - イメージ (p.19) : https://www.slideshare.net/xavigiro/d1l5-contentbased-image-retrieval-upc-2018-deep-learning-for-computer-vision
- metric learning
  - 基礎 : https://copypaste-ds.hatenablog.com/entry/2019/03/01/164155
  - 基礎 : https://qiita.com/gesogeso/items/547079f967d9bbf9aca8
  - 最新の Loss のまとめ : https://qiita.com/yu4u/items/078054dfb5592cbb80cc
    - ArcFace の チューニング tips : https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/109987#latest-633349
  - 実装 : https://github.com/KevinMusgrave/pytorch-metric-learning
- snapshot ensemble (cosine annealing with warmup)
  - 実装 : https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

## メモ (コンペ中)

### ひとりごと

- まずはベースラインを作ることに徹する
  - シングルモデルでスコアを向上させることに徹したい
  - データの意味も理解できていないので、ベースができたら理解する
    - https://bengali.ai/wp-content/uploads/CV19-COCO-Grapheme.pdf
- 自分としてのコンペのポイントの予想
  - grapheme roots をとにかく精度良く当てること
    - クラスがかなり多い (160 程度)
    - 不均衡なデータ
    - スコアにも重みが付けられている
  - 問題はどうやって、imbalance や misslabeling に対処するか...?

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

### 行なった実験

#### ベース設定

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
- Augmentation
  - None
- Epoch
  - 25
- MultilabelStratifiedKFold (hold out)

#### 実験項目

この論文の内容をやる方針にする  
Bag of Tricks for Image Classification with Convolutional Neural Networks  
https://arxiv.org/pdf/1812.01187.pdf

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
- Scheduler & Optimizer
  - AdamW with OneCycle scheduler
