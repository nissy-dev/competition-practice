# bengaliai-cv19

## コンペのメモ

- まずはベースラインを作ることに徹する
  - シングルモデルでスコアを向上させることに徹したい
  - 問題はどうやって、imbalance や misslabeling に対処するか...?
    - データの意味も理解できていないので、ベースができたら理解する

## 疑問点

- 画像について
  - 224x224 の方が精度は良さそう
  - 3 channel にする
    - imagenet の重みを使うと入力が 3channel しか受け付けないからだった
  - Resize の際には、文字が中心になるように変換すべき (こういう細かな配慮は大事)
- モデルは...?
  - densenet
  - efficientnet-b3
  - se_resnext50_32x4d
  - 今の所、学習時間的にも efficientnet が良さそう
- Loss は...?
  - Reduced Focal Loss
  - OHEM
- Optmizer は..?
  - Adam with ReduceLROnPlateau
- Augment は...?
  - baseline: ShiftScaleRotate + CutOut
  - CutMix (= cutout + mixup), AugMix も試す
- CV は...?
  - MultilabelStratifiedKFold を使うで良い気がする
  - hold out で良いか...
- 細かいところは以下を参照するのが良い
  - 下手に Discussion を見るよりは、ここのやつを参考にするのが良い
  - https://www.kaggle.com/c/bengaliai-cv19/discussion/12797
  - https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065
  - https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926
  - https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117210
  - https://www.kaggle.com/c/understanding_cloud_organization/discussion/118080
  - https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110543

# 実験

## 1 周目

- モデル
  - 学習が遅すぎる...
    1 epoch 40 min score も監視する
  - Efficienet-b4 とかでも良いかも
- モデルの構造
  - Target を分ける
- loss の自作
  - OHEM が気になる
  - 不均衡データに対する loss
