# bengaliai-cv19

## コンペのメモ

- まずはベースラインを作ることに徹する
  - シングルモデルでスコアを向上させることに徹したい
  - ResNet ベースでもなんとかなる
  - serene xt50 or efficientnet b0
  - 問題はどうやって、imbalance や misslabeling に対処するか
    - データの意味も理解できていないので、ベースができたら理解する
  - まず画像をを全て pickle object に直すところからか...?

## 疑問点

- 画像について
  - 137x236 to 224x224 using F.interpolate() at the input らしい
  - 3 channel にする
    - imagenet の重みを使うと入力が 3channel しか受け付けないからだった
  - Resize も文字が中心になるように変換すべき
- モデルは...?
  - efficient-net
  - Densenet121
  - se_resnext50_32x4d
- Optmizer は..?
  - Adam with reducelronplateau で 40 epoch でもそこそこ出るっぽい
- Augment は...?
  - cutout (baseline)
  - cutmix = cutout + mixup
- CV は...?
  - MultilabelStratifiedKFold を使うで良い気がする
- 細かいところは以下を参照するのが良い
  - 下手に Discussion を見るよりは、ここのやつを参考にするのが良い
  - https://www.kaggle.com/c/bengaliai-cv19/discussion/12797
  - https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065
  - https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107926
  - https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117210
  - https://www.kaggle.com/c/understanding_cloud_organization/discussion/118080
  - https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110543
