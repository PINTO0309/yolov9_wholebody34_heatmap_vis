# セグメンテーション用中間表現の分析レポート

## 分析概要

YOLOv9の活性化後（Swish: Sigmoid + Mul）の中間表現から、インスタンスセグメンテーションに最適な層を分析しました。

## 推奨層

### 1. 最優先推奨: `/model.22/cv3/cv3.0/cv3/act/Mul_output_0`
- **解像度**: 60×80 (8倍ダウンサンプリング)
- **チャンネル数**: 128
- **スコア**: 0.7047
- **特徴**:
  - 空間解像度と意味情報のバランスが最適
  - 人物の輪郭を捉えるのに十分な解像度
  - cv3モジュールによるマルチスケール特徴の集約

### 2. 高解像度代替案: `/model.19/cv3/cv3.0/cv3/act/Mul_output_0`
- **解像度**: 120×160 (4倍ダウンサンプリング)
- **チャンネル数**: 64
- **特徴**:
  - より高い空間解像度で細かい詳細を保持
  - 小さな人物や重なりのある場合に有効
  - 計算コストは高め

### 3. 意味的セグメンテーション向け: `/model.25/cv3/cv3.0/cv3/act/Mul_output_0`
- **解像度**: 30×40 (16倍ダウンサンプリング)
- **チャンネル数**: 256
- **特徴**:
  - より深い意味的特徴
  - 人物全体の理解に適している
  - 細かい境界線の精度は低下

## 実行コマンド

```bash
# 推奨層のヒートマップ生成
python generate_heatmaps_unified.py --activated --layers "/model.22/cv3/cv3.0/cv3/act/Mul_output_0" --alpha 0.6

# 3つの層の比較
python generate_heatmaps_unified.py --activated --layers "/model.19/cv3/cv3.0/cv3/act/Mul_output_0" "/model.22/cv3/cv3.0/cv3/act/Mul_output_0" "/model.25/cv3/cv3.0/cv3/act/Mul_output_0" --alpha 0.6
```

## 分析結果

画像内の13人の人物セグメンテーションにおいて、`/model.22/cv3/cv3.0/cv3/act/Mul_output_0`が最適な選択です。この層は：

1. 人物の境界を明確に識別
2. 重なり合う人物の分離が可能
3. 計算効率と精度のバランスが良好

生成されたヒートマップは`heatmaps/`および`overlays_60/`ディレクトリに保存されています。

![overlay__model 22_cv3_cv3 0_cv3_act_Mul_output_0](https://github.com/user-attachments/assets/2691a384-0c7c-43bc-9464-acc87b05ae5c)
