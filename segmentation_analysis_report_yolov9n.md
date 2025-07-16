# セグメンテーション用中間表現の分析レポート（YOLOv9n）

## 分析概要

YOLOv9nの活性化後（Swish: Sigmoid + Mul）の中間表現およびConcat層（特徴融合層）から、インスタンスセグメンテーションに最適な層を分析しました。

## 推奨層（更新版）

### 1. 最優先推奨: `/model.22/Concat_output_0` (Concat層)
- **解像度**: 80×80 (8倍ダウンサンプリング)
- **チャンネル数**: 89
- **スコア**: 0.7293
- **特徴**:
  - 検出ヘッドの特徴融合層で、マルチスケール情報を統合
  - 空間解像度と意味情報の最適なバランス
  - 人物の輪郭と詳細を同時に捉える

### 2. FPN特徴融合層: `/model.14/Concat_output_0` (Concat層)
- **解像度**: 80×80 (8倍ダウンサンプリング)
- **チャンネル数**: 80
- **スコア**: 0.7117
- **特徴**:
  - Feature Pyramid Networkのアップサンプリング経路
  - 低レベルと高レベルの特徴を効果的に結合
  - インスタンスの分離に優れる

### 3. 高解像度代替案: `/model.2/cv3/act/Mul_output_0` (Swish層)
- **解像度**: 160×160 (4倍ダウンサンプリング)
- **チャンネル数**: 8
- **スコア**: 0.4975
- **特徴**:
  - より高い空間解像度で細かい詳細を保持
  - 小さな人物や重なりのある場合に有効
  - 浅い層のため意味的情報は限定的

### 4. 意味的セグメンテーション向け: `/model.11/Concat_output_0` (Concat層)
- **解像度**: 40×40 (16倍ダウンサンプリング)
- **チャンネル数**: 112
- **スコア**: 0.6501
- **特徴**:
  - FPN横方向接続による豊富な特徴表現
  - 人物全体の理解に適している
  - より深い意味的情報を保持

## 実行コマンド

```bash
# 推奨層（Concat層）のヒートマップ生成
python generate_heatmaps_unified.py --model yolov9_n_wholebody25_0100_1x3x640x640.onnx --layers "/model.22/Concat_output_0" "/model.14/Concat_output_0" "/model.11/Concat_output_0" --alpha 0.6

# Concat層とSwish層の比較
python generate_heatmaps_unified.py --model yolov9_n_wholebody25_0100_1x3x640x640.onnx --layers "/model.22/Concat_output_0" "/model.14/Concat_output_0" "/model.2/cv3/act/Mul_output_0" --alpha 0.6 --activated

# 最適なConcat層単体での可視化
python generate_heatmaps_unified.py --model yolov9_n_wholebody25_0100_1x3x640x640.onnx --layers "/model.22/Concat_output_0" --alpha 0.6
```

## 分析結果

YOLOv9nモデルにおいて、Concat層を含めた分析により、`/model.22/Concat_output_0`が最適な選択となりました。この層は：

1. **マルチスケール特徴の統合**: 検出ヘッドでの特徴融合により、様々なスケールの情報を活用
2. **高いセグメンテーションスコア**: 0.7293（Swish層の最高スコア0.5543を大幅に上回る）
3. **バランスの取れた解像度**: 80×80の解像度で、詳細と全体像のバランスが最適
4. **豊富なチャンネル数**: 89チャンネルにより、多様な特徴表現が可能

Concat層は、複数の経路からの特徴を統合するため、インスタンスセグメンテーションにおいて単一のConv層よりも優れた性能を発揮します。

生成されたヒートマップは`heatmaps/`および`overlays_60/`ディレクトリに保存されています。