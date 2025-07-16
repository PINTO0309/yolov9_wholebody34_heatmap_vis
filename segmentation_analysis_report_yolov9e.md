# セグメンテーション用中間表現の分析レポート（YOLOv9e）

## 分析概要

YOLOv9e（拡張版）の活性化後（Swish: Sigmoid + Mul）の中間表現およびConcat層（特徴融合層）から、インスタンスセグメンテーションに最適な層を分析しました。YOLOv9eはYOLOv9nと比較してより深く複雑なアーキテクチャを持ち、総計991層（Concat層58個、Swish活性化層250個）を含んでいます。

## 推奨層

### 1. 最優先推奨: `/model.3/Concat_output_0` (Concat層)
- **解像度**: 160×160 (4倍ダウンサンプリング)
- **チャンネル数**: 256
- **スコア**: 0.76
- **特徴**:
  - 早期段階での特徴融合により、詳細な空間情報を保持
  - 中程度のチャンネル数で豊富な特徴表現
  - 高解像度により小さな人物や細かい輪郭の検出に優れる
  - 浅い層のため計算効率が良い

### 2. 高解像度FPN融合層: `/model.19/Concat_output_0` (Concat層)
- **解像度**: 160×160 (4倍ダウンサンプリング)
- **チャンネル数**: 256
- **スコア**: 0.76
- **特徴**:
  - Feature Pyramid Networkの中間段階での特徴融合
  - エンコーダとデコーダの情報を効果的に結合
  - 人物の詳細な境界線検出に適している

### 3. 意味的特徴融合層: `/model.5/Concat_output_0` (Concat層)
- **解像度**: 80×80 (8倍ダウンサンプリング)
- **チャンネル数**: 512
- **スコア**: 0.70
- **特徴**:
  - 空間解像度と意味情報の最適なバランス
  - 深いチャンネル数により複雑な特徴を表現
  - インスタンス分離に最適な解像度
  - YOLOv9nの最適層と同等の性能

### 4. デコーダ特徴融合層: `/model.22/Concat_output_0` (Concat層)
- **解像度**: 80×80 (8倍ダウンサンプリング)
- **チャンネル数**: 512
- **スコア**: 0.70
- **特徴**:
  - デコーダパスでの重要な特徴融合点
  - マルチスケール情報の効果的な統合
  - 人物全体と詳細の両方を捉える

### 5. 深層特徴融合層: `/model.34/Concat_output_0` (Concat層)
- **解像度**: 80×80 (8倍ダウンサンプリング)
- **チャンネル数**: 1024
- **スコア**: 0.70
- **特徴**:
  - 非常に深いチャンネル数で最も豊富な特徴表現
  - 複雑なシーンでの人物検出に強い
  - 計算コストは高いが精度重視の場合に推奨

## Swish活性化層の代替案

### 高解像度Swish層: `/model.3/cv4/act/Mul_output_0`
- **解像度**: 160×160 (4倍ダウンサンプリング)
- **チャンネル数**: 256
- **スコア**: 0.76
- **特徴**: Concat層と同等の性能だが、単一パスの特徴のみ

### 中解像度Swish層: `/model.5/cv4/act/Mul_output_0`
- **解像度**: 80×80 (8倍ダウンサンプリング)
- **チャンネル数**: 512
- **スコア**: 0.70
- **特徴**: バランスの取れた特徴抽出

## 実行コマンド

```bash
# 推奨Concat層のヒートマップ生成
python generate_heatmaps_all_layers.py --model yolov9_e_wholebody25_0100_1x3x640x640.onnx --layers "/model.3/Concat_output_0" "/model.19/Concat_output_0" "/model.5/Concat_output_0" "/model.22/Concat_output_0" --layer-types Concat --alpha 0.6

# 高解像度層の比較（Concat vs Swish）
python generate_heatmaps_all_layers.py --model yolov9_e_wholebody25_0100_1x3x640x640.onnx --layers "/model.3/Concat_output_0" "/model.3/cv4/act/Mul_output_0" --layer-types Concat Mul --alpha 0.6

# 最適な単一層での可視化
python generate_heatmaps_all_layers.py --model yolov9_e_wholebody25_0100_1x3x640x640.onnx --layers "/model.3/Concat_output_0" --layer-types Concat --alpha 0.6

# 全推奨層の統合可視化
python generate_heatmaps_all_layers.py --model yolov9_e_wholebody25_0100_1x3x640x640.onnx --layers "/model.3/Concat_output_0" "/model.19/Concat_output_0" "/model.5/Concat_output_0" "/model.22/Concat_output_0" "/model.34/Concat_output_0" --layer-types Concat --alpha 0.6
```

## YOLOv9nとの比較

### アーキテクチャの違い
- **層の総数**: YOLOv9e (991層) vs YOLOv9n (より少ない)
- **Concat層**: YOLOv9e (58個) vs YOLOv9n (より少ない)
- **Swish層**: YOLOv9e (250個) vs YOLOv9n (より少ない)

### 推奨層の比較
1. **解像度の選択**: YOLOv9eでは160×160の高解像度層が最高スコア（0.76）を獲得
2. **チャンネル数**: YOLOv9eは全体的により深いチャンネル数を持つ
3. **最適なダウンサンプリング率**: 両モデルとも8倍が最適だが、YOLOv9eは4倍でも高性能

## 用途別推奨

### 1. 高精度・詳細重視
- 使用層: `/model.3/Concat_output_0` または `/model.19/Concat_output_0`
- 理由: 160×160の高解像度で細かい境界線を正確に検出

### 2. バランス重視（推奨）
- 使用層: `/model.5/Concat_output_0` または `/model.22/Concat_output_0`
- 理由: 80×80の解像度で精度と効率のバランスが最適

### 3. 複雑なシーン・重なり対応
- 使用層: `/model.34/Concat_output_0`
- 理由: 1024チャンネルの深い特徴表現で複雑な状況に対応

## 分析結果のまとめ

YOLOv9eモデルの分析により、以下の知見が得られました：

1. **高解像度の優位性**: 4倍ダウンサンプリング（160×160）の層が最高スコアを獲得し、詳細な境界線検出に優れている
2. **Concat層の重要性**: 特徴融合を行うConcat層は、単一のConv層やSwish層よりもセグメンテーションに適している
3. **マルチスケール活用**: 異なる解像度の層を組み合わせることで、様々なサイズの人物に対応可能
4. **計算効率**: 浅い層（model.3）を使用することで、高い性能を維持しながら計算効率も確保

生成されたヒートマップは`heatmaps/`および`overlays_60/`ディレクトリに保存されています。