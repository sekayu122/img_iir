# img_iir

TIFF16連番画像にIIR系フィルタを適用し、GT画像と比較して画質評価するための実験用リポジトリです。主な用途は、`iir_filters.py` に実装したフィルタアルゴリズムを `run_experiment.py` で繰り返し評価することです。

## セットアップ

```bash
pip install -r requirements.txt
```

主な依存パッケージ:

- `numpy`
- `opencv-python`
- `imageio`, `imageio-ffmpeg`
- `scikit-image`
- `gdown`

## 基本ワークフロー

```bash
python run_experiment.py \
  data/validation_synthetic/test \
  output/validation_synthetic_ai \
  data/validation_synthetic/gt \
  --eval-before data/validation_synthetic/src \
  --filter AIExpFilter \
  --overwrite
```

内部では次の順に実行します。

```text
apply_filter.py input -> run_dir/filtered
evaluate_img_quarity.py eval-before run_dir/filtered gt
```

`--eval-before` を省略した場合は、フィルタ入力と同じディレクトリがbeforeとして使われます。

## 主要スクリプト

- `iir_filters.py` - フィルタアルゴリズム実装。AI開発では主にこのファイルを変更します。
- `apply_filter.py` - TIFF16連番にフィルタを適用します。
- `run_experiment.py` - フィルタ適用、評価、ログ保存をまとめて実行します。
- `evaluate_img_quarity.py` - before/after/GTの3入力で画質評価します。
- `generate_validation_dataset.py` - validation用の実在動画風合成データを生成します。
- `generate_test_video.py` - train評価向けの単純な矩形テスト動画/連番を生成します。
- `concat_video.py` - 動画または画像連番を左右に並べて比較用出力を作ります。
- `dumpimg.py` - 画像の画素値をCSVへ出力します。

## フィルタ開発

`apply_filter.py` と `run_experiment.py` は、`iir_filters.py` の `FILTER_REGISTRY` に登録されたフィルタ名を `--filter` で指定します。

```bash
python apply_filter.py input_tiffs output_tiffs --filter AIExpFilter
```

現在の主なフィルタ:

- `alpha` - 固定alphaの1次IIR。`--alpha` で係数を指定できます。
- `AIExpFilter` - AI開発用の実験フィルタ。

新しいアルゴリズムを追加する場合は、`iir_filters.py` に `AIExpFilterV2` のようなclassを追加し、`FILTER_REGISTRY` に登録します。これにより `apply_filter.py` や `run_experiment.py` を変更せずに呼び出せます。

## validationデータ生成

`generate_validation_dataset.py` は `src/`, `test/`, `gt/` と確認用MP4を生成します。

```bash
python generate_validation_dataset.py data/validation_synthetic \
  --frames 60 \
  --width 640 \
  --height 360 \
  --overwrite
```

出力:

- `gt/` - ノイズなし正解画像。評価のGTです。
- `src/` - ノイズあり画像。評価時のbeforeです。
- `test/` - フィルタ入力画像。現状は `src/` と同じ内容です。
- `gt_video.mp4`, `src_video.mp4`, `test_video.mp4` - 目視確認用MP4です。

背景モード:

- `--background-mode synthetic` - 背景グラデーション、低周波テクスチャ、線/矩形などの人工構造を生成します。
- `--background-mode skimage` - scikit-imageのサンプル静止画像を背景素材として使います。

scikit-image背景の例:

```bash
python generate_validation_dataset.py data/validation_skimage \
  --background-mode skimage \
  --skimage-image coffee \
  --frames 60 \
  --width 640 \
  --height 360 \
  --overwrite
```

どちらの背景モードでも、カメラ揺れ、動体、露光変動、高輝度点、輝度依存ノイズ、RGB別ノイズ、固定パターンノイズを追加します。

主なオプション:

- `--bit-depth` - 8/10/12/14/16bit相当のデータをTIFF16に保存します。
- `--noise-strength` - 正規化ノイズ強度です。
- `--fixed-pattern-strength` - 固定パターンノイズ強度です。
- `--temporal-correlation` - ノイズの時間相関です。
- `--video-quality` - 確認用MP4の品質です。
- `--no-videos` - 確認用MP4を生成しません。

## 評価

`evaluate_img_quarity.py` は3つのTIFF連番ディレクトリを比較します。

```bash
python evaluate_img_quarity.py before_dir after_dir gt_dir
```

評価指標:

- `noise_score` - `before - gt` に対して `after - gt` の残差ノイズがどれだけ減ったか。
- `blur_score` - `after` が `gt` のエッジ強度をどれだけ保持したか。
- `motion_score` - 動き領域での `after - gt` 誤差が小さいか。
- `psnr_score` - `after` と `gt` のPSNRを0..100点化したもの。
- `ssim_score` - `after` と `gt` のSSIMを0..100点化したもの。

`total_score` は、dark/normal/highの各輝度帯について、上記5項目の重み付き平均です。デフォルトでは全項目の重みが `1.0` です。

主なオプション:

- `--target-noise-db` - 100点に対応するノイズ低減dBです。
- `--blur-min-ratio` - blurが0点になる `after/GT` エッジ比率です。
- `--motion-error-ref` - motionが0点になる正規化誤差です。
- `--psnr-min-db`, `--psnr-target-db` - PSNRの0点/100点換算値です。デフォルトは25dB/40dBです。
- `--ssim-min`, `--ssim-target` - SSIMの0点/100点換算値です。デフォルトは0.75/0.95です。
- `--weight name=value` - 例: `--weight dark_psnr=0` のように重みを変更できます。
- `--json` - JSON形式で出力します。

## train用テストデータ生成

`generate_test_video.py` は矩形ベースの単純なtrain評価用フレーム列を生成します。`src`/`gt` のセットをまとめて作るスクリプトではないため、必要に応じてノイズあり/なしを別々に生成します。

MP4出力:

```bash
python generate_test_video.py test.mp4 \
  --width 640 \
  --height 360 \
  --frames 120 \
  --noise-strength 8 \
  --output-format mp4
```

TIFF16連番出力:

```bash
python generate_test_video.py data/test/src \
  --frames 100 \
  --output-format tiff16 \
  --bit-depth 16
```

主なオプション:

- `--brightnesses` - 矩形輝度のリストです。
- `--motion-max-speed` - 右側の矩形ほど速く動きます。
- `--noise-mode` - `gaussian` または `uniform` です。
- `--temporal-correlation` - ノイズの時間相関です。
- `--clean-squares` - 矩形内にはノイズを乗せません。

## その他の補助ツール

画像値をCSVへ出力:

```bash
python dumpimg.py input.tiff output.csv --channel 0 --roi 10,20,100,80
```

左右比較動画/画像連番を生成:

```bash
python concat_video.py left_input right_input output.mp4 --height 360 --crf 18
```

画像連番を左右に並べてTIFF/PNG連番として出すこともできます。

## dataディレクトリ例

現在の主なデータ:

- `data/test/src`, `data/test/gt` - train評価用データ。
- `data/validation_synthetic/src`, `test`, `gt` - synthetic背景のvalidationデータ。
- `data/validation_skimage/src`, `test`, `gt` - scikit-image背景のvalidationデータ。
- `data/crvd_tiff16/scene1_ISO12800_noisy0` - CRVD由来のnoisyデータ。
- `data/crvd_tiff16/scene1_ISO12800_gt` - CRVD由来のGTデータ。

CRVDデータは元データのライセンスに注意してください。
