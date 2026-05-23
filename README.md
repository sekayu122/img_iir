# img_iir

このリポジトリには、テスト動画の生成、TIFF連番へのIIRフィルタ適用、フィルタ品質の評価、結果の可視化に使うPythonスクリプトが含まれています。

## 必要要件

`requirements.txt` に記載されたPython依存パッケージをインストールしてください。

```bash
pip install -r requirements.txt
```

依存パッケージ:

- `numpy`
- `opencv-python`
- `imageio`
- `imageio-ffmpeg`
- `gdown`

## リポジトリ構成

- `generate_test_video.py` - 暗いノイズ付きのテストMP4動画またはTIFF連番を生成
- `dumpimg.py` - 画像の画素値をCSVに出力
- `apply_filter.py` - TIFF画像連番にIIRフィルタを適用
- `iir_filters.py` - 再利用可能なIIRフィルタ実装
- `concat_video.py` - 2つの動画または画像列を左右並びに合成
- `evaluate_img_quarity.py` - before/after/GT TIFF連番をノイズ・ブラー・モーションで評価
- `data/` - サンプルデータフォルダ

## スクリプト詳細

### generate_test_video.py

以下の要素を持つ合成テスト動画またはTIFF連番を生成します。

- 暗い背景
- 複数の固定輝度矩形
- 時変ノイズ
- 四角形の任意のモーション
- ビット深度と出力形式の指定

使い方:

```bash
python generate_test_video.py output.mp4
```

主なオプション:

- `--width`, `--height` - 出力フレームサイズ
- `--fps` - フレームレート
- `--frames` - フレーム数
- `--brightnesses` - カンマ区切りの矩形輝度値
- `--noise-strength` - ノイズ強度
- `--grain-size` - ノイズの粒度
- `--noise-mode` - `gaussian` または `uniform`
- `--bit-depth` - 保存するビット深度
- `--format` - `mp4` または `tiff`
- `--clean-squares` - ノイズ後も矩形の輝度を固定

例:

```bash
python generate_test_video.py test.mp4 --width 640 --height 360 --frames 120 --noise-strength 8 --format mp4
```

### dumpimg.py

`PNG` や `TIFF` などの画像を元のビット深度で読み込み、画素値をCSVに保存します。

使い方:

```bash
python dumpimg.py input.tiff output.csv
```

主なオプション:

- `--channel` - 出力するチャンネル番号
- `--roi` - `x,y,width,height` 形式の領域指定
- `--delimiter` - CSV区切り文字

例:

```bash
python dumpimg.py input.tif output.csv --channel 0 --roi 10,20,100,80
```

### apply_filter.py

TIFF画像のディレクトリにIIRフィルタを適用し、別の出力ディレクトリにフィルタ結果をTIFF連番で保存します。現在のところサポートしているフィルタは `alpha` のみです。

使い方:

```bash
python apply_filter.py input_dir output_dir
```

主なオプション:

- `--filter` - 現在は `alpha` のみ
- `--alpha` - フィルタのブレンド係数 (`0.0..1.0`)

例:

```bash
python apply_filter.py data/test/noisy_tiffs data/test/filtered_tiffs --alpha 0.5
```

### iir_filters.py

画像連番向けの再利用可能なフィルタ実装を提供します。

- `AlphaBlendIirFilter` - 現在フレームと前フレーム出力をアルファブレンドする1次IIR
  `output[n] = alpha * input[n] + (1 - alpha) * output[n-1]`
- `create_filter(name, alpha)` - 名前からフィルタオブジェクトを生成

このモジュールは `apply_filter.py` で使用されます。

### concat_video.py

2つの動画または画像列を左右に並べて出力します。

使い方:

```bash
python concat_video.py left_input right_input output.mp4
```

両方の入力がディレクトリの場合、画像連番として扱います。出力パスが動画拡張子の場合はMP4を書き出し、そうでなければ画像連番を出力します。

主なオプション:

- `--mode` - `auto`, `video`, `images`
- `--height` - 出力フレーム高さ
- `--fps` - 出力フレームレート
- `--crf` - 動画出力時のH.264品質
- `--image-extension` - ディレクトリ出力時の画像形式 (`tiff`, `png`, `bmp`)

例:

```bash
python concat_video.py left.mp4 right.mp4 side_by_side.mp4 --height 360 --crf 18
```

画像連番の場合の例:

```bash
python concat_video.py left_frames/ right_frames/ output_frames/ --mode images --image-extension png
```

### evaluate_img_quarity.py

`before`, `after`, `gt` の3つのTIFF連番を比較し、次の3つの輝度帯 (`dark`, `normal`, `high`) について評価します。

- ノイズ低減
- エッジ保持 / ブラー
- モーション誤差

使い方:

```bash
python evaluate_img_quarity.py before_dir after_dir gt_dir
```

主なオプション:

- `--target-noise-db` - 100点へ対応するノイズ低減dB
- `--blur-min-ratio` - ブラー評価で0点になるエッジ比率閾値
- `--motion-error-ref` - モーション誤差が0点になる参照値
- `--motion-threshold` - モーション検出の閾値
- `--noise-floor` - 既にクリーンとみなすノイズ下限
- `--min-pixels` - 各指標に必要な最小画素数
- `--dark-roi` - 暗部ノイズ測定用の静的領域指定
- `--weight` - `name=value` 形式でスコア重みを上書き
- `--json` - JSON形式で出力

例:

```bash
python evaluate_img_quarity.py before_tiffs after_tiffs gt_tiffs --json
```

## data ディレクトリ

`data/` にはサンプルデータフォルダが含まれています。

- `crvd_tiff16/` - TIFFテスト用データ
- `scene1_ISO12800_gt/` - グラウンドトゥルース画像
- `scene1_ISO12800_noisy0/` - ノイズ画像
- `scene1_ISO12800_noisy0_iir/` - フィルタ結果画像
- `test/` - 追加テストデータ

## 備考

- `generate_test_video.py` はMP4またはTIFF連番のどちらでも出力できます。
- `concat_video.py` は動画同士、または画像連番同士の左右並び出力をサポートします。
- `apply_filter.py` は入力ディレクトリと出力ディレクトリを別にする必要があります。
- `evaluate_img_quarity.py` はTIFF連番をフレーム単位で比較し、自動処理用にJSON出力も可能です。
