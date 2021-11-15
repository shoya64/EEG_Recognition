# EEG_Recognition

### ファイル構成

EEG_Recognition
┣ FeaturePlot: 特徴抽出層の２次元プロット
┃ ┣ Domain_plot.py
┃ ┣ Feature_plot.py
┃ ┗ Feature_plot_noUMAP.py
┣  Function: データ整形やモデル定義など必要な関数の定義
┃ ┣ data_load.py
┃ ┣ dataset.py
┃ ┣ functions.py
┃ ┣ make_another.py
┃ ┣ mk_data.py
┃ ┣ model.py
┃ ┣ param.py
┃ ┗ plt_func.py
┣  train: 学習・検証ファイルを格納
┃ ┣ validation.py
┃ ┗ train.py
┣  CFM_plot.py: 混同行列生成の実行ファイル
┣  Feature_plot.py: 特徴抽出層の2次元プロット
┣  main.py: 学習及び検証ファイルの実行ファイル
┗  run_plot.py: 特徴抽出層の2次元プロットの実行ファイル

