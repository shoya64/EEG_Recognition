# EEG_Recognition

### ファイル構成

EEG_Recognition <br>
┣ FeaturePlot: 特徴抽出層の２次元プロット <br>
┃ ┣ Domain_plot.py <br>
┃ ┣ Feature_plot.py <br>
┃ ┗ Feature_plot_noUMAP.py <br>
┣  Function: データ整形やモデル定義など必要な関数の定義 <br>
┃ ┣ data_load.py <br>
┃ ┣ dataset.py <br>
┃ ┣ functions.py <br>
┃ ┣ make_another.py <br>
┃ ┣ mk_data.py <br>
┃ ┣ model.py <br>
┃ ┣ param.py <br>
┃ ┗ plt_func.py <br>
┣  train: 学習・検証ファイルを格納 <br>
┃ ┣ validation.py <br>
┃ ┗ train.py <br>
┣  CFM_plot.py: 混同行列生成の実行ファイル <br>
┣  Feature_plot.py: 特徴抽出層の2次元プロット <br>
┣  main.py: 学習及び検証ファイルの実行ファイル <br>
┗  run_plot.py: 特徴抽出層の2次元プロットの実行ファイル <br>
