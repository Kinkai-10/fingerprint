# 機械学習
機械学習におけるパラメータ調整による学習の精度がどのように変化するのかを調査した。その際、CNNを用いて600人の指紋の学習を行っている[こちら](https://www.kaggle.com/brianzz/subjectid-finger-cnnrecognizer)を基に調査を行った。

# 今回使用した環境、ライブラリ
`python 3.7.9`  
`Keras 2.4.3` `tensorflow 2.2.0` `scikit-learn 0.23.2` `seaborn 0.11.0` `matplotlib 3.3.1`

# インストール
**python 3.7.9のインストール方法**<br>
pythonの[公式サイト](https://www.python.org/downloads/release/python-379/)から自分のOSや環境に合わせたインストーラーをダウンロードした後、インストーラーを起動しインストールする。<br>
**ライブラリのインストール方法**<br>
`pip install Keras==2.4.3`のようにして上記のライブラリをインストールする。

# 使い方
まず、指紋のデータセットを[Kaggle](https://www.kaggle.com/ruizgara/socofing)から作業用ディレクトリにダウンロードする。その際、アカウントを作成する必要がある。  
次に、作業用ディレクトリに[main.py](https://github.com/Kinkai-10/fingerprint/blob/main/main.py)をダウンロードし、`python main.py`を実行する。なお本実験では、main.pyの106行目から125行目までの畳み込み層の生成を行っている際の活性化関数、132行目のepoch、133行目のbatch_sizeを変更して変化を観察した。
 

# グループメンバー
大学の学籍番号をメンバーの情報とする。  
e185710  
e185714  
e185745  
e185752  
e185763  

# License
no License

# 謝辞
元のコードを書いてくださったMr.Brian Zhangに深く感謝いたします。