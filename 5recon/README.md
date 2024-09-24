⑤三次元再構成

python 2.7

CUDA 8.0

```sh
conda create -n py27 python=2.7 #python2.7なら不要(anaconda環境である必要はありません)
conda activate py27
pip install tensorflow-gpu==1.3.0 tflearn==0.3.2 scikit-image==0.14.5
sudo dpkg -i libcudnn6_6.0*+cuda8.0_amd64.deb
sudo dpkg -i libcudnn6-dev_6.0*+cuda8.0_amd64.deb
sudo dpkg -i libcudnn6-doc_6.0*+cuda8.0_amd64.deb
python train.py
python demo.py --image Data/examples/000.png
```
- 各現実写真にdemo.pyを実行（000.png~019.png）
- Data/examples内に.objファイルが生成される
- 論文の実験のデータセットはmensaからダウンロードしてData/ShapeNetP2M以下に配置

元のコード：https://github.com/nywang16/Pixel2Mesh

評価用のコード:https://colab.research.google.com/drive/1zkxUYKysM2qBoqE_lag1tVadOkY6uP7G?usp=sharing
- /content以下に〇〇〇.objと〇〇〇_gt.obj（krazに保存してあるgtモデル）を配置し、パスを以下のように書き換える
    - trg_obj = os.path.join('./〇〇〇.obj')
    - src_obj = os.path.join('./〇〇〇_gt.obj')
