# cuda,cud
FROM nvidia/cuda:8.0-cudnn7-devel


# 必要なパッケージのインストール。Advanceにて書き換える箇所。
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    vim \
    sudo \
    curl \
    tmux && \
    rm -rf /var/lib/apt/lists/*
    
# user設定
ARG user=${user:-user}
ARG uid=${uid:-uid}
ARG gid=${gid:-gid}

# Anaconda環境構築
ENV ANACONDA_ROOT=/opt/conda

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ${ANACONDA_ROOT} && \
    rm ~/miniconda.sh && \
    ${ANACONDA_ROOT}/bin/conda clean -ya 

# condaのpathを通す
ENV PATH ${ANACONDA_ROOT}/bin:$PATH

# condaのpathが通っているか確認
ARG PYTHON_VERSION=3.7.9
RUN conda install -y python=$PYTHON_VERSION

# root権限で作成したディレクトリなどがsudo権限がないと操作できない問題対策
# dockerという仮のグループを作成し、ユーザーを追加することで権限を下げる。
RUN groupadd -g ${gid} docker && \
    useradd -g docker -u ${uid} -s /bin/bash ${user}

# sudoerに自分を追加、パスワードなしでsudoコマンドを使えるようにする
RUN echo ${user} ALL=NOPASSWD: ALL >> /etc/sudoers

# ホームディレクトリを最初に表示する
WORKDIR /home/${user}

# ホームディレクトリの権限をユーザに下げる
RUN chown -R ${user} /home/${user}

# 作成したユーザーに切り替える
USER ${uid}

# pipする際に警告を出さないためにpathを通す
ENV PATH /home/${user}/.local/bin:$PATH

#-----jupyterを使いたい人は以下のRUNのコメントアウトを解除する-----

# jupyter notebookをhttp://127.0.0.1:5****/で操作できるように設定する。
    #jupyterのインストール
#RUN python -m pip install jupyter
    # 初期設定作成
#RUN jupyter notebook --generate-config
    # どこからでもアクセスできるように
#RUN echo c.NotebookApp.ip = \'0.0.0.0\' >> /home/${user}/.jupyter/jupyter_notebook_config.py
    # 自動でブラウザを開かないように
#RUN echo c.NotebookApp.open_browser = False >> /home/${user}/.jupyter/jupyter_notebook_config.py
    # トークン(パスワード)をなしに
#RUN echo c.NotebookApp.token = u\'\' >> /home/${user}/.jupyter/jupyter_notebook_config.py
    # ポートをdocker-compose.ymlで設定したものに
#RUN echo c.NotebookApp.port = $((${uid}%10000+50000)) >> /home/${user}/.jupyter/jupyter_notebook_config.py
