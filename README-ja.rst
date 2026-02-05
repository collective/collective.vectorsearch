.. This README is meant for consumption by humans and PyPI. PyPI can render rst files so please do not use Sphinx features.
   If you want to learn more about writing documentation, please check out: http://docs.plone.org/about/documentation_styleguide.html
   This text does not appear on PyPI or github. It is a comment.

**Language**: `English <README.rst>`_ | 日本語

.. image:: https://github.com/collective/collective.vectorsearch/actions/workflows/plone-package.yml/badge.svg
    :target: https://github.com/collective/collective.vectorsearch/actions/workflows/plone-package.yml

.. image:: https://coveralls.io/repos/github/collective/collective.vectorsearch/badge.svg?branch=main
    :target: https://coveralls.io/github/collective/collective.vectorsearch?branch=main
    :alt: Coveralls

.. image:: https://img.shields.io/pypi/v/collective.vectorsearch.svg
    :target: https://pypi.python.org/pypi/collective.vectorsearch/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/status/collective.vectorsearch.svg
    :target: https://pypi.python.org/pypi/collective.vectorsearch
    :alt: Egg Status

.. image:: https://img.shields.io/pypi/pyversions/collective.vectorsearch.svg?style=plastic
    :alt: Supported - Python Versions

.. image:: https://img.shields.io/pypi/l/collective.vectorsearch.svg
    :target: https://pypi.python.org/pypi/collective.vectorsearch/
    :alt: License


=======================
collective.vectorsearch
=======================

LLM埋め込みを使用したセマンティックベクトル検索機能を提供するPloneアドオンです。
テキストをベクトル埋め込みに変換し、コサイン類似度を使用して意味的に類似したコンテンツを検索できます。


機能
----

- **ZCatalog用VectorIndex**: ベクトル埋め込みを格納するカスタムカタログインデックス
- **複数の埋め込みモデル対応**: 以下のモデルをサポート:

  - MiniLM L6 v2（デフォルト、軽量、英語、FastEmbed）
  - E5-base multilingual（100+言語、FastEmbed）
  - E5-base multilingual GPU（GPU高速化、 ``[gpu]`` extras が必要）

- **デフォルトでFastEmbed**: CPUフレンドリーなONNX最適化埋め込み、GPUは不要
- **オプションのGPUサポート**: ``[gpu]`` extras を追加してGPU高速化処理に対応
- **コントロールパネル**: サイト設定から埋め込みモデルや検索設定を構成可能
- **バックエンド状態表示**: 利用可能なバックエンドとインストール状態を確認可能
- **遅延モデル読み込み**: モデルはパッケージインストール時ではなく、最初の使用時に読み込まれます
- **プラグイン可能なアーキテクチャ**: 新しい埋め込みモデルプロバイダーを簡単に追加可能


動作要件
--------

- Plone 6.0 または 6.1
- Python 3.10 - 3.13


インストール
------------

buildoutを使用してcollective.vectorsearchをインストール::

    [buildout]

    ...

    eggs =
        collective.vectorsearch


``bin/buildout`` を実行します。

GPUサポート（オプション）
~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorchとSentence TransformersによるGPU高速化埋め込みを使用する場合::

    [buildout]

    ...

    eggs =
        collective.vectorsearch [gpu]


クイックスタート
----------------

1. サイト設定 → アドオンからパッケージをインストール
2. サイト設定 → Vector Search で埋め込みモデルを設定
3. 「Embedding Backend Status」セクションでバックエンドが利用可能か確認
4. ``llm_vector`` インデックスが自動的に ``portal_catalog`` に追加されます
5. コントロールパネルまたはZMIからコンテンツを再インデックス


設定
----

サイト設定 → Vector Search から以下を設定できます:

- **Embedding Model**: 埋め込み生成に使用するモデルを選択（利用可能なモデルのみ表示）
- **Text Chunk Size**: 長いドキュメントのチャンク最大文字数
- **Storage Backend**: 現在はBTrees（内部ストレージ）をサポート
- **Approximation Algorithm**: 検索アルゴリズム（現在は網羅的コサイン類似度）

コントロールパネルには以下が表示されます:

- **Embedding Backend Status**: インストールされているバックエンドを表示（FastEmbed、Sentence Transformers）
- **Vector Index Statistics**: 各インデックスのドキュメント数とベクトル数


利用可能な埋め込みモデル
~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+--------+------+------------------------+
| モデル                    | 次元数 | GPU  | Extras                 |
+===========================+========+======+========================+
| MiniLM L6 v2 (FastEmbed)  | 384    | 不要 | (デフォルト)           |
+---------------------------+--------+------+------------------------+
| E5 Base Multilingual      | 768    | 不要 | (デフォルト)           |
| (FastEmbed)               |        |      |                        |
+---------------------------+--------+------+------------------------+
| E5 Base Multilingual      | 768    | 必要 | ``[gpu]``              |
| (GPU)                     |        |      |                        |
+---------------------------+--------+------+------------------------+


使い方
------

このパッケージは ``llm_vector`` という名前の ``VectorIndex`` をポータルカタログに追加します。
プログラムから以下のようにクエリできます::

    from plone import api

    catalog = api.portal.get_tool('portal_catalog')
    index = catalog.Indexes['llm_vector']

    # クエリは類似度スコア付きのドキュメントIDを返します
    results = index.query_index(record)


カスタムVectorIndexの追加
~~~~~~~~~~~~~~~~~~~~~~~~~

ZMIから追加のVectorIndexインスタンスを作成できます:

1. ``/Plone/portal_catalog/manage_main`` に移動
2. インデックスタイプのドロップダウンから「VectorIndex」を選択
3. IDを入力し、オプションでインデックス対象属性を指定


カスタムモデルプロバイダーの追加
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

外部パッケージは ``IEmbeddingModelProvider`` を実装して新しい埋め込みモデルを追加できます::

    from collective.vectorsearch.model_providers import BaseEmbeddingModelProvider

    class MyCustomProvider(BaseEmbeddingModelProvider):
        id = 'my-custom-model'
        title = u'My Custom Model'
        description = u'Custom model description'
        model_name = 'my-org/my-model'
        vector_dimensions = 768

        # バックエンド設定
        backend = 'fastembed'  # または 'sentence_transformers'
        backend_name = u'FastEmbed (CPU/ONNX)'
        requires_gpu = False
        extras_name = None  # または 'gpu' で [gpu] extras

パッケージの ``configure.zcml`` で登録::

    <utility
        factory=".providers.MyCustomProvider"
        provides="collective.vectorsearch.interfaces.IEmbeddingModelProvider"
        name="my-custom-model"
    />


重要な注意事項
--------------

再インストールとアップグレード
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

パッケージを再インストールまたはアップグレードした後は、**必ずPlone/Zopeサーバーを再起動**してください。
再起動しないと、モデルプロバイダーユーティリティが正しく登録されず、
再インデックスが正常に動作しない場合があります。

**推奨手順:**

1. サイト設定 → アドオンからパッケージを再インストールまたはアップグレード
2. Plone/Zopeサーバーを再起動
3. サイト設定 → Vector Search に移動
4. 「Reindex All」をクリックしてベクトルインデックスを再構築

アンインストール時の動作
~~~~~~~~~~~~~~~~~~~~~~~~

**警告:** このパッケージをアンインストールすると、カタログからすべてのベクトルデータが削除されます。
``llm_vector`` インデックスとすべての埋め込みは完全に削除されます。

パッケージのコードを更新しながらベクトルデータを保持する必要がある場合は、
アンインストール/再インストールではなく **アップグレード** 機能を使用してください。


開発
----

開発環境のセットアップ::

    git clone https://github.com/collective/collective.vectorsearch.git
    cd collective.vectorsearch
    make install

テストの実行::

    make test

詳細な開発手順については ``DEVELOP.rst`` を参照してください。


作者
----

- 寺田 学 (`@terapyon <https://github.com/terapyon>`_)


コントリビューター
------------------

- （ここにあなたの名前を）


コントリビュート
----------------

- Issue Tracker: https://github.com/collective/collective.vectorsearch/issues
- ソースコード: https://github.com/collective/collective.vectorsearch


サポート
--------

問題がある場合は、GitHubでIssueを作成してください:
https://github.com/collective/collective.vectorsearch/issues


ライセンス
----------

このプロジェクトはGPLv2でライセンスされています。
