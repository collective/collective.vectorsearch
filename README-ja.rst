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

Plone CMSにセマンティックベクトル検索を追加するアドオンです。
LLMベースのモデルを使用してテキストをベクトル埋め込みに変換し、多段階近似最近傍探索により意味的に類似したコンテンツを検索します。


機能
----

- **ZCatalog用VectorIndex**: Ploneの既存インデックスと共存するベクトル埋め込み用カスタムカタログインデックス
- **多段階近似検索**: 自動フォールバック付きの3つの検索アルゴリズム:

  - **Exhaustive Cosine**（デフォルト）: 全ドキュメントに対する総当たりコサイン類似度
  - **ITQ-LSH 2段階**: ITQバイナリハッシュによるハミング距離ランキング → 上位K件にコサイン類似度
  - **ITQ-LSH 3段階**: Pivotベースの三角不等式フィルタリング → ハミングランキング → コサイン類似度

- **複数の埋め込みモデル対応**:

  - All-MiniLM-L6-v2（デフォルト、384次元、英語、FastEmbed/CPU）
  - E5-Base Multilingual（768次元、100+言語、FastEmbed/CPU）
  - E5-Base Multilingual GPU（768次元、GPU高速化、 ``[gpu]`` extras が必要）

- **アノテーションベースのデータ保存**: ベクトルデータをコンテンツアノテーションにSingle Source of Truthとして保存
- **デフォルトでFastEmbed**: CPUフレンドリーなONNX最適化埋め込み、GPUは不要
- **オプションのGPUサポート**: ``[gpu]`` extras でPyTorchとSentence TransformersによるGPU高速化処理に対応
- **コントロールパネル**: サイト設定からモデル・検索アルゴリズム・パラメータを構成可能
- **プラグイン可能なアーキテクチャ**: 外部パッケージから新しい埋め込みモデルプロバイダーを追加可能


動作要件
--------

- Plone 6.0 or 6.1+
- Python 3.10 - 3.13


インストール
------------

buildoutを使用してcollective.vectorsearchをインストール::

    [buildout]

    ...

    eggs =
        collective.vectorsearch


``bin/buildout`` を実行します。

またはpipでインストール::

    pip install collective.vectorsearch

GPUサポート（オプション）
~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorchとSentence TransformersによるGPU高速化埋め込みを使用する場合::

    pip install collective.vectorsearch[gpu]

またはbuildout::

    [buildout]

    ...

    eggs =
        collective.vectorsearch [gpu]


クイックスタート
----------------

1. サイト設定 → アドオンからパッケージをインストール
2. サイト設定 → Vector Search で埋め込みモデルを設定
3. ``llm_vector`` インデックスが自動的に ``portal_catalog`` に追加されます
4. コンテンツの作成・変更時に自動的にベクトル化されます
5. 既存コンテンツのベクトル化にはコントロールパネルの「Reindex All」ボタンを使用


仕組み
------

アーキテクチャ
~~~~~~~~~~~~~~

コンテンツが作成・変更されると、イベントサブスクライバが自動的に埋め込みを計算し、
コンテンツアノテーションに保存します。カタログインデクサーはアノテーションからデータを読み取り、
VectorIndexおよびサポートインデックス（pivot1-8、itq_hashes）を更新します。

::

    コンテンツの作成/変更
      |
      +-- イベントサブスクライバ: compute_and_store_vectors()
      |     +-- 設定されたモデルでテキストを埋め込み
      |     +-- ITQバイナリハッシュを計算（128ビット）
      |     +-- Pivot距離を計算（8 Pivot）
      |     +-- 全データをコンテンツアノテーションに保存
      |
      +-- カタログインデクシング
            +-- VectorIndex: アノテーションからベクトルを読み込み
            +-- pivot1-8 KeywordIndex: Pivot距離を読み込み
            +-- itq_hashes メタデータ: ITQハッシュを読み込み

多段階検索
~~~~~~~~~~

本パッケージは `lsh-cascade-poc <https://github.com/cmscom/lsh-cascade-poc>`_ の研究成果に基づく
多段階近似最近傍探索を実装しています:

**Exhaustive Cosine** (``exhaustive_cosine``):
  全インデックス済みドキュメントに対してコサイン類似度を計算。
  最も正確だが、大規模データセットでは低速。

**ITQ-LSH 2段階** (``itq_lsh_2stage``):
  1. クエリのITQハッシュを計算し、全ドキュメントをハミング距離でランキング
  2. 上位K件の候補（ ``itq_candidates`` 、デフォルト: 100）にコサイン類似度を計算

**ITQ-LSH 3段階** (``itq_lsh_3stage``):
  1. **Pivotフィルタリング**: 8つのPivot距離と三角不等式を使用し、KeywordIndexのレンジクエリで候補を絞り込み
  2. **ハミングランキング**: 残りの候補をITQハミング距離でランキングし、上位K件を選択
  3. **コサイン類似度**: 最終候補に対して精密なスコアリング

ITQまたはPivotデータが利用できない場合、システムは自動的にフォールバック:
3段階 → 2段階 → Exhaustive。


設定
----

サイト設定 → Vector Search から以下を設定できます:

- **Embedding Model**: 埋め込み生成に使用するモデルを選択
- **Text Chunk Size**: チャンク最大文字数（100-10,000、デフォルト: 500）
- **Approximation Algorithm**: 検索戦略（exhaustive_cosine、itq_lsh_2stage、itq_lsh_3stage）
- **Pivot Threshold（Stage 1）**: Pivotベース検索のフィルタリング閾値（コサイン距離×1000、デフォルト: 200）
- **ITQ Candidates（Stage 2）**: ハミングランキング後の候補数（デフォルト: 100）
- **Storage Backend**: 現在はBTrees（内部ストレージ）をサポート


利用可能な埋め込みモデル
~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+--------+------+------------------------+
| モデル                    | 次元数 | GPU  | Extras                 |
+===========================+========+======+========================+
| All-MiniLM-L6-v2          | 384    | 不要 | (デフォルト)           |
| (FastEmbed)               |        |      |                        |
+---------------------------+--------+------+------------------------+
| E5 Base Multilingual      | 768    | 不要 | (デフォルト)           |
| (FastEmbed)               |        |      |                        |
+---------------------------+--------+------+------------------------+
| E5 Base Multilingual      | 768    | 必要 | ``[gpu]``              |
| (GPU)                     |        |      |                        |
+---------------------------+--------+------+------------------------+


使い方
------

プログラムからの検索
~~~~~~~~~~~~~~~~~~~~

このパッケージは ``llm_vector`` という名前の ``VectorIndex`` をポータルカタログに追加します。
プログラムから以下のようにクエリできます::

    from plone import api

    catalog = api.portal.get_tool('portal_catalog')
    index = catalog.Indexes['llm_vector']

    # 類似コンテンツの検索
    results = index.query_index(record)


カスタムVectorIndexの追加
~~~~~~~~~~~~~~~~~~~~~~~~~

ZMIから追加のVectorIndexインスタンスを作成できます:

1. ``/Plone/portal_catalog/manage_main`` に移動
2. インデックスタイプのドロップダウンから「VectorIndex」を選択
3. IDを入力し、オプションでインデックス対象属性を指定（カンマ区切り）


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


オフラインモデルダウンロード
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FastEmbedは初回使用時にモデルをダウンロードします。オフライン環境では、
CLIコマンドを使用してモデルを事前にダウンロードできます::

    vectorsearch-download

これにより、サポートされているすべてのモデルが ``~/.cache/fastembed`` にダウンロードされます。
別の場所を使用するには ``FASTEMBED_CACHE_PATH`` 環境変数を設定してください。


重要な注意事項
--------------

再インストールとアップグレード
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

パッケージを再インストールまたはアップグレードした後は、**必ずPlone/Zopeサーバーを再起動**してください。
再起動しないと、モデルプロバイダーユーティリティが正しく登録されない場合があります。

**推奨手順:**

1. サイト設定 → アドオンからパッケージを再インストールまたはアップグレード
2. Plone/Zopeサーバーを再起動
3. サイト設定 → Vector Search に移動
4. 「Reindex All」をクリックしてベクトルインデックスを再構築

アンインストール時の動作
~~~~~~~~~~~~~~~~~~~~~~~~

**警告:** このパッケージをアンインストールすると、カタログからすべてのベクトルデータが削除されます。
``llm_vector`` インデックス、Pivotインデックス、およびすべての埋め込みは完全に削除されます。

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
