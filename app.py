""" streamlit_demo
streamlitでIrisデータセットの分析結果を Web アプリ化するモジュール

【Streamlit 入門 1】Streamlit で機械学習のデモアプリ作成 – DogsCox's tech. blog
https://dogscox-trivial-tech-blog.com/posts/streamlit_demo_iris_decisiontree/
"""

from itertools import chain
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 
# import graphviz
# import plotly.graph_objects as go
# irisデータセットでテストする
# from sklearn.datasets import load_iris

# 決定木
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# 精度評価用
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# データを水増しするライブラリを読み込む
from imblearn.over_sampling import SMOTE

# ロゴの表示用
from PIL import Image

# ディープコピー
import copy

sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def st_display_table(df: pd.DataFrame):

    # データフレームを表示
    st.subheader('データの確認')
    # st.caption('最初の10件のみ表示しています')
    st.table(df)

    # Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/


def st_display_histogram(df: pd.DataFrame, x_col, hue_col):

    fig, ax = plt.subplots()
    # plt.title("ヒストグラム", fontsize=20)     # (3) タイトル
    # plt.xlabel("Age", fontsize=20)          # (4) x軸ラベル
    # plt.ylabel("Frequency", fontsize=20)      # (5) y軸ラベル
    plt.grid(True)                            # (6) 目盛線の表

    if hue_col == 'null':
        unique_cnt = len(df[x_col].value_counts())
        if unique_cnt > 10:
            plt.xlabel(x_col, fontsize=12)          # x軸ラベル
            plt.hist(df[x_col])   # 単なるヒストグラム
        else:
            sns.countplot(data=df, x=x_col, ax=ax)
    else:
        sns.countplot(data=df, x=x_col, hue=hue_col, ax=ax)

    st.pyplot(fig)

    # seabornでグラフを複数のグラフを描画する - Qiita
    # https://qiita.com/tomokitamaki/items/b954e26be739bee5621e



def ml_dtree(
    X: pd.DataFrame,
    y: pd.Series,
    depth: int) -> list:
    """ 決定木で学習、予測を行う関数
    Irisデータセット全体で学習し、学習データの予測値を返す関数
    Args:
        X(pd.DataFrame): 説明変数郡
        y(pd.Series): 目的変数
    
    Returns:
        List: [モデル, 学習データを予測した予測値, accuracy]のリスト
    """
    # 学習
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X, y)

    # 予測
    pred = clf.predict(X)

    # accuracyで精度評価
    score = accuracy_score(y, pred)

    return [clf, pred, score]


def st_display_dtree(clf, features):
    """決定木可視化関数
    streamlitでDtreeVizによる決定木を可視化する関数
    Args:
        clf(sklearn.DecisionTreeClassifier): 学習済みモデル
    Return:
    """
    # # graphvizで決定木を可視化
    # dot = tree.export_graphviz(clf, out_file=None)
    # # stで表示する
    # st.graphviz_chart(dot)

    dot = tree.export_graphviz(clf, 
                               out_file=None, # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                               filled=True, # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                               rounded=True, # Trueにすると、ノードの角を丸く描画する。
                            #    feature_names=['あ', 'い', 'う', 'え'], # これを指定しないとチャート上で特徴量の名前が表示されない
                               feature_names=features, # これを指定しないとチャート上で特徴量の名前が表示されない
                            #    class_names=['setosa' 'versicolor' 'virginica'], # これを指定しないとチャート上で分類名が表示されない
                               special_characters=True # 特殊文字を扱えるようにする
                               )

    # stで表示する
    st.graphviz_chart(dot)


def st_display_rtree(clf, features):

    # 重要度の抽出
    feature_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
    feature_importances = feature_importances.to_frame(name='重要度').sort_values(by='重要度', ascending=False)

    # TOP20可視化
    feature_importances[0:20].sort_values(by='重要度').plot.barh()
    plt.legend(loc='lower right')
    # plt.show()
    st.pyplot(plt)


def st_display_confusion_matrix(y_true, y_pred, labels):
    """混同行列を計算して表示する関数"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # グラフの描画
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues')
    plt.xticks(rotation=45)
    st.pyplot(fig)


def ml_drtree_pred(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm,
    t_size: float,
    use_smote: bool,
    params: dict) -> list:

    # train_test_split関数を利用してデータを分割する
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, train_size=t_size, random_state=0, stratify=y)

    # use_smoteがTrueの場合のみ、SMOTE処理を実行
    if use_smote:
        st.info("SMOTEを有効にして、少数派クラスのデータを補正しました。")
        try:
            oversample = SMOTE(random_state=0)
            train_x, train_y = oversample.fit_resample(train_x, train_y)
        except ValueError:
            # クラス数が1つの場合や、サンプル数が少なくSMOTEが適用できない場合にエラーになることがある
            st.warning("SMOTEによるオーバーサンプリングが適用できませんでした。クラスの比率が不均衡な場合、精度が偏る可能性があります。")
            pass
    
    if algorithm == 'dtree':
        # 分類器の設定
        clf = DecisionTreeClassifier(random_state=0, **params)

    elif algorithm == 'rtree':
        # 分類器の設定
        clf = RandomForestClassifier(random_state=0, **params)

    # 学習
    clf.fit(train_x, train_y)

    # 戻り値の初期化
    train_scores = []
    valid_scores = []

    # 訓練データで予測 ＆ 精度評価
    train_pred = clf.predict(train_x)
    
    # スコア計算 (多クラス分類に対応)
    train_scores.append(round(accuracy_score(train_y, train_pred), 3))
    train_scores.append(round(recall_score(train_y, train_pred, average='macro', zero_division=0), 3))
    train_scores.append(round(precision_score(train_y, train_pred, average='macro', zero_division=0), 3))
    train_scores.append(round(f1_score(train_y, train_pred, average='macro', zero_division=0), 3))
    
    # 検証データで予測 ＆ 精度評価
    valid_pred = clf.predict(valid_x)
    valid_scores.append(round(accuracy_score(valid_y, valid_pred), 3))
    valid_scores.append(round(recall_score(valid_y, valid_pred, average='macro', zero_division=0), 3))
    valid_scores.append(round(precision_score(valid_y, valid_pred, average='macro', zero_division=0), 3))
    valid_scores.append(round(f1_score(valid_y, valid_pred, average='macro', zero_division=0), 3))

    return [clf, train_y, train_pred, train_scores, valid_y, valid_pred, valid_scores]


@st.cache_data # Streamlitのキャッシュ機能で、同じファイルの場合は再読み込みしない
def load_csv_with_encoding_detection(uploaded_file):
    """
    アップロードされたCSVファイルを様々な文字コードで読み込みを試み、データフレームとして返す。
    """
    if uploaded_file is None:
        return None

    # 試行する文字コードのリスト
    encodings = ['utf-8', 'utf-8-sig', 'shift_jis', 'cp932']
    
    for enc in encodings:
        try:
            # ファイルの読み込み位置を先頭に戻す
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            # 読み込みに成功したら、どの文字コードが使われたか表示してデータフレームを返す
            st.success(f"ファイルは文字コード '{enc}' で正常に読み込まれました。")
            return df
        except Exception:
            # 失敗した場合は次の文字コードを試す
            continue
            
    # 全ての文字コードで読み込みに失敗した場合
    st.error("ファイルの読み込みに失敗しました。対応している文字コード (UTF-8, SHIFT-JIS) か確認してください。")
    return None


def main():
    """ メインモジュール
    """
    # stのタイトル表示
    st.title("Simple AutoML Demo\n（Maschine Learning)")

    # サイドメニューの設定
    activities = ["データ確認", "要約統計量", "グラフ表示", "学習と検証", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    # About画面はファイルアップロードに依存しないので先に処理
    if choice == 'About':
        image = Image.open('logo_nail.png')
        st.image(image)
        st.markdown("Built by [Nail Team]")
        st.text("Version 0.4") # バージョンアップ
        st.markdown("For More Information check out   (https://nai-lab.com/)")
        return # About画面を表示したら処理を終了

    # ファイルのアップローダー
    uploaded_file = st.sidebar.file_uploader("訓練用データのアップロード", type='csv') 

    # ファイルがアップロードされるまでここで処理を中断
    if uploaded_file is None:
        st.subheader('サイドバーから訓練用データをアップロードしてください')
        return

    # ファイルをデータフレームとして読み込み
    df = load_csv_with_encoding_detection(uploaded_file)
    
    # 読み込みに失敗した場合はここで処理を中断
    if df is None:
        return

    # --- 各機能の表示 ---
    if choice == 'データ確認':
        st.subheader('データの確認')
        cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)
        st.table(df.head(int(cnt)))

    elif choice == '要約統計量':
        st.subheader('要約統計量')
        st.table(df.describe())

    elif choice == 'グラフ表示':
        st.subheader('グラフ表示')
        # hue_col = df.columns[0]     # '退職'
        x_col = st.sidebar.selectbox("グラフのX軸", df.columns)
        st_display_histogram(df, x_col, 'null')

    elif choice == '学習と検証':
        st.subheader('学習と検証')

        # --- 1. 分析設定 ---
        st.sidebar.markdown("### 1. 分析設定")
        # 目的変数の選択
        target_col = st.sidebar.selectbox(
            '目的変数（予測したい項目）',
            df.columns,
            help="予測の対象となる列を選びます。分類問題に適した、カテゴリや少数のユニークな値を持つ列が推奨されます。"
        )
        # 説明変数の選択
        feature_cols = st.sidebar.multiselect(
            '説明変数（予測に使う項目）',
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col],
            help="予測のために使用する列を複数選択します。目的変数として選んだ列は、ここでは選択できません。"
        )

        # --- 2. モデル設定 ---
        st.sidebar.markdown("### 2. モデル設定")
        algorithm = st.sidebar.selectbox("学習アルゴリズム", ["決定木", "ランダムフォレスト"])

        # --- 任意設定 ---
        with st.sidebar.expander("（任意）データの前処理設定"):
            train_size_percentage = st.slider(
                '訓練データの割合 (%)', 
                min_value=10, max_value=90, value=70, step=5,
                help="訓練データと検証データの分割比率を指定します。"
            )
            use_smote = st.checkbox(
                "不均衡データを補正する (SMOTE)", 
                value=False,
                help="データセット内の少数派クラスのサンプルを人工的に増やし、クラス間の不均衡を是正します。"
            )

        with st.sidebar.expander("（任意）モデルのハイパーパラメータ設定"):
            params = {}
            if algorithm == '決定木':
                params = {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 1}
                params['criterion'] = st.selectbox('分割基準', ['gini', 'entropy'], key='dt_criterion')
                params['max_depth'] = st.slider('木の深さ', 1, 20, params['max_depth'], key='dt_max_depth')
                params['min_samples_leaf'] = st.slider('葉の最小サンプル数', 1, 50, params['min_samples_leaf'], key='dt_min_samples_leaf')
            elif algorithm == 'ランダムフォレスト':
                params = {'n_estimators': 100, 'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 1}
                params['n_estimators'] = st.slider('木の数', 10, 300, params['n_estimators'], 10, key='rf_n_estimators')
                params['criterion'] = st.selectbox('分割基準', ['gini', 'entropy'], key='rf_criterion')
                params['max_depth'] = st.slider('各木の最大深度', 1, 20, params['max_depth'], key='rf_max_depth')
                params['min_samples_leaf'] = st.slider('葉の最小サンプル数', 1, 50, params['min_samples_leaf'], key='rf_min_samples_leaf')

        # --- 3. 学習実行 ---
        st.sidebar.markdown("### 3. 学習実行")
        if st.sidebar.button('学習を実行', key='train_button'):
            # --- データ準備と妥当性チェック ---
            if not feature_cols:
                st.warning("説明変数を1つ以上選択してください。")
                return
            if df[target_col].nunique() < 2:
                st.error(f"目的変数 '{target_col}' のユニークな値が1つしかありません。")
                return
            target_dtype = df[target_col].dtype
            target_nunique = df[target_col].nunique()
            if np.issubdtype(target_dtype, np.number) and target_nunique > 20:
                st.error(f"目的変数 '{target_col}' は連続値のようです。このツールは分類専用です。")
                return
            if target_nunique > 10:
                 st.warning(f"目的変数 '{target_col}' のユニークな値が10個を超えています。")
            
            # カテゴリ変数をダミー変数に変換
            original_cols = df[feature_cols].columns
            train_X = pd.get_dummies(df[feature_cols], drop_first=True)
            encoded_cols = train_X.columns
            newly_added_cols = set(encoded_cols) - set(original_cols)
            if newly_added_cols:
                st.info(f"以下のカテゴリ変数を数値データに変換しました:\n {', '.join(sorted(list(newly_added_cols)))}")
            train_Y = df[target_col]
            
            try:
                # --- 学習と予測 ---
                train_size_float = train_size_percentage / 100.0
                algorithm_key = 'dtree' if algorithm == '決定木' else 'rtree'
                
                (
                    clf, train_y_resampled, train_pred, train_scores,
                    valid_y, valid_pred, valid_scores
                ) = ml_drtree_pred(
                    train_X, train_Y, algorithm_key, train_size_float, use_smote, params
                )
                
                # --- 結果の表示 ---
                st.subheader("📊 予測精度の評価")
                
                tab1, tab2 = st.tabs(["訓練データでの評価", "検証データでの評価"])

                with tab1:
                    st.write("#### 評価指標")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric('正解率', train_scores[0])
                    col2.metric('再現率', train_scores[1])
                    col3.metric('適合率', train_scores[2])
                    col4.metric('F1スコア', train_scores[3])
                    
                    st.write("#### 混同行列 (Confusion Matrix)")
                    if use_smote:
                        st.caption("SMOTEによって少数派クラスのデータが水増しされた後のデータで評価しています。")
                    labels = sorted(train_Y.unique())
                    st_display_confusion_matrix(train_y_resampled, train_pred, labels)

                with tab2:
                    st.write(f"#### 評価指標 (訓練{train_size_percentage}% / 検証{100-train_size_percentage}%)")
                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric('正解率', valid_scores[0])
                    col6.metric('再現率', valid_scores[1])
                    col7.metric('適合率', valid_scores[2])
                    col8.metric('F1スコア', valid_scores[3])

                    st.write("#### 混同行列 (Confusion Matrix)")
                    st_display_confusion_matrix(valid_y, valid_pred, labels)

                st.subheader("🧠 モデルの可視化")
                if algorithm == '決定木':
                    st.write("#### 決定木の可視化")
                    st_display_dtree(clf, train_X.columns.tolist())
                elif algorithm == 'ランダムフォレスト':
                    st.write("#### 特徴量の重要度")
                    st_display_rtree(clf, train_X.columns.tolist())

            except Exception as e:
                st.error("学習中に予期せぬエラーが発生しました。")
                st.info("目的変数の選択が適切でない可能性があります。")
                with st.expander("エラー詳細の表示"):
                    st.exception(e)
        

if __name__ == "__main__":
    main()

