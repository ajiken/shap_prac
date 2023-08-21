# This python file is a practice project of SHAP.
# ref: https://dev.classmethod.jp/articles/ml-xai-shap-merry-christmas/

import shap
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# カリフォルニアの住宅価格予測用データセットを取得
housing = fetch_california_housing(as_frame=True)

# 回帰モデルにRandomForestを使用
reg = RandomForestRegressor()
reg.fit(housing['data'], housing['target'])

# 説明モデルのインスタンスを作成
explainer = shap.TreeExplainer(model=reg,data=housing['data'])

# データ数が多すぎると計算に時間を要するため，nの値を適当に設定
shap_values = explainer(housing['data'].head(n=100))

# とある単一の入力に対しての貢献度を確認できる
shap.plots.waterfall(shap_values[0])

# 全体の貢献度の平均を確認できる
shap.plots.bar(shap_values=shap_values)

# 全体の貢献度の分布を確認できる
shap.plots.beeswarm(shap_values)