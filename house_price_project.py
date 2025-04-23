jisuhan
jaewonpark

import numpy as np
import pandas as pd


df = pd.read_csv('./train.csv')
# df = pd.read_csv('./test.csv')
# df = pd.read_csv('./sample_submission.csv')


df["SalePrice"]
df["OverallQual"]
df["OverallCond"]





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.api import OLS
from statsmodels.formula.api import ols
import statsmodels.api as sm

# df = pd.read_csv("./train.csv")
# num_df = df.select_dtypes(include=[np.number])

# # 3. SalePrice와의 상관계수 계산
# correlation = num_df.corr()["SalePrice"].sort_values(ascending=False)

# # 4. 상위 10개 변수 추출 (SalePrice 본인은 제외)
# top_corr_features = correlation[1:11]

# # 5. 시각화
# plt.figure(figsize=(10, 6))
# sns.barplot(x=top_corr_features.values, y=top_corr_features.index)
# plt.title("Top 10 Numerical Features Correlated with SalePrice")
# plt.xlabel("Correlation Coefficient")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.show()



###################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train 셋

x = df['OverallQual']
y = df['SalePrice']
data_for_learning = pd.DataFrame({'x': x, 'y': y})
k = np.linspace(x.min(), x.max(), 100)


# train 셋 나누기 -> train, valid
from sklearn.model_selection import train_test_split
train, valid = train_test_split(data_for_learning, test_size=0.3, random_state=1234)
print(train.shape)
print(valid.shape)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# i=3    # i = 1에서 변동시키면서 MSE 체크 할 것
perform_train = []
perform_valid = []
for i in range(1, 21):
    poly = PolynomialFeatures(degree=i, include_bias=True)
    X_train = poly.fit_transform(train[['x']])
    X_valid = poly.transform(valid[['x']])
    model = LinearRegression().fit(X_train, train['y'])
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    mse_train = mean_squared_error(train['y'], y_train_pred)
    mse_valid = mean_squared_error(valid['y'], y_valid_pred)
    perform_train.append(mse_train)
    perform_valid.append(mse_valid)
best_degree = np.argmin(perform_valid) + 1
print("Best polynomial degree:", best_degree)

i=best_degree


plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), perform_train, marker='o', label='Train MSE')
plt.plot(range(1, 21), perform_valid, marker='s', label='Valid MSE')
plt.axvline(x=best_degree, color='red', linestyle='--', label=f'Best Degree: {best_degree}')
plt.title('Polynomial Degree vs. MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



poly1 = PolynomialFeatures(degree=i, include_bias=True)
train_X = poly1.fit_transform(train[['x']])
model1 = LinearRegression().fit(train_X, train['y'])
model_line_blue = model1.predict(poly1.transform(k.reshape(-1, 1)))

# 예측값 계산
train_y_pred = model1.predict(poly1.transform(train[['x']]))
valid_y_pred = model1.predict(poly1.transform(valid[['x']]))

# MSE 계산
mse_train = mean_squared_error(train['y'], train_y_pred)
mse_valid = mean_squared_error(valid['y'], valid_y_pred)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1행 2열 서브플롯

# 왼쪽: 학습 데이터와 모델 피팅 결과
axes[0].scatter(train['x'], train['y'], color='black', label='Train Observed')
axes[0].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[0].text(0.05, -1.8, f'MSE: {mse_train:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[0].set_title(f'{i}-degree Polynomial Regression (Train)')
axes[0].legend()
axes[0].grid(True)

# 오른쪽: 검증 데이터
axes[1].scatter(valid['x'], valid['y'], color='green', label='Valid Observed')
axes[1].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[1].text(0.05, -1.8, f'MSE: {mse_valid:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[1].set_title(f'{i}-degree Polynomial Regression (Valid)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

###################################################################
# 시각화


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.api import OLS
from statsmodels.formula.api import ols
import statsmodels.api as sm

ames_df = pd.read_csv("./plotly_data/ames.csv")
num_df = ames_df.select_dtypes(include=[np.number])

ames_df.columns



fig = px.scatter_mapbox(
    ames_df,
    lat="Latitude",
    lon="Longitude",
    size="SalePrice",
    color="Neighborhood",  # 자치구 느낌
    hover_name="Prop_Addr",  # 마우스 올리면 주소 뜸
    hover_data={"SalePrice": True, "YearBuilt": True},
    zoom=11,
    height=700,
    mapbox_style="open-street-map"
)

fig.update_layout(
    width=1000,
    margin={"r":0,"t":0,"l":0,"b":0}
)
fig.show()




pd.set_option('display.max_columns', None)
import geopandas as gpd
gdf = gpd.read_file("./plotly_data/서울시군구/TL_SCCO_SIG_W.shp")
gdf.head(7)
gdf.shape
gdf.info()
gdf["geometry"][0]
gdf["geometry"][1]
print(gdf.crs)
gdf = gdf.to_crs(epsg=4326)
gdf.to_file("./plotly_data/seoul_districts.geojson",
            driver="GeoJSON")


import json
with open('./plotly_data/seoul_districts.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f) 
print(geojson_data.keys())

agg_df = (lcd_df.groupby("자치구",
as_index=False)["LCD거치대수"]
.sum())
agg_df.columns = ["자치구", "LCD합계"]
# 컬럼 이름을 GeoJSON과 맞추기
agg_df = agg_df.rename(columns={"자치구": "SIG_KOR_NM"})
print(agg_df.head(2))



fig = px.choropleth_mapbox(
agg_df,
geojson=geojson_data,
locations="SIG_KOR_NM",
featureidkey="properties.SIG_KOR_NM",
color="LCD합계",
color_continuous_scale="Blues",
mapbox_style="carto-positron",
center={"lat": 37.5665, "lon": 126.9780},
zoom=10,
opacity=0.7,
title="서울시 자치구별 LCD 거치대 수"
)
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


df_bar = (df.groupby("day", as_index=False)["total_bill"].mean())
# 막대그래프 생성
fig = px.bar(df_bar, 
x="day",  
y="total_bill",
title="Average Total Bill by Day",
labels={"day": "Day","total_bill": "Avg Total Bill ($)"},
color="day"
)
fig.show()

fig = px.scatter_mapbox(
lcd_df,
lat="lat",
lon="long",
size="LCD거치대수",
color="자치구",
hover_name="대여소명",
hover_data={"LCD거치대수": True, "자치구": True, "lat": False, "long": False},
text="text", # 마커 텍스트
zoom=10,
height=650,
title="서울시 대여소별 LCD 거치대 수"
)
fig.update_layout(
mapbox_style="carto-positron",
mapbox_layers=[
{
"sourcetype": "geojson",
"source": geojson_data,
"type": "line", "color": "black", "line": {"width": 1}
}
],
mapbox_center={"lat": 37.5665, "lon": 126.9780},
margin={"r":0,"t":30,"l":0,"b":0})
fig.show()








#############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.api import OLS
from statsmodels.formula.api import ols
import statsmodels.api as sm





df = pd.read_csv("./plotly_data/ames.csv")
df.isna().sum()
df.isna().sum()[df.isna().sum() > 0]






# 1. 결측치가 '없음'을 의미하는 항목은 "None"으로 채우기
none_fill_cols = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual',
    'MasVnrType'
]
df[none_fill_cols] = df[none_fill_cols].fillna("None")

# 2. MasVnrArea는 벽돌 마감이 없으면 0으로 처리
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# 3. GarageYrBlt: 차고가 없는 경우는 0으로 처리
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

# 4. LotFrontage: 같은 Neighborhood 내 평균으로 채우기
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.mean())
)

# 5. Electrical: 최빈값으로 채우기
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# 다시 결측치 있는지 확인
remaining_na = df.isnull().sum()
remaining_na = remaining_na[remaining_na > 0]
remaining_na




df = pd.read_csv("./plotly_data/ames.csv")
num_df = df.select_dtypes(include=[np.number])
num_df.isna().sum()
num_df = num_df.dropna()
num_df = num_df.drop(columns=['PID'])
x = num_df.drop(columns=['SalePrice'])
y=num_df['SalePrice']


from sklearn.linear_model import LassoCV
alphas=np.linspace(0,0.5,1000)
model_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
model_cv.fit(x, y)
model_cv.alpha_
model_cv.mse_path_
model_cv.mse_path_.shape

num_df



lasso_coef = model_cv.coef_

# 2. 사용된 변수 확인 (계수가 0이 아닌 변수들만)
selected_features = x.columns[lasso_coef != 0]

# 3. 사용된 변수 이름과 계수 함께 보기
import pandas as pd
pd.Series(lasso_coef, index=x.columns)[lasso_coef != 0]







# 데이터 재로드
import pandas as pd


# 사용자가 제시한 변수 목록 정리
target_columns = [
    'OverallQual', 'OverallCond', 'RoofStyle', 'ExterQual', 'ExterCond',
    'Exterior1st', 'HeatingQC', 'GarageCond', 'BsmtCond', 'PavedDrive',
    'YearBuilt', 'YearRemodAdd', 'BsmtQual', 'GarageYrBlt',
    'TotRmsAbvGrd', 'Latitude', 'Longitude', 'SalePrice'
]

# 결측치 개수 확인
missing_info = df[target_columns].isnull().sum()
missing_info = missing_info[missing_info > 0]
missing_info


df = df.dropna(subset=['Latitude', 'Longitude'])
# 1. 카테고리형 결측치: 지하실/차고가 없는 경우 "None" 처리
df['GarageCond'] = df['GarageCond'].fillna("None")
df['BsmtCond'] = df['BsmtCond'].fillna("None")
df['BsmtQual'] = df['BsmtQual'].fillna("None")

# 2. 수치형 결측치: 차고 건축연도 → 차고 없음이면 0으로
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
df = df.drop(columns=['PID'])


df.isna().sum()
df = df[target_columns]