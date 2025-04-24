# jisuhan



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
# 사용자가 제시한 변수 목록 정리
target_columns = [
    # 품질/상태 관련
    'OverallQual', 'OverallCond', 'RoofStyle', 'ExterQual', 'ExterCond',
    'Exterior1st', 'HeatingQC', 'GarageCond', 'BsmtCond', 'BsmtQual',
    'KitchenQual', 'GarageQual', 'Foundation', 'PavedDrive',
    
    # 연도 관련
    'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',
    
    # 욕실 관련
    'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
    
    # 면적 관련
    'GrLivArea', 'TotalBsmtSF', 'GarageArea', 'WoodDeckSF',
    
    # 위치 관련
    'TotRmsAbvGrd', 'Latitude', 'Longitude', 'SalePrice'
]

# 결측치 개수 확인
missing_info = df[target_columns].isnull().sum()
missing_info = missing_info[missing_info > 0]
missing_info
df = df.dropna(subset=['Latitude', 'Longitude'])



# 4. 결측치 처리 
# 4-1. 카테고리형 결측치: 지하실/차고가 없는 경우 "None" 처리
df['GarageCond'] = df['GarageCond'].fillna("None")
df['BsmtCond'] = df['BsmtCond'].fillna("None")
df['BsmtQual'] = df['BsmtQual'].fillna("None")
df['GarageQual'] = df['BsmtQual'].fillna("None")

# 4-2. 수치형 결측치: 차고 건축연도 → 차고 없음이면 0으로
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
df['GarageArea'] = df['GarageArea'].fillna(0)
df = df.drop(columns=['PID'])


df = df[target_columns]

# 5. 범주형 변수 처리하기
# 5️-1. 범주형 변수 → 수치형 변환 (Ordinal Encoding)
# 범주형 변수 수치화 (Ordinal Encoding)
ordinal_mappings = {
    'ExterQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
}

# 매핑 적용
for col, mapping in ordinal_mappings.items():
    df[col] = df[col].map(mapping)

df = pd.get_dummies(df, columns=['RoofStyle', 'Exterior1st', 'Foundation', 'PavedDrive'], drop_first=True)


# 6️. 결측치 확인
print(df.isnull().sum())


# LassoCV (변수 39개)
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# 독립변수(X), 종속변수(y) 분리
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# 1. 스케일링 (표준화)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # X를 평균 0, 표준편차 1로 변환

# 2️. LassoCV 학습 (교차검증으로 alpha 찾기)
lasso = LassoCV(cv=5, random_state=2025)
lasso.fit(X_scaled, y)

# 3. 결과 확인
best_alpha = lasso.alpha_
coefficients = lasso.coef_

# 4. 계수와 변수명을 데이터프레임으로 정리
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(f"Best alpha: {best_alpha}")
print(coef_df)



# Lasso alpha
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 기준 alpha (LassoCV에서 나온 값)
base_alpha = lasso.alpha_

# 실험할 alpha 배수들
alpha_multipliers = [0.5, 1, 2, 5, 10, 15, 20, 25]
results = []

# 반복해서 모델 돌리기
for m in alpha_multipliers:
    alpha = base_alpha * m
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_scaled, y)
    coef = model.coef_
    num_nonzero = np.sum(coef != 0)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)

    results.append({
        'alpha': round(alpha, 5),
        'multiplier': m,
        'num_features': num_nonzero,
        'mse': round(mse, 2)
    })

    # 남아 있는 변수 출력
    print(f"\n[alpha = {alpha:.5f}]")
    print(f"남은 변수 개수: {num_nonzero}")
    print("선택된 변수:")
    print(pd.Series(X.columns[coef != 0]).values)

# 결과 DataFrame으로 정리
results_df = pd.DataFrame(results)
print("\n📊 Alpha별 결과:")
print(results_df)

# 그래프로 보기
plt.figure(figsize=(10, 5))
plt.plot(results_df['alpha'], results_df['num_features'], marker='o', label='Selected Features')
plt.xlabel('Alpha')
plt.ylabel('Number of Selected Features')
plt.title('Alpha 값에 따른 변수 선택 개수 변화')
plt.grid(True)
plt.legend()
plt.show()

from sklearn.model_selection import GridSearchCV, KFold

# alpha로 찾기
selected_features = ['OverallQual', 'ExterQual', 'KitchenQual' ,'YearBuilt', 'BsmtFullBath',
'GrLivArea', 'TotalBsmtSF' ,'GarageArea' ,'WoodDeckSF','RoofStyle_Hip' ]




# 절대값으로 10개 찾기
# 1. 종속변수 로그 변환
y = (df['SalePrice'])  # log(1 + y)

# 2. 상위 10개 피처 선택 (기존 coef_df 기준)
top_features = coef_df[coef_df["Coefficient"] != 0]['Feature'].head(10).tolist()
X = df[top_features]

# 3. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Lasso + GridSearchCV 설정
lasso = Lasso(max_iter=10000)
alphas = np.linspace(0.01, 1, 100)
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
grid = GridSearchCV(estimator=lasso, param_grid={'alpha': alphas},
                    cv=cv, scoring='neg_mean_squared_error')
grid.fit(X_scaled, y)

# 5. 최적 alpha 및 성능 확인
print("Best alpha:", grid.best_params_['alpha'])
print("Best CV Score (MSE):", -grid.best_score_)

# 6. 회귀 계수 출력
coef_df_top = pd.DataFrame({
    'Feature': top_features,
    'Coefficient': grid.best_estimator_.coef_}).sort_values(by='Coefficient', key=abs, ascending=False)

print(coef_df_top)




# 모델 비교
# model 1 알파값
from statsmodels.formula.api import ols
model1 = ols(
    'SalePrice ~ OverallQual + ExterQual + KitchenQual + YearBuilt + BsmtFullBath + GrLivArea + TotalBsmtSF + GarageArea + WoodDeckSF + RoofStyle_Hip',
    data=df
).fit()

model1.summary()


# model 2 절대값
model2 = ols(
    formula='SalePrice ~ GrLivArea + OverallQual + TotalBsmtSF + GarageArea + YearBuilt + OverallCond + BsmtFullBath + ExterQual + KitchenQual + BsmtQual',
    data=df
).fit()

model2.summary()


import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm




df

# 최종 선택 10개 변수
top_features 
df[top_features]



import matplotlib.pyplot as plt
import seaborn as sns



# 시각화
plt.figure(figsize=(16, 12))
for i, col in enumerate(top_features):
    plt.subplot(4, 3, i + 1)
    sns.scatterplot(data=df, x=col, y='SalePrice', s=20)
    plt.xlabel(col)
    plt.ylabel('')

plt.tight_layout()
plt.show()

# 이상치 조건
condition = (
    (df['GrLivArea'] <= 3500) &
    (df['GarageArea'] <= 1200) &
    (df['TotalBsmtSF'] <= 2500)
)

# 이상치 제거
df_clean = df[condition].copy()

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 스케일링할 변수 목록
scale_columns = [
    'GrLivArea', 'TotalBsmtSF', 'GarageArea',   # 면적
    'OverallQual', 'KitchenQual', 'BsmtQual', 'ExterQual',  # 품질
    'OverallCond', 'BsmtFullBath',  # 상태
    'YearBuilt'  # 연식
]

# MinMaxScaler 적용 (0~1 범위)
scaler = MinMaxScaler()
df_clean[scale_columns] = scaler.fit_transform(df_clean[scale_columns])
df_clean = df_clean[scale_columns]


# 면적 그대로
area_score = df_clean['GrLivArea'] + df_clean['TotalBsmtSF'] + df_clean['GarageArea']

# 품질/상태/연식 뒤집기 (1 - 값)
quality_score = (1 - df_clean['OverallQual']) + (1 - df_clean['KitchenQual']) + (1 - df_clean['BsmtQual']) + (1 - df_clean['ExterQual'])
condition_score = (1 - df_clean['OverallCond']) + (1 - df_clean['BsmtFullBath'])
year_score = 1 - df_clean['YearBuilt']  # 연식도 반전

# 가중치 곱하고 합산
df['MaintenanceScore'] = (
    area_score * 0.2 +
    quality_score * 0.3 +
    condition_score * 0.2 +
    year_score * 0.3
)




df['Latitude']










# import statsmodels.api as sm

# def forward_stepwise(X, y, threshold_in=0.05, verbose=True):
#     initial_features = []
#     remaining_features = list(X.columns)
#     selected_features = []

#     while remaining_features:
#         aic_with_candidates = []
#         for candidate in remaining_features:
#             try:
#                 model = sm.OLS(y, sm.add_constant(X[initial_features + [candidate]])).fit()
#                 aic_with_candidates.append((model.aic, candidate))
#             except:
#                 continue
        
#         if not aic_with_candidates:
#             if verbose:
#                 print("⚠️ 남은 변수로 더 이상 개선 불가. 종료합니다.")
#             break

#         aic_with_candidates.sort()
#         best_aic, best_candidate = aic_with_candidates[0]

#         if verbose:
#             print(f"Try adding {best_candidate:>20} | AIC: {best_aic:.2f}")

#         initial_features.append(best_candidate)
#         remaining_features.remove(best_candidate)
#         selected_features = initial_features[:]

#     return selected_features




df_copy = df[condition].copy()



mean = df['MaintenanceScore'].mean()
std = df['MaintenanceScore'].std()


# 이건 정규분포 또는 비슷한 분포를 가진 점수값들을
# 상대적으로 나누기 위한 일반적인 전략
# 점수가 몇 점이 높고 낮은지 명확히 기준을 제시할 수 있어.
# 분포의 중심값(평균)을 기준으로 판단하니, 전체 데이터 비교에 유리.
# 절대값 기준보다 덜 왜곡됨, 특히 스케일링된 데이터에서는 더욱!


def hybrid_grade(score):
    if score >= mean + std:
        return "D (시급)"
    elif score >= mean:
        return "C (개선 필요)"
    elif score >= mean - std:
        return "B (양호)"
    else:
        return "A (최우수)"

df_copy['MaintenanceGrade'] = df['MaintenanceScore'].apply(hybrid_grade)

sns.scatterplot(data=df_copy, x='MaintenanceScore', y='SalePrice', hue='MaintenanceGrade')
plt.title("SalePrice vs MaintenanceScore")
plt.xlabel("Maintenance Score")
plt.ylabel("Sale Price")
plt.grid(True)
plt.show()




### D등급 중 낮은 유지보수를 가지면서 가격이 저렴한 top 10


# df['MaintenanceScore'] = df_clean['MaintenanceScore']
# df.info()
# df_copy=df.copy()
# df_copy = df_copy.dropna(subset=['MaintenanceScore'])
# df_copy.info()

# 1. D등급만 필터링 + 정렬 (로컬에서 다시 정의)
df_copy_d = df_copy[df_copy['MaintenanceGrade'] == 'A (최우수)'].copy()
df_d_sorted = df_copy_d.sort_values(by='SalePrice', ascending=True)

# 2. 상위 10개 선택
top10_d = df_d_sorted.head(10)
import plotly.express as px
top10_d['SalePrice']
top10_d[selected_features]
# 3. 지도 시각화
fig = px.scatter_mapbox(
    top10_d,
    lat="Latitude",
    lon="Longitude",
    color="SalePrice",
    size="SalePrice",
    color_continuous_scale="Reds",
    size_max=15,
    zoom=11,
    mapbox_style="carto-positron",
    hover_data=["SalePrice", "MaintenanceGrade", "YearBuilt", "GrLivArea"]
)

fig.update_layout(
    title="Top 10 Critical D-Grade Houses (Low Price & Low Maintenance Score)",
    width=950,
    height=600,
    margin={"r":0,"t":40,"l":0,"b":0}
)

fig.show()




fig = px.scatter_mapbox(
    df_copy,
    lat="Latitude",
    lon="Longitude",
    color="MaintenanceGrade",  # 등급별 색상 구분
    category_orders={
        "MaintenanceGrade": ["A (최우수)", "B (양호)", "C (개선 필요)", "D (시급)"]
    },
    mapbox_style="carto-positron",
    zoom=11,
    size="SalePrice",  # 혹은 GrLivArea도 가능
    size_max=12,
    hover_data=["SalePrice", "MaintenanceGrade", "YearBuilt", "GrLivArea"]
)

fig.update_layout(
    title="Maintenance Grade of All Properties in Ames Housing",
    width=950,
    height=600,
    margin={"r":0,"t":40,"l":0,"b":0}
)

fig.show()




###########################################
import seaborn as sns
pivot = df_copy.pivot_table(index="Neighborhood", values="MaintenanceScore", aggfunc="mean")
plt.figure(figsize=(8, 10))
sns.heatmap(pivot.sort_values("MaintenanceScore", ascending=False), cmap="RdYlGn_r", annot=True)
plt.title("Neighborhood-wise Average Maintenance Score")
plt.show()





origin_data = pd.read_csv("./plotly_data/ames.csv")
df_neigh = origin_data[[ 'Neighborhood']]  # 또는 'Id'가 기준이라면 그걸로
df_copy = df_copy.merge(origin_data[['Neighborhood']], left_index=True, right_index=True)

###########################################
# Neighborhood별 지도 시각화
# 1. 등급 비율 계산
grade_pct = pd.crosstab(df_copy['Neighborhood'], df_copy['MaintenanceGrade'])
grade_pct = grade_pct.div(grade_pct.sum(axis=1), axis=0) * 100

# 2. 가장 많은 등급 선택
grade_pct['DominantGrade'] = grade_pct.idxmax(axis=1)

# 3. 좌표 평균
coords = df_copy.groupby('Neighborhood')[['Latitude', 'Longitude']].mean()

# 4. 병합
import plotly.express as px
house_counts = df_copy['Neighborhood'].value_counts().rename("NumHouses")
# 5. 병합 후 표본 수까지 추가
neigh_dominant = grade_pct[['DominantGrade']].merge(coords, left_index=True, right_index=True)
neigh_dominant = neigh_dominant.merge(house_counts, left_on='Neighborhood', right_index=True).reset_index()

fig = px.scatter_mapbox(
    neigh_dominant,
    lat="Latitude",
    lon="Longitude",
    size="NumHouses", 
    color="DominantGrade", 
    color_discrete_sequence=px.colors.qualitative.Set2,
    category_orders={
    "DominantGrade": ["A (최우수)", "B (양호)", "C (개선 필요)", "D (시급)"]
    },
    mapbox_style="carto-positron",
    zoom=11,
    size_max=30,
    hover_name="Neighborhood",
    hover_data={"DominantGrade": True, "Latitude": False, "Longitude": False}
)

fig.update_layout(
    title="지역별 가장 많이 분포된 유지보수 등급 (Dominant Grade by Neighborhood)",
    width=950,
    height=600,
    margin={"r":0,"t":40,"l":0,"b":0}
)

fig.show()




# # 기준점 설정 (평균 위도/경도 기준)
# lat_mid = df_copy['Latitude'].median()
# lon_mid = df_copy['Longitude'].median()

# # 동서남북 지역 구분 함수
# def assign_direction(lat, lon):
#     if lat >= lat_mid and lon < lon_mid:
#         return 'NW'
#     elif lat >= lat_mid and lon >= lon_mid:
#         return 'NE'
#     elif lat < lat_mid and lon < lon_mid:
#         return 'SW'
#     else:
#         return 'SE'

# # 새로운 컬럼으로 저장
# df_copy['Direction'] = df_copy.apply(lambda row: assign_direction(row['Latitude'], row['Longitude']), axis=1)

# # df['MaintenanceGrade'] = df_clean['MaintenanceScore']
# plt.figure(figsize=(14, 6))
# sns.countplot(
#     data=df_copy,
#     x='MaintenanceGrade',
#     hue='Direction',
#     order=["A (최우수)", "B (양호)", "C (개선 필요)", "D (시급)"],
#     palette="Set2"
# )
# plt.title("Distribution of Maintenance Grades by Region (Grouped by Latitude/Longitude)")
# plt.xlabel("Maintenance Grade")
# plt.ylabel("Number of Houses")
# plt.xticks(rotation=0)
# plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()





# # 지도 시각화 - Direction 색상, SalePrice 크기
# fig = px.scatter_mapbox(
#     df_copy,
#     lat="Latitude",
#     lon="Longitude",
#     color="Direction",
#     size="SalePrice",
#     size_max=15,
#     zoom=11,
#     mapbox_style="carto-positron",
#     hover_data=["SalePrice", "YearBuilt", "OverallQual", "MaintenanceGrade"]
# )

# fig.update_layout(
#     title="Map of Ames Housing by Direction & Maintenance Grade",
#     width=950,
#     height=600,
#     margin={"r":0,"t":40,"l":0,"b":0}
# )

# fig.show()




# fig = px.scatter_mapbox(
#     df_copy,
#     lat="Latitude",
#     lon="Longitude",
#     color="MaintenanceGrade",  # A~D 등급별 색상
#     size="SalePrice",
#     size_max=15,
#     zoom=11,
#     mapbox_style="carto-positron",
#     hover_data=["SalePrice", "YearBuilt", "OverallQual", "Direction"]
# )

# fig.update_layout(
#     title="Map of Maintenance Grades by Direction (Hover)",
#     width=950,
#     height=600,
#     margin={"r":0,"t":40,"l":0,"b":0}
# )

# fig.show()




# plt.figure(figsize=(8, 5))
# sns.boxplot(
#     data=df_copy,
#     x="MaintenanceGrade",
#     y="SalePrice",
#     order=["A (최우수)", "B (양호)", "C (개선 필요)", "D (시급)"],
#     palette="Set2"
# )
# plt.title("SalePrice by Maintenance Grade")
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.show()






####  게이지 차트 시각화 : 중간 숫자는 평균 가격

# 평균 계산
species_avg = df_copy.groupby("MaintenanceGrade")["SalePrice"].mean().reindex(
    ["A (최우수)", "B (양호)", "C (개선 필요)", "D (시급)"]
)

# 최대값 설정
max_val = df_copy["SalePrice"].max()
import plotly.graph_objects as go
# 그래프 생성
fig = go.Figure()

for i, (grade, avg_price) in enumerate(species_avg.items()):
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=round(avg_price / 1000),  # 천 단위 k로 표시
        number={"suffix": "k", "font": {"size": 40}},
        title={'text': f"<b>{grade}</b>", "font": {"size": 18}},
        domain={'row': i // 2, 'column': i % 2},  # 2행 2열
        gauge={
            'axis': {'range': [0, round(max_val / 1000)], 'tickformat': 'k'},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, round(max_val * 0.5 / 1000)], 'color': '#d0f0f5'},
                {'range': [round(max_val * 0.5 / 1000), round(max_val / 1000)], 'color': '#9dddf2'}
            ],
        }
    ))

# 레이아웃 적용
fig.update_layout(
    grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
    title="<b>MaintenanceGrade vs. Average SalePrice</b>",
    height=700,
    width=1000,
    font=dict(family="Arial", size=14)
)

fig.show()









############ [EDA] 시각화 #########
#### EDA   시계열 그래프 1
# 사용할 변수 리스트
features = X.columns.tolist()

df.groupby("YearBuilt")["SalePrice"].mean().plot(
    kind="line", figsize=(10, 4), marker="o", color="royalblue"
)
plt.title("Average Sale Price by Year Built")
plt.xlabel("Year Built")
plt.ylabel("Average Sale Price")
plt.grid(True)
plt.tight_layout()
plt.show()



#### EDA   연식과 가격을 통한 지도 시각화 2

import plotly.express as px

fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="YearBuilt",           # 색상으로 연식 표현
    size="SalePrice",            # 크기로 가격 표현
    color_continuous_scale="Blues",
    size_max=15,
    zoom=11,
    mapbox_style="carto-positron",
    hover_data=["SalePrice", "YearBuilt", "GrLivArea", "OverallQual"]
)

fig.update_layout(
    title="Housing Age & Price on Map",
    width=950,
    height=600,
    margin={"r":0,"t":40,"l":0,"b":0}
)

fig.show()


#### EDA   품질과 가격을 통한 박스플롯 3
sns.boxplot(x="OverallQual", y="SalePrice", data=df)
plt.title("SalePrice by Overall Quality")
plt.grid(True)
plt.show()
