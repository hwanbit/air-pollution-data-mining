""" * File: dust.py
    * Author: Miseon Lee
    * Date: 2024-12-09
    * Description: 이 파일은 날짜별 대기 오염 수치 데이터를 통해 오염 요소 별 상관관계를 파악합니다."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Dataframe 형태로 파일 읽어오기
df = pd.read_csv('./Data/measuredData.csv')
df.head()
print(df)
print(df.columns)
print()
print()

# 데이터 기본 정보 및 기초 통계량 확인
print(df.info())
print()
print()
print(df.describe())
print()
print()

# 각 오염 물질의 최빈값 계산
for col in df.columns[1:]:
    mode_value = df[col].mode()[0]
    print(f"{col} 최빈값: {mode_value}")

# 한글 컬럼명을 영문명으로 변경
df.rename(columns={'날짜':'Datetime', '아황산가스':'SO2', '일산화탄소':'CO', '오존':'O3', '이산화질소':'NO2'}, inplace=True)
print(df.columns)
print()
print()

# 데이터 타입 변경
df['Datetime'] = df['Datetime'].str.replace(
    r"(\d{4}-\d{2}-\d{2}) 24",
    lambda x: (pd.Timestamp(x.group(1)) + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00'),
    regex=True)
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(df['Datetime'])
print()
print()
print(df.info())
print()
print()

# 결측치 확인 및 처리
print(df.isnull().any(axis=1))
print(df.isnull().any(axis=0))
df.dropna(subset=['O3'], inplace=True)
print(df.isnull().sum())
print(df.isnull().any(axis=1))
print(df.isnull().any(axis=0))

# 상관계수 함수를 이용하여 요소별 상관관계 분석
correlation_matrix = df.drop(columns=['Datetime']).corr()
print(correlation_matrix)
print()
print()

# 히스토그램으로 시각화
# 변수 개수에 맞춰 subplot 크기 계산
n_vars = len(correlation_matrix.columns)
ncols = 2  # 열의 수
nrows = (n_vars + 1) // ncols  # 행의 수 (반올림하여 충분히 큰 행을 확보)

# 그래프 크기 설정
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10))

# axes 배열이 2차원 배열로 구성되기 때문에, 이를 1차원으로 변환하여 사용할 수 있도록 처리
axes = axes.flatten()

# 각 변수에 대해 상관계수 막대그래프를 생성
for i, var in enumerate(correlation_matrix.columns):
    ax = axes[i]  # 각 그래프에 대해 적절한 위치 선택
    correlation_with_var = correlation_matrix[var].drop(var)  # 자기 자신과의 상관계수 제거

    # 상관계수의 최대값 찾기
    max_corr_value = correlation_with_var.max()

    # 상관계수의 색상을 다르게 설정 (최대값을 노란색으로 표현)
    colors = ['yellow' if corr == max_corr_value else 'skyblue' for corr in correlation_with_var]

    # 막대그래프 생성
    correlation_with_var.plot(kind='bar', ax=ax, color=colors, edgecolor='black')

    # 기준선 추가 (0 = 선 개수)
    ax.axhline(0, color='black', linewidth=1)

    # 그래프 꾸미기
    ax.set_title(f"Correlation with {var}", fontsize=14)
    ax.set_xlabel("Variables", fontsize=12)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.set_ylim(-1, 1)  # 상관계수 범위
    ax.set_xticklabels(correlation_with_var.index, rotation=0)

# 레이아웃 조정
plt.tight_layout()

# 그래프 출력
plt.show()

# 막대그래프로 일별 현황 그래프 출력
# 기존에 열려있는 플롯 창 모두 닫기
plt.close('all')

# 날짜별 평균값으로 집계
daily_avg = df.groupby(df['Datetime'].dt.date)[['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM2.5']].mean()

# 정규화
scaler = MinMaxScaler()
daily_avg_normalized = pd.DataFrame(scaler.fit_transform(daily_avg),
                                    columns=daily_avg.columns,
                                    index=daily_avg.index)

# 시각화 설정
fig, ax = plt.subplots(figsize=(15, 6))
daily_avg_normalized.plot(kind='bar', stacked=False, rot=45, ax=ax)

plt.title('Daily Report', fontsize=15)
plt.xlabel('Datetime', fontsize=12)
plt.ylabel('Scaled Values', fontsize=12)
plt.legend(title='Pollutant', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 상관관계 히트맵
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# 미세먼지(PM10)과 초미세먼지(PM2.5) 관계를 산점도 그래프로 확인

# PM10과 PM2.5 산점도 그래프
plt.figure(figsize=(8, 6))
plt.scatter(df['PM10'], df['PM2.5'], alpha=0.7)
plt.xlabel('PM10')
plt.ylabel('PM2.5')
plt.title('PM10 vs PM2.5 Scatter Plot')

# 추세선 추가
z = np.polyfit(df['PM10'], df['PM2.5'], 1)
p = np.poly1d(z)
plt.plot(df['PM10'], p(df['PM10']), color='lightsteelblue', linestyle='--', label='Trend Line')

plt.legend()
plt.tight_layout()
plt.show()

"""데이터 분석 정리
    * PM10과 PM2.5는 양의 상관관계를 갖는다.
    * CO와 NO2는 양의 상관관계를 갖는다."""