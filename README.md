# Air Pollution Correlation Analysis

This project analyzes air pollution data to identify correlations between different pollutants and visualize trends over time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)

## Overview

The script processes air pollution data from a CSV file, cleans and normalizes it, and generates visualizations including correlation matrices, histograms, and scatter plots to reveal patterns between pollutants.

## Features

- Reads air pollution data from `measuredData.csv`.
- Handles missing and malformed data entries.
- Computes and displays pollutant correlations.
- Visualizes data with:
  - Correlation bar charts.
  - Daily pollutant averages with normalization.
  - Correlation heatmaps.
  - Scatter plots (e.g., PM10 vs PM2.5).
- Highlights key pollutant relationships.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/air-pollution-analysis.git
   cd air-pollution-analysis
   ```
2. Install dependencies.

## Usage

1. Place your air pollution CSV file at `./Data/measuredData.csv`.
2. Run the script:
   ```bash
   python dust.py
   ```
3. Visualizations of pollutant correlations and daily trends will be displayed.

## Dependencies

- Python 3.8
- matplotlib
- seaborn
- pandas
- numpy
- scikit-learn

## Project Structure

```
.
├── Data
│   └── measuredData.csv
├── dust.py
└── README.md
```

---

## 데이터 전처리 방법 및 시각화

날짜별로 아황산가스(SO₂), 일산화탄소(CO), 오존(O₃), 이산화질소(NO₂), 미세먼지(PM10), 초미세먼지(PM2.5)의 수치가 들어있는 데이터를 탐색하고, 데이터 정제 과정을 거쳐 수치별 상관관계를 파악하는 것이 목표입니다.

### 1. 파일 읽어오기
Pandas의 read_csv를 통해서 csv 파일을 읽어와 Dataframe 형태로 만듭니다. <br/>
Dataframe 형태는 Numpy 구조를 기반으로 하여 연산 작업에서 빠른 속도를 보이기 때문에 데이터 분석에 자주 사용되는 형태입니다.
```python
df = pd.read_csv('./Data/measuredData.csv')
```

### 2. 데이터 기본 구조 및 기초통계량 확인
맨 앞의 데이터 5개와 마지막 데이터 5개를 콘솔창에 출력하여 데이터 구조를 간단히 확인하고 데이터에 어떤 식으로 접근하면 좋을지 파악합니다.
```python
print(df)
```
데이터 전처리 전에 데이터의 타입 확인이 우선이므로 데이터의 열과 행 개수를 확인하며, 열의 이름과 결측치 여부, 데이터의 타입을 확인합니다.
```python
df.info()
```
데이터의 각종 통계량을 요약하여 출력해보며 데이터 전처리를 어떤 방식으로 진행해야 할지 파악합니다.
```python
print(df.describe())
```
최빈값은 df.describe()를 통해 출력되지 않으므로 아래 코드를 통해 각 열의 최빈값을 파악합니다.<br/>
최빈값 확인이 결측치를 어떤 방식으로 처리할 지 결정하는 데 도움을 줍니다.
```python
for col in df.columns[1:]:
    mode_value = df[col].mode()[0]
    print(f"{col} 최빈값: {mode_value}")
```

### 3. 한글 컬럼명을 영문으로 변경
한글 사용으로 발생할 수 있는 잠재적 오류를 방지하기 위해 한글로 되어 있던 column 명을 영어로 변경해줍니다.
```python
df.rename(columns={'날짜':'Datetime', '아황산가스':'SO2', '일산화탄소':'CO', '오존':'O3', '이산화질소':'NO2'}, inplace=True)
print(df.columns)
```

### 4. 데이터 타입 변경
날짜 열 데이터를 살펴보면 01부터 24까지의 시간으로 이루어져 있습니다. 24는 날짜가 하루 더해지며 00시로 변경되는 것으로 바꿔야 합니다.<br/>
아래 코드를 통해 문자열 데이터에 대해 정규표현식을 사용하여 특정 패턴을 찾고, lambda 함수를 통해 변환할 수 있습니다. 마지막으로 pandas의 datetime 형식으로 변환하여 Datetime 열을 저장합니다.
```python
df['Datetime'] = df['Datetime'].str.replace(
    r"(\d{4}-\d{2}-\d{2}) 24",
    lambda x: (pd.Timestamp(x.group(1)) + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00'),
    regex=True)
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(df['Datetime'])
```

### 5. 결측치 확인 및 처리
결측치를 처리하지 않은 채로 데이터를 분석하게 되면 결과가 왜곡될 수 있습니다. Datetime을 제외한 모든 열에서 결측치가 존재하므로 결측치에 일정 값을 입력해주는 것보다 삭제하는 편을 택하였습니다.<br/>
따라서 결측치가 가장 많은 열인 O3 열에 맞추어 결측치를 제거합니다.
```python
df.dropna(subset=['O3'], inplace=True)
```

### 6. 상관계수 함수를 이용하여 요소별 상관관계 분석
날짜를 제외한 각 열의 상관관계를 파악하기 위해 피어슨 상관계수를 계산합니다.<br/>
* 양의 상관관계 : 값이 1에 가까움, 두 변수가 같은 방향으로 변화함
* 음의 상관관계 : 값이 -1에 가까움, 두 변수가 반대 방향으로 변화함
* 상관관계가 없음 : 값이 0
```python
correlation_matrix = df.drop(columns=['Datetime']).corr()
print(correlation_matrix)
```

### 7. 히스토그램 시각화
각 열에 대한 변수 간 상관계수를 막대 그래프로 시각화합니다.
<img src="https://github.com/user-attachments/assets/e334479d-f43b-4729-8e22-c10aaefaaee8" height="400"/>

대기오염 현황을 일별 그래프로 출력하기 위해 Datetime을 기준으로 묶은 뒤, 정규화 과정을 거쳐 각 열의 평균값을 그래프로 나타냅니다.
<img src="https://github.com/user-attachments/assets/c34579b5-97c3-423b-be71-d6561af9a6df" height="400"/>

상관계수로 히트맵을 나타냅니다. 콘솔 창에 출력된 상관계수를 보는 것보다 시각화하여 색으로 표시하니 한 눈에 중요한 지점을 파악하기 쉬워집니다.<br/>
연한 색으로 갈수록 상관관계가 높은 것인데, PM10과 PM2.5의 상관관계가 가장 높으며, 다음으로는 CO와 NO2의 상관관계가 높은 것을 알 수 있습니다.
<img src="https://github.com/user-attachments/assets/77e3f293-40b5-4e52-af0c-6cf6dd451406" height="400"/>

가장 높은 양의 상관관계를 가지는 PM10과 PM2.5의 관계를 산점도 그래프로 시각화하여 확인합니다.
<img src="https://github.com/user-attachments/assets/f7366471-a834-4e0f-97dd-a4240555b891" height="400"/>

---
## 결론
데이터 분석을 통해 아황산가스(SO₂), 일산화탄소(CO), 오존(O₃), 이산화질소(NO₂), 미세먼지(PM10), 초미세먼지(PM2.5) 간의 상관관계를 파악할 수 있습니다.<br/>
상관계수 행렬을 통해 각 오염 물질 간의 상관성을 확인한 결과, 특히 다음과 같은 양의 상관관계가 두드러졌습니다:
* CO와 NO₂: 주로 화석연료의 연소 과정에서 함께 배출되는 물질입니다.
* PM10과 PM2.5: 두 입자상 물질은 발생 원인이 유사하며, 대기 중에서 함께 존재하는 경우가 많습니다.
