```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

# Pclass가 1인 사람들 중에 생존한 사람들의 비율 계산
Pclass1 = train_data.loc[train_data.Pclass == 1]["Survived"]
rate_Pclass1 = sum(Pclass1) / len(Pclass1)

print("% of Pclass1 who survived:", rate_Pclass1)

# Pclass가 2인 사람들 중에 생존한 사람들의 비율 계산
Pclass2 = train_data.loc[train_data.Pclass == 2]["Survived"]
rate_Pclass2 = sum(Pclass2) / len(Pclass2)

print("% of Pclass2 who survived:", rate_Pclass2)
# Pclass가 3인 사람들 중에 생존한 사람들의 비율 계산
Pclass3 = train_data.loc[train_data.Pclass == 3]["Survived"]
rate_Pclass3 = sum(Pclass3) / len(Pclass3)

print("% of Pclass3 who survived:", rate_Pclass3)
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
import pandas as pd
import matplotlib.pyplot as plt

# 나이 결측치 처리 (중간값으로 채움)
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

# 연령대를 나타내는 새로운 열 생성
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=bins, labels=labels, right=False)

# 각 연령대별 생존율 계산
age_group_survival_rate = train_data.groupby('AgeGroup', observed=True)['Survived'].mean()

# 결과 출력
print(age_group_survival_rate)

# 연령대별 생존율 시각화
plt.figure(figsize=(10, 6))
age_group_survival_rate.plot(kind='bar')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# SibSp 값에 따른 생존율 계산
sibsp_survival_rate = train_data.groupby('SibSp')['Survived'].mean()

# 결과 출력
print(sibsp_survival_rate)

# SibSp 값에 따른 생존율 시각화
plt.figure(figsize=(10, 6))
sibsp_survival_rate.plot(kind='bar')
plt.title('Survival Rate by SibSp')
plt.xlabel('Number of Siblings/Spouses aboard')
plt.ylabel('Survival Rate')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Cabin의 알파벳 접두사 추출
train_data['CabinPrefix'] = train_data['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'Unknown')

# 각 접두사에 따른 생존율 계산
cabin_prefix_survival_rate = train_data.groupby('CabinPrefix')['Survived'].mean()

# 결과 출력
print(cabin_prefix_survival_rate)

# Cabin 접두사에 따른 생존율 시각화
plt.figure(figsize=(10, 6))
cabin_prefix_survival_rate.plot(kind='bar')
plt.title('Survival Rate by Cabin Prefix')
plt.xlabel('Cabin Prefix')
plt.ylabel('Survival Rate')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt


# Embarked 결측치 처리 (가장 빈번한 값으로 채움)
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Embarked 값에 따른 생존율 계산
embarked_survival_rate = train_data.groupby('Embarked')['Survived'].mean()

# 결과 출력
print(embarked_survival_rate)

# Embarked 값에 따른 생존율 시각화
plt.figure(figsize=(10, 6))
embarked_survival_rate.plot(kind='bar', color='skyblue')
plt.title('Survival Rate by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()
import pandas as pd
from sklearn.metrics import accuracy_score

# 데이터 로드
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

# Pclass 점수 부여
pclass_score = {1: 62, 2: 47, 3: 24}
train_data['PclassScore'] = train_data['Pclass'].map(pclass_score)

# Sex 점수 부여
sex_score = {'female': 148, 'male': 60}
train_data['SexScore'] = train_data['Sex'].map(sex_score)

# Age 점수 부여
def age_score(age):
    if age < 10:
        return 61
    elif age < 20:
        return 40
    elif age < 30:
        return 32
    elif age < 40:
        return 44
    elif age < 50:
        return 38
    elif age < 60:
        return 42
    elif age < 90:
        return 30
    else:
        return None

train_data['AgeScore'] = train_data['Age'].apply(age_score)

# SibSp 점수 부여
def sibsp_score(sibsp):
    if sibsp == 0:
        return 40
    elif sibsp == 1:
        return 60
    elif sibsp == 2:
        return 50
    elif sibsp == 3:
        return 30
    elif sibsp == 4:
        return 20
    else:
        return 0

train_data['SibSpScore'] = train_data['SibSp'].apply(sibsp_score)

# Parch 점수 부여
def parch_score(parch):
    if parch == 0:
        return 34
    elif parch == 1:
        return 55
    elif parch == 2:
        return 45


train_data['ParchScore'] = train_data['Parch'].apply(parch_score)

# Cabin 점수 부여
cabin_score = {'A': 47, 'B': 75, 'C': 59, 'D': 76, 'E': 75, 'F': 61}
train_data['CabinPrefix'] = train_data['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'Unknown')
train_data['CabinScore'] = train_data['CabinPrefix'].map(cabin_score)

# 각 승객의 총점과 평균 점수 계산
def calculate_average_score(row):
    scores = [
        row['PclassScore'],
        row['SexScore'],
        row['AgeScore'],
        row['SibSpScore'],
        row['ParchScore'],
        row['CabinScore']
    ]
    valid_scores = [score for score in scores if pd.notna(score)]
    if len(valid_scores) > 0:
        return sum(valid_scores) / len(valid_scores)
    else:
        return None

train_data['AverageScore'] = train_data.apply(calculate_average_score, axis=1)

# 점수가 50 이상이면 Survived = 1, 미만이면 Survived = 0
train_data['PredictedSurvived'] = train_data['AverageScore'].apply(lambda x: 1 if x >= 50 else 0)

# 정확도 계산
accuracy = accuracy_score(train_data['Survived'], train_data['PredictedSurvived'])

# 결과 출력
print(f'Accuracy: {accuracy:.2f}')
print(train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'AverageScore', 'PredictedSurvived', 'Survived']].head(30))
import pandas as pd
import matplotlib.pyplot as plt

# Parch 값에 따른 생존율 계산
parch_survival_rate = train_data.groupby('Parch')['Survived'].mean()

# 결과 출력
print(parch_survival_rate)

# Parch 값에 따른 생존율 시각화
plt.figure(figsize=(10, 6))
parch_survival_rate.plot(kind='bar', color='skyblue')
plt.title('Survival Rate by Parch')
plt.xlabel('Number of Parents/Children aboard')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# 데이터 불러오기
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

# 데이터 전처리 함수 정의
def preprocess_data(data, is_train=True):
    data = data.copy()
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    if 'Fare' in data.columns:
        data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    if 'Cabin' in data.columns:
        data = data.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'])
    else:
        data = data.drop(columns=['PassengerId'])
    label_encoders = {}
    for column in ['Sex', 'Embarked']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data

# 훈련 데이터 전처리
train_data = preprocess_data(train_data)

# 특성과 타겟 변수 분리
X_train = train_data.drop(columns=['Survived'])
y_train = train_data['Survived']

# 그래디언트 부스팅 모델 학습
gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb_model.fit(X_train, y_train)

# 테스트 데이터 전처리
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')  # 다시 불러오기
test_data_raw = test_data.copy()  # 원본 데이터를 저장
test_data = preprocess_data(test_data, is_train=False)

# 결측치 확인 및 처리
print(test_data.isnull().sum())

# 예측
test_data['Survived'] = gb_model.predict(test_data)

# 결과를 submission_gradient_boosting.csv 형식으로 저장
submission = pd.DataFrame({
    'PassengerId': test_data_raw['PassengerId'],
    'Survived': test_data['Survived']
})
submission.to_csv('submission_gradient_boosting.csv', index=False)

print("Prediction completed and saved to /kaggle/working/submission_gradient_boosting.csv")
```
