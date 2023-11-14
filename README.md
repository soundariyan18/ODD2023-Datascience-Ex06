# EX-06-FEATURE TRANSFORMATION

## AIM

To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM

STEP 1: Read the given Data

STEP 2: Clean the Data Set using Data Cleaning Process

STEP 3: Apply Feature Transformation techniques to all the features of the data set

STEP 4: Save the data to the file




## PROGRAM:

```
Developed by : M.N.SOUNDARIYAN

REG : 212222230146

```
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df

df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Positive Skew"])

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()

df['Highly Negative Skew_1']=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```

## OUTPUT:

```
```
![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/26efb87b-7cc0-46f4-960c-f37269841add)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/fa2c987c-f929-448a-9819-20686679d629)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/01c2f997-6a49-4188-bd4d-59d121cbf910)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/af75ed4a-4dc0-4865-80ad-98fe4ac6f43c)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/722f9e2b-eb50-4ba5-85cb-96f64290f24e)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/4586d3d8-2333-4cb6-8645-8b7498167e44)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/617a41e5-d768-4080-9bb9-962302331b94)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/0d3c68e8-5a33-4ad3-9ea8-0b2c405980ca)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/1f19fde8-ca41-43e9-a117-a3aa9770d32c)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/a88dcd76-2cb3-4012-83ac-1a41d89ab3ad)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/c6d3bad8-ca05-4ff4-a78a-668bcbcbf32c)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex06/assets/119393307/6ce7db32-fcd2-418a-9fa6-716ecfc5b002)
```

## RESULT:
Thus,Feature transformation is performed and executed successfully for the given dataset.
