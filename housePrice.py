import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import joblib


df = pd.read_csv("zameen-updated.csv")

df['area_marla'] = df['Area Size']
df.loc[df['Area Type'] == 'kanal' , 'area_marla'] *= 20 

df = df.drop(columns=['property_id',
                       'location_id',
                        'page_url' ,
                        'agency',
                        'agent',
                        'date_added',
                        'Area Category',
                        'province_name',
                        'area',
                        'Area Size',
                        'Area Type',
                        'latitude',
                        'longitude'])

df = df[df['purpose']== 'For Sale']
df =df.drop(columns='purpose')

df['city'].unique()
city_location_map = df.groupby('city')['location'].unique().to_dict()
print(city_location_map)
# df = pd.get_dummies(df , columns=['property_type'], drop_first=True)
# df = pd.get_dummies(df , columns=['city'], drop_first=True)
# df = df.astype({col: 'int' for col in df.columns if df[col].dtype == 'bool'})


# # print(df.shape)
# # print(df.isna().sum())

# #  linear model  ...............
# # model = LinearRegression()

# #  random forest ...........
# # model = RandomForestRegressor(n_estimators=200, max_depth=10 ,min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)

# other model
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

cat_features = ['location', 'city', 'property_type']

y = np.log1p(df['price'])
x= df.drop(columns='price')


print(df.info())

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42 )

model.fit(x_train , y_train ,cat_features=cat_features)

#  saving my model with the help of joblib........................
joblib.dump(model, "houe_price_predict.pkl")
joblib.dump(x.columns.tolist(), "column.pkl")

importance= model.feature_importances_
feature =x.columns

importantFeature = pd.Series(importance , index=feature).sort_values(ascending=False)
# print(importantFeature)

y_trainPred = model.predict(x_train)
y_testPred = model.predict(x_test)


mseTrain = mean_squared_error(y_train , y_trainPred)
print("Training MSE: ", mseTrain)

mseTest = mean_squared_error(y_test , y_testPred)
print("Test MSE: ", mseTest)

rmseTrain = np.sqrt(mseTrain)
rmseTest = np.sqrt(mseTest)

print("Training rmse: ", rmseTrain)
print("Testing rmse: ", rmseTest)

r2_train = r2_score(y_train, y_trainPred)
r2_test = r2_score(y_test, y_testPred)

print("Train R2:", r2_train)
print("Test R2:", r2_test)


