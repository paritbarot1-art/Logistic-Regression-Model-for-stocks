import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. DEFINE THE STRATEGY PARAMETERS
ticker = 'TSLA'         #stockticker (e.g., Apple Inc.)
start_date = '2024-12-01'

# This will automatically set end_date to today's date whenever you run the script
end_date = (datetime.date.today()+ datetime.timedelta(days=1)).strftime('%Y-%m-%d')
print(f"Fetching data for {ticker} from {start_date} to {end_date}")
today = datetime.date.today()
day_of_week = today.weekday() # 0 is Monday, 6 is Sunday

if day_of_week >= 5:
    print("Market is closed. Using latest available data from Friday.")
else:
    print("Market is open or was open today.")

data = yf.download(ticker, start=start_date, end=end_date)
df = pd.DataFrame(data)
print(df.tail())
print(df.shape)

# Generate log return
data["returns"] = np.log((data['Close']/data['Close'].shift(1)))
#df1 = pd.DataFrame(data)
#print(df1.head())
#print(df1.shape)

# Create target value

data['target'] = np.where(data.returns.shift(-1) > 0, 1, 0)
data.dropna(inplace=True)
df2 = pd.DataFrame(data)
print(df2.tail())
print(df2.shape)

print(data['target'].value_counts())

# Creating Features
features_list = []

# SD based features
for i in range (5,20,5):
    col_name = 'std' + str(i)
    data[col_name] = data['Close'].rolling(window=i).std()
    features_list.append(col_name)

# MA based features
for i in range (10,30,5):
    col_name = 'ma' + str(i)
    data[col_name] = data['Close'].rolling(window=i).mean()
    features_list.append(col_name)

# Daily pct change based features
for i in range(3,12,3):
    col_name = 'ret_sum'+str(i)
    data[col_name] = data['returns'].rolling(i).sum()
    features_list.append(col_name)

# Features based on volume
col_name = 'vma_4'
data[col_name] = data['Volume'].rolling(4).mean()
features_list.append(col_name)

# Intraday movement
col_name = 'co'
data[col_name] = data['Close'] - data['Open']
features_list.append(col_name)

scaler = StandardScaler()
# Scale only the features we are using for the model
data[features_list] = scaler.fit_transform(data[features_list])

print(features_list)

# Remove nan values

data.dropna(inplace=True)

print(data[features_list+['target']].tail())

# Generate Training and Testing Datasets

x_train , x_test , y_train, y_test = train_test_split(data[features_list].iloc[:-1],data.iloc[:-1]['target'], test_size= 0.25 , shuffle=False)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
print (x_train.shape, y_train.shape, x_test.shape, y_test.shape)

print(x_train.columns)
#sns.pairplot(x_train[['std5','ma10','vma_4']])
#plt.show()

print(x_train.describe().T.round(2))

model = LogisticRegression(random_state=1)

# Train Model

model.fit(x_train,y_train)

# Predict using the train model

y_pred_train = model.predict(x_train)

print('Model accuracy on training data:',model.score(x_train,y_train))


y_pred_test = model.predict(x_test)

print("MOdel accuracy on testing data:",model.score(x_test,y_test))

# Isolate the data from Feb 10
latest_features = x_test.tail(1)

# Get the prediction for today (Feb 11)
today_signal = model.predict(latest_features)
today_prob = model.predict_proba(latest_features)[:, 1]

print(f"Prediction for {ticker} Today : {'UP' if today_signal[0] == 1 else 'DOWN'}")
print(f"Confidence: {today_prob[0]:.2%}")

# Printing the confusion matrix
print(confusion_matrix(y_test, y_pred_test))


# Printing the classification report
print(classification_report(y_test, y_pred_test))

# Create a DataFrame to see the weights clearly
importance = pd.DataFrame({'Feature': features_list, 'Weight': model.coef_[0]})

# Sort them to see the most influential features at the top
print(importance.sort_values(by='Weight', ascending=False))

# Plotting the Confustion matrix
cm = confusion_matrix(y_test, y_pred_test)
df = pd.DataFrame(cm, index=['Short', 'Long'], columns=['Short', 'Long'])
plt.figure(figsize=(5, 4))
sns.heatmap(df, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()









