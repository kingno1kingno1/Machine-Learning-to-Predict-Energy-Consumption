# %%
# # Data
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Importing Data
URL = 'events.csv'
df = pd.read_csv(URL)
# ## Data Exploration
print("Return first 5 rows.","\n")
df.head()
print("Return last 5 rows.","\n")
df.tail()
df.info()
#print("Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.", "\n")
print(df.describe(), "\n")
# ## Feature Extraction
del df["Start time UTC"]
del df["End time UTC"]
del df["Start time UTC+03:00"]
df.rename(columns={"End time UTC+03:00":"DateTime","Electricity consumption in Finland":"Consumption"},inplace=True)
print(df.head(5))
# since we are dealing with time series data we should edite the index from 1 2 3 ... --> DateTime format.
dataset = df
dataset["Month"] = pd.to_datetime(df["DateTime"]).dt.month
dataset["Year"] = pd.to_datetime(df["DateTime"]).dt.year
dataset["Date"] = pd.to_datetime(df["DateTime"]).dt.date
dataset["Time"] = pd.to_datetime(df["DateTime"]).dt.time
dataset["Week"] = pd.to_datetime(df["DateTime"]).dt.isocalendar().week
dataset["Day"] = pd.to_datetime(df["DateTime"]).dt.day_name()
dataset = df.set_index("DateTime")
dataset.index = pd.to_datetime(dataset.index)
dataset.head()
print("")
print("Total Number of Years: ", dataset.Year.nunique() )
print(dataset.Year.unique())
# By assuming week starts on Mondey and ends on Sunday.
# The closest start would be on Monday 4-1-2016 
# The closest end would be on sunday 26-12-2021
# So we should omit first 71 rows and last 121 rows.
dataset = dataset[71:-121]
dataset.tail()
# ## Data Visualizations
from matplotlib import style
fig = plt.figure()
axes1 = plt.subplot2grid((1,1), (0,0))

style.use("ggplot")
sns.lineplot(x= dataset["Year"], y= dataset["Consumption"], data = dataset)
sns.set(rc={'figure.figsize': (20,10)})
plt.title("Electricity consumption in Finland 2016-2021")
plt.xlabel("Date")
plt.ylabel("Energy in MW")
plt.grid(True)
plt.legend()

for label in axes1.xaxis.get_ticklabels():
    label.set_rotation(90)


 
plt.figure(figsize=(15,10))
plt.plot(dataset["Consumption"])

 
# Energy Consumption Each Year
from matplotlib import style

fig = plt.figure(figsize = (30,30))

ax1 = fig.add_subplot(611)
ax2 = fig.add_subplot(612)
ax3 = fig.add_subplot(613)
ax4 = fig.add_subplot(614)
ax5 = fig.add_subplot(615)
ax6 = fig.add_subplot(616)

style.use("ggplot")

y_2016 = dataset.loc["2016"]["Consumption"].to_list()
x_2016 = dataset.loc["2016"]["Date"].to_list()
ax1.plot(x_2016, y_2016, color= "blue", linewidth= 1.7)

y_2017 = dataset.loc["2017"]["Consumption"].to_list()
x_2017 = dataset.loc["2017"]["Date"].to_list()
ax2.plot(x_2017, y_2017, color= "blue", linewidth= 1.7)

y_2018 = dataset.loc["2018"]["Consumption"].to_list()
x_2018 = dataset.loc["2018"]["Date"].to_list()
ax3.plot(x_2018, y_2018, color= "blue", linewidth= 1.7)

y_2019 = dataset.loc["2019"]["Consumption"].to_list()
x_2019 = dataset.loc["2019"]["Date"].to_list()
ax4.plot(x_2019, y_2019, color= "blue", linewidth= 1.7)

y_2020 = dataset.loc["2020"]["Consumption"].to_list()
x_2020 = dataset.loc["2020"]["Date"].to_list()
ax5.plot(x_2020, y_2020, color= "blue", linewidth= 1.7)

y_2021 = dataset.loc["2021"]["Consumption"].to_list()
x_2021 = dataset.loc["2021"]["Date"].to_list()
ax6.plot(x_2021, y_2021, color= "blue", linewidth= 1.7)

plt.rcParams["figure.figsize"] = (30, 15)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=2.5, wspace=0.4, hspace=0.4)
plt.title("Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Energy in MW")
plt.grid(True, alpha=1)
plt.legend()

 
# Lets us see the Distribution off Energy Consumption so we have a idea about your Dataset a bit more
fig = plt.figure(figsize = (15,10))
sns.distplot(dataset["Consumption"])
plt.title("Energy Distribution")

 
fig = plt.figure(figsize = (15,10))
sns.boxplot(x=dataset["Month"], y=dataset["Consumption"], data= df)
plt.title("Energy Consumption VS Month")
plt.xlabel("Month")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)

 
dataset1 = dataset
fig = plt.figure(figsize = (15,10))
sns.boxplot(x=dataset1.index.hour, y=dataset1.Consumption, data= df)
plt.title("Energy Consumption VS Hour")
plt.xlabel("Hour")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)

 
fig = plt.figure(figsize = (15,10))
sns.boxplot(x=dataset1.index.year, y=dataset1.Consumption, data= df)
plt.title("Energy Consumption VS Year")
plt.xlabel("Year")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)

# %% [markdown]===============================================================
# ## Train, Validation and Test Dataset
# Downsampling involves decreasing the time-frequency of the data
# Downsapling the time-frequency from hours to days 
newDataSet = dataset.resample('D').agg({
    'Consumption': 'mean',   # 對數值欄位求平均
    'Month': 'first',        # 對其他欄位取第一個值
    'Year': 'first',
    'Date': 'first',
    'Time': 'first',
    'Week': 'first',
    'Day': 'first'
})

 
# We have 2193 row
# 2193 - 3 - 6 = 2184 row after omit first two rows and last six ones.  
# 2184 / 7 = 312 week  
# 312 * 80 %  250 week for train (1750 day)  
# 312 - 250 = 62 week for test (434 day)
print("Old Dataset: ", dataset.shape)
print("New Dataset: ", newDataSet.shape)

 
# Saving data in CSV new file
# newDataSet.to_csv("newDataSet.csv")
# from google.colab import files
# files.download("newDataSet.csv")

 
newDataSet.head()

 
y = newDataSet["Consumption"]
print(y[0])
y.shape

# %%
# Normalize data before model fitting
# it will boost the performance( in neural networks) + transform
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error

# Assuming 'y' is your series data you want to model
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Split data into training, validation, and test sets
training_size = int(len(y_scaled) * 0.80)
val_size = int(training_size * 0.20)
test_size = len(y_scaled) - training_size
train_data, val_data, test_data = y_scaled[:training_size-val_size], y_scaled[training_size-val_size:training_size], y_scaled[training_size:]

# Create datasets
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Convert to PyTorch tensors
X_train, y_train = torch.Tensor(X_train).unsqueeze(2), torch.Tensor(y_train)
X_val, y_val = torch.Tensor(X_val).unsqueeze(2), torch.Tensor(y_val)
X_test, y_test = torch.Tensor(X_test).unsqueeze(2), torch.Tensor(y_test)

# Data loaders
batch_size = 20
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# Instantiate the model, define loss function and optimizer
model = LSTMModel(input_dim=1, hidden_dim=50, layer_dim=4, output_dim=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=60):
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets.unsqueeze(1))
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return history

# Assuming the model, train_loader, val_loader, optimizer, and loss function are already defined
history = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=60)

# Plotting the training and validation loss
plt.figure(figsize=(10, 10))
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'], label='validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()




# Assuming model, train_loader, val_loader, and test_loader are already defined and the model is trained

# Function to make predictions
def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
    predictions = np.vstack(predictions)
    return predictions

# Predicting
train_predict = predict(model, train_loader)
val_predict = predict(model, val_loader)
test_predict = predict(model, test_loader)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
val_predict = scaler.inverse_transform(val_predict)
test_predict = scaler.inverse_transform(test_predict)

# Compute RMSE
train_rmse = np.sqrt(mean_squared_error(y_train.numpy(), train_predict))
print(f"Train RMSE: {train_rmse}")

# Reshape for consistency
y_train_inv = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))

# Plotting model loss over epochs (assuming history is available)
plt.figure(figsize=(10, 10))
plt.plot(history['train_loss'])  # Update with actual history logging
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting actual vs predictions for train set
plt.figure(figsize=(20, 10))
plt.plot(y_train_inv)
plt.plot(train_predict)
plt.legend(['Actual', 'Train Predictions'])
plt.xlabel('Time Steps')
plt.ylabel('Consumption MWh')
plt.show()

# Prediction and plotting for future values (the next 30 days)
def forecast(model, initial_input, steps=30):
    model.eval()
    input_seq = initial_input
    predictions = []

    for _ in range(steps):
        input_tensor = torch.Tensor(input_seq[-100:]).unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            pred = model(input_tensor)
        pred_value = pred.numpy()[0, 0]
        predictions.append(pred_value)
        input_seq.append(pred_value)
    
    return predictions

# Assuming test_data is the input for forecasting
last_100 = test_data[-100:].reshape(1, -1).tolist()[0]
forecasted = forecast(model, last_100, 30)
forecasted_inv = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))

# Plotting the forecasted values
plt.figure(figsize=(15, 10))
days = np.arange(1, 101)
future_days = np.arange(101, 131)
plt.plot(days, scaler.inverse_transform(y[-100:].values.reshape(-1,1)))
plt.plot(future_days, forecasted_inv)
plt.legend(['Historical', 'Forecasted'])
plt.show()
