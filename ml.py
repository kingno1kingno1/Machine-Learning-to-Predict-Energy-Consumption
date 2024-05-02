# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Importing Data
URL = 'events.csv'
df = pd.read_csv(URL)
# ## Data Exploration
print("Return first 5 rows.","\n")
df.head()
print("Return last 5 rows.","\n")
df.tail()
df.info()
print(df.describe(), "\n")
# ## Feature Extraction
del df["Start time UTC"]
del df["End time UTC"]
del df["Start time UTC+03:00"]
df.rename(columns={"End time UTC+03:00":"DateTime","Electricity consumption in Finland":"Consumption"},inplace=True)
print(df.head(5))
# %%
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

# %%
# By assuming week starts on Mondey and ends on Sunday.
# The closest start would be on Monday 4-1-2016 
# The closest end would be on sunday 26-12-2021
# So we should omit first 71 rows and last 121 rows.
dataset = dataset[71:-121]
dataset.tail()

# %% 
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


# %%
plt.figure(figsize=(15,10))
plt.plot(dataset["Consumption"])

# %%
# Energy Consumption Each Year

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

# %%
# Lets us see the Distribution off Energy Consumption so we have a idea about your Dataset a bit more
fig = plt.figure(figsize = (15,10))
sns.distplot(dataset["Consumption"])
plt.title("Energy Distribution")

# %%
fig = plt.figure(figsize = (15,10))
sns.boxplot(x=dataset["Month"], y=dataset["Consumption"], data= df)
plt.title("Energy Consumption VS Month")
plt.xlabel("Month")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)

# %%
dataset1 = dataset
fig = plt.figure(figsize = (15,10))
sns.boxplot(x=dataset1.index.hour, y=dataset1.Consumption, data= df)
plt.title("Energy Consumption VS Hour")
plt.xlabel("Hour")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)

# %%
fig = plt.figure(figsize = (15,10))
sns.boxplot(x=dataset1.index.year, y=dataset1.Consumption, data= df)
plt.title("Energy Consumption VS Year")
plt.xlabel("Year")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)
# %%
# # LSTM Model
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

# %%
# We have 2193 row
# 2193 - 3 - 6 = 2184 row after omit first two rows and last six ones.  
# 2184 / 7 = 312 week  
# 312 * 80 %  250 week for train (1750 day)  
# 312 - 250 = 62 week for test (434 day)
print("Old Dataset: ", dataset.shape)
print("New Dataset: ", newDataSet.shape)

# %%
# Saving data in CSV new file
# newDataSet.to_csv("newDataSet.csv")
# from google.colab import files
# files.download("newDataSet.csv")

# %%
newDataSet.head()

# %%
y = newDataSet["Consumption"]
print(y[0])
y.shape

# %%
# Normalize data before model fitting
# it will boost the performance( in neural networks) + transform
from sklearn.preprocessing import MinMaxScaler
# scale of the output and input inthe range 0-1 to match the scale of the layer of LSTM
scaler = MinMaxScaler(feature_range = (0,1)) 
# reshape: convert the univariate 1D array into 2D
y = scaler.fit_transform(np.array(y).reshape(-1,1))
print("Normalizing data before model fitting")
print(y[:10])


# %%
training_size = int(len(y)*0.80)
test_size = len(y)- training_size
val_size = int(training_size*0.20)
train_data , test_data , val_data = y[0:training_size-val_size,:] , y[training_size:len(y),:1], y[len(y)-test_size-val_size:len(y)-test_size,:1]


# %%
# building input variable
def create_dataset(dataset, time_step = 1):
  dataX, dataY = [] , []
  for i in range(len(dataset)-time_step-1):
    a = dataset[i:(i+time_step),0]
    dataX.append(a)
    dataY.append(dataset[i + time_step,0])
  return np.array(dataX), np.array(dataY)


# %%
time_step = 100 
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
X_val, yval = create_dataset(val_data, time_step)


# %%
# reshape train and input-output pairs
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1],1)

print("X_train shape: ", X_train.shape)
print("X_test shape: ",X_test.shape)
print("X_val shape: ",X_val.shape)


# %% [markdown]
# ## Model Structure
# 

# %%
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Defining the LSTM layer(s)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Defining the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # Forward pass through LSTM layer
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Only take the output from the final time step
        out = self.linear(out[:, -1, :])
        return out


## 將數據轉換為 PyTorch Tensors
train_data_tensor = torch.Tensor(X_train)
train_target_tensor = torch.Tensor(y_train)

test_data_tensor = torch.Tensor(X_test)
test_target_tensor = torch.Tensor(ytest)

val_data_tensor = torch.Tensor(X_val)
val_target_tensor = torch.Tensor(yval)

model = LSTMModel(input_dim=1, hidden_dim=50, num_layers=4, output_dim=1)

# 損失函數和優化器
criterion = torch.nn.MSELoss()    # 均方誤差損失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 使用 Adam 優化器

from torch.utils.data import DataLoader, TensorDataset

# 假設這些是你的數據
train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
val_dataset = TensorDataset(val_data_tensor, val_target_tensor)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

num_epochs = 60
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    # Average losses
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')
plt.figure(figsize=(10,10))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()
# %%
# 預測
model.eval()
with torch.no_grad():
    train_preds = model(train_data_tensor).detach().numpy()
    val_preds = model(val_data_tensor).detach().numpy()

# 反標準化預測
train_preds = scaler.inverse_transform(train_preds)
val_preds = scaler.inverse_transform(val_preds)
plt.figure(figsize=(20, 10))
plt.plot(train_target_tensor.numpy(), label='Actual Train Data')
plt.plot(train_preds, label='Predicted Train Data')
plt.title('Training Data and Predictions')
plt.xlabel('Time Step')
plt.ylabel('Consumption')
plt.legend()
plt.show()
# torch.save(model.state_dict(), 'LSTM_model.pth')

# %%
