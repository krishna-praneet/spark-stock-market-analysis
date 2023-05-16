from pyspark.sql import SparkSession
import json
import os
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, DataType
from pyspark.ml.feature import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import math

print(os.getcwd())

top_10 = ['GT', 'TXN', 'DIOD', 'MSEX', 'VLGEA', 'KLIC', 'OTTR', 'SGC', 'PHI', 'APOG']
# top_10 = ['GT']
mse_metric = {}
rmse_metric = []

for stock in top_10:
    # Set the path to the JSON file in your machine
    json_file_path = os.getcwd() + '/../../dataset/' + stock + '-dataset.json/data.json'
    model_path = os.getcwd() + '/../../Model/lstm-regression/'+stock
    plot_path = os.getcwd() + '/../../outputs/lstm-regression/' + stock
    loss_plot_path = os.getcwd() + '/../../outputs/lstm-regression/loss/' + stock

   # Create a SparkSession
    spark = SparkSession.builder.appName("CreateDataFrameFromJSON").config("spark.executor.memory", "64g").config("spark.driver.memory", "64g").getOrCreate()

    # Read the JSON file into a DataFrame
    df = spark.read.json(json_file_path)

    # Select specific columns and drop any rows with missing values
    df = df.select('Date', 'Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume').dropna()

    df.show()

    df_sort_final = df.sample(fraction=1.00, seed=42)
    df_sort_final.orderBy('Date').collect()
    

    # df_sample.show()
    w = Window.partitionBy().orderBy("Date")
    # df_sort_final = df.withColumn('diffOpenClose', df_sort_final.Open - df_sort_final.Close)
    # df_sort_final = df_sort_final.withColumn('diffHighLow', df_sort_final.High - df_sort_final.Low)
    # df_sort_final = df_sort_final.withColumn('target', when(lag(df.Close).over(w) < df.Close, 1).otherwise(0))
    # Extract date features
    days = lambda i: i * 86400
    w1 = (Window.orderBy(col("Date").cast('long')).rangeBetween(-days(1), 0))
    w14 = (Window.orderBy(col("Date").cast('long')).rangeBetween(-days(14), -days(1)))
    w141 = (Window.orderBy(col("Date").cast('long')).rangeBetween(-days(14), 0))

    df_sort_final =df_sort_final.withColumn('Date', col('Date').cast('timestamp'))
    df_sort_final = df_sort_final.withColumn("Year", year(df_sort_final["date"]))
    df_sort_final = df_sort_final.withColumn("Month", month(df_sort_final["date"]))
    df_sort_final = df_sort_final.withColumn("Day", dayofmonth(df_sort_final["date"]))

    df_sort_final = df_sort_final.withColumn("SMA", avg("Close").over(w141))
    df_sort_final = df_sort_final.withColumn("TP", (col("High") + col("Low") + col("Close"))/ 3)
    df_sort_final = df_sort_final.withColumn("14DTP", avg(col("TP")).over(w141))
    df_sort_final = df_sort_final.withColumn("14MD", avg(abs(col("TP") - col("14DTP"))).over(w141))
    df_sort_final = df_sort_final.withColumn("label", col('Close'))
    df_sort_final = df_sort_final.na.drop(subset=['SMA', 'TP'])
    avgVolume = df_sort_final.agg({'Volume': 'avg'}).collect()[0][0]
    df_sort_final = df_sort_final.withColumn("Volume", col('Volume')/avgVolume)


    df_sort_final.show(20)
    numRows = df_sort_final.count()
    print("Total number of rows: ", numRows)

    inputCols = ['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year', 'SMA', 'TP', '14DTP', '14MD']

    # Assemble the feature vector using VectorAssembler
    assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
    data = assembler.transform(df_sort_final).select("features", "label", "Date")
    data.show()

    labelCol = "label"

    # Split the data into training and testing sets
    train = 0.7
    trainRows = int((0.7 * numRows) // 1)
    trainingData = data.limit(trainRows)
    testingData = data.subtract(trainingData)
    print('Done splitting data')


    # Convert the PySpark DataFrame to PyTorch tensors
    train_features = torch.tensor(np.array(trainingData.select('features').collect())).float()
    train_targets = torch.tensor(np.array(trainingData.select(labelCol).collect())).float()
    test_features = torch.tensor(np.array(testingData.select('features').collect())).float()
    test_targets = torch.tensor(np.array(testingData.select(labelCol).collect())).float()

    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

            out, (hn, cn) = self.lstm(x, (h0, c0))
            out, (hn, _) = self.lstm2(out, (h0, c0))

            out = self.fc(out[:, -1, :])

            return out


    # Define the hyperparameters for the LSTM model
    input_size = len(inputCols)
    hidden_size = 64
    output_size = 1
    num_layers = 1
    batch_size = 64
    seq_len = 2
    learning_rate = 0.001
    num_epochs = 200

    # Instantiate the model
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers= num_layers, output_size=output_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Create a PyTorch DataLoader for the training data
    dataset = TensorDataset(train_features, train_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {'train_loss': [], 'rmse_loss': []}

    # Train the model
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = lstm(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            
            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print loss every 10 iterations
            if (i+1) % 10 == 0:
                print('Stock {}, Epoch [{}/{}], Step [{}/{}], MSE Loss: {:.4f}'.format(stock, epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
        
        print("Epoch %d: train MSE loss: %.4f" % (epoch, loss.item()))
        
        train_loss /= len(dataloader.dataset)
        history['train_loss'].append(train_loss)
        history['rmse_loss'].append(math.sqrt(train_loss))

    # Get the training and validation loss and accuracy from the training history
    train_loss = history['train_loss']
    rmse_loss = history['rmse_loss']

    # Plot the training and validation loss
    plt.figure()
    plt.plot(rmse_loss, label='Training RMSE Loss')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE Loss')
    plt.title(f'RMSE Training Loss v/s epochs for stock {stock}')
    plt.legend()
    plt.savefig(loss_plot_path)


    # Create a PyTorch DataLoader for the test data
    test_dataset = TensorDataset(test_features, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Test the LSTM model
    test_loss = 0.0

    actual = []
    predictions = []
    lstm.eval()
    
    for i, (inputs, targets) in enumerate(test_dataloader):
        with torch.no_grad():
            # Forward pass
            outputs = lstm(inputs)
            test_loss += criterion(outputs, targets.view(-1, 1)).item()
            
            predictions.append(outputs.squeeze())
            actual.append(targets.squeeze())

        
    # Calculate the mean squared error (MSE) loss
    test_loss /= len(test_dataloader.dataset)
    print('Test MSE Loss: {}'.format(test_loss))
    
    actual = torch.cat(actual).numpy()
    predictions = torch.cat(predictions).numpy()
    mse_metric[stock] = test_loss
    rmse = math.sqrt(test_loss)
    rmse_metric.append(rmse)
    
    # Plot the training and validation loss
    dates = np.array(testingData.select('Date').collect())
    plt.figure()
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predictions, label='Predicted')
    plt.xlabel('Dates')
    plt.ylabel('Closing Price')
    plt.title(f'Actual vs Predicted Closing price for stock {stock}, RMSE: {rmse}')
    plt.legend()
    plt.savefig(plot_path)
    
    # Save the model
    torch.save(lstm.state_dict(), model_path)



    
print("Stock Name\tRMSE")
for i in range(len(top_10)):
    print(f'{top_10[i]}\t{rmse_metric[i]}')