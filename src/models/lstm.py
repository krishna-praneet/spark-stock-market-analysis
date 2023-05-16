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

print(os.getcwd())

top_10 = ['GT', 'TXN', 'DIOD', 'MSEX', 'VLGEA', 'KLIC', 'OTTR', 'SGC', 'PHI', 'APOG']
mse_metric = {}

for stock in top_10:
    # Set the path to the JSON file in your machine
    json_file_path = os.getcwd() + '/../../dataset/' + stock + '-dataset.json/data.json'
    model_path = os.getcwd() + '/../../Model/lstm/'+stock

    # Create a SparkSession
    spark = SparkSession.builder.appName("CreateDataFrameFromJSON").config("spark.executor.memory", "64g").config("spark.driver.memory", "64g").getOrCreate()

    # Read the JSON file into a DataFrame
    df = spark.read.json(json_file_path)

    # Select specific columns and drop any rows with missing values
    df = df.select('Date', 'Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume', 'Stock').dropna()

    df.show()

    df_sort_final = df.sample(fraction=1.00, seed=123)

    # Show the sampled DataFrame
    # df_sample.show()
    w = Window.partitionBy('Stock').orderBy("Date")
    df_sort_final = df_sort_final.filter(df_sort_final['Stock']=='GT')
    df_sort_final = df_sort_final.withColumn('diffOpenClose', df_sort_final.Open - df_sort_final.Close)
    df_sort_final = df_sort_final.withColumn('diffHighLow', df_sort_final.High - df_sort_final.Low)
    df_sort_final = df_sort_final.withColumn("year", year(df_sort_final["date"]))
    df_sort_final = df_sort_final.withColumn("month", month(df_sort_final["date"]))
    df_sort_final = df_sort_final.withColumn("day", dayofmonth(df_sort_final["date"]))
    df_sort_final = df_sort_final.withColumn('target', F.when(F.lag(df.Close).over(w) < df.Close, 1).otherwise(0))
    df_sort_final.show(20)
    print(df_sort_final.count())

    labelCol = "target"

    labelCol = "target"

    assembler = VectorAssembler(inputCols=['Open', 'High', 'Low', 'Volume','diffOpenClose','diffHighLow','year','month','day'], outputCol="features")
    df_sort_final2 = assembler.transform(df_sort_final).select(labelCol, "features")

    # Split the data into training and testing sets
    (trainingData, testingData) = df_sort_final2.randomSplit([0.7, 0.3], seed=42)

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
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            out = self.fc(out[:, -1, :])

            return out

    input_cols=['Open', 'High', 'Low', 'Volume','diffOpenClose','diffHighLow','year','month','day']

    # Define the hyperparameters for the LSTM model
    input_size = len(input_cols)
    hidden_size = 16
    output_size = 1
    num_layers = 1
    batch_size = 2
    seq_len = 2
    learning_rate = 0.0005
    num_epochs =50

    # Instantiate the model
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers= num_layers, output_size=output_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Create a PyTorch DataLoader for the training data
    dataset = TensorDataset(train_features, train_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {'train_loss': [], 'train_acc': []}

    # Train the model
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = lstm(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            
            train_loss += loss.item()
            train_acc += ((outputs > 0.5) == (targets > 0.5)).sum().item() / targets.size(0)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print loss every 10 iterations
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
        
        print("Epoch %d: train loss: %.4f" % (epoch, loss.item()))
        
        train_loss /= len(dataloader.dataset)
        train_acc /= len(dataloader.dataset)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

    # Get the training and validation loss and accuracy from the training history
    train_loss = history['train_loss']
    train_acc = history['train_acc']

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

    # Plot the training accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.show()

    # Save the model
    torch.save(lstm.state_dict(), model_path)

    lstm = LSTM(input_size, hidden_size, output_size, num_layers)

    # Create a PyTorch DataLoader for the test data
    test_dataset = TensorDataset(test_features, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    lstm = LSTM(input_size, hidden_size, output_size, num_layers)

    # Load the trained model parameters into the new LSTM object
    lstm.load_state_dict(torch.load(model_path))

    # Test the LSTM model
    lstm.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            # Forward pass
            outputs = lstm(inputs)
            test_loss += criterion(outputs, targets.view(-1, 1)).item()
        
        # Calculate the mean squared error (MSE) loss
        test_loss /= len(test_dataloader.dataset)
        print('Test MSE Loss: {:.4f}'.format(test_loss))
    
    mse_metric[stock] = test_loss