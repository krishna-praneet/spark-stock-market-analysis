from pyspark.sql import SparkSession
import json
import os
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

print(os.getcwd())

top_10 = ['GT', 'TXN', 'DIOD', 'MSEX', 'VLGEA', 'KLIC', 'OTTR', 'SGC', 'PHI', 'APOG']
rmse_metric = {}
auc_metric = {}

for stock in top_10:
    # Set the path to the JSON file in your machine
    json_file_path = os.getcwd() + '/../../dataset/' + stock + '-dataset.json/data.json'
    model_path = os.getcwd() + '/../../Model/random-forest-regression/' + stock
    plot_path = os.getcwd() + '/../../outputs/random-forest-regression/' + stock

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

    # Create a Random Forest Regressor object and fit it to the training data
    rf = RandomForestRegressor(numTrees=100, maxDepth=5, seed=42, labelCol=labelCol)
    model = rf.fit(trainingData)
    # # Use the model to make predictions on the testing data
    predictions = model.transform(testingData)

    # # Evaluate the performance of the model using Mean Squared Error
    evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    print(f"Root Mean Squared Error = {rmse}")

    model.write().overwrite().save(model_path)

    model = RandomForestRegressionModel.load(model_path)
    # # Use the model to make predictions on the testing data
    predictions = model.transform(testingData)

    # # Evaluate the performance of the model using Mean Squared Error
    evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    print(f"Root Mean Squared Error from saved model= {rmse}")

    evaluator = BinaryClassificationEvaluator(labelCol=labelCol, rawPredictionCol="prediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"AUC: {auc}")

    dates = [row['Date'] for row in predictions.select('Date').collect()]
    actual = [row['label'] for row in predictions.select('label').collect()]
    predicted = [row['prediction'] for row in predictions.select('prediction').collect()]

    plt.figure()
    plt.plot(dates, actual, label='actual')
    plt.plot(dates, predicted, label='predicted')
    plt.legend()
    plt.xlabel('Dates')
    plt.ylabel('Closing Price')
    plt.title(f'Actual vs Predicted Closing price for stock {stock}, RMSE: {rmse}')
    plt.savefig(plot_path)

    rmse_metric[stock] = rmse
    auc_metric[stock] = auc


print("Stock Name\tMSE")
for stock in top_10:
    print(f"{stock}\t{rmse_metric[stock]}")