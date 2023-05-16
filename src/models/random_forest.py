from pyspark.sql import SparkSession
import json
import os
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer
from pyspark.sql import functions as F
from pyspark.sql.window import Window

print(os.getcwd())

top_10 = ['GT', 'TXN', 'DIOD', 'MSEX', 'VLGEA', 'KLIC', 'OTTR', 'SGC', 'PHI', 'APOG']
mse_metric = {}
auc_metric = {}

for stock in top_10:
    # Set the path to the JSON file in your machine
    json_file_path = os.getcwd() + '/../../dataset/' + stock + '-dataset.json/data.json'
    model_path = os.getcwd() + '/../../Model/random-forest/'+stock

    # Create a SparkSession
    spark = SparkSession.builder.appName("CreateDataFrameFromJSON").config("spark.executor.memory", "64g").config("spark.driver.memory", "64g").getOrCreate()

    # Read the JSON file into a DataFrame
    df = spark.read.json(json_file_path)

    # Select specific columns and drop any rows with missing values
    df = df.select('Date', 'Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume').dropna()

    df.show()

    df_sort_final = df.sample(fraction=0.05, seed=42)

    # Show the sampled DataFrame
    # df_sample.show()
    w = Window.partitionBy().orderBy("Date")
    df_sort_final = df_sort_final.withColumn('diffOpenClose', df_sort_final.Open - df_sort_final.Close)
    df_sort_final = df_sort_final.withColumn('diffHighLow', df_sort_final.High - df_sort_final.Low)
    df_sort_final = df_sort_final.withColumn('target', F.when(F.lag(df.Close).over(w) < df.Close, 1).otherwise(0))
    df_sort_final = df_sort_final.withColumn("year", F.year(df_sort_final["date"]))
    df_sort_final = df_sort_final.withColumn("month", F.month(df_sort_final["date"]))
    df_sort_final = df_sort_final.withColumn("day", F.dayofmonth(df_sort_final["date"]))
    df_sort_final.show(20)
    print(df_sort_final.count())

    labelCol = "target"

    # # Assemble the feature vector using VectorAssembler
    assembler = VectorAssembler(inputCols=['Open', 'High', 'Low', 'Volume','diffOpenClose','diffHighLow','year','month','day'], outputCol="features")
    df_sort_final2 = assembler.transform(df_sort_final).select(labelCol, "features")

    # Split the data into training and testing sets
    (trainingData, testingData) = df_sort_final2.randomSplit([0.8, 0.2], seed=42)

    # Create a Random Forest Regressor object and fit it to the training data
    rf = RandomForestRegressor(numTrees=10, maxDepth=5, seed=42, labelCol=labelCol)
    model = rf.fit(trainingData)
    # # Use the model to make predictions on the testing data
    predictions = model.transform(testingData)

    # # Evaluate the performance of the model using Mean Squared Error
    evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="mse")
    mse = evaluator.evaluate(predictions)

    print(f"Mean Squared Error = {mse}")

    model.write().overwrite().save(model_path)

    model = RandomForestRegressionModel.load(model_path)
    # # Use the model to make predictions on the testing data
    predictions = model.transform(testingData)

    # # Evaluate the performance of the model using Mean Squared Error
    evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="mse")
    mse = evaluator.evaluate(predictions)

    print(f"Mean Squared Error from saved model= {mse}")

    evaluator = BinaryClassificationEvaluator(labelCol=labelCol, rawPredictionCol="prediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"AUC: {auc}")

    mse_metric[stock] = mse
    auc_metric[stock] = auc


print("Stock Name\tMSE\tAUC\n")
for stock in top_10:
    print(f"{stock}\t{mse_metric[stock]}\t{auc_metric[stock]}")