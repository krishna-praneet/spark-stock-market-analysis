# spark-stock-market-analysis
Stock Market Prediction using Python

# Collaborators
1. Krishna Praneet Gudipaty
2. Archana Ganesh 
3. Shubham Patel 

# Division of work
1. Shubham Patel - Data collection and preprocessing, Random Forest Classification
2. Krishna Praneet Gudipaty - Feature Generation, FM and RF Regression, GPU  
3. Archana Ganesh - LSTM Classification and  Regression, Dashboard

Pair programming in all stages of the project

# Dataset used: 
Nasdaq dataset from Kaggle

# Goal: 
The goal is to build a stock recommendation system using Python to predict the closing price and suggest stocks.

# Machine Learning models used: 
The following models are to be used and compared to finalize the best performing one for the prediction model.

1. Random forest regressor
2. LSTM
3. Factorization Machines regressor

# Software tech stack:
1. Python: <br>
   a. SparkSQL <br>
   b. SparkMLLib <br>
   c. Pandas <br>
   d. Numpy <br>
   e. Matplotlib <br>
   f. Torch <br>
2. DynamoDB

# Instructions followed for DynamoDb.
Here's a step-by-step guide on how to create a DynamoDB table, load data into it from a JSON file using the AWS CLI and boto3(library), and access the data from Python using the boto3 library:
1. First, make sure you have an AWS account and have installed the AWS CLI and boto3 library on your machine.
2. Next, create a new DynamoDB table. You can do this through the AWS Management Console or using the AWS CLI. Here's an example command to create a table with the name "myTable", a partition key  and a sort key:<br>
   a. aws dynamodb create-table --table-name myTable
3. Once the table is created, you can load data into it from a JSON file using the AWS CLI or boto3(Replace ACCESS_KEY and SECRET_KEY with your actual access and secret keys for DynamoDB.). Make sure you have a JSON file containing an array of JSON objects. Each object represents a single item to be added to the table.
4. You can then access the data from Python using the boto3 library.

# Instructions to run the code:
1. Install the necessary packages listed in the requirements.txt file using the following command:

   `pip install -r requirements.txt`

2. Download dataset and add it to path `stock_market_data/` relative to the `data_preprocess.py` file.

3. To generate the data set in the suitable format, run in src/processing after editing the path the data needs to be stored, in the file:

   `python3 dataset_generator.py`

4. To run the LSTM regressor, check the path of the dataset and modify the path to store the model and run in src/models:

   `python3 lstm_regression.py`

5. To run the FM regressor, check the path of the dataset and modify the path to store the model and run in src/models:

   `python3 fm_regression.py`

6. To run the Random Forest regressor, check the path of the dataset and modify the path to store the model and run in src/models:

   `python3 random_forest_regression.py`

7. To view the dashboard, open `dashbord.html ` in the browser. Select a stock from the dropdown to view its details and trend
