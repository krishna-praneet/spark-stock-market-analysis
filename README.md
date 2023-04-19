# spark-stock-market-analysis
Stock Market Prediction using Python

# Collaborators
1. Krishna Praneet Gudipaty
2. Archana Ganesh 
3. Shubham Patel 

Equal contribution from all the team members through pair programming. 

# Dataset used: 
Nasdaq dataset

# Goal: 
To build a stock recommendation system using Python to predict and suggest stocks with be best ROI.

# Machine Learning models used: 
The following models are to be used and compared to finalize the best performing one for the prediction model.

1. CNN
2. LSTM
3. Decision Tree Regressor

# Software tech stack:
1. Python: <br>
   a. SparkSQL <br>
   b. SparkMLLib <br>
   c. Boto3 <br>
   d. Pandas <br>
   e. Numpy <br>
   f. Matplotlib <br>
   g. Seaborn <br>
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

# To do:
1. Train the ML models to compare and tune the best-performing model
2. Build a dashboard that users may view that shows stocks in the decreasing order of ROI
