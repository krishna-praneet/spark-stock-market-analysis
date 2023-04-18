import os
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct,col
from pyspark.sql.types import DoubleType, TimestampType
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd

print(os.getcwd())
paths = {}

print('Fetching data file paths...')
for dirname, _, filenames in os.walk('stock_market_data/nasdaq/csv'):
    ps = dirname.split('/')
    for filename in filenames:
        paths[filename.replace('.csv', '')] = os.path.join(dirname, filename)

print(f'Number of stocks in nasdaq: {len(paths.keys())}')
    
print('Creating spark session...')
spark = SparkSession.builder.appName('json-to-dynamodb').getOrCreate()
print('Created.')
STOCKS = ['AMZN']

for data in STOCKS:
    # Input into csv data
    df = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(paths[data])
    df = df.withColumn('AdjClose', col('Adjusted Close')).drop('Adjusted Close')

    # Preprocessing Drop Nulls and duplicates
    df = df.na.drop()
    df = df.dropDuplicates()
    df.show()

    # Proper typecasting of the columns
    df = df.withColumn('Date', col('Date').cast(TimestampType()))
    df = df.withColumn('Open', col('Open').cast(DoubleType()))
    df = df.withColumn('Close', col('Close').cast(DoubleType()))
    df = df.withColumn('High', col('High').cast(DoubleType()))
    df = df.withColumn('Low', col('Low').cast(DoubleType()))
    df = df.withColumn('AdjClose', col('AdjClose').cast(DoubleType()))
    df = df.withColumn('Volume', col('Volume').cast(DoubleType()))
    
    # Print dtypes to check for correctness
    print(df.dtypes)

    # Drop date check numerical types for correlation matrix
    corr_df = df.drop('Date')
    vector_col = "corr_features"
    corr_df.show()

    # Calculate correlation matrix
    assembler = VectorAssembler(inputCols=corr_df.columns, outputCol=vector_col)
    df_vector = assembler.transform(corr_df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col).collect()[0][0] 
    corr_matrix = matrix.toArray().tolist() 
    corr_matrix_df = pd.DataFrame(corr_matrix, columns=corr_df.columns, index=corr_df.columns)
    print(corr_matrix_df)

    # Plot heatmap of correlation matrix
    plt.figure(figsize=(12,5))
    sns.heatmap(corr_matrix_df, 
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values, annot=True, fmt='.6f')
    figname = data + '_corr.png'
    plt.savefig(f'images/{figname}')
    # plt.show()

    # Plot Scatter and density plots of each column-wise pairs
    pdf = df.drop('Date').toPandas()
    axes = pd.plotting.scatter_matrix(pdf, alpha=0.75, figsize=[12,12], diagonal='kde')
    corrs = pdf.corr().values
    for i, j in zip(*plt.np.triu_indices_from(axes, k = 1)):
        axes[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=8)
    plt.suptitle('Scatter plots')
    figname = data + '_sca.png'
    plt.savefig(f'images/{figname}')
    # plt.show()

    dynamodf = pdf.to_dict('records')
    dynamodb = boto3.resource('dynamodb', aws_access_key_id='AKIA4YX3MJOJUKUMDZ4M', 
                              aws_secret_access_key='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',region_name='us-east-1')
    table_name = 'Nasdaq'
    table = dynamodb.Table(table_name)
    
    for item in dynamodf:
        table.put_item(Item=item)


