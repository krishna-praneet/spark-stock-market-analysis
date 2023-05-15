import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_replace, to_timestamp
from pyspark.sql.types import *

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

complete_df = []
first = True
count = 0

full_path = 'stock_market_data/nasdaq/csv/*.csv'
dire = os.getcwd() + '/' + 'stock_market_data/nasdaq/csv/'

schema = StructType([StructField('Date', StringType(), True), StructField('Low', DoubleType(), True), 
                     StructField('Open', DoubleType(), True), StructField('Volume',DoubleType(), True), 
                     StructField('High', DoubleType(), True), StructField('Close', DoubleType(), True), 
                     StructField('Adj Close', DoubleType(), True), StructField('Stock', StringType(), True)])

df = spark.read.format('csv').option('header', 'true').schema(schema).load(full_path).withColumn('Stock', input_file_name())
df = df.withColumn('Stock', regexp_replace('Stock', 'file:' + dire, ''))
df = df.withColumn('Stock', regexp_replace('Stock', '.csv', ''))
df = df.withColumn('Date', to_timestamp('Date', 'd-M-yyyy'))

df.show()
print(f'Number of rows: {df.count()}')
df.coalesce(1).write.format('json').save('dataset.json')