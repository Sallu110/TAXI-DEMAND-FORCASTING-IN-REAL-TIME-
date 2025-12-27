## STEPS

1. Import libraries 
2. Data Ingestion
3. Data Preprocessing
4. Feature Engineering
5. Data Splitting
6. Linear Regression Model Training
7. Linear regression model testing and evaluation 
8. Random Forest Model Training
9. Random Forest Model Testing and Evaluation
10. Streaming Pipeline


### Import libraries
```python
# importing required libraries

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
from pyspark.sql.functions import input_file_name, col, log1p, expm1, avg, sum as spark_sum
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
```

### Data Ingestion 

```python
# This part sets up Spark for the project. It creates a Spark session, sets the log level to show only important messages
# defines where the streaming data and checkpoints are stored, specifies the structure of the taxi data,

and prepares a streaming DataFrame to read new CSV files as they arrive.

spark = SparkSession.builder \
    .appName("Streaming-ML-TaxiDemand-Full-Improved") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# HDFS Directories (stream input and checkpoint directories)
stream_dir = "hdfs:///taxi_demand/stream_input"
checkpoint_dir = "file:///C:/spark-temp"
# Schema
schema = StructType([
    StructField("hour", IntegerType(), True),
    StructField("weekday", IntegerType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("RatecodeID", IntegerType(), True),
    StructField("payment_type", StringType(), True),  
    StructField("fare_amount", DoubleType(), True),
    StructField("extra", DoubleType(), True),
    StructField("tip_amount", DoubleType(), True),
    StructField("tolls_amount", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),  
    StructField("congestion_surcharge", DoubleType(), True),
    StructField("Airport_fee", DoubleType(), True),
    StructField("trip_count", IntegerType(), True),  
    StructField("Zone_num", IntegerType(), True),
    StructField("Borough_num", IntegerType(), True), 
    StructField("service_zone_num", IntegerType(), True)
])

# Streaming DataFrame 
stream_df = spark.readStream \
    .option("header", "true") \
    .schema(schema) \
    .option("maxFilesPerTrigger", 5) \
    .option("recursiveFileLookup", "true") \
    .csv(stream_dir)
```

### Data Preprocessing
```python

# This function processes each batch of streaming data. It first prints which batch is being processed and skips it if it’s empty.
# Then, it adds the source file name, removes unnecessary columns,
# keeps only valid RatecodeID values. Missing numeric values are filled with 0.
# The data is then grouped by hour, weekday, and zones to calculate totals and averages,
# and the target variable (trip_count) is log-transformed.
# Finally, the weekday column is converted into a machine learning friendly format using one-hot encoding

# Batch Processing 
def process_batch(batch_df, batch_id):
    print("==============================")
    stream_name = f"Stream-{batch_id + 1}" 
    print(f"Processing {stream_name}")
    print("==============================")
    
    if batch_df.rdd.isEmpty():
        print(f"[Batch {batch_id + 1}] Empty batch — skipping")
        return

    batch_df = batch_df.withColumn("input_file", input_file_name())

    # Drop irrelevant columns
    batch_df = batch_df.drop("payment_type", "total_amount", "Borough_num")

    # Filter invalid RatecodeID
    batch_df = batch_df.filter(col("RatecodeID").isin([1, 2, 3]))

    # Fill nulls
    numeric_cols = [c for c, t in batch_df.dtypes if t in ("int", "double")]
    batch_df = batch_df.fillna(0, subset=numeric_cols)

    #  Aggregate per (hour, weekday, Zone_num, service_zone_num)
    agg_df = batch_df.groupBy("hour", "weekday", "Zone_num", "service_zone_num") \
        .agg(
            spark_sum("trip_count").alias("trip_count"),
            avg("trip_distance").alias("avg_trip_distance"),
            avg("fare_amount").alias("avg_fare_amount"),
            avg("extra").alias("avg_extra"),
            avg("tip_amount").alias("avg_tip_amount")
        )

    # Log-transform target
    agg_df = agg_df.withColumn("log_trip_count", log1p("trip_count"))

    print(f"\n[Batch {batch_id + 1}] Aggregated row count: {agg_df.count()}")
    agg_df.show(5, truncate=False)

    # One-hot encode weekday 
    indexer = StringIndexer(inputCol="weekday", outputCol="weekday_index")
    agg_df = indexer.fit(agg_df).transform(agg_df)
    encoder = OneHotEncoder(inputCols=["weekday_index"], outputCols=["weekday_vec"])
    agg_df = encoder.fit(agg_df).transform(agg_df)
```
### Feature Engineering
```python
# This part selects the columns that will be used as features for the machine learning models.
# It combines them into a single vector using VectorAssembler and
# then standardizes the values using StandardScaler so that all features have a similar scale.
# The resulting DataFrame contains the prepared features and the target variable (log_trip_count)
# which are ready for training and testing the models.
# A preview of the first few rows is displayed to check the prepared features.

# Feature columns
    feature_cols = ["hour", "Zone_num", "service_zone_num", "avg_trip_distance",
                    "avg_fare_amount", "avg_extra", "avg_tip_amount", "weekday_vec"]

    # VectorAssembler + Scaling
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled", handleInvalid="keep")
    assembled_df = assembler.transform(agg_df)

    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withMean=True, withStd=True)
    scaled_df = scaler.fit(assembled_df).transform(assembled_df).select("features", "log_trip_count")

    print(f"[Batch {batch_id + 1}] Prepared features preview:")
    scaled_df.show(5, truncate=False)
```
<img width="1181" height="279" alt="image" src="https://github.com/user-attachments/assets/e85f2dde-54df-44e1-a653-28375659a7e4" />


### Data Splitting
``` python
# Data is splitted into training and testing set for models training and evaluations

train, test = scaled_df.randomSplit([0.8, 0.2], seed=42)
```

### Linear Regression Model Training 
``` python
# This part sets up the evaluation metric for the models.
# It uses RegressionEvaluator to measure how well the predictions match the actual values of log_trip_count.
# The chosen metric is RMSE, which shows the average difference between the predicted and actual values.

evaluator = RegressionEvaluator(
    labelCol="log_trip_count",
    predictionCol="prediction",
    metricName="rmse"
)

# This part creates and trains a linear regression model.
# It uses the prepared features to predict log_trip_count
# fits the model to the training data so it can learn the relationship between the features and the target.
    

# Linear Regression 
    lr = LinearRegression(
     featuresCol="features",
    labelCol="log_trip_count"
    )

    lr_model = lr.fit(train)
```
### Linear regression model testing and evaluation 
``` python

# Now the model is evaluating with the help of testing data and generating predictions
# these predictions are then evaluated and used to calculate the RMSE value for verifying model performance
 
lr_preds = lr_model.transform(test)
    lr_rmse = evaluator.evaluate(lr_preds)

    print("========================================================")
    print(f"[Batch {batch_id + 1}] Linear Regression RMSE: {lr_rmse}")
    print("Linear Regression Predictions:")
    lr_preds.select("log_trip_count", "prediction").show(5, truncate=False)
    print("========================================================")
```
### Random Forest Model Training 
``` python

# Now this part is about to train the random forest model  and setting some hyperparameters of random forest model

# Random Forest 
    rf_model = RandomForestRegressor(
    featuresCol="features",
    labelCol="log_trip_count",
    numTrees=50,
    maxDepth=12)

    rf_model = rf_model.fit(train)
```
### Random Forest Model Testing and Evaluation 
``` python
# this part is making prediction by random forest model with the help of test data
# and then evaluating the random forest model by generated predictions
# and calculate its RMSE value for verifying model performance

rf_preds = rf_model.transform(test)
    rf_rmse = evaluator.evaluate(rf_preds)

    print("========================================================")
    print(f"[Batch {batch_id + 1}] Random Forest RMSE: {rf_rmse}")
    print("Random Forest Predictions:")
    rf_preds.select("log_trip_count", "prediction").show(5, truncate=False)
    print("========================================================")
```
### Streaming Pipeline 
```python
# This section starts the Spark structured streaming query, processing incoming taxi data in real-time using the
# process_batch` function. It uses a checkpoint for fault tolerance and triggers every 10 seconds.

query = stream_df.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", checkpoint_dir) \
    .trigger(processingTime="10 seconds") \
    .start()

query.awaitTermination()
```












