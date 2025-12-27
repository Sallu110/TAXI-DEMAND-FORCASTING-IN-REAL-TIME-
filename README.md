# STEPS

1. Import libraries 
1. Data Ingestion
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Testing & Evaluation
6. Model Prediction
7. Streaming Pipeline
8. Results

## IMPORT LIBRARIES 
```python 
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
from pyspark.sql.functions import input_file_name, col, log1p, expm1, avg, sum as spark_sum
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
```

