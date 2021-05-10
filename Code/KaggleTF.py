# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import DecisionTreeRegressor

# COMMAND ----------

IS_SPARK_SUBMIT_CLI = True

if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

file_location = "/user/jliu2/KaggleTaxi.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df1 = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# COMMAND ----------

temp_table_name1 = "KaggleTaxi_csv"

df1.createOrReplaceTempView(temp_table_name1)

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
    KaggleTaxi = spark.read.csv('/user/jliu2/KaggleTaxi.csv', inferSchema=True, header=True)
else:
    KaggleTaxi = sqlContext.sql("select * from KaggleTaxi_csv")
    
KaggleTaxi.show(5)

# COMMAND ----------

timediff=KaggleTaxi.select('tpep_pickup_datetime', 'tpep_dropoff_datetime','passenger_count','trip_distance',col('fare_amount').alias('label'))

timediff.show(5)

# COMMAND ----------

df2=timediff.withColumn('tpep_pickup_datetime',to_timestamp(col('tpep_pickup_datetime')))\
  .withColumn('tpep_dropoff_datetime', to_timestamp(col('tpep_dropoff_datetime')))\
  .withColumn('trip_time_in_secs',col("tpep_dropoff_datetime").cast("long") - col('tpep_pickup_datetime').cast("long"))
df2.show(5)

# COMMAND ----------

databe=df2.select('passenger_count','trip_distance','trip_time_in_secs','label')
data=databe.dropna()
data.show(5)

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['passenger_count', 'trip_time_in_secs', 'trip_distance'], outputCol="features")
gbt = GBTRegressor(labelCol="label")

# COMMAND ----------

paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 2])\
  .addGrid(gbt.maxIter, [10, 10])\
  .build()
  
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

pipeline = Pipeline(stages=[assembler, cv])
pipelineModel = pipeline.fit(train)

# COMMAND ----------

predictions = pipelineModel.transform(test)

# COMMAND ----------

predicted = predictions.select("features", "prediction", "trueLabel")
predicted.show(10)

# COMMAND ----------

predicted.createOrReplaceTempView("regressionPredictions")

# COMMAND ----------

dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")

dataPred.show(10)

# COMMAND ----------

evaluator  = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse1 = evaluator.evaluate(predictions)
print( "Root Mean Square Error (RMSE) for GBT Regression :", rmse1)

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['passenger_count', 'trip_time_in_secs', 'trip_distance'], outputCol="features")
lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=10, regParam=0.3)
pipeline1 = Pipeline(stages=[assembler, lr])

# COMMAND ----------

paramGrid1 = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.01]).addGrid(lr.maxIter, [10, 5]).build()
trainval = TrainValidationSplit(estimator=pipeline1, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid1, trainRatio=0.8)

# COMMAND ----------
pipelineModel = trainval.fit(train)


predictions = pipelineModel.transform(test)

# COMMAND ----------

predicted = predictions.select("features", "prediction", "trueLabel")
predicted.show(10)

# COMMAND ----------

predicted.createOrReplaceTempView("regressionPredictions")

# COMMAND ----------

dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")

dataPred.show(10)

# COMMAND ----------

evaluator  = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse2 = evaluator.evaluate(predictions)
print ("Root Mean Square Error (RMSE) for Linear Regression :", rmse2)

# COMMAND ----------
assembler = VectorAssembler(inputCols = ['passenger_count', 'trip_time_in_secs', 'trip_distance'], outputCol="features")
dt = DecisionTreeRegressor(labelCol="label",featuresCol="features")
# COMMAND ----------
paramGrid2 = ParamGridBuilder()\
  .addGrid(dt.maxDepth, [2,2])\
  .addGrid(dt.maxBins, [10,10])\
  .build()

evaluator = RegressionEvaluator(metricName="rmse", labelCol=dt.getLabelCol(), predictionCol=dt.getPredictionCol())

dtcv = CrossValidator(estimator = dt, estimatorParamMaps = paramGrid2, evaluator = evaluator, numFolds=2)
# COMMAND ----------

pipeline2 = Pipeline(stages=[assembler, dtcv])
pipelineModel = pipeline2.fit(train)


# COMMAND ----------
predictions = pipelineModel.transform(test)

predicted = predictions.select("features","prediction","truelabel")
predicted.show(10)
# COMMAND ----------
predicted.createOrReplaceTempView("regressionPredictions")

dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")

dataPred.show(5)
# COMMAND ----------

evaluator  = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse3 = evaluator.evaluate(predictions)
print( "Root Mean Square Error (RMSE) for Decision Forest Regression :", rmse3)


print( "Root Mean Square Error (RMSE) for GBT Regression :", rmse1)
print ("Root Mean Square Error (RMSE) for Linear Regression :", rmse2)
print( "Root Mean Square Error (RMSE) for Decision Forest Regression :", rmse3)
