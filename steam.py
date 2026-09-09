# Databricks notebook source
# MAGIC %md
# MAGIC

# COMMAND ----------

# curl https://full-stack-bigdata-datasets.s3.amazonaws.com/Big_Data/Project_Steam/steam_game_output.json
# record a file in local wget -O steam_game_output.json https://full-stack-bigdata-datasets.s3.amazonaws.com/Big_Data/Project_Steam/steam_game_output.json

# COMMAND ----------

from pyspark.sql import SparkSession

filepath = "s3://full-stack-bigdata-datasets/Big_Data/Project_Steam/steam_game_output.json"

df = spark.read.format('json').load(filepath)

# COMMAND ----------

# Noumber of elements in dataframe
df.count()


# COMMAND ----------

type(df)

# COMMAND ----------

df.take(1)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df_flat = df.select("data.*")
df_flat.display()

# COMMAND ----------

import pyspark.pandas as ps

# COMMAND ----------

# MAGIC %md
# MAGIC