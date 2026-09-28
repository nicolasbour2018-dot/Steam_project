# Databricks notebook source
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

type(df_flat)

# COMMAND ----------

df_flat.printSchema()

# COMMAND ----------

from pyspark.sql.functions import explode
df_flat = df_flat.select("*", explode("categories").alias("category"))
df_flat.display()

# COMMAND ----------

df_flat.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

df_flat = df_flat.withColumn(
    "platform",
    F.expr("concat_ws(', ', IF(platforms.linux, 'linux', NULL), IF(platforms.mac, 'mac', NULL), IF(platforms.windows, 'windows', NULL))")
)
df_flat.display()

# COMMAND ----------

df_flat.printSchema()

# COMMAND ----------

df_flat = df_flat.withColumn(
    "tag",
    F.concat_ws(", ", F.map_keys(F.from_json(F.to_json("tags"), "map<string,bigint>")))
)
df_flat.display()

# COMMAND ----------

df_flat = df_flat.drop( "categories", "platforms", "tags")

# COMMAND ----------

df_flat.printSchema()

# COMMAND ----------

# import pyspark.pandas as ps

# COMMAND ----------

# MAGIC %md
# MAGIC