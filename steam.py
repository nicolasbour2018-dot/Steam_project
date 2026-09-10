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

# MAGIC %md
# MAGIC Dataset to big for json_normalize method

# COMMAND ----------

type(df)

# COMMAND ----------

df.take(1)

# COMMAND ----------

type(df)

# COMMAND ----------



# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Categories, platforms ans tags are nested. Let's flatened the dataframe to range datas at same level.

# COMMAND ----------

from pyspark.sql.functions import explode
flat_df = df.select("data.*", explode("data.categories").alias("category"))
flat_df.display()

# COMMAND ----------

flat_df = flat_df.withColumn(
    "platform",
    F.concat_ws(', ', *[F.when(F.col(f"platforms.{f}"), F.lit(f)) for f in ['linux', 'mac', 'windows']])
)
flat_df.display()

# COMMAND ----------

'''df_flat = df_flat.withColumn(
    "platform",
    F.concat_ws(', ', 
        F.when(F.col("platforms.linux"), F.lit('linux')),
        F.when(F.col("platforms.mac"), F.lit('mac')),
        F.when(F.col("platforms.windows"), F.lit('windows'))
    )
)
df_flat.display()'''

# COMMAND ----------

flat_df = flat_df.withColumn("tag", F.to_json(F.col("tags")))
flat_df.display()

# COMMAND ----------

flat_df = flat_df.drop( "categories", "platforms", "tags")

# COMMAND ----------

flat_df.printSchema()

# COMMAND ----------

flat_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC # Explore dataset

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's query on the dataframe. Spark lets us run classic SQL queries on your tables, however, using classic SQL in Spark requires you to load the data in memory before running any query. We will use the .createOrReplaceTempView Spark DataFrame method in order to load the data in memory under a certain table name, we will then be able to run SQL queries on it.

# COMMAND ----------

flat_df.createOrReplaceTempView('table') # Creates a temporary view of the spark dataframe table in memory under the name
# my_table, which we can now query!

# COMMAND ----------

# MAGIC %md
# MAGIC The .sql method lets you write queries in SQL while benefiting from the distributed computing advantages of Spark.

# COMMAND ----------

result = spark.sql("SELECT * FROM table") # filters elements from my_table where position
# show everything
result.show()


# COMMAND ----------

# MAGIC %md
# MAGIC We observe :
# MAGIC - additionnal spaces. Yet, they don't affect results. It's only a display problem.
# MAGIC - missing values written '' on website

# COMMAND ----------

# test request on values with spaces succeed
result = spark.sql("SELECT * from table where genre = 'Action' and appid = 10 and developer = 'Valve' and initialprice = 999 and name = 'Counter-Strike' and publisher = 'Valve' and website =''")
result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We observe some problems :
# MAGIC - empty entries 
# MAGIC - asiatic characters. We need information even if the langage changes.

# COMMAND ----------

flat_df.select('developer', 'publisher').sample(fraction=0.00001).distinct().show()

# COMMAND ----------

spark.sql("select * from table limit 1").show()

# COMMAND ----------

# MAGIC %md
# MAGIC .isNull() detects SQL NULL, not float NaN. For numeric NaN, use functions.isnan.

# COMMAND ----------

result = spark.sql("select distinct name from table where name like'Counter-Strike%'")
result.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Analysis at the "macro" level

# COMMAND ----------

# MAGIC %md
# MAGIC Which publisher has released the most games on Steam?

# COMMAND ----------

result = flat_df \
    .select(flat_df['publisher'],'name') \
    .distinct() \
    .groupBy('publisher') \
    .count() \
    .orderBy(desc('count')) \
    .limit(1)
result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC What is Valve's position as publisher ?

# COMMAND ----------

from pyspark.sql.functions import desc, row_number
from pyspark.sql.window import Window

result = flat_df \
    .select(flat_df['publisher'],'name') \
    .distinct() \
    .groupBy('publisher') \
    .count() \
    .orderBy(desc('count')) \
    .withColumn('rank', row_number().over(Window \
    .orderBy(desc('count')))) \
    .filter(flat_df.publisher .isin(['Big Fish Games','Valve']))
result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC How many games on steam ?

# COMMAND ----------

result = flat_df \
    .select(flat_df['name']) \
    .distinct() \
    .count() 
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC How many developers indicated on steam ?

# COMMAND ----------

result = flat_df \
    .select(flat_df['developer']) \
    .distinct() \
    .count()
print(f"Developer number : {result}")



# COMMAND ----------

# MAGIC %md
# MAGIC Which developer contributed to the most games on steam ?

# COMMAND ----------

result = flat_df \
    .select(flat_df['developer'],'name') \
    .distinct() \
    .groupBy('developer') \
    .count() \
    .orderBy(desc('count')) \
    .limit(1)
result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC What are the best rated games?

# COMMAND ----------

result = flat_df \
    .select(flat_df['name']) \
    .groupBy() \

# COMMAND ----------

result = flat_df \
    .select(flat_df['name'],'positive') \
    .groupBy('name') \
    .agg(sum('positive').alias('count')) \
    .orderBy(desc('count')) \
    .limit(10)
result.show()

# COMMAND ----------

from pyspark.sql.functions import col, try_divide
result = flat_df \
    .select(flat_df['name'],'positive', 'negative') \
    .groupBy('name') \
    .agg( \
        sum('positive').alias('positive_sum'), \
        sum('negative').alias('negative_sum') \
        ) \
    .filter((col('positive_sum') > 0) & (col('negative_sum') > 0)) \
    .withColumn('ratio', col('positive_sum') /  col('negative_sum')) \
    .orderBy(desc('positive_sum')) \
    .limit(10)
result.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC In absolute value, Counter-Strike: Global Offensive has more positive advices (65,4M). Yet Terraria has a better ratio : 45,34 positive advices for 1 negative advice against 7,5 onpour Counter-Strike. Then, Terraria is proportianaly more liked.

# COMMAND ----------

result = spark.sql("SELECT * FROM table") # filters elements from my_table where position
# show everything
result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Are there years with more releases? Were there more or fewer game releases during the Covid, for example?

# COMMAND ----------

# to continue
result = flat_df \
    .select(flat_df['name'],'release_date') \
    .distinct() \
    .groupBy('name') \
    .count() \
    .orderBy(desc('count')) \
    .limit(10)
result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC How are the prizes distributed? Are there many games with a discount?

# COMMAND ----------

# MAGIC %md
# MAGIC What are the most represented languages?

# COMMAND ----------

# MAGIC %md
# MAGIC Are there many games prohibited for children under 16/18?