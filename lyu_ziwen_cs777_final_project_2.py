from __future__ import print_function

import re
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import to_date, year, hour
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, countDistinct, split, month
import pandas as pd
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, PolynomialExpansion
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.stat import ChiSquareTest
import numpy as np

sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
sqlContext = SQLContext(sc)


# Function to convert time to a continuous variable
def time_to_float(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours + minutes / 60  # Convert minutes to a fraction of an hour


if __name__ == "__main__":

    # import the data
    rdd = sys.argv[1]
    df = sqlContext.read.format('csv').options(header='true', inferSchema='true', sep=',').load(rdd)
    # only choose the year 2022
    df = df.withColumn("CRASH DATE", to_date(df["CRASH DATE"], "M/d/yyyy"))
    df = df.filter(year(df["CRASH DATE"]) == 2022)
    # select only interested columns
    columns_of_interest = ['CRASH DATE', 'CRASH TIME', 'BOROUGH', 'NUMBER OF PERSONS INJURED',
                           'NUMBER OF PERSONS KILLED', 'CONTRIBUTING FACTOR VEHICLE 1', 'VEHICLE TYPE CODE 1']
    df = df.select(columns_of_interest)
    df = df.cache()

    # Data Cleaning
    ###Check how many data we have
    row_counts = df.count()
    print(f"Total number of rows: {row_counts}")  # 103886
    ###Check how many null values it have
    null_counts = df.select([F.sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
    null_counts.show()  # BOROUGH - 35099, Contributing factors - 592, Vehicle type - 1460
    ###Check how many specific values we have for each columns
    unique_counts = df.select([countDistinct(col(c)).alias(c) for c in df.columns])
    unique_counts.show()
    ###Check how many rows correspond to each specific value in features contributing factor and vehicle type
    df.groupBy('CONTRIBUTING FACTOR VEHICLE 1').count().orderBy("count", ascending=False).show(100)
    df.groupBy('VEHICLE TYPE CODE 1').count().orderBy("count", ascending=False).show(100)

    ##We can see that 35099 data points miss borough, takes up over 33% of the data. We will code this to "Other"
    df = df.fillna({"BOROUGH": "other"})
    ##We can see contirbuting factors have 55 unique values, and vehicle type has 355 unique values. We need to reduce them.
    ###I binned contributing factors into 6 categories: Driver's improper operation/health issue, Illegal behavior, Road issue, Third party issue, Vehicle defection, and others
    category_mapping = {
        "Following Too Closely": "driver's improper operation",
        "Traffic Control Disregarded": "driver's improper operation",
        "Driverless/Runaway Vehicle": "vehicle defection",
        "Accelerator Defective": "vehicle defection",
        "Windshield Inadequate": "vehicle defection",
        "Using On Board Navigation Device": "driver's improper operation",
        "Unsafe Speed": "driver's improper operation",
        "Shoulders Defective/Improper": "road issue",
        "Tinted Windows": "vehicle defection",
        "Oversized Vehicle": "vehicle defection",
        "Passing or Lane Usage Improper": "driver's improper operation",
        "Lane Marking Improper/Inadequate": "road issue",
        "Other Lighting Defects": "road issue",
        "Aggressive Driving/Road Rage": "driver's improper operation",
        "Other Vehicular": "third party issue",
        "Driver Inexperience": "driver's improper operation",
        "Texting": "driver's improper operation",
        "Tow Hitch Defective": "vehicle defection",
        "Illnes": "health issue",
        "Drugs (illegal)": "illegal behavior",
        "Unspecified": "other",
        "Pavement Defective": "road issue",
        "Prescription Medication": "health issue",
        "View Obstructed/Limited": "road issue",
        "Lost Consciousness": "health issue",
        "Reaction to Uninvolved Vehicle": "driver's improper operation",
        "Fell Asleep": "health issue",
        "Tire Failure/Inadequate": "vehicle defection",
        "Outside Car Distraction": "distraction",
        "Fatigued/Drowsy": "health issue",
        "Driver Inattention/Distraction": "distraction",
        "Cell Phone (hand-Held)": "driver's improper operation",
        "Listening/Using Headphones": "illegal behavior",
        "Failure to Keep Right": "driver's improper operation",
        "Obstruction/Debris": "road issue",
        "Unsafe Lane Changing": "driver's improper operation",
        "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion": "third party issue",
        "Passenger Distraction": "distraction",
        "Alcohol Involvement": "illegal behavior",
        "Traffic Control Device Improper/Non-Working": "road issue",
        "Brakes Defective": "vehicle defection",
        "Backing Unsafely": "driver's improper operation",
        "Eating or Drinking": "driver's improper operation",
        "Headlights Defective": "vehicle defection",
        "Failure to Yield Right-of-Way": "driver's improper operation",
        "Animals Action": "third party issue",
        "Turning Improperly": "driver's improper operation",
        "Steering Failure": "vehicle defection",
        "Glare": "road issue",
        "Passing Too Closely": "driver's improper operation",
        "Vehicle Vandalism": "vehicle defection",
        "Pavement Slippery": "road issue",
        "Physical Disability": "health issue",
        "Other Electronic Device": "driver's improper operation",
        "Cell Phone (hands-free)": "driver's improper operation"}

    binned_column = None

    for value, category in category_mapping.items():
        condition = (col("CONTRIBUTING FACTOR VEHICLE 1") == value)
        if binned_column is None:
            binned_column = when(condition, category)
        else:
            binned_column = binned_column.when(condition, category)

    df = df.withColumn("contributing_factor_binned", binned_column)
    df.groupBy("contributing_factor_binned").count().show()
    df = df.cache()

    ##Binned all vehicle types into: Sedan, SUV, bus, truck, bike/motorcycle, other
    df = df.withColumn(
        "vehicle_type",
        when(F.col("VEHICLE TYPE CODE 1").ilike("%sedan%"), "sedan")
        .when(F.col("VEHICLE TYPE CODE 1").ilike("%station wagon%"), "SUV")
        .when(F.col("VEHICLE TYPE CODE 1").ilike("%bus%"), "bus")
        .when(F.col("VEHICLE TYPE CODE 1").ilike("%truck%"), "truck")
        .when(F.col("VEHICLE TYPE CODE 1").ilike("%bike%"), "bike")
        .when(F.col("VEHICLE TYPE CODE 1").ilike("%motor%"), "motorcycle")
        .when(F.col("VEHICLE TYPE CODE 1").ilike("%taxi%"), "taxi")
        .when(F.col("VEHICLE TYPE CODE 1").ilike("%cab%"), "taxi")
        .when(F.col("VEHICLE TYPE CODE 1").ilike("%van%"), "van")
        .when(F.col("VEHICLE TYPE CODE 1").ilike("%scooter%"), "scooter")
        .otherwise("other")  # Keep other values as they are
    )
    df.groupBy("vehicle_type").count().show()
    df = df.cache()

    ##Bin all day time into 4 categories: morning, afternoon, evening, midnight
    df = df.withColumn('month', month(df['CRASH DATE']))
    df = df.withColumn("crash_hour", hour(col("CRASH TIME")))
    df = df.withColumn("time_of_day",
                       when((F.col("crash_hour") >= 6) & (F.col("crash_hour") < 12), "morning")
                       .when((F.col("crash_hour") >= 12) & (F.col("crash_hour") < 18), "afternoon")
                       .when((F.col("crash_hour") >= 18) & (F.col("crash_hour") < 24), "evening")
                       .otherwise("midnight"))
    df = df.cache()
    df.groupBy('time_of_day').count().orderBy("count",
                                              ascending=False).show()  # Check how many data points distrbute in four categories

    ###Check again - how many null values we have in the dataset
    null_counts = df.select([F.sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
    null_counts.show()
    ##Null values only take up less than 1% of the data in vehicle type and contributing factors, remove them
    df = df.dropna()

    ###continous time
    df = df.withColumn(
        "time_continuous",
        (split(F.col("CRASH TIME"), ":").getItem(0).cast("int") +
         split(F.col("CRASH TIME"), ":").getItem(1).cast("int") / 60))
    ### set the severity to 3 level
    df = df.withColumn(
        "severity_level",
        when((F.col('NUMBER OF PERSONS KILLED') > 0) | (F.col('NUMBER OF PERSONS INJURED') > 4), 'serious crash')
        .when((F.col('NUMBER OF PERSONS INJURED') > 0) & (F.col('NUMBER OF PERSONS INJURED') <= 4), 'moderate crash')
        .otherwise('no injury crash')
    )
    df.groupBy("severity_level").count().show()
    df = df.cache()

    # After the cleaning of data, start to explore the relationship between features
    # 1. Question: Are certain boroughs in NYC and certain periods of a day more prone to have vehicle collisions in 2022?

    ##Borough vs. Collision Severity
    df_borough = df.filter(F.col('BOROUGH') != 'other')
    df_borough_level = df_borough.select('BOROUGH', 'severity_level').groupBy("BOROUGH","severity_level").count().orderBy("count",ascending=False).show(truncate=False)

    ##day_of_time vs. Collision Severity
    df_time = df.select('time_of_day', 'severity_level').groupBy("time_of_day", "severity_level").count().orderBy("count", ascending=False).show(truncate=False)

    ## month vs. Collision Severity
    df_month = df.select('month', 'severity_level').groupBy('month', 'severity_level').count().orderBy("count",ascending=False).show(100, truncate=False)

    ## time_of_day & Borough vs. Collision severity score
    df_agg_score = df_borough.select('BOROUGH', 'time_of_day', 'severity_level').groupBy("BOROUGH", "time_of_day","severity_level").count().orderBy("count", ascending=False).show(truncate=False)

    # 2. Quesion: How do certain human behaviors (driver's improper operation, illegal behavior), vehicle defection, road issue, third party impact crash severith?
    df.groupBy('contributing_factor_binned', 'severity_level').count().orderBy("count", ascending=False).show(100, truncate=False)

    # 3. Question: how do vehicle type impact the collision severity?
    df.groupBy('vehicle_type', 'severity_level').count().orderBy("count", ascending=False).show(100, truncate=False)

    # Build the machine learning model to predict the crash collision severity score
    # Decision Tree model
    ##Index the unique values
    indexer_borough = StringIndexer(inputCol="BOROUGH", outputCol="borough_index")
    indexer_crash_factors = StringIndexer(inputCol="contributing_factor_binned", outputCol="factor_index")
    indexer_vehicle_type = StringIndexer(inputCol="vehicle_type", outputCol="vehicle_type_index")
    indexer_month = StringIndexer(inputCol='month', outputCol='month_index')
    df = df.withColumn("level_index",
                       when(F.col("severity_level") == 'no injury crash', 0)
                       .otherwise(1))

    ### due to imbalance, give the weight to logistic regression model
    class_counts = df.groupBy("level_index").agg(F.count("*").alias("class_count"))
    total_count = df.count()
    class_weights = class_counts.withColumn("weight", F.lit(total_count) / F.col("class_count"))
    df = df.join(class_weights.select("level_index", "weight"), on="level_index", how="left")

    ##one-hot encoding
    encoder_borough = OneHotEncoder(inputCol="borough_index", outputCol="borough_one")
    encoder_crash_factors = OneHotEncoder(inputCol="factor_index", outputCol="factor_one")
    encoder_month = OneHotEncoder(inputCol="month_index", outputCol="month_one")
    encoder_vehicle_type = OneHotEncoder(inputCol="vehicle_type_index", outputCol="vehicle_type_one")

    ###Aseemble features to vectors
    vector_assembler = VectorAssembler(
        inputCols=["borough_one", "time_continuous", "crash_hour", "month_one", "factor_one", "vehicle_type_one"],
        outputCol="features"
    )
    pipeline = Pipeline(stages=[indexer_borough, indexer_crash_factors, indexer_vehicle_type,
                                indexer_month, encoder_borough, encoder_crash_factors,
                                encoder_month, encoder_vehicle_type, vector_assembler])
    model = pipeline.fit(df)
    preprocessed_df = model.transform(df)
    ###Chi-square test
    chi_square_result = ChiSquareTest.test(preprocessed_df, "features", "level_index")
    chi_square_result.show(truncate=False)

    polynomial_expansion = PolynomialExpansion(degree=3, inputCol="features", outputCol="expanded_features")
    # decision tree
    dt = DecisionTreeClassifier(featuresCol="expanded_features", labelCol="level_index", predictionCol="dt_prediction",
                                maxDepth=5, maxBins=32)
    pipline = Pipeline(stages=[indexer_borough, indexer_crash_factors, indexer_vehicle_type,
                               indexer_month, encoder_borough, encoder_crash_factors,
                               encoder_month, encoder_vehicle_type, vector_assembler, polynomial_expansion, dt])
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    dt_model = pipline.fit(train_df)
    dt_predictions = dt_model.transform(test_df)
    prediction_test_true_positives = dt_predictions.filter((F.col("level_index") == 1) & (F.col("dt_prediction") == 1))
    prediction_test_false_positives = dt_predictions.filter((F.col("level_index") == 0) & (F.col("dt_prediction") == 1))
    prediction_test_false_negatives = dt_predictions.filter((F.col("level_index") == 1) & (F.col("dt_prediction") == 0))
    prediction_test_true_negatives = dt_predictions.filter((F.col("level_index") == 0) & (F.col("dt_prediction") == 0))
    TP = prediction_test_true_positives.count()
    FP = prediction_test_false_positives.count()
    FN = prediction_test_false_negatives.count()
    TN = prediction_test_true_negatives.count()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("\nPerformance Metrics: Descision Tree")
    print("Accuracy", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Confusion Matrix:\n")
    print(f"TP: {TP} FP: {FP}\nFN: {FN} TN: {TN}")

    ### Logistic Regression (Weighted)
    ###ploynomial

    lr = LogisticRegression(featuresCol='expanded_features', labelCol='level_index', weightCol='weight', regParam=1,
                            maxIter=150)  # L2 regularization
    lr = lr.setThreshold(0.48)
    pipline = Pipeline(stages=[indexer_borough, indexer_crash_factors, indexer_vehicle_type,
                               indexer_month, encoder_borough, encoder_crash_factors,
                               encoder_month, encoder_vehicle_type, vector_assembler, polynomial_expansion, lr])

    ###tsplit, predict, and evaluate
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    lr_model = pipline.fit(train_df)
    predictions = lr_model.transform(test_df)
    prediction_test_true_positives = predictions.filter((F.col("level_index") == 1) & (F.col("prediction") == 1))
    prediction_test_false_positives = predictions.filter((F.col("level_index") == 0) & (F.col("prediction") == 1))
    prediction_test_false_negatives = predictions.filter((F.col("level_index") == 1) & (F.col("prediction") == 0))
    prediction_test_true_negatives = predictions.filter((F.col("level_index") == 0) & (F.col("prediction") == 0))
    TP = prediction_test_true_positives.count()
    FP = prediction_test_false_positives.count()
    FN = prediction_test_false_negatives.count()
    TN = prediction_test_true_negatives.count()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("\nPerformance Metrics: Logisitic Regression")
    print("Accuracy", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Confusion Matrix:\n")
    print(f"TP: {TP} FP: {FP}\nFN: {FN} TN: {TN}")