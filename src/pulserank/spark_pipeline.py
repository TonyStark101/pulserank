"""Spark Structured Streaming + Delta Lake deployment path.

This module imports distributed dependencies lazily so the laptop reference
implementation stays dependency-free. Run on Python 3.10+ with the
``distributed`` project extra and a Java 17 runtime.
"""

import argparse
import os
import sys
from pathlib import Path


def build_spark(app_name="pulserank-streaming"):
    # PySpark otherwise falls back to the first `python3` on PATH for workers,
    # which can silently differ from the driver environment on macOS.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    homebrew_java = Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home")
    if "JAVA_HOME" not in os.environ and homebrew_java.exists():
        os.environ["JAVA_HOME"] = str(homebrew_java)
    try:
        from delta import configure_spark_with_delta_pip
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError(
            "Install PulseRank with the distributed extra on Python 3.10+: "
            "pip install -e '.[distributed]'"
        ) from exc

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("PULSERANK_SPARK_MASTER", "local[2]"))
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.streaming.stateStore.providerClass", "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider")
        .config("spark.sql.shuffle.partitions", "6")
    )
    return configure_spark_with_delta_pip(
        builder,
        extra_packages=["org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"],
    ).getOrCreate()


def start_pipeline(spark, brokers, topic, lake, checkpoint, watermark="10 minutes"):
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

    lake, checkpoint = Path(lake), Path(checkpoint)
    schema = StructType([
        StructField("schema_version", IntegerType(), False),
        StructField("event_id", StringType(), False),
        StructField("user_id", StringType(), False),
        StructField("item_id", StringType(), False),
        StructField("action", StringType(), False),
        StructField("event_time", StringType(), False),
        StructField("watch_pct", DoubleType(), False),
        StructField("session_id", StringType(), True),
        StructField("trace_id", StringType(), True),
    ])

    kafka = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", brokers)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "true")
        .load()
    )
    bronze = kafka.select(
        F.col("topic"), F.col("partition"), F.col("offset"),
        F.col("timestamp").alias("broker_time"),
        F.col("key").cast("string").alias("message_key"),
        F.col("value").cast("string").alias("raw_json"),
    )
    bronze_path = str(lake / "bronze" / "viewing_events")
    silver_path = str(lake / "silver" / "viewing_events")
    # Initialize the Delta log before the downstream streaming reader starts.
    # This removes a startup race when the Kafka topic is initially empty.
    spark.createDataFrame([], bronze.schema).write.format("delta").mode("ignore").save(bronze_path)
    bronze_query = (
        bronze.writeStream.format("delta")
        .queryName("pulserank-bronze")
        .option("checkpointLocation", str(checkpoint / "bronze"))
        .outputMode("append")
        .start(bronze_path)
    )

    bronze_stream = spark.readStream.format("delta").load(bronze_path)
    parsed = bronze_stream.select(
        "topic", "partition", "offset", "broker_time", "raw_json",
        F.from_json("raw_json", schema).alias("event"),
    )
    valid_actions = ["impression", "play", "complete", "like", "skip"]
    silver = (
        parsed.filter(F.col("event").isNotNull())
        .select("topic", "partition", "offset", "broker_time", "event.*")
        .withColumn("event_time", F.to_timestamp("event_time"))
        .filter(
            (F.col("schema_version") == 1)
            & F.col("action").isin(valid_actions)
            & F.col("watch_pct").between(0.0, 1.0)
            & F.col("event_time").isNotNull()
        )
        .withWatermark("event_time", watermark)
        .dropDuplicatesWithinWatermark(["event_id"])
        .withColumn("processed_at", F.current_timestamp())
    )
    silver_query = (
        silver.writeStream.format("delta")
        .queryName("pulserank-silver")
        .option("checkpointLocation", str(checkpoint / "silver"))
        .outputMode("append")
        .start(silver_path)
    )
    return bronze_query, silver_query


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run PulseRank's Spark/Delta streaming pipeline")
    parser.add_argument("--brokers", default="localhost:19092")
    parser.add_argument("--topic", default="viewing-events.v1")
    parser.add_argument("--lake", default="data/delta")
    parser.add_argument("--checkpoint", default="data/checkpoints")
    parser.add_argument("--watermark", default="10 minutes")
    args = parser.parse_args(argv)
    spark = build_spark()
    queries = start_pipeline(spark, args.brokers, args.topic, args.lake, args.checkpoint, args.watermark)
    try:
        spark.streams.awaitAnyTermination()
    finally:
        for query in queries:
            query.stop()
        spark.stop()


if __name__ == "__main__":
    main()
