
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "monthly_foreign_arrivals.csv"

OUTPUT_FILE = (
    PROCESSED_DIR / "monthly_foreign_arrivals_features.csv"
)


# ==========================================
# 2. CREATE SPARK SESSION
# ==========================================

spark = (
    SparkSession.builder
    .appName("TourismIntelligence-FeatureEngineering")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


try:

    print("\n==========================================")
    print(" TOURISM INTELLIGENCE - FEATURE ENGINEERING")
    print("==========================================")

    # ==========================================
    # 3. LOAD MONTHLY TOURISM DATA
    # ==========================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(INPUT_FILE))
    )

    print("\nDataset loaded successfully!")

    # ==========================================
    # 4. DATA VALIDATION
    # ==========================================

    required_columns = [
        "year",
        "month",
        "foreign_tourist_arrivals"
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = (
        df
        .withColumn("year", F.col("year").cast("int"))
        .withColumn("month", F.col("month").cast("int"))
        .withColumn(
            "foreign_tourist_arrivals",
            F.col("foreign_tourist_arrivals").cast("long")
        )
        .dropna(subset=required_columns)
        .filter(
            (F.col("year") >= 1900) &
            (F.col("month").between(1, 12)) &
            (F.col("foreign_tourist_arrivals") >= 0)
        )
        .dropDuplicates(["year", "month"])
    )

    # ==========================================
    # 5. CREATE DATE COLUMN
    # ==========================================

    df = df.withColumn(
        "date",
        F.to_date(
            F.concat_ws(
                "-",
                F.col("year"),
                F.lpad(F.col("month"), 2, "0"),
                F.lit("01")
            )
        )
    )

    # ==========================================
    # 6. TEMPORAL WINDOW
    # ==========================================
    # Partition by year so lag features never
    # incorrectly cross the gaps between years.

    window_spec = (
        Window
        .partitionBy("year")
        .orderBy("month")
    )

    rolling_3 = (
        window_spec
        .rowsBetween(-3, -1)
    )

    rolling_6 = (
        window_spec
        .rowsBetween(-6, -1)
    )

    # ==========================================
    # 7. FEATURE ENGINEERING
    # ==========================================

    print("\nCreating machine learning features...")

    df = (
        df

        # Previous month arrivals
        .withColumn(
            "lag_1",
            F.lag("foreign_tourist_arrivals", 1)
            .over(window_spec)
        )

        # Previous 3 months arrivals
        .withColumn(
            "lag_3",
            F.lag("foreign_tourist_arrivals", 3)
            .over(window_spec)
        )

        # Previous 12 months
        .withColumn(
            "lag_12",
            F.lag("foreign_tourist_arrivals", 12)
            .over(window_spec)
        )

        # Previous 3-month average
        .withColumn(
            "rolling_mean_3",
            F.avg("foreign_tourist_arrivals")
            .over(rolling_3)
        )

        # Previous 6-month average
        .withColumn(
            "rolling_mean_6",
            F.avg("foreign_tourist_arrivals")
            .over(rolling_6)
        )

        # Seasonal features
        .withColumn(
            "month_sin",
            F.sin(
                2 * F.lit(3.141592653589793) *
                F.col("month") / 12
            )
        )

        .withColumn(
            "month_cos",
            F.cos(
                2 * F.lit(3.141592653589793) *
                F.col("month") / 12
            )
        )
    )

    # ==========================================
    # 8. DISPLAY FEATURES
    # ==========================================

    print("\n========== FEATURE ENGINEERED DATA ==========")

    df.orderBy("date").show(36, truncate=False)

    print("\nDataset Schema:")

    df.printSchema()

    # ==========================================
    # 9. DATASET SUMMARY
    # ==========================================

    print("\n========== DATASET SUMMARY ==========")

    print("Total Records:", df.count())

    print(
        "Available Years:",
        df.select("year").distinct().count()
    )

    print(
        "Available Features:",
        len(df.columns)
    )

    # ==========================================
    # 10. EXPORT FEATURE DATASET
    # ==========================================

    print("\nExporting ML dataset...")

    # This is a small starter dataset.
    # Pandas is used only for single-file export.

    pandas_df = df.orderBy("date").toPandas()

    pandas_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n==========================================")
    print(" FEATURE ENGINEERING COMPLETED")
    print("==========================================")

    print("Output File:")
    print(OUTPUT_FILE)

    print("Total Records:", len(pandas_df))

    print("\nML Dataset Generated Successfully!")

finally:

    spark.stop()

    print("\nSpark Session Closed Successfully.")
