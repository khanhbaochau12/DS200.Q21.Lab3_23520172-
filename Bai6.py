import os
import datetime
from pyspark import SparkConf, SparkContext

app_conf = SparkConf().setAppName("YearlyAvgRating").setMaster("local[*]")
sc = SparkContext(conf=app_conf)
sc.setLogLevel("ERROR")

INPUT_DIR  = "hdfs://localhost:9000/movie/input"
PATH_RATE1 = f"{INPUT_DIR}/ratings_1.txt"
PATH_RATE2 = f"{INPUT_DIR}/ratings_2.txt"

HDFS_OUT  = "hdfs://localhost:9000/movie/output/bai6"
LOCAL_OUT = "/mnt/d/UIT 3rd year/BigData/ThucHanh/Lab3/output/bai6.txt"

def extract_year(ts_str):
    return datetime.datetime.fromtimestamp(int(ts_str)).year

# --- Map mỗi rating → (năm, (rating, 1)) ---
def to_year_key(cols):
    year  = extract_year(cols[3])
    score = float(cols[2])
    return (year, (score, 1))

def merge_scores(a, b):
    return (a[0] + b[0], a[1] + b[1])

rating_cols = sc.textFile(f"{PATH_RATE1},{PATH_RATE2}") \
                .map(lambda line: line.split(","))

result = rating_cols.map(to_year_key) \
                    .reduceByKey(merge_scores) \
                    .map(lambda r: (r[0], round(r[1][0] / r[1][1], 4), r[1][1])) \
                    .sortBy(lambda r: r[0]) \
                    .collect()

# --- In kết quả ---
header  = f"\n  {'Năm':<8} {'Avg Rating':>12} {'Tổng lượt':>12}"
divider = "  " + "-" * 34
print(header)
print(divider)
for year, avg, cnt in result:
    print(f"  {year:<8} {avg:>12.4f} {cnt:>12}")

# --- Ghi file local ---
output_lines = [header, divider]
for year, avg, cnt in result:
    output_lines.append(f"  {year:<8} {avg:>12.4f} {cnt:>12}")

# --- Lưu HDFS ---
sc.parallelize(result) \
  .map(lambda r: f"{r[0]}::{r[1]}::{r[2]}") \
  .saveAsTextFile(HDFS_OUT)
print(f"\n>>> Đã lưu kết quả lên HDFS: {HDFS_OUT}")

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)
with open(LOCAL_OUT, "w", encoding="utf-8") as fout:
    fout.write("\n".join(output_lines))
print(f">>> Đã lưu kết quả về local: {LOCAL_OUT}")
print("=" * 55 + "\n")

sc.stop()
