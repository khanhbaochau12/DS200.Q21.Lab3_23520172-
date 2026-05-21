import os
from pyspark import SparkConf, SparkContext

app_conf = SparkConf().setAppName("OccupationAvgRating").setMaster("local[*]")
sc = SparkContext(conf=app_conf)
sc.setLogLevel("ERROR")

INPUT_DIR  = "hdfs://localhost:9000/movie/input"
PATH_RATE1 = f"{INPUT_DIR}/ratings_1.txt"
PATH_RATE2 = f"{INPUT_DIR}/ratings_2.txt"
PATH_USERS = f"{INPUT_DIR}/users.txt"
PATH_OCC   = f"{INPUT_DIR}/occupation.txt"

HDFS_OUT  = "hdfs://localhost:9000/movie/output/bai5"
LOCAL_OUT = "/mnt/d/UIT 3rd year/BigData/ThucHanh/Lab3/output/bai5.txt"

# --- Bước 1: Đọc occupation.txt → map occId → tên nghề ---
def parse_occupation(line):
    parts = line.split(",")
    return (int(parts[0]), parts[1].strip())

occ_id_to_name = sc.textFile(PATH_OCC).map(parse_occupation).collectAsMap()

# --- Bước 2: Đọc users.txt → map userId → tên nghề ---
def parse_user_occ(line):
    parts = line.split(",")
    occ_name = occ_id_to_name.get(int(parts[3]), "unknown")
    return (int(parts[0]), occ_name)

uid_to_occ = sc.textFile(PATH_USERS).map(parse_user_occ).collectAsMap()
occ_bc = sc.broadcast(uid_to_occ)

# --- Bước 3: Map mỗi rating → (occupation, (rating, 1)) ---
def to_occ_key(cols):
    uid   = int(cols[0])
    score = float(cols[2])
    occ   = occ_bc.value.get(uid, "unknown")
    return (occ, (score, 1))

def merge_scores(a, b):
    return (a[0] + b[0], a[1] + b[1])

rating_cols = sc.textFile(f"{PATH_RATE1},{PATH_RATE2}") \
                .map(lambda line: line.split(","))

result = rating_cols.map(to_occ_key) \
                    .reduceByKey(merge_scores) \
                    .map(lambda r: (r[0], round(r[1][0] / r[1][1], 4), r[1][1])) \
                    .sortBy(lambda r: -r[1]) \
                    .collect()

# --- In kết quả ---
header  = f"\n  {'STT':<5} {'Occupation':<30} {'Avg Rating':>10} {'Số lượt':>10}"
divider = "  " + "-" * 57
print(header)
print(divider)
for i, (occ, avg, cnt) in enumerate(result, 1):
    print(f"  {i:<5} {occ:<30} {avg:>10.4f} {cnt:>10}")

# --- Ghi file local ---
output_lines = [header, divider]
for i, (occ, avg, cnt) in enumerate(result, 1):
    output_lines.append(f"  {i:<5} {occ:<30} {avg:>10.4f} {cnt:>10}")

# --- Lưu HDFS ---
sc.parallelize(result) \
  .map(lambda r: f"{r[0]}::{r[1]}::{r[2]}") \
  .saveAsTextFile(HDFS_OUT)
print(f"\n>>> Đã lưu kết quả lên HDFS: {HDFS_OUT}")

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)
with open(LOCAL_OUT, "w", encoding="utf-8") as fout:
    fout.write("\n".join(output_lines))
print(f">>> Đã lưu kết quả về local: {LOCAL_OUT}")
print("=" * 60 + "\n")

sc.stop()
