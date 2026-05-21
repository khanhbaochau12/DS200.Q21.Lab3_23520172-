import os
from collections import defaultdict
from pyspark import SparkConf, SparkContext

app_conf = SparkConf().setAppName("GenderAvgRating").setMaster("local[*]")
sc = SparkContext(conf=app_conf)
sc.setLogLevel("ERROR")

INPUT_DIR  = "hdfs://localhost:9000/movie/input"
PATH_MOVIES = f"{INPUT_DIR}/movies.txt"
PATH_RATE1  = f"{INPUT_DIR}/ratings_1.txt"
PATH_RATE2  = f"{INPUT_DIR}/ratings_2.txt"
PATH_USERS  = f"{INPUT_DIR}/users.txt"

HDFS_OUT  = "hdfs://localhost:9000/movie/output/bai3"
LOCAL_OUT = "/mnt/d/UIT 3rd year/BigData/ThucHanh/Lab3/output/bai3.txt"

# --- Bước 1: Tạo map userId → gender ---
def parse_user_gender(line):
    parts = line.split(",")
    return (int(parts[0]), parts[1])

uid_to_gender = sc.textFile(PATH_USERS).map(parse_user_gender).collectAsMap()
gender_bc = sc.broadcast(uid_to_gender)

# Lấy tên phim
def parse_movie_title(line):
    parts = line.split(",", 2)
    return (int(parts[0]), parts[1])

id_to_title = sc.textFile(PATH_MOVIES).map(parse_movie_title).collectAsMap()

# --- Bước 2: Map mỗi rating → key=(movieId, gender), value=(rating,1) ---
def to_gender_key(cols):
    uid   = int(cols[0])
    mid   = int(cols[1])
    score = float(cols[2])
    gender = gender_bc.value.get(uid, "Unknown")
    return ((mid, gender), (score, 1))

def merge_scores(a, b):
    return (a[0] + b[0], a[1] + b[1])

rating_cols = sc.textFile(f"{PATH_RATE1},{PATH_RATE2}") \
                .map(lambda line: line.split(","))

raw = rating_cols.map(to_gender_key) \
                 .reduceByKey(merge_scores) \
                 .map(lambda r: (r[0][0], r[0][1], round(r[1][0] / r[1][1], 4), r[1][1])) \
                 .collect()

# --- Bước 3: Nhóm theo movieId, điền đủ 2 giới tính ---
GENDERS = ["F", "M"]
per_movie = defaultdict(dict)
for mid, gender, avg, cnt in raw:
    per_movie[mid][gender] = (avg, cnt)

result = []
for mid in sorted(per_movie):
    for g in GENDERS:
        if g in per_movie[mid]:
            avg, cnt = per_movie[mid][g]
        else:
            avg, cnt = float("nan"), 0
        result.append((mid, g, avg, cnt))

# --- In kết quả ---
header  = f"\n  {'MovieID':<10} {'Tên phim':<40} {'Gender':<8} {'Avg':>7} {'Count':>7}"
divider = "  " + "-" * 74
print(header)
print(divider)
for mid, g, avg, cnt in result:
    title   = id_to_title.get(mid, "Unknown")
    avg_str = f"{avg:>7.4f}" if cnt > 0 else f"{'NaN':>7}"
    print(f"  {mid:<10} {title:<40} {g:<8} {avg_str} {cnt:>7}")

# --- Ghi file local ---
output_lines = [header, divider]
for mid, g, avg, cnt in result:
    title   = id_to_title.get(mid, "Unknown")
    avg_str = f"{avg:>7.4f}" if cnt > 0 else f"{'NaN':>7}"
    output_lines.append(f"  {mid:<10} {title:<40} {g:<8} {avg_str} {cnt:>7}")

# --- Lưu HDFS ---
sc.parallelize(result) \
  .map(lambda r: f"{r[0]},{id_to_title.get(r[0], 'Unknown')},{r[1]},{r[2]},{r[3]}") \
  .saveAsTextFile(HDFS_OUT)
print(f"\n>>> Đã lưu kết quả lên HDFS: {HDFS_OUT}")

os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)
with open(LOCAL_OUT, "w", encoding="utf-8") as fout:
    fout.write("\n".join(output_lines))
print(f">>> Đã lưu kết quả về local: {LOCAL_OUT}")
print("=" * 75 + "\n")

sc.stop()
