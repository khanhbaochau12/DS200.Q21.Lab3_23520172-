import os
from collections import defaultdict
from pyspark import SparkConf, SparkContext

app_conf = SparkConf().setAppName("AgeGroupAvgRating").setMaster("local[*]")
sc = SparkContext(conf=app_conf)
sc.setLogLevel("ERROR")

INPUT_DIR  = "hdfs://localhost:9000/movie/input"
PATH_MOVIES = f"{INPUT_DIR}/movies.txt"
PATH_RATE1  = f"{INPUT_DIR}/ratings_1.txt"
PATH_RATE2  = f"{INPUT_DIR}/ratings_2.txt"
PATH_USERS  = f"{INPUT_DIR}/users.txt"

HDFS_OUT  = "hdfs://localhost:9000/movie/output/bai4"
LOCAL_OUT = "/mnt/d/UIT 3rd year/BigData/ThucHanh/Lab3/output/bai4.txt"

AGE_GROUPS = ["1. Under 18", "2. 18-24", "3. 25-34", "4. 35-44", "5. 45-54", "6. 55+"]

def classify_age(age_str):
    age = int(age_str)
    if age < 18:   return "1. Under 18"
    elif age < 25: return "2. 18-24"
    elif age < 35: return "3. 25-34"
    elif age < 45: return "4. 35-44"
    elif age < 55: return "5. 45-54"
    else:          return "6. 55+"

# --- Bước 1: Map userId → nhóm tuổi ---
def parse_user_age(line):
    parts = line.split(",")
    return (int(parts[0]), classify_age(parts[2]))

uid_to_group = sc.textFile(PATH_USERS).map(parse_user_age).collectAsMap()
group_bc = sc.broadcast(uid_to_group)

def parse_movie_title(line):
    parts = line.split(",", 2)
    return (int(parts[0]), parts[1])

id_to_title = sc.textFile(PATH_MOVIES).map(parse_movie_title).collectAsMap()

# --- Bước 2: Map mỗi rating → key=(movieId, ageGroup), value=(rating,1) ---
def to_age_key(cols):
    uid   = int(cols[0])
    mid   = int(cols[1])
    score = float(cols[2])
    grp   = group_bc.value.get(uid, "Unknown")
    return ((mid, grp), (score, 1))

def merge_scores(a, b):
    return (a[0] + b[0], a[1] + b[1])

rating_cols = sc.textFile(f"{PATH_RATE1},{PATH_RATE2}") \
                .map(lambda line: line.split(","))

raw = rating_cols.map(to_age_key) \
                 .reduceByKey(merge_scores) \
                 .map(lambda r: (r[0][0], r[0][1], round(r[1][0] / r[1][1], 4), r[1][1])) \
                 .collect()

# --- Bước 3: Nhóm theo movieId, điền đủ 6 nhóm tuổi ---
per_movie = defaultdict(dict)
for mid, grp, avg, cnt in raw:
    per_movie[mid][grp] = (avg, cnt)

result = []
for mid in sorted(per_movie):
    for grp in AGE_GROUPS:
        if grp in per_movie[mid]:
            avg, cnt = per_movie[mid][grp]
        else:
            avg, cnt = float("nan"), 0
        result.append((mid, grp, avg, cnt))

# --- In kết quả ---
header  = f"\n  {'MovieID':<10} {'Tên phim':<35} {'Nhóm tuổi':<14} {'Avg':>7} {'Count':>7}"
divider = "  " + "-" * 75
print(header)
print(divider)
for mid, grp, avg, cnt in result:
    title   = id_to_title.get(mid, "Unknown")
    avg_str = f"{avg:>7.4f}" if cnt > 0 else f"{'NaN':>7}"
    print(f"  {mid:<10} {title:<35} {grp:<14} {avg_str} {cnt:>7}")

# --- Ghi file local ---
output_lines = [header, divider]
for mid, grp, avg, cnt in result:
    title   = id_to_title.get(mid, "Unknown")
    avg_str = f"{avg:>7.4f}" if cnt > 0 else f"{'NaN':>7}"
    output_lines.append(f"  {mid:<10} {title:<35} {grp:<14} {avg_str} {cnt:>7}")

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
