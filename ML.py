# ==============================================================
# BÀI TOÁN: PHÂN LOẠI XẾP LOẠI HIỆU QUẢ HOẠT ĐỘNG CỦA ĐVKD (HDBank)
# Phiên bản Python thường (pandas + scikit-learn), không dùng Spark.
# Biến mục tiêu: xep_loai = Chưa hoàn thành (0) / Hoàn thành (1) / Hoàn thành xuất sắc (2)
# Thuật toán   : Random Forest (chính), Logistic Regression (so sánh), Baseline
# ==============================================================
import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, auc, f1_score, balanced_accuracy_score,
                             accuracy_score, precision_score, recall_score)

warnings.filterwarnings("ignore")

# CẤU HÌNH CHUNG & KIỂM TRA FILE DỮ LIỆU

DATA  = "survey-dicore-giaodichtaiquay.xlsx"
LABEL = "xep_loai.xlsx"
MEDIA = "media"
SEED, K = 42, 5
CLASSES = ["Chưa hoàn thành", "Hoàn thành", "Hoàn thành xuất sắc"]
os.makedirs(MEDIA, exist_ok=True)

missing_files = [f for f in [DATA, LABEL] if not os.path.exists(f)]
if missing_files:
    raise FileNotFoundError(
        f"\n[LỖI THIẾU FILE] Không tìm thấy file: {missing_files}\n"
        f"Hãy đặt các file {', '.join([DATA, LABEL])} cùng thư mục với file .py này rồi chạy lại."
    )

print("=" * 60)
print("BÀI TOÁN: DỰ ĐOÁN XẾP LOẠI HIỆU QUẢ HOẠT ĐỘNG CỦA ĐVKD")
print("=" * 60)

# BƯỚC 1: NẠP DỮ LIỆU

print("\n[BƯỚC 1] Nạp dữ liệu...")
raw = pd.read_excel(DATA, sheet_name="Sheet1")
raw.columns = raw.columns.astype(str).str.replace("\xa0", " ", regex=False).str.strip()
for c in ["Tên ĐVKD", "Loại hình ĐVKD"]:
    raw[c] = raw[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
import unicodedata
def _norm(s):
    return unicodedata.normalize("NFC", str(s)).replace("\xa0", " ").strip()

lab = pd.read_excel(LABEL)
print("Các cột trong file xếp loại:", [repr(c) for c in lab.columns])
lab.columns = [_norm(c) for c in lab.columns]
# Tự dò cột tên ĐVKD và cột xếp loại (phòng khi tên cột khác nhẹ so với dự kiến)
def _key(s):   # bỏ dấu + hạ chữ thường + bỏ khoảng trắng/gạch dưới để so khớp tên cột
    s = unicodedata.normalize("NFD", str(s)).replace("đ", "d").replace("Đ", "D")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"[\s_]+", "", s).lower()

col_name = next(c for c in lab.columns if _key(c) == "tendvkd")
col_rank = next((c for c in lab.columns if _key(c) in ("xeploai", "xep_loai")), None)
if col_rank is None:
    col_rank = next(c for c in lab.columns if "xep" in _key(c) and "loaihinh" not in _key(c))
print(f"Dùng cột tên: {col_name!r} | cột xếp loại: {col_rank!r}")
lab = lab[[col_name, col_rank]].rename(columns={col_name: "Tên ĐVKD", col_rank: "Xếp loại"})
lab["Tên ĐVKD"] = lab["Tên ĐVKD"].map(_norm)
lab["Xếp loại"] = lab["Xếp loại"].map(_norm)
raw["Tên ĐVKD"] = raw["Tên ĐVKD"].map(_norm)
print("Các giá trị xếp loại trong file:", lab["Xếp loại"].unique().tolist())
print(f"khảo sát : {raw.shape[0]:,} dòng, {raw.shape[1]} cột | {raw['Tên ĐVKD'].nunique()} ĐVKD")
print(f"xếp loại : {len(lab):,} dòng")

# BƯỚC 2: TÍCH HỢP VÀ TIỀN XỬ LÝ DỮ LIỆU

print("\n[BƯỚC 2] Tích hợp và tiền xử lý dữ liệu...")

# --- 2.1: Bỏ trùng (1 người khảo sát nhiều lần: giữ bản mới nhất) ---
ma_nv = [c for c in raw.columns if c.startswith("Mã NV")][0]
df0 = (raw.sort_values("Completion time").drop_duplicates(ma_nv, keep="last").reset_index(drop=True))
print(f"Sau khi bỏ trùng mã NV: {len(df0):,} dòng")

# --- 2.2: Nhận diện 3 loại cột nghiệp vụ theo nội dung ---
biz = [c for c in df0.columns if re.match(r"^\d+\.\s", c)]
def ctype(s):
    v = " ".join(s.dropna().astype(str).unique())
    return ("time" if "phút" in v else "result" if "GD thực hiện" in v
            else "error" if ("Lỗi" in v or "Không phát sinh lỗi" in v) else "unk")
ct = {c: ctype(df0[c]) for c in biz}
num_of = lambda c: int(re.match(r"^(\d+)\.", c).group(1))
GROUPS = {"cif_tk": range(1, 11), "tien": range(11, 18), "quy": range(18, 22),
          "tsdb": range(22, 28), "tin_dung": range(28, 48)}
grp = lambda c: next(g for g, r in GROUPS.items() if num_of(c) in r)
TM = [("<5", 1), (">5 - 15", 2), (">15 - 30", 3), (">30", 4)]
def tscore(x):
    # Chuẩn hóa khoảng trắng và chữ trước khi ánh xạ nhãn thời gian.
    # Thứ tự kiểm tra giúp phân biệt đúng các khoảng thời gian.
    s = re.sub(r"\s+", " ", str(x)).strip().lower()
    if s in {"<5", "< 5", "dưới 5 phút", "duoi 5 phut"}:
        return 1
    if re.match(r"^(>\s*5|trên 5|tren 5).*15", s):
        return 2
    if re.match(r"^(>\s*15|trên 15|tren 15).*30", s):
        return 3
    if s.startswith((">30", "> 30", "trên 30", "tren 30")):
        return 4
    return np.nan

# --- 2.3: Feature Engineering (gộp 47 nghiệp vụ thành 5 nhóm x 5 chỉ số) ---
#   so_nv_phat_sinh : quy mô hoạt động (số nghiệp vụ đã phát sinh)
#   ty_le_tu_lam    : % giao dịch ĐVKD tự thao tác, không cần hỗ trợ
#   ty_le_cntt      : % giao dịch cần CNTT hỗ trợ
#   thoi_gian_tb    : thời gian xử lý trung bình (1 = <5 phút ... 4 = >30 phút)
#   ty_le_loi       : % nghiệp vụ có phát sinh lỗi
# "Chưa phát sinh" = không áp dụng -> loại khỏi mẫu số, để trống (NaN)
feat = pd.DataFrame(index=df0.index)
for g in GROUPS:
    R = [c for c in biz if ct[c] == "result" and grp(c) == g]
    T = [c for c in biz if ct[c] == "time" and grp(c) == g]
    E = [c for c in biz if ct[c] == "error" and grp(c) == g]
    Rs = df0[R].astype(str); app = ~Rs.apply(lambda s: s.str.contains("Chưa phát sinh"))
    n = app.sum(axis=1).replace(0, np.nan)
    feat[f"{g}_so_nv_phat_sinh"] = app.sum(axis=1)
    feat[f"{g}_ty_le_tu_lam"] = Rs.apply(lambda s: s.str.contains("ĐVKD tự thao tác")).sum(axis=1) / n
    feat[f"{g}_ty_le_cntt"] = Rs.apply(lambda s: s.str.contains("CNTT")).sum(axis=1) / n
    feat[f"{g}_thoi_gian_tb"] = df0[T].apply(lambda s: s.map(tscore)).mean(axis=1)
    Es = df0[E].astype(str); ans = ~Es.apply(lambda s: s.str.contains("Chưa phát sinh"))
    feat[f"{g}_ty_le_loi"] = Es.apply(lambda s: s.str.startswith("Lỗi")).sum(axis=1) / ans.sum(axis=1).replace(0, np.nan)
num_cols = list(feat.columns)

# --- 2.4: Gộp theo ĐVKD (mỗi ĐVKD 1 dòng = 1 mẫu) ---
feat["don_vi"] = df0["Tên ĐVKD"]
unit = feat.groupby("don_vi")[num_cols].mean()
unit["so_nguoi_khao_sat"] = feat.groupby("don_vi").size()
unit["loai_hinh"] = df0.groupby("Tên ĐVKD")["Loại hình ĐVKD"].agg(lambda s: s.mode().iloc[0])
num_cols = num_cols + ["so_nguoi_khao_sat"]

# --- 2.5: Ghép nhãn xếp loại ---
# Nếu một ĐVKD có nhiều nhãn khác nhau, dừng để kiểm tra thay vì tự giữ một nhãn.
label_conflicts = (lab.groupby("Tên ĐVKD")["Xếp loại"].nunique(dropna=True) > 1)
if label_conflicts.any():
    names = label_conflicts[label_conflicts].index.tolist()
    raise ValueError(f"ĐVKD có nhiều nhãn xếp loại khác nhau: {names[:20]}")
lab = lab.drop_duplicates("Tên ĐVKD").set_index("Tên ĐVKD")["Xếp loại"]
unit["xep_loai"] = unit.index.map(lab)
unit = unit.dropna(subset=["xep_loai"]).reset_index()
LABEL_MAP = {"chuahoanthanh": 0, "hoanthanh": 1, "hoanthanhxuatsac": 2,
             "0": 0, "1": 1, "2": 2, "0.0": 0, "1.0": 1, "2.0": 2}
unit["label"] = unit["xep_loai"].map(lambda v: LABEL_MAP.get(_key(v)))
bad = unit.loc[unit["label"].isna(), "xep_loai"].unique().tolist()
if bad:
    raise ValueError(f"Giá trị xếp loại không nhận diện được: {bad}. "
                     f"Hãy thêm vào LABEL_MAP (các giá trị hiện có: {unit['xep_loai'].unique().tolist()})")
unit["label"] = unit["label"].astype(int)
unit["xep_loai"] = unit["label"].map(dict(enumerate(CLASSES)))   # tên lớp chuẩn để in/vẽ

# --- 2.6: Chia fold phân tầng (giữ tỉ lệ 3 lớp ở mọi fold) ---
unit["fold"] = -1
for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED).split(unit, unit["label"])):
    unit.loc[te, "fold"] = k

# --- 2.7: Thống kê mất cân bằng lớp ---
# Trọng số lớp sẽ được tính riêng trên tập huấn luyện của từng fold,
# không tính từ toàn bộ dữ liệu để tránh dùng thông tin fold đánh giá.
cnt = unit["label"].value_counts()
total = len(unit)
print(f"\nSố mẫu (ĐVKD): {total:,}")
print("Phân phối nhãn:")
for name, n_ in unit["xep_loai"].value_counts().items():
    print(f"  {name:22s}: {n_:4d} ({n_/total*100:.1f}%)")
print("Lưu ý: trọng số lớp được tính riêng từ tập huấn luyện trong từng fold.")
print("Tỉ lệ thiếu (NaN) cao nhất:")
miss = {c: unit[c].isna().mean() for c in num_cols}
for c, v in sorted(miss.items(), key=lambda x: -x[1])[:4]:
    print(f"  {c:28s}: {v*100:.1f}%")

# BƯỚC 3: XÂY DỰNG PIPELINE scikit-learn

print("\n[BƯỚC 3] Xây dựng Pipeline scikit-learn...")

def build_pipeline(kind="rf", n_train=None):
    # 3.1 Imputer: điền giá trị thiếu bằng trung vị
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    # 3.2 StandardScaler: chuẩn hóa về cùng thang đo
    scaler = StandardScaler()
    # 3.3 Mô hình
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3,
                                     max_features="sqrt", random_state=SEED, n_jobs=-1)
    else:
        # Spark: regParam=0.01 (tính trên loss trung bình)  <->  sklearn: C = 1 / (regParam * n)
        clf = LogisticRegression(C=1.0 / (0.01 * n_train), max_iter=200)
    return Pipeline([("imputer", imputer), ("scaler", scaler), ("clf", clf)])

# BƯỚC 4: HUẤN LUYỆN (5-fold cross-validation phân tầng)

print(f"\n[BƯỚC 4] Huấn luyện bằng {K}-fold cross-validation (mỗi ĐVKD được dự đoán bởi mô hình chưa thấy nó)...")

def cross_val(cols, kind):
    parts = []
    for k in range(K):
        tr, te = unit[unit["fold"] != k], unit[unit["fold"] == k]
        model = build_pipeline(kind, len(tr))

        # Tính class weight CHỈ trên tập huấn luyện của fold hiện tại.
        train_counts = tr["label"].value_counts()
        n_train, n_classes = len(tr), tr["label"].nunique()
        class_weights = {
            int(cls): n_train / (n_classes * int(count))
            for cls, count in train_counts.items()
        }
        sample_weight = tr["label"].map(class_weights).to_numpy()
        model.fit(tr[cols], tr["label"], clf__sample_weight=sample_weight)
        proba = model.predict_proba(te[cols])
        pred = model.predict(te[cols])
        parts.append(pd.DataFrame({
            "don_vi": te["don_vi"].values,
            "label": te["label"].values,
            "prediction": pred,
            "prob": list(proba),
            "fold": k,
        }))
    return pd.concat(parts, ignore_index=True)


def cross_val_baseline():
    # Baseline được xác định bằng lớp đa số của tập huấn luyện ở từng fold.
    parts = []
    for k in range(K):
        tr, te = unit[unit["fold"] != k], unit[unit["fold"] == k]
        majority_class = int(tr["label"].value_counts().idxmax())
        parts.append(pd.DataFrame({
            "don_vi": te["don_vi"].values,
            "label": te["label"].values,
            "prediction": majority_class,
            "fold": k,
        }))
    return pd.concat(parts, ignore_index=True)

oof = {}
oof["Random Forest"] = cross_val(num_cols, "rf")
oof["Logistic Regression"] = cross_val(num_cols, "lr")
oof["Baseline (lớp đông nhất trong tập huấn luyện)"] = cross_val_baseline()
print("Huấn luyện hoàn tất!")

# BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH

print("\n[BƯỚC 5] Đánh giá mô hình...")
rows = []
for name, o in oof.items():
    y, p = o["label"], o["prediction"]
    rows.append((name, accuracy_score(y, p), balanced_accuracy_score(y, p), f1_score(y, p, average="macro")))
res = pd.DataFrame(rows, columns=["Mô hình", "Accuracy", "Balanced acc", "F1-macro"])
print("\n" + "=" * 62)
print(f"  SO SÁNH MÔ HÌNH (out-of-fold, {total} ĐVKD)")
print("=" * 62)
print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("=" * 62)
best = res[~res["Mô hình"].str.startswith("Baseline")].sort_values("F1-macro", ascending=False).iloc[0]["Mô hình"]
print(f"Mô hình tốt nhất theo F1-macro: {best}")

# Chỉ số trung bình có trọng số theo lớp (tương đương Spark MulticlassClassificationEvaluator)
yb, pb = oof[best]["label"], oof[best]["prediction"]
print(f"\n--- {best} (trung bình có trọng số theo lớp) ---")
print(f"  Accuracy : {accuracy_score(yb, pb):.4f}")
print(f"  F1       : {f1_score(yb, pb, average='weighted'):.4f}")
print(f"  Precision: {precision_score(yb, pb, average='weighted', zero_division=0):.4f}")
print(f"  Recall   : {recall_score(yb, pb, average='weighted', zero_division=0):.4f}")

# Kiểm tra rò rỉ nhãn: chỉ dùng feature quy mô (không tham gia tạo nhãn)
scale_cols = [c for c in num_cols if "so_nv_phat_sinh" in c or c == "so_nguoi_khao_sat"]
ab = cross_val(scale_cols, "rf")
print(f"\nKiểm tra đối chứng (chỉ feature quy mô): Accuracy={accuracy_score(ab['label'], ab['prediction']):.3f}  "
      f"F1-macro={f1_score(ab['label'], ab['prediction'], average='macro'):.3f}")
print("  -> Đây là kiểm tra đối chứng về đặc trưng quy mô; kết quả gần Baseline không tự nó chứng minh có rò rỉ nhãn.")

# BƯỚC 6: TRỰC QUAN HÓA

print("\n[BƯỚC 6] Vẽ đồ thị...")
o = oof[best]
y_true, y_pred = o["label"].astype(int).values, o["prediction"].astype(int).values
P = np.vstack(o["prob"].values)

# --- 6.0: Phân phối nhãn mục tiêu ---
label_counts = unit["xep_loai"].value_counts().reindex(CLASSES, fill_value=0)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(label_counts.index, label_counts.values)
for b, n_ in zip(bars, label_counts.values):
    pct = n_ / len(unit) * 100
    ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{n_} ({pct:.1f}%)", ha="center", va="bottom")
ax.set_ylabel("Số lượng ĐVKD")
ax.set_title("Phân phối nhãn xếp loại trong bộ dữ liệu")
ax.tick_params(axis="x", rotation=10)
plt.tight_layout(); plt.savefig(f"{MEDIA}/fig_00_label_distribution.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved: fig_00_label_distribution.png")

# --- 6.0b: Sơ đồ quy trình đánh giá mô hình ---
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.axis("off")
flow = [
    (0.02, "Dữ liệu khảo sát\n+ nhãn ĐVKD"),
    (0.22, "Tiền xử lý\n+ tạo đặc trưng"),
    (0.42, "Stratified 5-Fold\n4 fold train / 1 fold đánh giá"),
    (0.65, "Pipeline + mô hình\nRF / Logistic Regression"),
    (0.84, "OOF dự đoán\n+ chỉ số đánh giá"),
]
for x, label in flow:
    ax.text(x, 0.52, label, transform=ax.transAxes, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray"))
for i in range(len(flow)-1):
    x1 = flow[i][0] + 0.075
    x2 = flow[i+1][0] - 0.075
    ax.annotate("", xy=(x2, 0.52), xytext=(x1, 0.52), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax.set_title("Quy trình xây dựng và đánh giá mô hình phân loại ĐVKD", pad=12)
plt.tight_layout(); plt.savefig(f"{MEDIA}/fig_00_model_workflow.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved: fig_00_model_workflow.png")

# --- 6.0c: So sánh các mô hình theo ba chỉ số ---
metric_cols = ["Accuracy", "Balanced acc", "F1-macro"]
plot_res = res.set_index("Mô hình")[metric_cols]
ax = plot_res.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=10)
ax.set_ylabel("Giá trị chỉ số")
ax.set_title("So sánh Random Forest, Logistic Regression và Baseline")
ax.legend(title="Chỉ số", loc="lower right")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{MEDIA}/fig_01_model_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved: fig_01_model_comparison.png")

# --- 6.1: Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=CLASSES, yticklabels=CLASSES)
ax.set_title(f"Confusion Matrix – {best} (Xếp loại ĐVKD)", fontsize=13, fontweight="bold")
ax.set_xlabel("Dự đoán (Predicted)"); ax.set_ylabel("Thực tế (Actual)")
plt.tight_layout(); plt.savefig(f"{MEDIA}/fig_01a_confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved: fig_01a_confusion_matrix.png")

# --- 6.2: ROC Curve (One-vs-Rest cho 3 lớp) ---
fig, ax = plt.subplots(figsize=(7, 5))
for i, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), P[:, i])
    ax.plot(fpr, tpr, lw=2, label=f"{c} (AUC = {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Ngẫu nhiên (AUC = 0.50)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC One-vs-Rest – {best}", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{MEDIA}/fig_01b_roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved: fig_01b_roc_curve.png")

# --- 6.3: Feature Importance (Random Forest huấn luyện trên toàn bộ dữ liệu) ---
rf_full = build_pipeline("rf")
full_counts = unit["label"].value_counts()
full_weights = {int(cls): len(unit) / (unit["label"].nunique() * int(count))
                for cls, count in full_counts.items()}
full_sample_weight = unit["label"].map(full_weights).to_numpy()
rf_full.fit(unit[num_cols], unit["label"], clf__sample_weight=full_sample_weight)
imp = rf_full.named_steps["clf"].feature_importances_
top = sorted(zip(num_cols, imp), key=lambda x: -x[1])[:15]
names, vals = [t[0] for t in top][::-1], [t[1] for t in top][::-1]
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(names, vals, color=sns.color_palette("viridis", len(names))[::-1])
for b, v in zip(bars, vals):
    ax.text(b.get_width() + 0.001, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlabel("Importance Score"); ax.grid(axis="x", alpha=0.3)
ax.set_title("Top 15 Feature Importances – Random Forest (Xếp loại ĐVKD)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{MEDIA}/fig_01c_feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved: fig_01c_feature_importance.png")

# --- 6.4: Phân phối xác suất "Chưa hoàn thành" theo nhãn thực tế ---
fig, ax = plt.subplots(figsize=(8, 5))
for i, (c, col) in enumerate(zip(CLASSES, ["tomato", "orange", "steelblue"])):
    ax.hist(P[y_true == i, 0], bins=20, alpha=0.6, color=col, density=True, label=f"{c} (thực tế)")
ax.set_xlabel("Xác suất dự đoán là 'Chưa hoàn thành'"); ax.set_ylabel("Density")
ax.set_title("Phân phối xác suất dự đoán theo nhãn thực tế", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{MEDIA}/fig_01d_prob_distribution.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved: fig_01d_prob_distribution.png")

# --- Báo cáo chi tiết theo lớp ---
from sklearn.metrics import classification_report
report_by_class = classification_report(
    y_true, y_pred, labels=[0, 1, 2], target_names=CLASSES,
    output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_by_class).transpose().reset_index().rename(columns={"index": "Lớp"})
report_df.to_excel(f"{MEDIA}/chi_so_theo_lop.xlsx", index=False)
# Biểu đồ Precision, Recall, F1-score theo từng lớp
class_metrics = report_df[report_df["Lớp"].isin(CLASSES)].set_index("Lớp")[["precision", "recall", "f1-score"]]
ax = class_metrics.plot(kind="bar", figsize=(9, 5), ylim=(0, 1), rot=10)
ax.set_ylabel("Giá trị chỉ số")
ax.set_title(f"Precision, Recall và F1-score theo lớp – {best}")
ax.legend(title="Chỉ số", loc="lower right")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{MEDIA}/fig_01e_per_class_metrics.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved: fig_01e_per_class_metrics.png")
print("Đã lưu Precision/Recall/F1 theo lớp: chi_so_theo_lop.xlsx")

# --- Lưu kết quả dự đoán ---
out = o.copy()
out["thuc_te"] = out["label"].astype(int).map(dict(enumerate(CLASSES)))
out["du_doan"] = out["prediction"].astype(int).map(dict(enumerate(CLASSES)))
out[["don_vi", "thuc_te", "du_doan"]].to_excel(f"{MEDIA}/ket_qua_du_doan.xlsx", index=False)
res.to_excel(f"{MEDIA}/so_sanh_mo_hinh.xlsx", index=False)
print("Saved: ket_qua_du_doan.xlsx, so_sanh_mo_hinh.xlsx, chi_so_theo_lop.xlsx")

print("\n[HOÀN TẤT] Dự đoán xếp loại hiệu quả hoạt động ĐVKD")