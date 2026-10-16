import io
import glob
import os
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ======================================================================
# KONFIGURASI HALAMAN
# ======================================================================
st.set_page_config(
    page_title="UTS BI - Liliyosa Jingga Luahane",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS KUSTOM INTERAKTIF (PINK HATI & KUPU-KUPU) ---
st.markdown("""
    <style>
    .main, .stApp {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%);
    }
    [data-testid="stSidebar"], .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FFC0CB 0%, #FFD6E0 100%);
    }
    h1, h2, h3 {
        color: #C71585;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h4, h5 { color: #8B008B; }
    .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
        background: linear-gradient(45deg, #C71585, #DB7093);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 10px 28px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(199, 21, 133, 0.3);
        transition: 0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {
        background: linear-gradient(45deg, #900C3F, #C71585);
        color: #FFF0F5;
        box-shadow: 0 6px 15px rgba(144, 12, 63, 0.4);
        transform: translateY(-2px);
    }
    .stButton>button:active, .stDownloadButton>button:active { transform: scale(0.97); }
    .stButton>button p, .stDownloadButton>button p, .stFormSubmitButton>button p { color: white; }
    .card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 6px 15px rgba(199, 21, 133, 0.12);
        margin-bottom: 20px;
        border-left: 6px solid #C71585;
        transition: 0.3s;
    }
    .card:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(199, 21, 133, 0.2); }
    .card-soft {
        background: linear-gradient(135deg, #ffffff 0%, #FFF5EE 100%);
        padding: 16px 20px;
        border-radius: 15px;
        border: 2px dashed #FFC0CB;
        margin-bottom: 14px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #FFF5EE 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(199, 21, 133, 0.1);
        text-align: center;
        border: 2px solid #FFC0CB;
        transition: 0.3s;
    }
    .kpi-card:hover { transform: translateY(-4px) scale(1.02); box-shadow: 0 8px 20px rgba(199, 21, 133, 0.25); border-color: #DB7093; }
    .kpi-title {
        color: #8B008B;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .kpi-value {
        color: #C71585;
        font-size: 24px;
        font-weight: 800;
    }
    .kpi-sub { color: #DB7093; font-size: 12px; margin-top: 2px; }
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-right: 6px;
    }
    .badge-fact { background: #E6F9EE; color: #1B7F46; border: 1px solid #1B7F46; }
    .badge-est { background: #FFF4DB; color: #A66A00; border: 1px solid #A66A00; }
    .badge-act { background: #EAF0FF; color: #2B4DB5; border: 1px solid #2B4DB5; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #FFE4EC; border-radius: 14px 14px 0 0; padding: 8px 16px; color: #8B008B; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background: linear-gradient(45deg, #C71585, #DB7093); color: white !important; }
    div[data-testid="stMetric"] {
        background: white; border-radius: 14px; padding: 10px 14px; border: 2px solid #FFC0CB;
        box-shadow: 0 3px 8px rgba(199,21,133,0.08);
    }
    div[data-testid="stExpander"] { background: rgba(255,255,255,0.7); border-radius: 14px; border: 1px solid #FFC0CB; }
    </style>
""", unsafe_allow_html=True)

PINK = ['#C71585', '#DB7093', '#FF69B4', '#8B008B', '#FFB6C1', '#BA55D3', '#900C3F', '#E75480']
UNKNOWN = "Tidak Diketahui"


# ======================================================================
# HELPER TAMPILAN
# ======================================================================
def fmt_idr(val):
    """Format angka ribuan dengan titik (.)"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Rp 0"
    return f"Rp {val:,.0f}".replace(",", ".")


def fmt_int(val):
    return f"{val:,.0f}".replace(",", ".")


def card(html):
    st.markdown(f"<div class='card'>{html}</div>", unsafe_allow_html=True)


def soft_card(html):
    st.markdown(f"<div class='card-soft'>{html}</div>", unsafe_allow_html=True)


def kpi(title, value, sub=""):
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>{title}</div>"
        f"<div class='kpi-value'>{value}</div>{sub_html}</div>",
        unsafe_allow_html=True)


def show_df(df, height=None, **kw):
    """st.dataframe yang kompatibel dengan versi Streamlit lama maupun baru."""
    args = dict(hide_index=True)
    if height:
        args["height"] = height
    args.update(kw)
    try:
        st.dataframe(df, width="stretch", **args)
    except TypeError:
        st.dataframe(df, use_container_width=True, **args)


def show_plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def style_fig(fig, h=380, title=None):
    fig.update_layout(
        template="plotly_white", height=h, colorway=PINK,
        font=dict(family="Helvetica Neue, sans-serif", color="#5a1040"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.7)",
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        hoverlabel=dict(bgcolor="#FFF0F5", font_color="#8B008B"),
        legend=dict(bgcolor="rgba(255,255,255,0.6)"),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color="#C71585", size=16)))
    return fig


def dl_csv(df, label, fname, key):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=fname,
                       mime="text/csv", key=key)


# ======================================================================
# ====================  FUNGSI ANALISIS (BACKEND)  =====================
# ======================================================================
@st.cache_data(show_spinner="📖 Membaca file Excel...")
def read_workbook(file_bytes):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)


def build_catalog(raw, prod):
    """Katalog produk dari sheet Data_Produk; fallback dari Data_Raw bila sheet tidak ada."""
    if prod is not None and {"Product_ID", "Product_Name", "Category", "Reference_Unit_Price_Rp"} <= set(prod.columns):
        return prod[["Product_ID", "Product_Name", "Category", "Reference_Unit_Price_Rp"]].copy()
    g = raw.dropna(subset=["Product_ID", "Product_Name"]).groupby("Product_ID")
    return pd.DataFrame({
        "Product_ID": g.size().index,
        "Product_Name": g["Product_Name"].agg(lambda s: s.mode().iloc[0]).values,
        "Category": g["Category"].agg(lambda s: str(s.mode().iloc[0]).strip().title()).values,
        "Reference_Unit_Price_Rp": g["Unit_Price"].median().values,
    })


@st.cache_data(show_spinner="🔍 Mengaudit kualitas data mentah...")
def audit_raw(raw, prod):
    cat = build_catalog(raw, prod)
    out = {}
    miss = raw.isna().sum()
    out["missing"] = pd.DataFrame({
        "Kolom": miss.index, "Jumlah Missing": miss.values,
        "Persentase (%)": (miss.values / len(raw) * 100).round(3)
    })
    out["dup_exact"] = int(raw.duplicated().sum())

    def variants(col):
        s = raw[col].dropna().astype(str)
        vc = s.value_counts()
        norm = s.str.strip().str.lower()
        groups = s.groupby(norm).nunique()
        bad = groups[groups > 1].index
        rows = vc[vc.index.map(lambda v: v.strip().lower() in set(bad))]
        return rows.reset_index().rename(columns={"index": col, col: "Nilai", "count": "Jumlah"}) if len(rows) else pd.DataFrame()

    inc = []
    for col in ["Category", "Payment_Method"]:
        s = raw[col].dropna().astype(str)
        canon = s.str.strip().str.lower()
        for key, sub in s.groupby(canon):
            vals = sub.value_counts()
            if len(vals) > 1:
                inc.append({"Kolom": col, "Kelompok": key,
                            "Variasi Penulisan": " | ".join([f"'{v}' ({n})" for v, n in vals.items()]),
                            "Baris Terdampak": int(len(sub) - vals.max())})
    out["inconsistent"] = pd.DataFrame(inc)

    price_ref = raw["Product_ID"].map(cat.set_index("Product_ID")["Reference_Unit_Price_Rp"])
    q = raw["Quantity"]
    up = raw["Unit_Price"]
    expected = raw["Quantity"] * raw["Unit_Price"] * (1 - raw["Discount_Rate"])
    out["outlier"] = pd.DataFrame([
        {"Pemeriksaan": "Quantity ≤ 0", "Jumlah Baris": int((q <= 0).sum())},
        {"Pemeriksaan": "Quantity > 50 (tidak wajar)", "Jumlah Baris": int((q > 50).sum())},
        {"Pemeriksaan": "Unit_Price ≤ 0", "Jumlah Baris": int((up <= 0).sum())},
        {"Pemeriksaan": "Unit_Price > 3× harga referensi", "Jumlah Baris": int((up > price_ref * 3).sum())},
        {"Pemeriksaan": "Line_Total ≠ Qty × Harga × (1 − Diskon)", "Jumlah Baris": int(((expected - raw["Line_Total"]).abs() > 1).sum())},
    ])
    return out


@st.cache_data(show_spinner="🧹 Membersihkan data (Data_Clean)...")
def preprocess(raw, prod, cust):
    """Membuat Data_Clean dari salinan Data_Raw (Data_Raw tidak ditimpa)."""
    log = []
    d = raw.copy()
    cat = build_catalog(raw, prod)

    def add(masalah, n, keputusan, before, after):
        log.append({"Masalah": masalah, "Baris Terdampak": int(n), "Keputusan": keputusan,
                    "Baris Sebelum": int(before), "Baris Sesudah": int(after)})

    # 1. Format tanggal & ID transaksi
    d["Transaction_Date"] = pd.to_datetime(d["Transaction_Date"], errors="coerce")
    b = len(d)
    bad = d["Transaction_Date"].isna() | d["Transaction_ID"].isna()
    d = d[~bad].copy()
    add("Format tanggal / Transaction_ID tidak valid", bad.sum(), "Dikonversi ke datetime; baris tidak valid dihapus", b, len(d))

    # 2. Teks tidak konsisten (kapital / spasi)
    text_cols = ["Transaction_ID", "Customer_ID", "Region", "Customer_Type", "Product_ID",
                 "Product_Name", "Category", "Payment_Method"]
    for c in text_cols:
        s = d[c].astype("object")
        cleaned = s.where(s.isna(), s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
        d[c] = cleaned.replace("", np.nan)
    cat_map = {c.lower(): c for c in cat["Category"].unique()}
    before = d["Category"].copy()
    d["Category"] = d["Category"].str.lower().map(cat_map).fillna(d["Category"])
    n_cat = int((before != d["Category"]).sum())
    pay_map = {"qris": "QRIS", "tunai": "Tunai", "debit": "Debit", "transfer": "Transfer"}
    before = d["Payment_Method"].copy()
    d["Payment_Method"] = d["Payment_Method"].str.lower().map(pay_map).fillna(d["Payment_Method"])
    n_pay = int((before.fillna("") != d["Payment_Method"].fillna("")).sum())
    add("Inkonsistensi teks Category (kapital/spasi)", n_cat, "Diperbaiki: distandarkan sesuai katalog", len(d), len(d))
    add("Inkonsistensi teks Payment_Method (ejaan/kapital/spasi)", n_pay, "Diperbaiki: distandarkan (QRIS, Tunai, Debit, Transfer)", len(d), len(d))

    # 3. Product_ID kosong & konsistensi Product_ID - Product_Name - Category
    id_by_name = cat.set_index("Product_Name")["Product_ID"]
    name_by_id = cat.set_index("Product_ID")["Product_Name"]
    cat_by_id = cat.set_index("Product_ID")["Category"]
    miss = d["Product_ID"].isna()
    d.loc[miss, "Product_ID"] = d.loc[miss, "Product_Name"].map(id_by_name)
    n_pid = int(miss.sum())
    d["Flag_Data"] = ""
    d.loc[miss & d["Product_ID"].notna(), "Flag_Data"] += "Product_ID diimputasi dari Product_Name; "
    b = len(d)
    nopid = d["Product_ID"].isna()
    d = d[~nopid].copy()
    add("Product_ID kosong", n_pid, "Diimputasi dari katalog berdasarkan Product_Name (sisa yang tidak dapat dipetakan dihapus)", b, len(d))
    mism = (d["Product_Name"] != d["Product_ID"].map(name_by_id)) | (d["Category"] != d["Product_ID"].map(cat_by_id))
    d["Product_Name"] = d["Product_ID"].map(name_by_id)
    d["Category"] = d["Product_ID"].map(cat_by_id)
    add("Product_Name / Category tidak sesuai katalog", mism.sum(), "Diperbaiki: mengikuti Data_Produk", len(d), len(d))

    # 4. Customer_ID, Region, Customer_Type kosong
    n_c0 = int(d["Customer_ID"].isna().sum())
    first_c = d.groupby("Transaction_ID")["Customer_ID"].transform("first")
    d["Customer_ID"] = d["Customer_ID"].fillna(first_c)
    n_c1 = int(d["Customer_ID"].isna().sum())
    add("Customer_ID kosong", n_c0,
        f"Diimputasi dari baris lain pada Transaction_ID yang sama ({n_c0 - n_c1} baris pulih); sisa {n_c1} baris dipertahankan untuk analisis penjualan tetapi DIKELUARKAN dari RFM/K-Means",
        len(d), len(d))
    if cust is not None and {"Customer_ID", "Region", "Customer_Type"} <= set(cust.columns):
        cm = cust.drop_duplicates("Customer_ID").set_index("Customer_ID")
    else:
        cm = None
    for c in ["Region", "Customer_Type"]:
        n0 = int(d[c].isna().sum())
        d[c] = d[c].fillna(d.groupby("Transaction_ID")[c].transform("first"))
        if cm is not None:
            d[c] = d[c].fillna(d["Customer_ID"].map(cm[c]))
        n1 = int(d[c].isna().sum())
        d[c] = d[c].fillna(UNKNOWN)
        add(f"{c} kosong", n0, f"Diimputasi dari transaksi yang sama / Data_Pelanggan; sisa {n1} baris diberi label '{UNKNOWN}'", len(d), len(d))
    n_p = int(d["Payment_Method"].isna().sum())
    d["Payment_Method"] = d["Payment_Method"].fillna(UNKNOWN)
    add("Payment_Method kosong", n_p, f"Ditandai '{UNKNOWN}' (metode bayar berbeda antar baris sehingga tidak dapat diimputasi)", len(d), len(d))

    # 5. Duplikasi
    b = len(d)
    d = d.drop_duplicates(subset=[c for c in d.columns if c != "Flag_Data"])
    add("Duplikasi baris persis sama", b - len(d), "Dihapus (dipertahankan 1 baris)", b, len(d))
    b = len(d)
    d["_k"] = d[["Quantity", "Unit_Price", "Line_Total"]].isna().sum(axis=1) + (d["Payment_Method"] == UNKNOWN)
    d = d.sort_values("_k", kind="stable").drop_duplicates(["Transaction_ID", "Product_ID"]).sort_index().drop(columns="_k")
    add("Duplikasi produk dalam 1 Transaction_ID (beda detail minor)", b - len(d),
        "Dihapus; dipertahankan baris dengan data paling lengkap", b, len(d))

    # 6. Unit_Price tidak valid
    ref = d["Product_ID"].map(cat.set_index("Product_ID")["Reference_Unit_Price_Rp"])
    bad_p = d["Unit_Price"].isna() | (d["Unit_Price"] <= 0) | (d["Unit_Price"] > ref * 3) | (d["Unit_Price"] < ref / 3)
    d.loc[bad_p, "Unit_Price"] = ref[bad_p]
    d.loc[bad_p, "Flag_Data"] += "Unit_Price diganti harga referensi; "
    add("Unit_Price kosong / ≤0 / tidak wajar (mis. 999.999)", bad_p.sum(), "Diperbaiki: diganti harga referensi katalog", len(d), len(d))

    # 7. Discount_Rate
    bad_d = d["Discount_Rate"].isna() | (d["Discount_Rate"] < 0) | (d["Discount_Rate"] > 0.5)
    d.loc[bad_d, "Discount_Rate"] = 0.0
    add("Discount_Rate di luar rentang 0–50%", bad_d.sum(), "Diperbaiki: diganti 0", len(d), len(d))

    # 8. Quantity tidak valid -> pulihkan dari Line_Total bila konsisten
    net_unit = d["Unit_Price"] * (1 - d["Discount_Rate"])
    bad_q = d["Quantity"].isna() | (d["Quantity"] <= 0) | (d["Quantity"] > 50) | (d["Quantity"] % 1 != 0)
    derived = (d["Line_Total"] / net_unit).round()
    ok = derived.between(1, 50) & ((derived * net_unit - d["Line_Total"]).abs() <= 1)
    fix = bad_q & ok
    d.loc[fix, "Quantity"] = derived[fix]
    d.loc[fix, "Flag_Data"] += "Quantity dipulihkan dari Line_Total; "
    b = len(d)
    drop = bad_q & ~ok
    d = d[~drop].copy()
    add("Quantity kosong / ≤0 / 999 (outlier tidak valid)", bad_q.sum(),
        f"{int(fix.sum())} baris diperbaiki (Quantity = Line_Total ÷ harga bersih), {int(drop.sum())} baris dihapus karena tidak dapat dipulihkan", b, len(d))
    d["Quantity"] = d["Quantity"].astype(int)

    # 9. Line_Total
    exp = d["Quantity"] * d["Unit_Price"] * (1 - d["Discount_Rate"])
    bad_l = d["Line_Total"].isna() | ((d["Line_Total"] - exp).abs() > 1)
    d.loc[bad_l, "Flag_Data"] += "Line_Total dihitung ulang; "
    d["Line_Total"] = exp.round(2)
    add("Line_Total kosong / tidak konsisten", bad_l.sum(),
        "Diperbaiki: dihitung ulang = Quantity × Unit_Price × (1 − Discount_Rate)", len(d), len(d))

    # 10. Outlier nilai transaksi (ditandai saja)
    q1, q3 = d["Line_Total"].quantile([0.25, 0.75])
    thr = q3 + 3 * (q3 - q1)
    d["Outlier_Nilai"] = d["Line_Total"] > thr
    add(f"Nilai baris sangat tinggi (> Rp {thr:,.0f}, batas Q3 + 3×IQR)".replace(",", "."), d["Outlier_Nilai"].sum(),
        "Ditandai (kolom Outlier_Nilai) untuk diperiksa; dipertahankan karena masih wajar (pembelian qty besar produk mahal)", len(d), len(d))

    # Fitur turunan
    d["basket_size"] = d.groupby("Transaction_ID")["Product_ID"].transform("nunique")
    d["Month"] = d["Transaction_Date"].dt.to_period("M").astype(str)
    d["Date"] = d["Transaction_Date"].dt.normalize()
    d["Hour"] = d["Transaction_Date"].dt.hour
    d["Weekday"] = d["Transaction_Date"].dt.dayofweek
    d["Flag_Data"] = d["Flag_Data"].str.strip("; ").replace("", "OK")
    d = d.reset_index(drop=True)
    return d, pd.DataFrame(log)


@st.cache_data(show_spinner="👥 Membentuk RFM per pelanggan...")
def build_rfm(clean):
    c = clean.dropna(subset=["Customer_ID"])
    ref_date = c["Transaction_Date"].max().normalize() + pd.Timedelta(days=1)
    g = c.groupby("Customer_ID").agg(
        Last_Date=("Transaction_Date", "max"),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Line_Total", "sum"),
    ).reset_index()
    g["Recency"] = (ref_date - g["Last_Date"].dt.normalize()).dt.days
    g["Monetary"] = g["Monetary"].round(2)
    return g[["Customer_ID", "Recency", "Frequency", "Monetary", "Last_Date"]], ref_date


@st.cache_data(show_spinner="🛒 Menyusun matriks transaksi × produk...")
def build_basket(clean):
    m = pd.crosstab(clean["Transaction_ID"], clean["Product_Name"]).clip(upper=1).astype(bool)
    return m


@st.cache_data(show_spinner="🛒 Menjalankan Apriori...")
def run_apriori(basket, min_sup, min_conf, min_lift, max_len):
    A = basket.values
    n = A.shape[0]
    items = list(basket.columns)
    m = len(items)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()
    sup = {}
    L = []
    for i in range(m):
        s = A[:, i].mean()
        if s >= min_sup:
            sup[(i,)] = s
            L.append((i,))
    frequent = list(L)
    current = sorted(L)
    k = 2
    while current and k <= max_len:
        cur_set = set(current)
        cand = set()
        for a in current:
            for b in current:
                if a[:-1] == b[:-1] and a[-1] < b[-1]:
                    c = a + (b[-1],)
                    if all(sub in cur_set for sub in itertools.combinations(c, k - 1)):
                        cand.add(c)
        nxt = []
        for c in cand:
            s = A[:, list(c)].all(axis=1).mean()
            if s >= min_sup:
                sup[c] = s
                nxt.append(c)
        frequent += nxt
        current = sorted(nxt)
        k += 1
    its = pd.DataFrame([{
        "Itemset": " + ".join(items[i] for i in t), "Ukuran": len(t),
        "Support": sup[t], "Jumlah Transaksi": int(round(sup[t] * n))} for t in frequent])
    if len(its):
        its = its.sort_values(["Ukuran", "Support"], ascending=[True, False]).reset_index(drop=True)
    rules = []
    for t in frequent:
        if len(t) < 2:
            continue
        for r in range(1, len(t)):
            for ant in itertools.combinations(t, r):
                cons = tuple(x for x in t if x not in ant)
                conf = sup[t] / sup[ant]
                lift = conf / sup[cons]
                if conf >= min_conf and lift >= min_lift:
                    rules.append({
                        "Antecedent": " + ".join(items[i] for i in ant),
                        "Consequent": " + ".join(items[i] for i in cons),
                        "Support": sup[t], "Confidence": conf, "Lift": lift,
                        "Ukuran Itemset": len(t),
                    })
    rules = pd.DataFrame(rules)
    return its, rules


def rank_rules(rules, criterion, one_way, top=10):
    if rules is None or len(rules) == 0:
        return rules
    r = rules.copy()
    if criterion.startswith("Lift"):
        r = r.sort_values(["Lift", "Confidence"], ascending=False)
    elif criterion == "Confidence":
        r = r.sort_values(["Confidence", "Lift"], ascending=False)
    elif criterion == "Support":
        r = r.sort_values(["Support", "Lift"], ascending=False)
    else:
        r["_s"] = r["Lift"] * r["Confidence"]
        r = r.sort_values("_s", ascending=False).drop(columns="_s")
    if one_way:
        r["_key"] = r.apply(lambda x: frozenset((x["Antecedent"] + " + " + x["Consequent"]).split(" + ")), axis=1)
        r = r.drop_duplicates("_key").drop(columns="_key")
    return r.head(top).reset_index(drop=True)


@st.cache_data(show_spinner="📐 Menguji kandidat k (Elbow & Silhouette)...")
def kmeans_eval(rfm, use_log):
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    if use_log:
        X = np.log1p(X)
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        rows.append({"k": k, "Inertia (WCSS)": km.inertia_, "Silhouette": silhouette_score(Xs, km.labels_)})
    return pd.DataFrame(rows)


def label_clusters(prof):
    labels = {}
    rem = list(prof.index)
    val = prof["Frequency"].rank() + prof["Monetary"].rank()
    loyal = val.idxmax()
    labels[loyal] = "Loyal Customers"
    rem.remove(loyal)
    if rem:
        churn = prof.loc[rem, "Recency"].idxmax()
        labels[churn] = "Churn Risk"
        rem.remove(churn)
    if rem:
        new = prof.loc[rem, "Frequency"].idxmin()
        labels[new] = "New Customers"
        rem.remove(new)
    for i, c in enumerate(rem):
        labels[c] = "Potential Customers" + (f" {chr(65 + i)}" if len(rem) > 1 else "")
    return labels


@st.cache_data(show_spinner="🎯 Menjalankan K-Means...")
def run_kmeans(rfm, k, use_log):
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    if use_log:
        X = np.log1p(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    out = rfm.copy()
    out["Cluster"] = km.labels_
    for i, c in enumerate(["z_Recency", "z_Frequency", "z_Monetary"]):
        out[c] = Xs[:, i]
    prof_mean = out.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    labels = label_clusters(prof_mean)
    out["Segmen"] = out["Cluster"].map(labels)
    return out


def segment_profile(km):
    g = km.groupby("Segmen")
    prof = pd.DataFrame({
        "Jumlah Pelanggan": g.size(),
        "% Pelanggan": (g.size() / len(km) * 100).round(1),
        "Recency Rata-rata": g["Recency"].mean().round(1),
        "Recency Median": g["Recency"].median(),
        "Frequency Rata-rata": g["Frequency"].mean().round(2),
        "Frequency Median": g["Frequency"].median(),
        "Monetary Rata-rata": g["Monetary"].mean().round(0),
        "Monetary Median": g["Monetary"].median().round(0),
        "Total Monetary": g["Monetary"].sum().round(0),
    }).reset_index()
    order = {"Loyal Customers": 0, "Potential Customers": 1, "New Customers": 2, "Churn Risk": 3}
    prof["_o"] = prof["Segmen"].map(lambda s: order.get(s.split(" ")[0] + " " + s.split(" ")[1] if " " in s else s, 9))
    return prof.sort_values("_o").drop(columns="_o").reset_index(drop=True)


STRATEGY = {
    "Loyal Customers": ("👑", "Pertahankan & tingkatkan nilai: program poin/membership tier, akses awal promo, bundling premium, ajak jadi 'duta' toko (referral)."),
    "Potential Customers": ("🌱", "Dorong naik kelas: voucher belanja berikutnya, rekomendasi cross-selling dari aturan Apriori, target frekuensi kunjungan lebih sering."),
    "New Customers": ("🎁", "Onboarding: kupon sambutan untuk kunjungan ke-2, edukasi katalog produk, ajak mendaftar member agar tercatat perilaku belanjanya."),
    "Churn Risk": ("⏰", "Re-engagement: kampanye 'kami rindu Anda' via WhatsApp/SMS, diskon terbatas waktu, survei singkat alasan berhenti belanja."),
}


def strategy_for(label):
    for key, v in STRATEGY.items():
        if label.startswith(key):
            return v
    return ("📌", "Pantau dan uji penawaran yang sesuai profil.")


def get_default_k(rfm, use_log):
    ev = kmeans_eval(rfm, use_log)
    cand = ev[ev["k"].between(3, 6)]
    return int(cand.loc[cand["Silhouette"].idxmax(), "k"])


def current_km(rfm):
    p = st.session_state.get("km_params", {"k": None, "log": False})
    k = p["k"] if p["k"] else get_default_k(rfm, p["log"])
    return run_kmeans(rfm, k, p["log"]), k, p["log"]


AP_DEFAULT = {"sup": 0.01, "conf": 0.20, "lift": 1.2, "len": 3, "crit": "Lift (tie-break Confidence)", "oneway": True}


def current_ap():
    return st.session_state.get("ap_params", AP_DEFAULT)


def network_fig(rules):
    nodes = sorted(set(itertools.chain.from_iterable(
        (r["Antecedent"] + " + " + r["Consequent"]).split(" + ") for _, r in rules.iterrows())))
    n = len(nodes)
    pos = {nm: (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i, nm in enumerate(nodes)}
    fig = go.Figure()
    lift_max = rules["Lift"].max()
    for _, r in rules.iterrows():
        ants = r["Antecedent"].split(" + ")
        cons = r["Consequent"].split(" + ")
        for a in ants:
            for c in cons:
                x0, y0 = pos[a]
                x1, y1 = pos[c]
                w = 1 + 5 * r["Lift"] / lift_max
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines", line=dict(width=w, color="rgba(199,21,133,0.45)"),
                    hoverinfo="text", showlegend=False,
                    text=f"{r['Antecedent']} → {r['Consequent']}<br>Lift {r['Lift']:.2f} | Conf {r['Confidence']:.2f}"))
                fig.add_annotation(x=x1 * 0.93 + x0 * 0.07, y=y1 * 0.93 + y0 * 0.07, ax=x0, ay=y0,
                                   xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                   arrowhead=3, arrowsize=1.4, arrowwidth=1.5, arrowcolor="#C71585", opacity=0.6)
    deg = {nm: 0 for nm in nodes}
    for _, r in rules.iterrows():
        for nm in (r["Antecedent"] + " + " + r["Consequent"]).split(" + "):
            deg[nm] += 1
    fig.add_trace(go.Scatter(
        x=[pos[nm][0] for nm in nodes], y=[pos[nm][1] for nm in nodes], mode="markers+text",
        text=nodes, textposition="top center",
        marker=dict(size=[18 + 6 * deg[nm] for nm in nodes], color="#FFB6C1", line=dict(width=2, color="#C71585")),
        hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False, range=[-1.6, 1.6])
    fig.update_yaxes(visible=False, range=[-1.4, 1.4])
    return style_fig(fig, 520, "Jaringan Aturan Asosiasi (tebal garis = Lift)")


# ======================================================================
# ===========================  HALAMAN / UI  ===========================
# ======================================================================
# --- SESSION STATE LOGIN ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- HALAMAN LOGIN DENGAN FOTO DI ATAS ---
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #C71585;'>🦋 PORTAL UJIAN TENGAH SEMESTER (UTS) 🦋</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #DB7093;'>Business Intelligence & Data Mining</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        # Tampilkan Foto Profil di Atas Kartu Login
        try:
            img = Image.open("Liliyosa.jpeg")
            st.image(img, width=180, caption="Liliyosa Jingga Luahane (51230087)")
        except Exception:
            st.info("💡 Pastikan file foto 'Liliyosa.jpeg' berada di direktori yang sama.")

        st.markdown("""
            <div class="card">
                <h4>💖 Identitas Mahasiswa 💖</h4>
                <p><b>Nama:</b> Liliyosa Jingga Luahane</p>
                <p><b>NPM:</b> 51230087</p>
                <p><b>Prodi:</b> Teknologi Informasi</p>
                <p><b>Universitas:</b> Universitas Timor</p>
            </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username_input = st.text_input("Username", value="liliyosa")
            password_input = st.text_input("Password", type="password", value="51230087")
            submit_login = st.form_submit_button("🔐 Login Sistem")

            if submit_login:
                if username_input == "liliyosa" and password_input == "51230087":
                    st.session_state.logged_in = True
                    st.success("Login Berhasil!")
                    st.rerun()
                else:
                    st.error("Username atau Password salah! Gunakan (liliyosa / 51230087)")
    st.stop()

# --- SIDEBAR NAVIGASI & FOTO PROFIL KECIL ---
MENU = [
    "🏠 Beranda & Upload Dataset",
    "📊 Tahap 1: EDA & Business Understanding",
    "🧹 Tahap 2: Data Preprocessing",
    "🛒 Tahap 3: Apriori (Market Basket Analysis)",
    "🎯 Tahap 4: K-Means (Segmentasi Pelanggan)",
    "📈 Tahap 5: BI Dashboard & Visualisasi",
    "💡 Tahap 6: Rekomendasi Keputusan Bisnis",
]
if 'menu' not in st.session_state:
    st.session_state.menu = MENU[0]


def go_to(i):
    st.session_state.menu = MENU[i]


st.sidebar.markdown("<h2>🦋 Menu Navigasi UTS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

try:
    sidebar_img = Image.open("Liliyosa.jpeg")
    st.sidebar.image(sidebar_img, width=120)
except Exception:
    pass

st.sidebar.markdown("""
**Identitas Mahasiswa:**
* **Nama:** Liliyosa Jingga Luahane
* **NPM:** 51230087
* **Prodi:** Teknologi Informasi
* **Univ:** Universitas Timor
""")
st.sidebar.markdown("---")

menu = st.sidebar.radio("Pilih Tahapan UTS:", MENU, key="menu")
idx_menu = MENU.index(menu)
st.sidebar.progress(idx_menu / (len(MENU) - 1), text=f"Progres tahapan: {idx_menu}/{len(MENU) - 1}")
if st.session_state.get("xl_bytes") is not None:
    st.sidebar.success("📂 Dataset termuat")
else:
    st.sidebar.warning("📂 Dataset belum diunggah")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()


def nav_buttons():
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 3, 1])
    if idx_menu > 0:
        c1.button("⬅️ Sebelumnya", key=f"prev_{idx_menu}", on_click=go_to, args=(idx_menu - 1,))
    if idx_menu < len(MENU) - 1:
        c3.button("Lanjut ➡️", key=f"next_{idx_menu}", on_click=go_to, args=(idx_menu + 1,))


# --- GLOBAL DATASET STORAGE ---
if 'xl_bytes' not in st.session_state:
    st.session_state.xl_bytes = None
    st.session_state.xl_name = None


# ----------------------------------------------------------------------
# BERANDA
# ----------------------------------------------------------------------
if menu == MENU[0]:
    st.markdown("<h1>🏠 Beranda Utama & Upload Dataset UTS</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class="card">
            <h3>📂 Upload Dataset Transaksi Toko (.xlsx)</h3>
            <p>Silakan upload file dataset Excel UTS Anda di bawah ini untuk memulai seluruh proses analisis, pemodelan, dan dashboard interaktif.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Pilih file Excel Dataset UTS", type=["xlsx", "xls"])
    if uploaded_file is not None:
        st.session_state.xl_bytes = uploaded_file.getvalue()
        st.session_state.xl_name = uploaded_file.name

    local_files = glob.glob("*Business*Intelligence*.xlsx") + glob.glob("*UTS*.xlsx")
    if st.session_state.xl_bytes is None and local_files:
        if st.button(f"⚡ Pakai file lokal: {local_files[0]}"):
            with open(local_files[0], "rb") as fh:
                st.session_state.xl_bytes = fh.read()
            st.session_state.xl_name = local_files[0]
            st.rerun()

    if st.session_state.xl_bytes is not None:
        try:
            xl = read_workbook(st.session_state.xl_bytes)
            if "Data_Raw" not in xl:
                raise ValueError("Sheet 'Data_Raw' tidak ditemukan di file ini.")
            st.success(f"✅ Dataset **{st.session_state.xl_name}** berhasil dimuat ke dalam sistem!")
            raw = xl["Data_Raw"]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi("Baris Data Mentah", fmt_int(len(raw)))
            with c2:
                kpi("Kolom", len(raw.columns))
            with c3:
                kpi("Sheet Tersedia", len(xl))
            with c4:
                kpi("Transaction_ID Unik", fmt_int(raw["Transaction_ID"].nunique()))
            st.markdown("---")
            t1, t2, t3, t4 = st.tabs(["📋 Preview Data_Raw", "📖 Petunjuk UAS", "📚 Data Dictionary", "🧺 Katalog Produk"])
            with t1:
                n_prev = st.slider("Jumlah baris preview", 5, 100, 10, step=5)
                st.markdown(f"#### 📋 Preview {n_prev} Baris Pertama Dataset Raw:")
                show_df(raw.head(n_prev))
            with t2:
                if "Petunjuk_UAS" in xl:
                    show_df(xl["Petunjuk_UAS"].fillna(""), height=500)
                else:
                    st.info("Sheet Petunjuk_UAS tidak tersedia.")
            with t3:
                if "Data_Dictionary" in xl:
                    show_df(xl["Data_Dictionary"])
                else:
                    st.info("Sheet Data_Dictionary tidak tersedia.")
            with t4:
                if "Data_Produk" in xl:
                    show_df(xl["Data_Produk"])
                else:
                    st.info("Sheet Data_Produk tidak tersedia.")
            st.button("🚀 Mulai Analisis → Tahap 1", on_click=go_to, args=(1,), key="start_btn")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.session_state.xl_bytes = None
    else:
        st.info("💡 Belum ada file yang di-upload. Silakan upload file `P8. UTS_Business Intelligence.xlsx` di atas.")

# Validasi jika belum upload data dan masuk ke menu lain
elif st.session_state.xl_bytes is None:
    st.warning("⚠️ Silakan kembali ke menu **Beranda & Upload Dataset** terlebih dahulu untuk meng-upload file Excel UTS Anda!")
    st.button("🏠 Ke Beranda", on_click=go_to, args=(0,))
    st.stop()

else:
    # ------------------------------------------------------------------
    # PIPELINE DATA (dicache): Data_Raw -> Data_Clean -> basket, RFM
    # ------------------------------------------------------------------
    xl = read_workbook(st.session_state.xl_bytes)
    if "Data_Raw" not in xl:
        st.error("Sheet 'Data_Raw' tidak ditemukan.")
        st.stop()
    df_raw = xl["Data_Raw"]
    prod_ref = xl.get("Data_Produk")
    cust_ref = xl.get("Data_Pelanggan")
    df_clean, prep_log = preprocess(df_raw, prod_ref, cust_ref)
    rfm, ref_date = build_rfm(df_clean)
    basket = build_basket(df_clean)

    # ==================================================================
    # TAHAP 1
    # ==================================================================
    if menu == MENU[1]:
        st.markdown("<h1>📊 Tahap 1: Business Understanding & EDA (10%)</h1>", unsafe_allow_html=True)
        audit = audit_raw(df_raw, prod_ref)
        tabs = st.tabs(["🎯 Business Understanding", "📋 Profil Dataset", "🔍 Audit Kualitas Data",
                        "📊 Visualisasi EDA", "💡 Insight Awal"])

        with tabs[0]:
            st.markdown("""
                <div class="card">
                    <h4>Soal 1: Permasalahan & Pertanyaan Bisnis</h4>
                    <p><b>Permasalahan:</b> Perusahaan ritel memiliki volume data besar namun belum dimanfaatkan secara optimal untuk pola pembelian dan segmentasi.</p>
                    <p><b>Pertanyaan Analitis:</b></p>
                    <ol>
                        <li>Produk apa yang sering dibeli bersama?</li>
                        <li>Bagaimana segmentasi pelanggan berdasarkan RFM?</li>
                        <li>Bagaimana tren pendapatan bulanan toko?</li>
                        <li>Segmen pelanggan mana yang paling bernilai dan mana yang berisiko tidak kembali?</li>
                        <li>Kategori dan produk apa yang menyumbang penjualan terbesar?</li>
                    </ol>
                    <p><b>Peran:</b> Data Analyst / BI Analyst yang menyampaikan hasil kepada manajemen toko.</p>
                </div>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.info("🛒 **Apriori** menjawab pertanyaan 1")
            c2.info("🎯 **K-Means (RFM)** menjawab pertanyaan 2 & 4")
            c3.info("📈 **Dashboard** menjawab pertanyaan 3 & 5")

        with tabs[1]:
            st.markdown("#### 📋 Gambaran Umum Data_Raw (sebelum dibersihkan)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi("Jumlah Baris", fmt_int(df_raw.shape[0]))
            with c2:
                kpi("Jumlah Kolom", df_raw.shape[1])
            with c3:
                kpi("Transaksi Unik", fmt_int(df_raw["Transaction_ID"].nunique()))
            with c4:
                kpi("Pelanggan Unik", fmt_int(df_raw["Customer_ID"].nunique()))
            st.markdown("<br>", unsafe_allow_html=True)
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                kpi("Produk Unik", fmt_int(df_raw["Product_ID"].nunique()))
            with c6:
                kpi("Rentang Tanggal", f"{df_raw['Transaction_Date'].min():%d %b %Y} – {df_raw['Transaction_Date'].max():%d %b %Y}")
            with c7:
                kpi("Total Penjualan Sementara", fmt_idr(df_raw["Line_Total"].sum()), "dari Data_Raw, belum bersih")
            with c8:
                kpi("Rata-rata Baris/Transaksi", f"{len(df_raw) / df_raw['Transaction_ID'].nunique():.2f}")
            st.markdown("#### 🧬 Tipe Data Tiap Kolom")
            dt = pd.DataFrame({"Kolom": df_raw.columns, "Tipe Data (pandas)": df_raw.dtypes.astype(str).values,
                               "Jumlah Nilai Unik": [df_raw[c].nunique() for c in df_raw.columns],
                               "Contoh Nilai": [str(df_raw[c].dropna().iloc[0]) for c in df_raw.columns]})
            show_df(dt)
            st.markdown("#### 📐 Statistik Deskriptif Numerik (Data_Raw)")
            show_df(df_raw[["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]].describe().T.round(2).reset_index().rename(columns={"index": "Kolom"}))

        with tabs[2]:
            st.markdown("#### 🔍 Audit Kualitas Data Mentah")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**1️⃣ Missing Values**")
                mv = audit["missing"]
                show_df(mv[mv["Jumlah Missing"] > 0])
                fig = px.bar(mv[mv["Jumlah Missing"] > 0], x="Jumlah Missing", y="Kolom", orientation="h",
                             color="Jumlah Missing", color_continuous_scale="PuRd")
                fig.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False)
                show_plot(style_fig(fig, 300, "Missing Values per Kolom"), key="aud_mv")
            with c2:
                st.markdown("**2️⃣ Duplikasi**")
                kpi("Baris Duplikat Persis", fmt_int(audit["dup_exact"]), f"{audit['dup_exact'] / len(df_raw) * 100:.2f}% dari seluruh baris")
                st.markdown("<br>**3️⃣ Inkonsistensi Kategori / Teks**", unsafe_allow_html=True)
                show_df(audit["inconsistent"]) if len(audit["inconsistent"]) else st.success("Tidak ada inkonsistensi.")
            st.markdown("**4️⃣ Nilai Tidak Wajar / Outlier**")
            c3, c4 = st.columns([1, 1.2])
            with c3:
                show_df(audit["outlier"])
            with c4:
                fig = px.box(df_raw, x="Quantity", points="outliers", color_discrete_sequence=["#C71585"])
                show_plot(style_fig(fig, 220, "Boxplot Quantity (terlihat nilai 999 & negatif)"), key="aud_box")

        with tabs[3]:
            st.markdown("#### 📊 Visualisasi EDA")
            st.caption("Visualisasi memakai **Data_Clean** agar tidak terdistorsi outlier (Qty 999, harga 999.999). Perbandingan sebelum/sesudah ada di Tahap 2.")
            metric = st.radio("Metrik utama:", ["Pendapatan (Rp)", "Jumlah Transaksi", "Jumlah Item"], horizontal=True, key="eda_metric")
            agg_map = {"Pendapatan (Rp)": ("Line_Total", "sum"), "Jumlah Transaksi": ("Transaction_ID", "nunique"), "Jumlah Item": ("Quantity", "sum")}
            col_m, fn_m = agg_map[metric]
            c1, c2 = st.columns(2)
            with c1:
                monthly = df_clean.groupby("Month")[col_m].agg(fn_m).reset_index()
                fig = px.line(monthly, x="Month", y=col_m, markers=True)
                fig.update_traces(line=dict(color="#C71585", width=3), marker=dict(size=9))
                fig.update_layout(xaxis_title="Bulan", yaxis_title=metric)
                show_plot(style_fig(fig, 380, f"📈 Tren Bulanan — {metric}"), key="eda1")
            with c2:
                bycat = df_clean.groupby("Category")[col_m].agg(fn_m).reset_index().sort_values(col_m)
                fig = px.bar(bycat, x=col_m, y="Category", orientation="h", color=col_m, color_continuous_scale="PuRd")
                fig.update_layout(coloraxis_showscale=False, xaxis_title=metric)
                show_plot(style_fig(fig, 380, f"🧺 {metric} per Kategori"), key="eda2")
            c3, c4 = st.columns(2)
            with c3:
                hm = df_clean.groupby(["Weekday", "Hour"])["Transaction_ID"].nunique().reset_index()
                pv = hm.pivot(index="Weekday", columns="Hour", values="Transaction_ID").fillna(0)
                pv.index = [["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][i] for i in pv.index]
                fig = px.imshow(pv, color_continuous_scale="PuRd", aspect="auto", labels=dict(x="Jam", y="Hari", color="Transaksi"))
                show_plot(style_fig(fig, 360, "🕒 Heatmap Transaksi: Hari × Jam"), key="eda3")
            with c4:
                fig = px.histogram(df_clean.drop_duplicates("Transaction_ID"), x="basket_size", nbins=6,
                                   color_discrete_sequence=["#DB7093"])
                fig.update_layout(xaxis_title="basket_size (produk unik / transaksi)", yaxis_title="Jumlah Transaksi", bargap=0.1)
                show_plot(style_fig(fig, 360, "🧺 Distribusi basket_size"), key="eda4")

        with tabs[4]:
            st.markdown("#### 💡 Insight Awal (dihitung otomatis dari Data_Clean)")
            monthly = df_clean.groupby("Month")["Line_Total"].sum()
            catrev = df_clean.groupby("Category")["Line_Total"].sum().sort_values(ascending=False)
            top_prod = df_clean.groupby("Product_Name")["Line_Total"].sum().sort_values(ascending=False)
            mem = df_clean[df_clean["Customer_Type"].isin(["Member", "Non-Member"])].drop_duplicates("Transaction_ID") \
                .assign(tot=lambda x: x["Transaction_ID"].map(df_clean.groupby("Transaction_ID")["Line_Total"].sum())) \
                .groupby("Customer_Type")["tot"].mean()
            days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
            wk = df_clean.drop_duplicates("Transaction_ID").groupby("Weekday").size()
            hr = df_clean.drop_duplicates("Transaction_ID").groupby("Hour").size()
            ins = [
                f"**Puncak penjualan** terjadi pada <b>{monthly.idxmax()}</b> ({fmt_idr(monthly.max())}), sedangkan terendah <b>{monthly.idxmin()}</b> ({fmt_idr(monthly.min())}) — selisih {((monthly.max() / monthly.min()) - 1) * 100:.0f}%.",
                f"Kategori <b>{catrev.index[0]}</b> menyumbang {catrev.iloc[0] / catrev.sum() * 100:.1f}% pendapatan, diikuti <b>{catrev.index[1]}</b> ({catrev.iloc[1] / catrev.sum() * 100:.1f}%). Produk terlaris berdasarkan nilai: <b>{top_prod.index[0]}</b>.",
                f"Rata-rata <b>basket_size</b> = {df_clean.drop_duplicates('Transaction_ID')['basket_size'].mean():.2f} produk/transaksi; hari tersibuk <b>{days[wk.idxmax()]}</b> dan jam tersibuk sekitar <b>{hr.idxmax()}:00</b>.",
            ]
            if len(mem) == 2:
                ins.append(f"Rata-rata nilai transaksi Member {fmt_idr(mem.get('Member', 0))} vs Non-Member {fmt_idr(mem.get('Non-Member', 0))}.")
            ins.append(f"Kualitas data: {fmt_int(audit['dup_exact'])} baris duplikat, nilai Quantity/Harga tidak wajar, dan variasi ejaan perlu dibersihkan sebelum Apriori dan K-Means.")
            for i, t in enumerate(ins, 1):
                card(f"<b>Insight {i}.</b> {t}")
        nav_buttons()

    # ==================================================================
    # TAHAP 2
    # ==================================================================
    elif menu == MENU[2]:
        st.markdown("<h1>🧹 Tahap 2: Data Preprocessing (25%)</h1>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="card">
                <h4>Ringkasan Pembersihan Data</h4>
                <ul>
                    <li><b>Baris Data Mentah:</b> {fmt_int(len(df_raw))}</li>
                    <li><b>Baris Data Bersih:</b> {fmt_int(len(df_clean))}</li>
                    <li><b>Fitur yang Dibentuk:</b> RFM (Recency, Frequency, Monetary) dan <code>basket_size</code>.</li>
                    <li><b>Prinsip:</b> Data_Raw tidak ditimpa — seluruh perbaikan dilakukan pada salinan kerja (<code>Data_Clean</code>).</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Baris Sebelum", fmt_int(len(df_raw)))
        with c2:
            kpi("Baris Sesudah", fmt_int(len(df_clean)))
        with c3:
            kpi("Baris Dibuang", fmt_int(len(df_raw) - len(df_clean)), f"{(len(df_raw) - len(df_clean)) / len(df_raw) * 100:.2f}%")
        with c4:
            kpi("Transaksi Unik", fmt_int(df_clean["Transaction_ID"].nunique()))
        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs(["📝 Log & Keputusan", "🛍️ basket_size & Matriks", "👥 RFM", "⚖️ Scaling & Big Data", "💾 Data & Unduhan"])
        with tabs[0]:
            st.markdown("#### 📝 Keputusan Penanganan Tiap Masalah (Tugas 4–5)")
            show_df(prep_log, height=480)
            fig = px.bar(prep_log[prep_log["Baris Terdampak"] > 0].sort_values("Baris Terdampak"),
                         x="Baris Terdampak", y="Masalah", orientation="h", color="Baris Terdampak", color_continuous_scale="PuRd")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            show_plot(style_fig(fig, 420, "Jumlah Baris Terdampak per Masalah"), key="prep_bar")
            st.markdown("**Perbandingan sebelum vs sesudah (Quantity & Unit_Price)**")
            cc1, cc2 = st.columns(2)
            with cc1:
                fig = px.box(df_raw, y="Unit_Price", points="outliers", color_discrete_sequence=["#DB7093"])
                show_plot(style_fig(fig, 300, "Unit_Price — Raw"), key="prep_b1")
            with cc2:
                fig = px.box(df_clean, y="Unit_Price", points="outliers", color_discrete_sequence=["#C71585"])
                show_plot(style_fig(fig, 300, "Unit_Price — Clean"), key="prep_b2")
        with tabs[1]:
            st.markdown("#### 🛍️ basket_size = jumlah produk unik per Transaction_ID (Tugas 6)")
            tx = df_clean.groupby("Transaction_ID").agg(
                Tanggal=("Transaction_Date", "first"), Customer_ID=("Customer_ID", "first"),
                basket_size=("Product_ID", "nunique"), Total_Belanja=("Line_Total", "sum")).reset_index()
            c1, c2 = st.columns([1, 1])
            with c1:
                show_df(tx.head(200), height=320)
            with c2:
                vc = tx["basket_size"].value_counts().sort_index().reset_index()
                vc.columns = ["basket_size", "Jumlah Transaksi"]
                fig = px.bar(vc, x="basket_size", y="Jumlah Transaksi", text="Jumlah Transaksi", color_discrete_sequence=["#C71585"])
                show_plot(style_fig(fig, 320, "Distribusi basket_size"), key="prep_bs")
            st.markdown("#### 🧮 Matriks Transaksi × Produk untuk Apriori (1 = dibeli, 0 = tidak)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Jumlah Transaksi (baris)", fmt_int(basket.shape[0]))
            c2.metric("Jumlah Produk (kolom)", basket.shape[1])
            c3.metric("Kepadatan Matriks", f"{basket.values.mean() * 100:.1f}%")
            show_df(basket.astype(int).head(15).reset_index(), height=320)
            st.caption("Satu baris = satu Transaction_ID (bukan satu baris item), sehingga transaksi tidak dihitung ganda.")
        with tabs[2]:
            st.markdown("#### 👥 RFM per Pelanggan (Tugas 7)")
            st.markdown(f"""
                <div class="card-soft">
                <b>📅 Tanggal acuan (reference date):</b> {ref_date:%d %B %Y} — satu hari setelah transaksi terakhir di Data_Clean ({(ref_date - pd.Timedelta(days=1)):%d %B %Y}).<br>
                <b>Recency</b> = hari sejak transaksi terakhir terhadap tanggal acuan · <b>Frequency</b> = jumlah Transaction_ID unik · <b>Monetary</b> = total Line_Total bersih.<br>
                Baris tanpa Customer_ID tidak dimasukkan ke RFM. Satu baris = satu pelanggan ({fmt_int(len(rfm))} pelanggan).
                </div>
            """, unsafe_allow_html=True)
            show_df(rfm.sort_values("Monetary", ascending=False).head(300), height=300)
            c1, c2, c3 = st.columns(3)
            for col, feat, clr in zip([c1, c2, c3], ["Recency", "Frequency", "Monetary"], ["#C71585", "#DB7093", "#8B008B"]):
                with col:
                    fig = px.histogram(rfm, x=feat, nbins=30, color_discrete_sequence=[clr])
                    show_plot(style_fig(fig, 280, f"Distribusi {feat}"), key=f"rfm_h_{feat}")
        with tabs[3]:
            st.markdown("#### ⚖️ Fitur K-Means & Standardisasi (Tugas 8)")
            log_opt = st.toggle("Terapkan log1p sebelum standardisasi (opsional, mengurangi skew)", value=False, key="prep_log")
            Xf = rfm[["Recency", "Frequency", "Monetary"]].copy()
            if log_opt:
                Xf = np.log1p(Xf)
            Xz = pd.DataFrame(StandardScaler().fit_transform(Xf), columns=["z_Recency", "z_Frequency", "z_Monetary"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Sebelum scaling**")
                show_df(rfm[["Recency", "Frequency", "Monetary"]].describe().T.round(2).reset_index())
            with c2:
                st.markdown("**Sesudah StandardScaler (mean≈0, std≈1)**")
                show_df(Xz.describe().T.round(2).reset_index())
            st.markdown("""
                <div class="card">
                    <h4>🧠 Mengapa harus distandardisasi?</h4>
                    <p>K-Means memakai jarak Euclidean. Monetary bernilai ratusan ribu hingga jutaan rupiah, sedangkan Frequency hanya satuan dan Recency puluhan hari. Tanpa scaling, <b>Monetary mendominasi jarak</b> sehingga klaster hanya mencerminkan besarnya belanja. <code>StandardScaler</code> menyamakan skala ketiga fitur. Customer_ID <b>tidak</b> dimasukkan sebagai fitur.</p>
                    <h4>🐘 Strategi untuk Data Besar</h4>
                    <ul>
                        <li><b>Agregasi:</b> data item (18 ribu baris) diringkas menjadi 1 baris/pelanggan (RFM) dan 1 baris/transaksi (basket) — dipakai di aplikasi ini.</li>
                        <li><b>Sampling:</b> bila jutaan pelanggan, jalankan elbow/silhouette pada sampel acak terstratifikasi, lalu terapkan model ke seluruh data.</li>
                        <li><b>Chunking / MiniBatchKMeans:</b> baca file bertahap (<code>chunksize</code>) dan latih <code>MiniBatchKMeans</code>; untuk Apriori gunakan matriks sparse atau FP-Growth.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        with tabs[4]:
            st.markdown("#### 💾 Data_Clean & Dokumentasi Perubahan")
            st.caption("Kolom Flag_Data menandai baris yang diperbaiki; Outlier_Nilai menandai baris bernilai sangat tinggi.")
            only_flag = st.checkbox("Tampilkan hanya baris yang diperbaiki (Flag_Data ≠ OK)", value=False)
            view_df = df_clean[df_clean["Flag_Data"] != "OK"] if only_flag else df_clean
            show_df(view_df.drop(columns=["Date", "Hour", "Weekday"]), height=360)
            c1, c2, c3 = st.columns(3)
            with c1:
                dl_csv(df_clean.drop(columns=["Date", "Hour", "Weekday"]), "⬇️ Unduh Data_Clean (CSV)", "Data_Clean.csv", "dl_clean")
            with c2:
                dl_csv(prep_log, "⬇️ Unduh Log Preprocessing", "Log_Preprocessing.csv", "dl_log")
            with c3:
                dl_csv(rfm, "⬇️ Unduh Tabel RFM", "RFM_Pelanggan.csv", "dl_rfm")
        nav_buttons()

    # ==================================================================
    # TAHAP 3
    # ==================================================================
    elif menu == MENU[3]:
        st.markdown("<h1>🛒 Tahap 3: Apriori / Market Basket Analysis (20%)</h1>", unsafe_allow_html=True)
        P = current_ap()
        with st.form("ap_form"):
            st.markdown("#### ⚙️ Parameter Apriori")
            c1, c2, c3 = st.columns(3)
            sup = c1.slider("Minimum Support", 0.005, 0.05, float(P["sup"]), 0.0025, format="%.4f")
            conf = c2.slider("Minimum Confidence", 0.05, 0.80, float(P["conf"]), 0.05)
            lift_min = c3.slider("Minimum Lift", 1.0, 3.0, float(P["lift"]), 0.1)
            c4, c5, c6 = st.columns(3)
            max_len = c4.selectbox("Ukuran itemset maksimum", [2, 3], index=[2, 3].index(P["len"]))
            crits = ["Lift (tie-break Confidence)", "Confidence", "Support", "Skor gabungan (Lift × Confidence)"]
            crit = c5.selectbox("Kriteria pemeringkatan", crits, index=crits.index(P["crit"]))
            oneway = c6.checkbox("1 arah terbaik per pasangan produk", value=P["oneway"])
            run = st.form_submit_button("🚀 Jalankan Apriori")
        if run:
            st.session_state.ap_params = {"sup": sup, "conf": conf, "lift": lift_min, "len": max_len, "crit": crit, "oneway": oneway}
            P = st.session_state.ap_params
        its, rules = run_apriori(basket, P["sup"], P["conf"], P["lift"], P["len"])
        top = rank_rules(rules, P["crit"], P["oneway"], 10)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Transaksi Dianalisis", fmt_int(basket.shape[0]))
        with c2:
            kpi("Frequent Itemsets", fmt_int(len(its)))
        with c3:
            kpi("Aturan Asosiasi", fmt_int(len(rules)))
        with c4:
            kpi("Lift Tertinggi", f"{rules['Lift'].max():.2f}" if len(rules) else "-")
        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs(["📖 Metode & Alasan Parameter", "🧩 Frequent Itemsets", "🏆 Top 10 Aturan", "📊 Visualisasi", "🛍️ Rekomendasi Cross-Selling"])
        with tabs[0]:
            n_trx = basket.shape[0]
            avg_item_sup = basket.values.mean()
            st.markdown(f"""
                <div class="card">
                    <h4>Aturan Asosiasi Produk</h4>
                    <p><b>Minimum Support:</b> {P['sup']:.4f} (≈ {int(P['sup'] * n_trx)} dari {fmt_int(n_trx)} transaksi) | <b>Minimum Confidence:</b> {P['conf']:.2f} | <b>Minimum Lift:</b> {P['lift']:.1f}</p>
                    <p><b>Alasan:</b> katalog hanya 30 produk dengan support satu produk ≈ {basket.mean().min() * 100:.1f}%–{basket.mean().max() * 100:.1f}%. Jika produk saling independen, support sebuah pasangan hanya ≈ {(avg_item_sup ** 2) * 100:.2f}%, sehingga <i>support 0.02 terlalu ketat</i> dan threshold confidence 0.5 tidak mungkin tercapai (confidence tertinggi data ini sekitar 0.3). Support {P['sup']:.4f} menyaring pasangan yang muncul ≥ {int(P['sup'] * n_trx)} kali, confidence {P['conf']:.2f} dan <b>lift &gt; 1</b> memastikan pasangan lebih sering muncul dibanding kebetulan.</p>
                    <p><b>Definisi:</b> Support = P(A∩B) · Confidence = P(B|A) · Lift = Confidence / P(B). Lift &gt; 1 berarti A dan B dibeli bersama lebih sering dari yang diharapkan. <i>Aturan asosiasi bukan bukti sebab-akibat.</i></p>
                </div>
            """, unsafe_allow_html=True)
            with st.expander("🧮 Lihat ringkasan matriks transaksi × produk"):
                st.write(f"Ukuran matriks: **{basket.shape[0]:,} transaksi × {basket.shape[1]} produk** (1 = dibeli).".replace(",", "."))
                show_df(basket.astype(int).head(10).reset_index())
        with tabs[1]:
            if len(its):
                f1, f2 = st.columns([1, 1])
                size_sel = f1.multiselect("Filter ukuran itemset", sorted(its["Ukuran"].unique()), default=sorted(its["Ukuran"].unique()))
                its_v = its[its["Ukuran"].isin(size_sel)].copy()
                its_v["Support"] = its_v["Support"].round(4)
                show_df(its_v, height=380)
                dl_csv(its_v, "⬇️ Unduh Frequent Itemsets", "frequent_itemsets.csv", "dl_its")
            else:
                st.warning("Tidak ada frequent itemset. Turunkan minimum support.")
        with tabs[2]:
            if top is not None and len(top):
                st.markdown(f"**Kriteria peringkat:** {P['crit']} — maksimal 10 aturan terbaik. Lift &gt; 1 = asosiasi positif.", unsafe_allow_html=True)
                t = top.copy()
                t.insert(0, "Peringkat", range(1, len(t) + 1))
                for c in ["Support", "Confidence"]:
                    t[c] = t[c].round(4)
                t["Lift"] = t["Lift"].round(3)
                show_df(t.drop(columns=["Ukuran Itemset"]))
                dl_csv(rules.round(4), "⬇️ Unduh Semua Aturan", "association_rules.csv", "dl_rules")
            else:
                st.warning("Tidak ada aturan yang memenuhi threshold. Turunkan minimum support / confidence / lift lalu jalankan ulang.")
        with tabs[3]:
            if top is not None and len(top):
                v1, v2 = st.columns([1.2, 1])
                with v1:
                    show_plot(network_fig(top), key="ap_net")
                with v2:
                    fig = px.scatter(rules, x="Support", y="Confidence", size="Lift", color="Lift", color_continuous_scale="PuRd",
                                     hover_data=["Antecedent", "Consequent"], size_max=22)
                    show_plot(style_fig(fig, 520, "Support vs Confidence (ukuran/warna = Lift)"), key="ap_sc")
                pair = rules[(~rules["Antecedent"].str.contains(r"\+", regex=True)) & (~rules["Consequent"].str.contains(r"\+", regex=True))]
                if len(pair):
                    pv = pair.pivot_table(index="Antecedent", columns="Consequent", values="Lift")
                    fig = px.imshow(pv, color_continuous_scale="PuRd", aspect="auto", labels=dict(color="Lift"))
                    show_plot(style_fig(fig, 520, "Heatmap Lift (Antecedent → Consequent)"), key="ap_hm")
                bar = top.assign(Aturan=top["Antecedent"] + " → " + top["Consequent"]).sort_values("Lift")
                fig = px.bar(bar, x="Lift", y="Aturan", orientation="h", color="Confidence", color_continuous_scale="PuRd", text=bar["Lift"].round(2))
                show_plot(style_fig(fig, 420, "Top Aturan berdasarkan Lift (warna = Confidence)"), key="ap_bar")
            else:
                st.info("Visualisasi muncul setelah ada aturan.")
        with tabs[4]:
            if top is not None and len(top):
                best = rank_rules(rules, "Lift (tie-break Confidence)", True, 3)
                st.markdown("#### 🛍️ 3 Rekomendasi Cross-Selling / Bundling")
                for i, r in best.iterrows():
                    card(
                        f"<h4>{['🥇', '🥈', '🥉'][i]} Paket \"{r['Antecedent']} + {r['Consequent']}\"</h4>"
                        f"<p><b>Bukti data:</b> Support {r['Support']:.3f} · Confidence {r['Confidence']:.2f} · Lift {r['Lift']:.2f} "
                        f"(dari pembeli <i>{r['Antecedent']}</i>, {r['Confidence'] * 100:.0f}% juga membeli <i>{r['Consequent']}</i>).</p>"
                        f"<p><b>Bundling:</b> tawarkan paket hemat dengan diskon kecil (mis. 3–5%) untuk kedua produk.<br>"
                        f"<b>Penempatan rak:</b> letakkan <i>{r['Consequent']}</i> bersebelahan dengan <i>{r['Antecedent']}</i> atau pada end-cap yang sama; "
                        f"kasir menawarkan produk pasangannya.</p>")
                st.caption("⚠️ Rekomendasi bersifat asosiatif (korelasi), dampak penjualan harus diuji lewat uji coba terkontrol (lihat Tahap 6).")
            else:
                st.info("Rekomendasi dibuat otomatis dari aturan terbaik. Longgarkan parameter bila belum ada aturan.")
        nav_buttons()

    # ==================================================================
    # TAHAP 4
    # ==================================================================
    elif menu == MENU[4]:
        st.markdown("<h1>🎯 Tahap 4: Segmentasi Pelanggan K-Means (20%)</h1>", unsafe_allow_html=True)
        KP = st.session_state.get("km_params", {"k": None, "log": False})
        with st.form("km_form"):
            st.markdown("#### ⚙️ Pengaturan K-Means")
            c1, c2, c3 = st.columns([1.2, 1.2, 1])
            use_log = c1.checkbox("Terapkan log1p sebelum StandardScaler", value=KP["log"])
            rec_k = get_default_k(rfm, KP["log"])
            k_choice = c2.slider(f"Jumlah klaster (k) — rekomendasi metrik: {rec_k}", 2, 8, int(KP["k"] or rec_k))
            c3.markdown("<br>", unsafe_allow_html=True)
            run_km = st.form_submit_button("🎯 Jalankan K-Means")
        if run_km:
            st.session_state.km_params = {"k": k_choice, "log": use_log}
            KP = st.session_state.km_params
        km_df, k_used, log_used = current_km(rfm)
        ev = kmeans_eval(rfm, log_used)
        prof = segment_profile(km_df)

        tabs = st.tabs(["📐 Elbow & Silhouette", "👥 Profil Klaster", "🔵 Scatter Plot", "📣 Strategi Pemasaran", "🔎 Cek Pelanggan"])
        with tabs[0]:
            st.markdown("#### 📐 Uji Kandidat k = 2–8 (Tugas 14)")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(ev, x="k", y="Inertia (WCSS)", markers=True)
                fig.update_traces(line=dict(color="#C71585", width=3), marker=dict(size=10))
                fig.add_vline(x=k_used, line_dash="dash", line_color="#8B008B", annotation_text=f"k terpilih = {k_used}")
                show_plot(style_fig(fig, 360, "Elbow Method"), key="km_elbow")
            with c2:
                fig = px.bar(ev, x="k", y="Silhouette", text=ev["Silhouette"].round(3), color="Silhouette", color_continuous_scale="PuRd")
                fig.update_layout(coloraxis_showscale=False)
                show_plot(style_fig(fig, 360, "Silhouette Score"), key="km_sil")
            show_df(ev.round(3))
            best_sil = ev.loc[ev["Silhouette"].idxmax()]
            st.markdown(f"""
                <div class="card">
                    <h4>Alasan Pemilihan k = {k_used}</h4>
                    <p>Silhouette tertinggi di seluruh kandidat adalah <b>k = {int(best_sil['k'])}</b> ({best_sil['Silhouette']:.3f}); skor untuk k = {k_used} adalah <b>{ev.loc[ev['k'] == k_used, 'Silhouette'].iloc[0]:.3f}</b>.
                    Rekomendasi otomatis memilih silhouette terbaik pada k = 3–6, karena k = 2 hanya membedakan 'aktif' vs 'pasif' (terlalu kasar untuk strategi pemasaran) dan k &gt; 6 menghasilkan segmen yang sulit ditindaklanjuti.
                    Titik siku (elbow) dibaca bersama silhouette dan <b>interpretasi bisnis</b> (segmen loyal, baru, potensial, berisiko) — Anda dapat mengubah k pada slider lalu menjalankan ulang.</p>
                </div>
            """, unsafe_allow_html=True)
        with tabs[1]:
            st.markdown("#### 👥 Profil Setiap Klaster (Tugas 15)")
            cols = st.columns(len(prof))
            for col, (_, r) in zip(cols, prof.iterrows()):
                with col:
                    kpi(f"{strategy_for(r['Segmen'])[0]} {r['Segmen']}", fmt_int(r["Jumlah Pelanggan"]), f"{r['% Pelanggan']}% pelanggan")
            st.markdown("<br>", unsafe_allow_html=True)
            show_df(prof)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(prof, names="Segmen", values="Jumlah Pelanggan", hole=0.45, color_discrete_sequence=PINK)
                fig.update_traces(textinfo="percent+label")
                show_plot(style_fig(fig, 360, "Komposisi Pelanggan per Segmen"), key="km_pie")
            with c2:
                zp = km_df.groupby("Segmen")[["z_Recency", "z_Frequency", "z_Monetary"]].mean()
                zp.columns = ["Recency", "Frequency", "Monetary"]
                fig = px.imshow(zp, color_continuous_scale="PuRd", aspect="auto", text_auto=".2f", labels=dict(color="z-score"))
                show_plot(style_fig(fig, 360, "Heatmap Profil (z-score rata-rata)"), key="km_hm")
            st.markdown("""
                <div class="card-soft">
                <b>🏷️ Dasar pelabelan (berbasis data):</b> <i>Loyal</i> = klaster dengan peringkat Frequency + Monetary tertinggi · <i>Churn Risk</i> = Recency rata-rata terbesar (lama tidak belanja) ·
                <i>New</i> = Frequency terendah di antara sisanya · sisanya <i>Potential</i>. Label hanya dipakai jika sesuai tabel profil di atas; tinjau ulang bila k diubah.
                </div>
            """, unsafe_allow_html=True)
            dl_csv(km_df.drop(columns=["z_Recency", "z_Frequency", "z_Monetary"]), "⬇️ Unduh Hasil Segmentasi", "segmentasi_pelanggan.csv", "dl_km")
        with tabs[2]:
            st.markdown("#### 🔵 Scatter Plot Klaster")
            mode = st.radio("Tampilan", ["2D", "3D"], horizontal=True, key="km_mode")
            feats = ["Recency", "Frequency", "Monetary"]
            if mode == "2D":
                ca, cb = st.columns(2)
                xa = ca.selectbox("Sumbu X", feats, index=0)
                ya = cb.selectbox("Sumbu Y", feats, index=2)
                fig = px.scatter(km_df, x=xa, y=ya, color="Segmen", hover_data=["Customer_ID", "Recency", "Frequency", "Monetary"],
                                 color_discrete_sequence=PINK, opacity=0.8)
                fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
            else:
                fig = px.scatter_3d(km_df, x="Recency", y="Frequency", z="Monetary", color="Segmen", hover_data=["Customer_ID"],
                                    color_discrete_sequence=PINK, opacity=0.85)
                fig.update_traces(marker=dict(size=4))
            show_plot(style_fig(fig, 520, f"Sebaran Pelanggan per Segmen (k = {k_used})"), key="km_sc")
        with tabs[3]:
            st.markdown("#### 📣 Strategi Pemasaran per Segmen (Tugas 16)")
            for _, r in prof.iterrows():
                icon, strat = strategy_for(r["Segmen"])
                card(f"<h4>{icon} {r['Segmen']} — {fmt_int(r['Jumlah Pelanggan'])} pelanggan ({r['% Pelanggan']}%)</h4>"
                     f"<p><b>Profil:</b> Recency rata-rata {r['Recency Rata-rata']:.0f} hari · Frequency {r['Frequency Rata-rata']:.2f}× · Monetary {fmt_idr(r['Monetary Rata-rata'])}</p>"
                     f"<p><b>Strategi:</b> {strat}</p>")
        with tabs[4]:
            st.markdown("#### 🔎 Cek Pelanggan Individual")
            cid = st.selectbox("Pilih / ketik Customer_ID", sorted(km_df["Customer_ID"].unique()))
            row = km_df[km_df["Customer_ID"] == cid].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Segmen", row["Segmen"])
            c2.metric("Recency", f"{row['Recency']:.0f} hari")
            c3.metric("Frequency", f"{row['Frequency']:.0f}×")
            c4.metric("Monetary", fmt_idr(row["Monetary"]))
            hist = df_clean[df_clean["Customer_ID"] == cid][["Transaction_ID", "Transaction_Date", "Product_Name", "Category", "Quantity", "Line_Total"]]
            show_df(hist.sort_values("Transaction_Date", ascending=False))
            st.info(f"Strategi: {strategy_for(row['Segmen'])[1]}")
        nav_buttons()

    # ==================================================================
    # TAHAP 5
    # ==================================================================
    elif menu == MENU[5]:
        st.markdown("<h1>📈 Tahap 5: Business Intelligence Dashboard (15%)</h1>", unsafe_allow_html=True)
        km_df, k_used, _ = current_km(rfm)
        seg_map = km_df.set_index("Customer_ID")["Segmen"]
        dfx = df_clean.copy()
        dfx["Segmen"] = dfx["Customer_ID"].map(seg_map).fillna(UNKNOWN)

        with st.expander("🎛️ Filter Dashboard (tanggal, kategori, segmen, wilayah, tipe pelanggan)", expanded=True):
            f1, f2 = st.columns(2)
            dmin, dmax = dfx["Date"].min().date(), dfx["Date"].max().date()
            dr = f1.date_input("Rentang tanggal", value=(dmin, dmax), min_value=dmin, max_value=dmax)
            cats = sorted(dfx["Category"].unique())
            sel_cat = f2.multiselect("Kategori produk", cats, default=cats)
            f3, f4, f5 = st.columns(3)
            segs = sorted(dfx["Segmen"].unique())
            sel_seg = f3.multiselect("Segmen pelanggan", segs, default=segs)
            regs = sorted(dfx["Region"].unique())
            sel_reg = f4.multiselect("Wilayah", regs, default=regs)
            types = sorted(dfx["Customer_Type"].unique())
            sel_typ = f5.multiselect("Tipe pelanggan", types, default=types)
        if isinstance(dr, (tuple, list)) and len(dr) == 2:
            d0, d1 = dr
        else:
            d0 = d1 = dr[0] if isinstance(dr, (tuple, list)) else dr
        f = dfx[(dfx["Date"].dt.date >= d0) & (dfx["Date"].dt.date <= d1) & dfx["Category"].isin(sel_cat)
                & dfx["Segmen"].isin(sel_seg) & dfx["Region"].isin(sel_reg) & dfx["Customer_Type"].isin(sel_typ)]

        if len(f) == 0:
            st.warning("Tidak ada data untuk kombinasi filter ini. Longgarkan filter.")
            nav_buttons()
            st.stop()

        total_rev = f['Line_Total'].sum()
        total_trx = f['Transaction_ID'].nunique()
        total_cust = f['Customer_ID'].nunique()
        avg_basket = f.groupby('Transaction_ID')['Product_ID'].nunique().mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Pendapatan</div><div class='kpi-value'>{fmt_idr(total_rev)}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Jumlah Transaksi</div><div class='kpi-value'>{fmt_int(total_trx)}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Pelanggan Unik</div><div class='kpi-value'>{fmt_int(total_cust)}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Rata-rata Basket Size</div><div class='kpi-value'>{avg_basket:.2f}</div></div>", unsafe_allow_html=True)
        st.caption(f"Menampilkan {fmt_int(len(f))} dari {fmt_int(len(dfx))} baris · rata-rata nilai transaksi {fmt_idr(total_rev / max(total_trx, 1))}")
        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs(["📈 Tren & Komposisi", "🏆 Top 10 Produk / Kategori", "🛒 Hasil Apriori", "🎯 Profil K-Means", "🧩 Produk per Segmen (Bonus)", "🧠 Insight Manajemen"])
        with tabs[0]:
            gran = st.radio("Granularitas tren", ["Bulanan", "Mingguan", "Harian"], horizontal=True, key="gran")
            ctype = st.radio("Jenis grafik", ["Garis", "Batang"], horizontal=True, key="ctype")
            if gran == "Bulanan":
                tr = f.groupby("Month")["Line_Total"].sum().reset_index().rename(columns={"Month": "Periode"})
            elif gran == "Mingguan":
                tr = f.groupby(f["Date"].dt.to_period("W").dt.start_time)["Line_Total"].sum().reset_index().rename(columns={"Date": "Periode"})
            else:
                tr = f.groupby("Date")["Line_Total"].sum().reset_index().rename(columns={"Date": "Periode"})
            if ctype == "Garis":
                fig = px.line(tr, x="Periode", y="Line_Total", markers=(gran == "Bulanan"))
                fig.update_traces(line=dict(color="#C71585", width=3), fill="tozeroy", fillcolor="rgba(219,112,147,0.18)")
            else:
                fig = px.bar(tr, x="Periode", y="Line_Total", color_discrete_sequence=["#C71585"])
            fig.update_layout(yaxis_title="Pendapatan (Rp)", xaxis_title="")
            show_plot(style_fig(fig, 380, f"📈 Tren Penjualan {gran}"), key="d_trend")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 🥧 Distribusi Penjualan per Kategori")
                cat_sales = f.groupby('Category')['Line_Total'].sum().reset_index()
                fig = px.pie(cat_sales, names="Category", values="Line_Total", hole=0.4, color_discrete_sequence=PINK)
                fig.update_traces(textposition="inside", textinfo="percent", pull=[0.03] * len(cat_sales),
                                  marker=dict(line=dict(color="white", width=2)))
                fig.update_layout(legend=dict(orientation="v"))
                show_plot(style_fig(fig, 400, "Persentase Penjualan Berdasarkan Kategori"), key="d_pie")
            with col_b:
                st.markdown("#### 💳 Metode Pembayaran")
                pay_counts = f.drop_duplicates(["Transaction_ID", "Payment_Method"])["Payment_Method"].value_counts().reset_index()
                pay_counts.columns = ["Metode", "Jumlah"]
                fig = px.bar(pay_counts.sort_values("Jumlah"), x="Jumlah", y="Metode", orientation="h", color="Jumlah",
                             color_continuous_scale="PuRd", text="Jumlah")
                fig.update_layout(coloraxis_showscale=False, xaxis_title="Jumlah Transaksi")
                show_plot(style_fig(fig, 400, "Frekuensi Metode Pembayaran"), key="d_pay")
            reg = f.groupby("Region")["Line_Total"].sum().reset_index().sort_values("Line_Total", ascending=False)
            fig = px.bar(reg, x="Region", y="Line_Total", color="Line_Total", color_continuous_scale="PuRd")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="Pendapatan (Rp)")
            show_plot(style_fig(fig, 320, "🗺️ Pendapatan per Wilayah"), key="d_reg")
        with tabs[1]:
            c1, c2 = st.columns([1, 1])
            lvl = c1.radio("Tingkat", ["Produk", "Kategori"], horizontal=True, key="top_lvl")
            met = c2.radio("Diurutkan berdasarkan", ["Pendapatan", "Jumlah Item Terjual"], horizontal=True, key="top_met")
            key = "Product_Name" if lvl == "Produk" else "Category"
            val_col, fn = ("Line_Total", "sum") if met == "Pendapatan" else ("Quantity", "sum")
            tp = f.groupby(key)[val_col].agg(fn).sort_values(ascending=False).head(10).reset_index().sort_values(val_col)
            fig = px.bar(tp, x=val_col, y=key, orientation="h", color=val_col, color_continuous_scale="PuRd", text=tp[val_col].map(lambda v: fmt_int(v)))
            fig.update_layout(coloraxis_showscale=False, xaxis_title=met, yaxis_title="")
            show_plot(style_fig(fig, 460, f"🏆 Top 10 {lvl} — {met}"), key="d_top")
        with tabs[2]:
            P = current_ap()
            bk = build_basket(f)
            if bk.shape[0] < 100:
                st.info("Transaksi terlalu sedikit untuk Apriori pada filter ini.")
            else:
                its_d, rules_d = run_apriori(bk, P["sup"], P["conf"], P["lift"], P["len"])
                topd = rank_rules(rules_d, P["crit"], P["oneway"], 10)
                st.caption(f"Apriori dihitung pada {fmt_int(bk.shape[0])} transaksi hasil filter · support ≥ {P['sup']:.4f}, confidence ≥ {P['conf']:.2f}, lift ≥ {P['lift']:.1f} (ubah di Tahap 3).")
                if topd is not None and len(topd):
                    t = topd.drop(columns=["Ukuran Itemset"]).copy()
                    t["Support"] = t["Support"].round(4)
                    t["Confidence"] = t["Confidence"].round(3)
                    t["Lift"] = t["Lift"].round(3)
                    show_df(t)
                    bar = topd.assign(Aturan=topd["Antecedent"] + " → " + topd["Consequent"]).sort_values("Lift")
                    fig = px.bar(bar, x="Lift", y="Aturan", orientation="h", color="Confidence", color_continuous_scale="PuRd")
                    show_plot(style_fig(fig, 400, "Top Aturan Asosiasi (Lift)"), key="d_ap")
                else:
                    st.info("Tidak ada aturan pada filter ini.")
        with tabs[3]:
            prof_f = segment_profile(km_df)
            st.caption(f"Profil K-Means (k = {k_used}) dihitung pada seluruh pelanggan; filter segmen memengaruhi KPI & grafik lain.")
            show_df(prof_f)
            fig = px.scatter(km_df, x="Frequency", y="Monetary", color="Segmen", size="Recency", color_discrete_sequence=PINK,
                             hover_data=["Customer_ID"], opacity=0.75, size_max=14)
            show_plot(style_fig(fig, 420, "Frequency vs Monetary (ukuran = Recency)"), key="d_km")
        with tabs[4]:
            st.markdown("#### 🧩 Pola Pembelian Kategori per Segmen (Bonus)")
            mode = st.radio("Tampilkan", ["Kategori", "Produk"], horizontal=True, key="seg_mode")
            kk = "Category" if mode == "Kategori" else "Product_Name"
            pv = f[f["Segmen"] != UNKNOWN].groupby(["Segmen", kk])["Line_Total"].sum().reset_index()
            pv["Share"] = pv["Line_Total"] / pv.groupby("Segmen")["Line_Total"].transform("sum") * 100
            pvt = pv.pivot(index="Segmen", columns=kk, values="Share").fillna(0)
            fig = px.imshow(pvt, color_continuous_scale="PuRd", aspect="auto", labels=dict(color="% pendapatan"), text_auto=".0f" if mode == "Kategori" else False)
            show_plot(style_fig(fig, 420, f"% Pendapatan {mode} dalam Tiap Segmen"), key="d_seg")
            ov = f[f["Segmen"] != UNKNOWN].groupby(["Segmen", "Category"])["Line_Total"].sum().reset_index()
            fig = px.bar(ov, x="Segmen", y="Line_Total", color="Category", color_discrete_sequence=PINK + ["#FFC0CB", "#6A0572"])
            show_plot(style_fig(fig, 400, "Pendapatan per Segmen (komposisi kategori)"), key="d_seg2")
        with tabs[5]:
            mo = f.groupby("Month")["Line_Total"].sum()
            cr = f.groupby("Category")["Line_Total"].sum().sort_values(ascending=False)
            sg = f.groupby("Segmen")["Line_Total"].sum().sort_values(ascending=False)
            pr = f.groupby("Product_Name")["Line_Total"].sum().sort_values(ascending=False)
            for t in [
                f"Pendapatan tertinggi pada <b>{mo.idxmax()}</b> ({fmt_idr(mo.max())}); terendah <b>{mo.idxmin()}</b>. Jadwalkan promo & stok sesuai pola ini.",
                f"<b>{cr.index[0]}</b> adalah kategori terbesar ({cr.iloc[0] / cr.sum() * 100:.1f}%); 3 produk teratas: {', '.join(pr.index[:3])}.",
                f"Segmen <b>{sg.index[0]}</b> menyumbang {sg.iloc[0] / sg.sum() * 100:.1f}% pendapatan pada filter ini — prioritaskan retensinya.",
                f"Rata-rata <b>basket_size {avg_basket:.2f}</b> produk/transaksi: peluang menaikkan ukuran keranjang lewat cross-selling berdasarkan aturan Apriori.",
            ]:
                card(t)
            st.info("📸 **Tugas 19:** ambil tangkapan layar dashboard ini (beserta filter yang dipakai) untuk dilampirkan di laporan.")
            dl_csv(f.drop(columns=["Date", "Hour", "Weekday"]), "⬇️ Unduh Data Hasil Filter", "data_filter_dashboard.csv", "dl_filter")
        nav_buttons()

    # ==================================================================
    # TAHAP 6
    # ==================================================================
    elif menu == MENU[6]:
        st.markdown("<h1>💡 Tahap 6: Rekomendasi Keputusan Bisnis (10%)</h1>", unsafe_allow_html=True)
        P = current_ap()
        its6, rules6 = run_apriori(basket, P["sup"], P["conf"], P["lift"], P["len"])
        top6 = rank_rules(rules6, "Lift (tie-break Confidence)", True, 3)
        km6, k6, _ = current_km(rfm)
        prof6 = segment_profile(km6)
        tot_rev = df_clean["Line_Total"].sum()
        n_tx = df_clean["Transaction_ID"].nunique()
        avg_tx_val = tot_rev / n_tx
        mo = df_clean.groupby("Month")["Line_Total"].sum()

        def seg_row(prefix):
            m = prof6[prof6["Segmen"].str.startswith(prefix)]
            return m.iloc[0] if len(m) else None

        loyal, churn, newc = seg_row("Loyal"), seg_row("Churn"), seg_row("New")

        st.markdown("""
            <div class="card">
                <h4>Rekomendasi Preskriptif Manajemen</h4>
                <ol>
                    <li><b>Penataan Produk / Bundling:</b> Letakkan barang-barang dengan aturan asosiasi tinggi secara berdampingan di etalase toko dan tawarkan paket hemat.</li>
                    <li><b>Retensi Pelanggan:</b> Jalankan program loyalitas khusus untuk pelanggan segmen <i>Loyal Customers</i> dan kampanye re-engagement untuk <i>Churn Risk</i>.</li>
                    <li><b>Tindakan Pemasaran:</b> Selaraskan promo dengan bulan/jam puncak dan dorong pelanggan baru kembali berbelanja.</li>
                    <li><b>Prioritas Tindakan:</b> Fokuskan pada optimasi tata letak etalase sebagai langkah awal berbiaya rendah dengan dampak instan.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs(["🥇 Rek. 1: Penempatan & Bundling", "🥈 Rek. 2: Retensi Pelanggan", "🥉 Rek. 3: Tindakan Pemasaran",
                        "🧪 Fakta vs Estimasi vs Aktual", "📄 Ringkasan Eksekutif"])

        est = {}
        with tabs[0]:
            if len(top6):
                r0 = top6.iloc[0]
                a, c = r0["Antecedent"], r0["Consequent"]
                n_ant = int(round(basket[a.split(" + ")].all(axis=1).sum())) if all(x in basket.columns for x in a.split(" + ")) else 0
                avg_cons = df_clean[df_clean["Product_Name"] == c.split(" + ")[0]]["Line_Total"].mean()
                st.markdown(f"""
                    <div class="card">
                        <span class="badge badge-fact">FAKTA</span><b>Bukti data:</b> pembeli <i>{a}</i> juga membeli <i>{c}</i> dengan confidence {r0['Confidence']:.2f} dan lift {r0['Lift']:.2f} (support {r0['Support']:.3f}); {fmt_int(n_ant)} transaksi memuat {a}.<br><br>
                        <b>Tindakan:</b> tempatkan {a} dan {c} berdampingan, buat paket bundling, dan latih kasir menawarkan pasangan produk.<br>
                        <b>KPI keberhasilan:</b> attach rate {c} pada transaksi {a}, rata-rata basket_size, rata-rata nilai transaksi.<br>
                        <b>Cara evaluasi:</b> uji coba A/B — sebagian rak/kasir diubah, sebagian tidak, selama 4–8 minggu; bandingkan attach rate (uji proporsi) pada periode yang sama.
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("##### 🧮 Simulator Estimasi Manfaat (berbasis asumsi)")
                s1, s2, s3 = st.columns(3)
                up = s1.slider("Asumsi kenaikan attach rate (poin persen)", 0, 20, 5, key="s_up")
                mg = s2.slider("Asumsi margin kotor (%)", 5, 40, 15, key="s_mg")
                cost = s3.number_input("Biaya program (Rp): rak/label/pelatihan", 0, 50_000_000, 1_000_000, step=250_000, key="s_cost")
                inc_rev = n_ant * (up / 100) * avg_cons
                inc_profit = inc_rev * mg / 100
                est["bundling"] = (inc_rev, inc_profit - cost)
                m1, m2, m3 = st.columns(3)
                m1.metric("Estimasi tambahan pendapatan", fmt_idr(inc_rev))
                m2.metric("Estimasi laba tambahan", fmt_idr(inc_profit))
                m3.metric("Estimasi manfaat bersih", fmt_idr(inc_profit - cost), delta="ESTIMASI" )
                st.markdown("<span class='badge badge-est'>ESTIMASI</span> Angka di atas hanya asumsi (periode data 1 tahun). Bukan hasil terukur sampai uji coba dijalankan.", unsafe_allow_html=True)
            else:
                st.info("Belum ada aturan asosiasi — turunkan threshold di Tahap 3.")
        with tabs[1]:
            if loyal is not None and churn is not None:
                st.markdown(f"""
                    <div class="card">
                        <span class="badge badge-fact">FAKTA</span><b>Bukti data:</b> segmen <i>Loyal Customers</i> ada {fmt_int(loyal['Jumlah Pelanggan'])} pelanggan dengan rata-rata belanja {fmt_idr(loyal['Monetary Rata-rata'])} ({loyal['Frequency Rata-rata']:.1f}× transaksi); segmen <i>Churn Risk</i> {fmt_int(churn['Jumlah Pelanggan'])} pelanggan dengan Recency rata-rata {churn['Recency Rata-rata']:.0f} hari.<br><br>
                        <b>Tindakan:</b> program poin/tier untuk Loyal; kampanye re-engagement (WhatsApp + voucher terbatas waktu) untuk Churn Risk.<br>
                        <b>KPI keberhasilan:</b> repeat purchase rate, penurunan Recency rata-rata, jumlah pelanggan Churn Risk yang kembali bertransaksi ≤ 60 hari.<br>
                        <b>Cara evaluasi:</b> kelompok kontrol (tidak dikirimi voucher) vs perlakuan; bandingkan reaktivasi dan pendapatan per pelanggan 30–90 hari setelah kampanye.
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("##### 🧮 Simulator Estimasi Manfaat (berbasis asumsi)")
                s1, s2, s3 = st.columns(3)
                react = s1.slider("Asumsi % Churn Risk yang kembali", 0, 30, 8, key="s_react")
                disc = s2.slider("Biaya voucher per pelanggan (Rp)", 0, 20000, 5000, step=1000, key="s_disc")
                mg2 = s3.slider("Asumsi margin kotor (%) ", 5, 40, 15, key="s_mg2")
                per_visit = churn["Monetary Rata-rata"] / max(churn["Frequency Rata-rata"], 1)
                back = churn["Jumlah Pelanggan"] * react / 100
                inc2 = back * per_visit
                cost2 = churn["Jumlah Pelanggan"] * disc
                est["retensi"] = (inc2, inc2 * mg2 / 100 - cost2)
                m1, m2, m3 = st.columns(3)
                m1.metric("Pelanggan kembali (asumsi)", fmt_int(back))
                m2.metric("Estimasi tambahan pendapatan", fmt_idr(inc2))
                m3.metric("Estimasi manfaat bersih", fmt_idr(inc2 * mg2 / 100 - cost2), delta="ESTIMASI")
                st.markdown("<span class='badge badge-est'>ESTIMASI</span> Mengasumsikan setiap pelanggan yang kembali berbelanja sekali dengan nilai rata-rata per kunjungan segmen tersebut.", unsafe_allow_html=True)
            else:
                st.info("Segmen Loyal / Churn Risk tidak terbentuk pada k yang dipilih. Gunakan k ≥ 3 di Tahap 4.")
        with tabs[2]:
            hr = df_clean.drop_duplicates("Transaction_ID").groupby("Hour").size()
            top_h = hr.sort_values(ascending=False).head(3).index.sort_values()
            mem_share = df_clean.drop_duplicates("Transaction_ID")["Customer_Type"].eq("Member").mean() * 100
            nc = f"{fmt_int(newc['Jumlah Pelanggan'])} pelanggan baru" if newc is not None else "pelanggan berfrekuensi rendah"
            st.markdown(f"""
                <div class="card">
                    <span class="badge badge-fact">FAKTA</span><b>Bukti data:</b> penjualan puncak pada <b>{mo.idxmax()}</b> ({fmt_idr(mo.max())}) dan terendah <b>{mo.idxmin()}</b> ({fmt_idr(mo.min())}); jam tersibuk sekitar {', '.join(f'{h}:00' for h in top_h)}; {mem_share:.0f}% transaksi dilakukan Member; terdapat {nc}.<br><br>
                    <b>Tindakan:</b> (1) siapkan stok & staf ekstra pada bulan/jam puncak; (2) promo bulan sepi ({mo.idxmin()}) untuk meratakan permintaan; (3) kupon kunjungan ke-2 dan ajakan menjadi Member untuk pelanggan baru.<br>
                    <b>KPI keberhasilan:</b> pendapatan bulan sepi, rasio pelanggan baru yang melakukan transaksi ke-2, pertumbuhan jumlah Member.<br>
                    <b>Cara evaluasi:</b> bandingkan bulan promo dengan baseline periode sebelumnya (kontrol musiman) dan pantau ROI kampanye = (laba tambahan − biaya) ÷ biaya.
                </div>
            """, unsafe_allow_html=True)
            fig = px.bar(mo.reset_index(), x="Month", y="Line_Total", color="Line_Total", color_continuous_scale="PuRd")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="Pendapatan (Rp)", xaxis_title="")
            show_plot(style_fig(fig, 320, "Pendapatan Bulanan (dasar penentuan timing promo)"), key="r_mo")
            m_up = st.slider("Asumsi kenaikan pendapatan pada bulan sepi (%)", 0, 30, 10, key="s_mo")
            gain = mo.min() * m_up / 100
            est["pemasaran"] = (gain, None)
            st.metric("Estimasi tambahan pendapatan bulan sepi", fmt_idr(gain), delta="ESTIMASI")
        with tabs[3]:
            st.markdown("#### 🧪 Membedakan Fakta, Estimasi, dan Dampak Aktual (Tugas 22)")
            rows = []
            if len(top6):
                r0 = top6.iloc[0]
                rows.append(["FAKTA", f"{r0['Antecedent']} & {r0['Consequent']} dibeli bersama (lift {r0['Lift']:.2f})", "Teramati pada data historis 2025"])
            rows += [
                ["FAKTA", f"Total pendapatan Data_Clean {fmt_idr(tot_rev)} dari {fmt_int(n_tx)} transaksi", "Dihitung langsung dari data bersih"],
                ["FAKTA", f"Pelanggan terbagi {len(prof6)} segmen RFM", "Hasil K-Means pada data historis"],
            ]
            if "bundling" in est:
                rows.append(["ESTIMASI", f"Tambahan pendapatan bundling ≈ {fmt_idr(est['bundling'][0])}", "Bergantung asumsi kenaikan attach rate & margin (simulator)"])
            if "retensi" in est:
                rows.append(["ESTIMASI", f"Tambahan pendapatan reaktivasi ≈ {fmt_idr(est['retensi'][0])}", "Bergantung asumsi persentase pelanggan kembali"])
            rows.append(["DAMPAK AKTUAL", "Kenaikan penjualan setelah implementasi", "BELUM DIUKUR — harus diuji lewat uji coba A/B / pre-post setelah program berjalan"])
            tb = pd.DataFrame(rows, columns=["Jenis", "Pernyataan", "Dasar / Catatan"])
            show_df(tb)
            st.markdown("""
                <div class="card-soft">
                <span class="badge badge-fact">FAKTA</span> hasil olah data historis ·
                <span class="badge badge-est">ESTIMASI</span> proyeksi berbasis asumsi yang bisa diubah ·
                <span class="badge badge-act">AKTUAL</span> hanya boleh diklaim setelah implementasi dan evaluasi.<br>
                <b>Asumsi & batasan:</b> dataset sintetis 1 tahun; aturan asosiasi dan klaster bersifat korelasional, bukan sebab-akibat; margin dan biaya adalah angka ilustrasi yang harus diganti dengan data keuangan toko.
                </div>
            """, unsafe_allow_html=True)
        with tabs[4]:
            lines = ["# Ringkasan Eksekutif — UTS Business Intelligence", "",
                     "Disusun oleh: Liliyosa Jingga Luahane (51230087) — Universitas Timor", "",
                     "## Gambaran Data",
                     f"- Data mentah {fmt_int(len(df_raw))} baris → data bersih {fmt_int(len(df_clean))} baris",
                     f"- Total pendapatan (bersih): {fmt_idr(tot_rev)} dari {fmt_int(n_tx)} transaksi, {fmt_int(df_clean['Customer_ID'].nunique())} pelanggan unik",
                     f"- Rata-rata basket_size: {df_clean.drop_duplicates('Transaction_ID')['basket_size'].mean():.2f}", "",
                     "## Temuan Apriori (FAKTA)"]
            for _, r in top6.iterrows():
                lines.append(f"- {r['Antecedent']} → {r['Consequent']}: support {r['Support']:.3f}, confidence {r['Confidence']:.2f}, lift {r['Lift']:.2f}")
            lines += ["", f"## Segmentasi K-Means (k = {k6})"]
            for _, r in prof6.iterrows():
                lines.append(f"- {r['Segmen']}: {fmt_int(r['Jumlah Pelanggan'])} pelanggan, Recency {r['Recency Rata-rata']:.0f} hari, Frequency {r['Frequency Rata-rata']:.2f}, Monetary {fmt_idr(r['Monetary Rata-rata'])}")
            lines += ["", "## Rekomendasi",
                      "1. Penempatan rak & bundling untuk pasangan produk berlift tertinggi (uji A/B).",
                      "2. Program loyalitas untuk Loyal Customers dan re-engagement untuk Churn Risk (kelompok kontrol).",
                      "3. Selaraskan promo dengan bulan/jam puncak dan konversi pelanggan baru.", "",
                      "> Estimasi manfaat adalah proyeksi berbasis asumsi; dampak aktual belum diuji."]
            md = "\n".join(lines)
            st.markdown(md)
            st.download_button("⬇️ Unduh Ringkasan Eksekutif (.md)", md.encode("utf-8"), file_name="ringkasan_eksekutif.md", mime="text/markdown", key="dl_md")
        nav_buttons()

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #C71585;'>💖 Dibuat dengan penuh ketelitian untuk UTS Business Intelligence — Liliyosa Jingga Luahane (51230087) | Universitas Timor 💖</p>", unsafe_allow_html=True)