import io
import glob
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_OK = True
except ImportError:  # pragma: no cover
    MLXTEND_OK = False

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="UTS BI - Liliyosa Jingga Luahane",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

PINK = ["#C71585", "#FF69B4", "#DB7093", "#BA55D3", "#8B008B", "#FFB6C1", "#900C3F", "#FFC0CB"]
SEG_COLORS = {
    "Loyal Customers": "#C71585",
    "Potential Customers": "#FF69B4",
    "Low-Spend Active": "#BA55D3",
    "New Customers": "#FFB6C1",
    "Churn Risk": "#8B008B",
}

MENU = [
    "🏠 Beranda & Upload Dataset",
    "📊 Tahap 1: EDA & Business Understanding",
    "🧹 Tahap 2: Data Preprocessing",
    "🛒 Tahap 3: Apriori (Market Basket Analysis)",
    "🎯 Tahap 4: K-Means (Segmentasi Pelanggan)",
    "📈 Tahap 5: BI Dashboard & Visualisasi",
    "💡 Tahap 6: Rekomendasi Keputusan Bisnis",
]

# =====================================================================
# CSS KUSTOM INTERAKTIF (PINK HATI & KUPU-KUPU)
# =====================================================================
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"], .main {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"], .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FFC0CB 0%, #FFE4E1 100%);
    }
    h1, h2, h3, h4 {
        color: #C71585;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .subtitle { color: #DB7093; font-size: 16px; margin: -8px 0 18px 0; }
    .badge {
        display: inline-block; background: linear-gradient(45deg, #C71585, #DB7093);
        color: white; padding: 2px 12px; border-radius: 20px; font-size: 12px;
        font-weight: bold; margin-left: 8px; vertical-align: middle;
    }

    /* ---------- Tombol ---------- */
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
        transform: translateY(-2px) scale(1.03);
    }
    .stButton>button:active { transform: scale(0.97); }
    .stButton>button p, .stDownloadButton>button p, .stFormSubmitButton>button p { color: white; }

    /* ---------- Kartu ---------- */
    .card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 6px 15px rgba(199, 21, 133, 0.12);
        margin-bottom: 20px;
        border-left: 6px solid #C71585;
        transition: 0.3s;
    }
    .card:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(199, 21, 133, 0.22); }
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #FFF5EE 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(199, 21, 133, 0.1);
        text-align: center;
        border: 2px solid #FFC0CB;
        transition: 0.3s;
    }
    .kpi-card:hover { transform: translateY(-5px) scale(1.02); border-color: #C71585; box-shadow: 0 10px 20px rgba(199, 21, 133, 0.25); }
    .kpi-title { color: #8B008B; font-size: 14px; font-weight: bold; margin-bottom: 5px; }
    .kpi-value { color: #C71585; font-size: 24px; font-weight: 800; }
    .insight {
        background: rgba(255, 255, 255, 0.85); border-left: 5px solid #FF69B4;
        padding: 12px 16px; border-radius: 12px; margin: 8px 0; color: #5a1040;
    }
    .fact { border-left-color: #C71585; }
    .estimate { border-left-color: #BA55D3; }
    .actual { border-left-color: #8B008B; background: #FFF0F5; }

    /* ---------- Menu sidebar bergaya tombol ---------- */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.78); border: 1px solid #FFC0CB; border-radius: 14px;
        padding: 9px 12px; margin-bottom: 8px; transition: 0.25s; width: 100%;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        transform: translateX(6px); box-shadow: 0 4px 12px rgba(199, 21, 133, 0.25);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(45deg, #C71585, #DB7093); border-color: #C71585;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color: white !important; font-weight: 700;
    }

    /* ---------- Tab, metric, expander ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.8); border-radius: 20px; padding: 6px 16px;
        border: 1px solid #FFC0CB; transition: 0.25s;
    }
    .stTabs [data-baseweb="tab"]:hover { background: #FFC0CB; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(45deg, #C71585, #DB7093); border-color: #C71585;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p { color: white !important; font-weight: 700; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
    [data-testid="stMetric"] {
        background: white; border: 2px solid #FFC0CB; border-radius: 15px; padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(199, 21, 133, 0.1);
    }
    [data-testid="stMetricValue"] { color: #C71585; }
    [data-testid="stExpander"] { background: rgba(255, 255, 255, 0.8); border-radius: 14px; border: 1px solid #FFC0CB; }

    /* ---------- Kupu-kupu melayang ---------- */
    .bf { position: fixed; bottom: -60px; font-size: 26px; opacity: 0.5; pointer-events: none; z-index: 0;
          animation: floaty 16s linear infinite; }
    @keyframes floaty {
        0%   { transform: translate(0, 0) rotate(0deg); }
        50%  { transform: translate(45px, -55vh) rotate(18deg); }
        100% { transform: translate(-30px, -115vh) rotate(-14deg); }
    }
    </style>
    <div class="bf" style="left:6%; animation-delay:0s">🦋</div>
    <div class="bf" style="left:24%; animation-delay:4s">💖</div>
    <div class="bf" style="left:47%; animation-delay:8s">🦋</div>
    <div class="bf" style="left:68%; animation-delay:2s">💗</div>
    <div class="bf" style="left:86%; animation-delay:6s">🦋</div>
""", unsafe_allow_html=True)

# =====================================================================
# UTILITAS TAMPILAN
# =====================================================================
def fmt_idr(val):
    """Format Rupiah dengan titik ribuan."""
    return f"Rp {val:,.0f}".replace(",", ".")


def fmt_int(val):
    return f"{val:,.0f}".replace(",", ".")


def fmt_short(val):
    """Format Rupiah ringkas untuk kartu KPI."""
    if abs(val) >= 1e9:
        return f"Rp {val / 1e9:,.2f} M".replace(".", ",")
    if abs(val) >= 1e6:
        return f"Rp {val / 1e6:,.1f} jt".replace(".", ",")
    return fmt_idr(val)


def hero(title, subtitle="", badge=None):
    b = f"<span class='badge'>{badge}</span>" if badge else ""
    st.markdown(f"<h1>{title}</h1><div class='subtitle'>{subtitle} {b}</div>", unsafe_allow_html=True)


def card(html):
    st.markdown(f"<div class='card'>{html}</div>", unsafe_allow_html=True)


def insight(text, kind=""):
    st.markdown(f"<div class='insight {kind}'>{text}</div>", unsafe_allow_html=True)


def kpi(title, value, tip=""):
    return f"<div class='kpi-card' title='{tip}'><div class='kpi-title'>{title}</div><div class='kpi-value'>{value}</div></div>"


def style_fig(fig, height=380, title=None):
    fig.update_layout(
        height=height, template="plotly_white", colorway=PINK,
        font=dict(color="#8B008B"),
        title=dict(text=title, font=dict(color="#C71585", size=16)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.85)",
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        hoverlabel=dict(bgcolor="#FFF0F5", font_color="#900C3F"),
    )
    return fig


def show_fig(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kwargs):
    kwargs.setdefault("hide_index", True)
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, **kwargs)


def csv_bytes(df, index=False):
    return df.to_csv(index=index).encode("utf-8-sig")


def goto(page):
    st.session_state["menu"] = page


# =====================================================================
# LAPISAN DATA: MEMBACA, AUDIT, PREPROCESSING
# =====================================================================
PAYMENT_MAP = {"QRIS": "QRIS", "TUNAI": "Tunai", "DEBIT": "Debit", "TRANSFER": "Transfer"}


@st.cache_data(show_spinner="📂 Membaca file Excel...")
def parse_excel(content: bytes):
    xl = pd.ExcelFile(io.BytesIO(content))
    sheets = xl.sheet_names
    if "Data_Raw" not in sheets:
        return None, None, None, None, sheets
    raw = xl.parse("Data_Raw")
    prod = xl.parse("Data_Produk") if "Data_Produk" in sheets else None
    cust = xl.parse("Data_Pelanggan") if "Data_Pelanggan" in sheets else None
    dic = xl.parse("Data_Dictionary") if "Data_Dictionary" in sheets else None
    return raw, prod, cust, dic, sheets


def to_datetime_safe(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def build_ref(raw, prod):
    """Katalog referensi produk (dari sheet Data_Produk, atau diturunkan dari Data_Raw)."""
    need = {"Product_ID", "Product_Name", "Category", "Reference_Unit_Price_Rp"}
    if prod is not None and need <= set(prod.columns):
        return prod[list(need)].drop_duplicates("Product_ID")
    r = raw.copy()
    r["Category"] = r["Category"].astype(str).str.strip().str.title().replace({"Atk": "ATK"})
    valid = r[(r["Unit_Price"] > 0) & (r["Unit_Price"] < 500000)]
    rows = []
    for name, g in valid.groupby("Product_Name"):
        ids = g["Product_ID"].dropna()
        rows.append({
            "Product_Name": name,
            "Product_ID": ids.mode().iloc[0] if len(ids) else np.nan,
            "Category": g["Category"].mode().iloc[0],
            "Reference_Unit_Price_Rp": g["Unit_Price"].median(),
        })
    return pd.DataFrame(rows)


def audit_raw(raw, ref, qty_max=50):
    """Audit kualitas Data_Raw."""
    cats_ok = set(ref["Category"].unique())
    q = raw["Quantity"]
    p = raw["Unit_Price"]
    pay = raw["Payment_Method"].dropna()
    audit = {
        "missing": raw.isna().sum(),
        "dup": int(raw.duplicated().sum()),
        "cat_incons": int((~raw["Category"].isin(cats_ok)).sum()),
        "pay_incons": int((~pay.isin(list(PAYMENT_MAP.values()))).sum()),
        "qty_invalid": int(((q <= 0) | (q > qty_max)).sum()),
        "price_invalid": int(((p <= 0) | (p > 500000)).sum()),
        "lt_mismatch": int(((raw["Quantity"] * raw["Unit_Price"] * (1 - raw["Discount_Rate"]) - raw["Line_Total"]).abs() > 1).sum()),
    }
    return audit


@st.cache_data(show_spinner="🦋 Membersihkan data (preprocessing)...")
def run_preprocessing(raw, prod, cust, qty_max):
    """Preprocessing lengkap. Data_Raw TIDAK ditimpa (bekerja pada salinan)."""
    ref = build_ref(raw, prod)
    id2cat = ref.set_index("Product_ID")["Category"].to_dict()
    id2price = ref.set_index("Product_ID")["Reference_Unit_Price_Rp"].to_dict()
    name2id = ref.set_index("Product_Name")["Product_ID"].to_dict()

    df = raw.copy()
    log = []

    def add(masalah, jumlah, keputusan, aksi, before, after):
        log.append({"Masalah": masalah, "Jumlah Terdeteksi": int(jumlah), "Keputusan": keputusan,
                    "Penanganan": aksi, "Baris Sebelum": int(before), "Baris Sesudah": int(after)})

    n = len(df)
    add("Salinan data kerja dibuat", 0, "Disalin", "Data_Raw tidak ditimpa; seluruh proses memakai salinan", n, n)

    # 1. Duplikasi baris penuh
    dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    add("Baris duplikat persis", dup, "Dihapus", "drop_duplicates() — duplikat menggandakan penjualan", n, len(df))
    n = len(df)

    # 2. Format tanggal
    df["Transaction_Date"] = to_datetime_safe(df["Transaction_Date"])
    bad_date = int(df["Transaction_Date"].isna().sum())
    df = df.dropna(subset=["Transaction_Date"])
    add("Format tanggal tidak valid", bad_date, "Dihapus bila gagal", "Dikonversi ke datetime; baris gagal konversi dibuang", n, len(df))
    n = len(df)

    # 3. Product_ID kosong -> dari Product_Name
    miss_pid = int(df["Product_ID"].isna().sum())
    df["Product_Name"] = df["Product_Name"].astype(str).str.strip()
    df["Product_ID"] = df["Product_ID"].fillna(df["Product_Name"].map(name2id))
    left = int(df["Product_ID"].isna().sum())
    df = df.dropna(subset=["Product_ID"])
    add("Product_ID kosong", miss_pid, "Diperbaiki", f"Diisi dari katalog Data_Produk berdasarkan Product_Name (sisa tak terpetakan: {left}, dihapus)", n, len(df))
    n = len(df)

    # 4. Kategori tidak konsisten
    ref_cat = df["Product_ID"].map(id2cat)
    fallback = df["Category"].astype(str).str.strip().str.title().replace({"Atk": "ATK"})
    changed = int(((df["Category"] != ref_cat) & ref_cat.notna()).sum())
    df["Category"] = ref_cat.fillna(fallback)
    add("Kategori tidak konsisten (kapital/spasi)", changed, "Diperbaiki", "Distandarkan mengikuti katalog Data_Produk berdasarkan Product_ID", n, len(df))

    # 5. Metode pembayaran
    pm_old = df["Payment_Method"]
    pm_std = pm_old.astype(str).str.strip().str.upper().map(PAYMENT_MAP)
    changed = int((pm_old.notna() & (pm_old != pm_std) & pm_std.notna()).sum())
    miss_pm = int(pm_old.isna().sum())
    df["Payment_Method"] = pm_std.fillna("Tidak Diketahui")
    add("Metode pembayaran tidak konsisten (ejaan/spasi)", changed, "Diperbaiki", "Distandarkan: QRIS, Tunai, Debit, Transfer", n, len(df))
    add("Metode pembayaran kosong", miss_pm, "Ditandai", "Diisi 'Tidak Diketahui' (tidak memengaruhi nilai transaksi)", n, len(df))

    # 6. Customer_ID kosong
    miss_c = int(df["Customer_ID"].isna().sum())
    first_c = df.groupby("Transaction_ID")["Customer_ID"].transform("first")
    df["Customer_ID"] = df["Customer_ID"].fillna(first_c)
    left = int(df["Customer_ID"].isna().sum())
    df["Customer_ID"] = df["Customer_ID"].fillna("UNKNOWN")
    add("Customer_ID kosong", miss_c, "Diimputasi", f"Diisi dari baris lain pada Transaction_ID yang sama; sisa {left} baris ditandai 'UNKNOWN' (dikeluarkan hanya dari RFM)", n, len(df))

    # 7. Region & Customer_Type kosong
    cm = cust.drop_duplicates("Customer_ID").set_index("Customer_ID") if cust is not None else None
    for col in ["Region", "Customer_Type"]:
        miss = int(df[col].isna().sum())
        df[col] = df[col].fillna(df.groupby("Transaction_ID")[col].transform("first"))
        if cm is not None and col in cm.columns:
            df[col] = df[col].fillna(df["Customer_ID"].map(cm[col]))
        df[col] = df[col].fillna("Tidak Diketahui")
        add(f"{col} kosong", miss, "Diimputasi", "Diisi dari transaksi yang sama, lalu dari Data_Pelanggan; sisanya 'Tidak Diketahui'", n, len(df))

    # 8. Harga satuan tidak valid
    price = df["Unit_Price"]
    refp = df["Product_ID"].map(id2price)
    bad_p = price.isna() | (price <= 0) | (price > 5 * refp)
    df.loc[bad_p, "Unit_Price"] = refp[bad_p]
    add("Unit_Price kosong / ≤ 0 / tidak wajar", bad_p.sum(), "Diperbaiki", "Diganti harga referensi katalog produk (Reference_Unit_Price_Rp)", n, len(df))

    # 9. Quantity tidak valid
    q = df["Quantity"]
    bad_q = q.isna() | (q <= 0) | (q > qty_max)
    est = (df["Line_Total"] / (df["Unit_Price"] * (1 - df["Discount_Rate"]))).round()
    fixable = bad_q & est.between(1, qty_max)
    df.loc[fixable, "Quantity"] = est[fixable]
    unfix = df["Quantity"].isna() | (df["Quantity"] <= 0) | (df["Quantity"] > qty_max)
    n_unfix = int(unfix.sum())
    df = df[~unfix]
    add(f"Quantity kosong / ≤ 0 / > {qty_max}", bad_q.sum(), "Diperbaiki / dihapus",
        f"{int(fixable.sum())} baris dihitung ulang dari Line_Total ÷ (harga × (1−diskon)); {n_unfix} baris tak dapat dipulihkan dihapus", n, len(df))
    n = len(df)

    # 10. Line_Total
    calc = df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Rate"])
    mism = int((df["Line_Total"].isna() | ((calc - df["Line_Total"]).abs() > 1)).sum())
    df["Line_Total"] = calc
    df = df[df["Line_Total"] > 0]
    add("Line_Total kosong / tidak konsisten", mism, "Dihitung ulang", "Line_Total = Quantity × Unit_Price × (1 − Discount_Rate)", n, len(df))
    n = len(df)

    # 11. Total transaksi tidak masuk akal (hanya ditandai)
    trx = df.groupby("Transaction_ID")["Line_Total"].sum()
    q1, q3 = trx.quantile([0.25, 0.75])
    fence = q3 + 3 * (q3 - q1)
    out_ids = set(trx[trx > fence].index)
    df["Flag_Outlier_Trx"] = df["Transaction_ID"].isin(out_ids)
    add("Total transaksi ekstrem (> Q3 + 3×IQR)", len(out_ids), "Ditandai", "Kolom Flag_Outlier_Trx; tetap disimpan karena belanja besar wajar", n, len(df))

    # 12. Produk ganda dalam satu transaksi
    dup_tp = int(df.duplicated(["Transaction_ID", "Product_ID"]).sum())
    add("Produk sama muncul 2× dalam satu transaksi", dup_tp, "Dibiarkan", "Basket Apriori memakai himpunan produk UNIK per transaksi", n, len(df))

    df["Month"] = df["Transaction_Date"].dt.to_period("M").astype(str)
    df["Customer_Unknown"] = df["Customer_ID"] == "UNKNOWN"
    df = df.sort_values(["Transaction_ID", "Product_ID"]).reset_index(drop=True)
    return df, pd.DataFrame(log), {"outlier_fence": float(fence), "n_outlier_trx": len(out_ids)}


@st.cache_data
def make_basket_matrix(df):
    """Matriks transaksi × produk (True jika produk dibeli) — 1 baris = 1 transaksi."""
    m = df.assign(v=1).pivot_table(index="Transaction_ID", columns="Product_Name", values="v", aggfunc="max", fill_value=0)
    return m.astype(bool)


@st.cache_data
def make_rfm(df):
    d = df[df["Customer_ID"] != "UNKNOWN"]
    ref = d["Transaction_Date"].max().normalize() + pd.Timedelta(days=1)
    g = d.groupby("Customer_ID")
    rfm = pd.DataFrame({
        "Recency": (ref - g["Transaction_Date"].max()).dt.days,
        "Frequency": g["Transaction_ID"].nunique(),
        "Monetary": g["Line_Total"].sum(),
        "Tenure": (ref - g["Transaction_Date"].min()).dt.days,
    })
    return rfm, ref


# =====================================================================
# APRIORI
# =====================================================================
def _fmt_set(s):
    return " + ".join(sorted(s))


@st.cache_data(show_spinner="🛒 Menjalankan Apriori...")
def run_apriori(matrix, min_support, min_conf, min_lift, max_len):
    n = len(matrix)
    fi = apriori(matrix, min_support=min_support, use_colnames=True, max_len=max_len)
    if fi.empty:
        return fi, pd.DataFrame()
    fi["Ukuran"] = fi["itemsets"].apply(len)
    fi["Itemset"] = fi["itemsets"].apply(_fmt_set)
    fi["Jumlah Transaksi"] = (fi["support"] * n).round().astype(int)
    if not (fi["Ukuran"] > 1).any():
        return fi, pd.DataFrame()
    rules = association_rules(fi[["support", "itemsets"]], metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return fi, rules
    rules["Jika (Antecedent)"] = rules["antecedents"].apply(_fmt_set)
    rules["Maka (Consequent)"] = rules["consequents"].apply(_fmt_set)
    rules["Aturan"] = rules["Jika (Antecedent)"] + "  →  " + rules["Maka (Consequent)"]
    rules["Jumlah Transaksi"] = (rules["support"] * n).round().astype(int)
    rules["n_ant"] = rules["antecedents"].apply(len)
    rules["n_con"] = rules["consequents"].apply(len)
    rules["Pasangan"] = rules.apply(lambda r: _fmt_set(set(r["antecedents"]) | set(r["consequents"])), axis=1)
    rules = rules.drop(columns=["antecedents", "consequents"])
    return fi, rules.reset_index(drop=True)


def rank_rules(rules, criteria, one_way=True, top=10):
    if rules.empty:
        return rules
    r = rules.copy()
    r["Skor"] = r["lift"] * r["confidence"]
    sort_map = {
        "Lift → Confidence → Support": ["lift", "confidence", "support"],
        "Confidence → Lift → Support": ["confidence", "lift", "support"],
        "Support → Lift → Confidence": ["support", "lift", "confidence"],
        "Skor gabungan (Lift × Confidence)": ["Skor", "lift", "support"],
    }
    r = r.sort_values(sort_map[criteria], ascending=False)
    if one_way:
        r = r.drop_duplicates("Pasangan", keep="first")
    return r.head(top).reset_index(drop=True)


RULE_COLS = {"Aturan": "Aturan (Jika → Maka)", "support": "Support", "confidence": "Confidence",
             "lift": "Lift", "Jumlah Transaksi": "Jml Transaksi"}


def rules_table(r):
    t = r[list(RULE_COLS.keys())].rename(columns=RULE_COLS).copy()
    t.insert(0, "Peringkat", range(1, len(t) + 1))
    return t


# =====================================================================
# K-MEANS
# =====================================================================
def scale_rfm(rfm, use_log):
    X = rfm[["Recency", "Frequency", "Monetary"]].astype(float)
    if use_log:
        X = np.log1p(X)
    return StandardScaler().fit_transform(X)


@st.cache_data(show_spinner="🎯 Menguji kandidat k (2–8)...")
def kmeans_eval(rfm, use_log):
    X = scale_rfm(rfm, use_log)
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"k": k, "Inertia (WCSS)": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def base_label(label):
    return label.split(" #")[0]


def label_clusters(rc):
    """Label klaster berbasis data (bandingkan rata-rata klaster dengan rata-rata seluruh pelanggan)."""
    prof = rc.groupby("Cluster").agg(R=("Recency", "mean"), F=("Frequency", "mean"),
                                     M=("Monetary", "mean"), T=("Tenure", "mean"))
    zR = (prof["R"] - rc["Recency"].mean()) / rc["Recency"].std()
    zF = (prof["F"] - rc["Frequency"].mean()) / rc["Frequency"].std()
    zM = (prof["M"] - rc["Monetary"].mean()) / rc["Monetary"].std()
    fm = (zF + zM) / 2
    labels = {}
    for c in prof.index:
        if zR[c] > 0.6:
            labels[c] = "Churn Risk"
        elif fm[c] > 0.5 and zR[c] < 0.3:
            labels[c] = "Loyal Customers"
        elif fm[c] < -0.3 and prof.loc[c, "T"] < 120:
            labels[c] = "New Customers"
        elif fm[c] < -0.3:
            labels[c] = "Low-Spend Active"
        else:
            labels[c] = "Potential Customers"
    cnt = Counter(labels.values())
    seen = {}
    for c in sorted(labels, key=lambda x: -fm[x]):
        b = labels[c]
        if cnt[b] > 1:
            seen[b] = seen.get(b, 0) + 1
            labels[c] = f"{b} #{seen[b]}"
    return labels


@st.cache_data(show_spinner="🎯 Menjalankan K-Means...")
def kmeans_fit(rfm, k, use_log):
    X = scale_rfm(rfm, use_log)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rc = rfm.copy()
    rc["Cluster"] = km.labels_
    labels = label_clusters(rc)
    rc["Segment"] = rc["Cluster"].map(labels)
    prof = rc.groupby("Segment").agg(
        Pelanggan=("Recency", "size"),
        Recency_Mean=("Recency", "mean"), Recency_Median=("Recency", "median"),
        Frequency_Mean=("Frequency", "mean"), Frequency_Median=("Frequency", "median"),
        Monetary_Mean=("Monetary", "mean"), Monetary_Median=("Monetary", "median"),
        Total_Monetary=("Monetary", "sum"), Tenure_Mean=("Tenure", "mean"),
    ).reset_index().sort_values("Monetary_Mean", ascending=False).reset_index(drop=True)
    prof["% Pelanggan"] = prof["Pelanggan"] / prof["Pelanggan"].sum() * 100
    prof["% Pendapatan"] = prof["Total_Monetary"] / prof["Total_Monetary"].sum() * 100
    return rc, prof


STRATEGY = {
    "Loyal Customers": ("Belanja sering, bernilai tinggi, dan masih aktif.",
                        ["Program loyalitas bertingkat (poin, tier Silver/Gold)", "Akses awal produk baru & bundling eksklusif",
                         "Program referral — jadikan duta toko", "Jaga ketersediaan produk favorit (Sembako, Bayi)"]),
    "Potential Customers": ("Nilai dan frekuensi menengah — berpotensi naik kelas menjadi loyal.",
                            ["Promo 'belanja kelipatan' untuk menaikkan frekuensi", "Cross-selling berdasarkan aturan Apriori",
                             "Voucher kunjungan berikutnya dengan batas waktu", "Tawarkan keanggotaan (member) bila masih Non-Member"]),
    "Low-Spend Active": ("Masih aktif membeli tetapi nilai belanja rendah.",
                         ["Paket hemat / bundling untuk menaikkan nilai keranjang", "Upselling ke produk bernilai lebih tinggi",
                          "Gratis ongkir atau diskon minimum belanja tertentu", "Edukasi manfaat belanja bulanan"]),
    "New Customers": ("Pelanggan dengan riwayat pembelian pendek dan nilai belanja rendah.",
                      ["Welcome voucher untuk pembelian kedua", "Onboarding keanggotaan (member)",
                       "Rekomendasi produk populer", "Follow-up via WhatsApp 7–14 hari setelah transaksi pertama"]),
    "Churn Risk": ("Lama tidak berbelanja — berisiko tidak kembali.",
                   ["Kampanye re-engagement (voucher 'kami rindu kamu')", "Survei singkat alasan berhenti berbelanja",
                    "Penawaran terbatas pada kategori yang dulu sering dibeli", "Batasi biaya: fokus pada yang pernah bernilai tinggi"]),
}

SEG_FALLBACK = ["#C71585", "#FF69B4", "#BA55D3", "#8B008B", "#DB7093", "#FFB6C1", "#900C3F", "#FFC0CB"]


def seg_color_map(segments):
    out = {}
    for i, s in enumerate(sorted(segments)):
        out[s] = SEG_COLORS.get(s, SEG_FALLBACK[i % len(SEG_FALLBACK)])
    return out

# =====================================================================
# IDENTITAS & SESSION STATE
# =====================================================================
NAMA = "Liliyosa Jingga Luahane"
NPM = "51230087"
FOTO = "Liliyosa.jpeg"

DEFAULTS = {
    "logged_in": False, "df_raw": None, "df_prod": None, "df_cust": None, "df_dic": None,
    "menu": MENU[0], "qty_max": 50,
    "ap_sup": 0.01, "ap_conf": 0.2, "ap_lift": 1.2, "ap_len": 3,
    "km_log": False,
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def img_b64(path):
    try:
        import base64
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


# =====================================================================
# HALAMAN LOGIN — FOTO & IDENTITAS BERDAMPINGAN (UKURAN SAMA)
# =====================================================================
def login_page():
    st.markdown("""
        <style>
        .profile-row { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: stretch;
                       height: 330px; margin-bottom: 22px; }
        .profile-photo { position: relative; border-radius: 18px; overflow: hidden; border: 4px solid #FFC0CB;
                         box-shadow: 0 6px 15px rgba(199, 21, 133, 0.18); background: #FFF0F5; height: 100%; }
        .profile-photo img { width: 100%; height: 100%; object-fit: cover; object-position: top; display: block; }
        .profile-photo .cap { position: absolute; left: 0; right: 0; bottom: 0; padding: 22px 10px 8px 10px;
                              background: linear-gradient(transparent, rgba(144, 12, 63, 0.75));
                              color: white; text-align: center; font-size: 13px; font-weight: 600; }
        .profile-photo .ph { height: 100%; display: flex; align-items: center; justify-content: center;
                             flex-direction: column; color: #C71585; text-align: center; padding: 12px; }
        .profile-card { height: 100%; margin-bottom: 0 !important; display: flex; flex-direction: column;
                        justify-content: center; box-sizing: border-box; }
        .profile-card p { margin: 6px 0; font-size: 16px; }
        @media (max-width: 700px) {
            .profile-row { grid-template-columns: 1fr; height: auto; }
            .profile-photo { height: 320px; }
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: #C71585;'>🦋 PORTAL UJIAN TENGAH SEMESTER (UTS) 🦋</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #DB7093;'>Business Intelligence & Data Mining</h3>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        b64 = img_b64(FOTO)
        if b64:
            photo_html = f"<img src='data:image/jpeg;base64,{b64}' alt='Foto profil'><div class='cap'>{NAMA} ({NPM})</div>"
        else:
            photo_html = (f"<div class='ph'><div style='font-size:46px'>🦋</div>"
                          f"<b>{NAMA}</b><small>Letakkan file <code>{FOTO}</code><br>di folder yang sama dengan app.py</small></div>")
        st.markdown(f"""
            <div class="profile-row">
                <div class="profile-photo">{photo_html}</div>
                <div class="card profile-card">
                    <h4>💖 Identitas Mahasiswa 💖</h4>
                    <p><b>Nama:</b> {NAMA}</p>
                    <p><b>NPM:</b> {NPM}</p>
                    <p><b>Prodi:</b> Teknologi Informasi</p>
                    <p><b>Universitas:</b> Universitas Timor</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username_input = st.text_input("Username", value="liliyosa")
            password_input = st.text_input("Password", type="password", value=NPM)
            submit_login = st.form_submit_button("🔐 Login Sistem")
            if submit_login:
                if username_input == "liliyosa" and password_input == NPM:
                    st.session_state.logged_in = True
                    st.success("Login Berhasil!")
                    st.rerun()
                else:
                    st.error("Username atau Password salah! Gunakan (liliyosa / 51230087)")


if not st.session_state.logged_in:
    login_page()
    st.stop()

# =====================================================================
# SIDEBAR NAVIGASI & FOTO PROFIL KECIL
# =====================================================================
st.sidebar.markdown("<h2>🦋 Menu Navigasi UTS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

try:
    st.sidebar.image(Image.open(FOTO), width=120)
except Exception:
    pass

st.sidebar.markdown(f"""
**Identitas Mahasiswa:**
* **Nama:** {NAMA}
* **NPM:** {NPM}
* **Prodi:** Teknologi Informasi
* **Univ:** Universitas Timor
""")
st.sidebar.markdown("---")

menu = st.sidebar.radio("Pilih Tahapan UTS:", MENU, key="menu")

st.sidebar.markdown("---")
if st.session_state.df_raw is not None:
    st.sidebar.success(f"📦 Dataset aktif: {fmt_int(len(st.session_state.df_raw))} baris")
else:
    st.sidebar.warning("📦 Dataset belum di-upload")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()


# =====================================================================
# KONTEKS DATA (dihitung sekali per tampilan, di-cache)
# =====================================================================
def get_ctx():
    raw = st.session_state.df_raw
    clean, log, meta = run_preprocessing(raw, st.session_state.df_prod, st.session_state.df_cust, int(st.session_state.qty_max))
    matrix = make_basket_matrix(clean)
    rfm, ref_date = make_rfm(clean)
    basket = clean.groupby("Transaction_ID")["Product_ID"].nunique().rename("basket_size")
    return {"raw": raw, "clean": clean, "log": log, "meta": meta, "matrix": matrix,
            "rfm": rfm, "ref_date": ref_date, "basket": basket}


def get_segments(ctx):
    rfm = ctx["rfm"]
    use_log = bool(st.session_state.km_log)
    ev = kmeans_eval(rfm, use_log)
    best = int(ev.loc[ev["Silhouette"].idxmax(), "k"])
    k = int(st.session_state.get("km_k", best))
    rc, prof = kmeans_fit(rfm, k, use_log)
    return rc, prof, ev, best, k


def get_rules(ctx):
    return run_apriori(ctx["matrix"], float(st.session_state.ap_sup), float(st.session_state.ap_conf),
                       float(st.session_state.ap_lift), int(st.session_state.ap_len))


# =====================================================================
# HALAMAN BERANDA (TEMPAT UPLOAD FILE EXCEL)
# =====================================================================
REQUIRED_COLS = ["Transaction_ID", "Transaction_Date", "Customer_ID", "Product_ID", "Product_Name",
                 "Category", "Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]

STAGES = [
    ("📊", "EDA & Business Understanding", "10%", "Pertanyaan bisnis, profil data, audit kualitas, insight awal"),
    ("🧹", "Data Preprocessing", "25%", "Pembersihan, basket_size, RFM, scaling"),
    ("🛒", "Apriori (Market Basket)", "20%", "Frequent itemsets, support, confidence, lift"),
    ("🎯", "K-Means Segmentasi", "20%", "Elbow, silhouette, profil & strategi segmen"),
    ("📈", "BI Dashboard", "15%", "KPI, tren, Top 10, filter interaktif"),
    ("💡", "Rekomendasi Keputusan", "10%", "Prioritas, KPI, estimasi biaya/manfaat"),
]


def ingest(content, name):
    raw, prod, cust, dic, sheets = parse_excel(content)
    if raw is None:
        st.error(f"Sheet 'Data_Raw' tidak ditemukan. Sheet yang tersedia: {', '.join(sheets)}")
        return
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan di Data_Raw: {', '.join(missing)}")
        return
    st.session_state.update(df_raw=raw, df_prod=prod, df_cust=cust, df_dic=dic, src_name=name)
    st.success("✅ Dataset berhasil di-upload dan dimuat ke dalam sistem!")


def page_home():
    hero("🏠 Beranda Utama & Upload Dataset UTS", "Unggah dataset lalu telusuri enam tahapan analisis secara interaktif.")
    card("""
        <h3>📂 Upload Dataset Transaksi Toko (.xlsx)</h3>
        <p>Silakan upload file dataset Excel UTS Anda di bawah ini untuk memulai seluruh proses analisis,
        pemodelan, dan dashboard interaktif.</p>
    """)
    c1, c2 = st.columns([3, 1])
    with c1:
        up = st.file_uploader("Pilih file Excel Dataset UTS", type=["xlsx", "xls"])
    with c2:
        local = sorted(glob.glob("P8*Business*Intelligence*.xlsx"))
        if local:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📁 Muat file lokal", help=local[0]):
                with open(local[0], "rb") as f:
                    ingest(f.read(), local[0])

    if up is not None and st.session_state.get("src_sig") != (up.name, up.size):
        ingest(up.getvalue(), up.name)
        st.session_state["src_sig"] = (up.name, up.size)

    raw = st.session_state.df_raw
    if raw is None:
        st.info("💡 Belum ada file yang di-upload. Silakan upload file `P8. UTS_Business Intelligence.xlsx` di atas.")
        return

    st.markdown("---")
    cols = st.columns(5)
    items = [("Baris Data", fmt_int(len(raw))), ("Kolom", str(raw.shape[1])),
             ("Transaksi Unik", fmt_int(raw["Transaction_ID"].nunique())),
             ("Pelanggan Unik", fmt_int(raw["Customer_ID"].nunique())),
             ("Produk Unik", fmt_int(raw["Product_Name"].nunique()))]
    for c, (t, v) in zip(cols, items):
        c.markdown(kpi(t, v), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["📋 Preview Data Raw", "📖 Kamus Data", "🛍️ Katalog Produk", "👥 Data Pelanggan"])
    with t1:
        n = st.slider("Jumlah baris yang ditampilkan", 5, 200, 10, step=5)
        st.markdown(f"#### 📋 Preview {n} Baris Pertama Dataset Raw:")
        show_df(raw.head(n))
    with t2:
        if st.session_state.df_dic is not None:
            show_df(st.session_state.df_dic)
        else:
            st.info("Sheet Data_Dictionary tidak ada di file ini.")
    with t3:
        if st.session_state.df_prod is not None:
            show_df(st.session_state.df_prod)
        else:
            st.info("Sheet Data_Produk tidak ada — katalog akan diturunkan otomatis dari Data_Raw.")
    with t4:
        if st.session_state.df_cust is not None:
            show_df(st.session_state.df_cust.head(100))
        else:
            st.info("Sheet Data_Pelanggan tidak ada di file ini.")

    st.markdown("### 🚀 Mulai Analisis")
    for row in range(2):
        cs = st.columns(3)
        for j, c in enumerate(cs):
            i = row * 3 + j
            icon, title, w, desc = STAGES[i]
            with c:
                card(f"<h4>{icon} Tahap {i + 1}: {title} <span class='badge'>{w}</span></h4><p>{desc}</p>")
                st.button("Buka tahap ➜", key=f"go_{i}", on_click=goto, args=(MENU[i + 1],))


# =====================================================================
# TAHAP 1 — EDA & BUSINESS UNDERSTANDING
# =====================================================================
def page_eda(ctx):
    raw, clean = ctx["raw"], ctx["clean"]
    hero("📊 Tahap 1: Business Understanding & EDA", "Memahami masalah bisnis dan kondisi data sebelum analisis.", "Bobot 10%")
    t1, t2, t3, t4 = st.tabs(["🎯 Pertanyaan Bisnis", "🔎 Profil Dataset", "🧪 Audit Kualitas Data", "📉 Visualisasi & Insight"])

    with t1:
        card("""
            <h4>Permasalahan Bisnis</h4>
            <p>Toko ritel memiliki ribuan transaksi namun belum dimanfaatkan untuk memahami pola pembelian,
            nilai pelanggan, dan waktu penjualan terbaik. Analisis ini bertujuan menghasilkan keputusan
            penempatan produk, retensi pelanggan, dan strategi pemasaran berbasis data.</p>
        """)
        qs = [("🛒", "Produk apa yang sering dibeli bersama?", "Dijawab dengan Apriori (Tahap 3) — dasar bundling & penataan rak."),
              ("👑", "Segmen pelanggan mana yang paling bernilai?", "Dijawab dengan RFM + K-Means (Tahap 4)."),
              ("📅", "Kapan penjualan meningkat?", "Dijawab melalui tren bulanan dan pola jam/hari (Tahap 1 & 5)."),
              ("🏷️", "Kategori dan produk apa yang menyumbang pendapatan terbesar?", "Dijawab melalui Top 10 produk/kategori di dashboard."),
              ("⚠️", "Pelanggan mana yang berisiko tidak kembali?", "Dijawab melalui profil segmen dengan Recency tinggi.")]
        for i, (ic, q, a) in enumerate(qs, 1):
            insight(f"<b>{ic} Pertanyaan {i}:</b> {q}<br><small>{a}</small>", "fact")

    with t2:
        dmin, dmax = raw["Transaction_Date"].min(), raw["Transaction_Date"].max()
        cols = st.columns(4)
        vals = [("Jumlah Baris", fmt_int(len(raw))), ("Jumlah Kolom", str(raw.shape[1])),
                ("Rentang Tanggal", f"{pd.to_datetime(dmin):%d %b %Y} – {pd.to_datetime(dmax):%d %b %Y}"),
                ("Transaksi Unik", fmt_int(raw["Transaction_ID"].nunique()))]
        for c, (t, v) in zip(cols, vals):
            c.markdown(kpi(t, v), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(3)
        vals = [("Pelanggan Unik", fmt_int(raw["Customer_ID"].nunique())),
                ("Produk Unik", fmt_int(raw["Product_ID"].nunique())),
                ("Total Penjualan Sementara", fmt_short(raw["Line_Total"].sum()))]
        for c, (t, v) in zip(cols, vals):
            c.markdown(kpi(t, v, fmt_idr(raw["Line_Total"].sum()) if "Total" in t else ""), unsafe_allow_html=True)
        st.markdown("#### 🧬 Tipe Data Setiap Kolom")
        dt = pd.DataFrame({"Kolom": raw.columns, "Tipe Data": raw.dtypes.astype(str).values,
                           "Non-Null": raw.notna().sum().values, "Nilai Unik": raw.nunique().values})
        show_df(dt)
        with st.expander("📐 Statistik deskriptif kolom numerik"):
            show_df(raw[["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]].describe().round(2).reset_index())
        st.caption("Catatan: total penjualan di atas masih sementara karena data mentah belum dibersihkan (duplikat & nilai tidak wajar masih ada).")

    with t3:
        a = audit_raw(raw, build_ref(raw, st.session_state.df_prod), int(st.session_state.qty_max))
        miss_total = int(a["missing"].sum())
        cols = st.columns(5)
        for c, (t, v) in zip(cols, [("Missing Values", miss_total), ("Baris Duplikat", a["dup"]),
                                    ("Kategori Tak Konsisten", a["cat_incons"]),
                                    ("Metode Bayar Tak Konsisten", a["pay_incons"]),
                                    ("Qty/Harga Tak Valid", a["qty_invalid"] + a["price_invalid"])]):
            c.metric(t, fmt_int(v))
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            ms = a["missing"][a["missing"] > 0].sort_values().reset_index()
            ms.columns = ["Kolom", "Jumlah"]
            fig = px.bar(ms, x="Jumlah", y="Kolom", orientation="h", text="Jumlah", color="Jumlah",
                         color_continuous_scale=["#FFC0CB", "#C71585"])
            fig.update_coloraxes(showscale=False)
            show_fig(style_fig(fig, 360, "Missing Values per Kolom"))
        with c2:
            q = raw["Quantity"].dropna()
            fig = px.histogram(q[q.between(-5, 20)], nbins=26, color_discrete_sequence=["#C71585"])
            fig.update_layout(showlegend=False, xaxis_title="Quantity (dipotong −5…20 agar terbaca)", yaxis_title="Jumlah baris")
            show_fig(style_fig(fig, 360, "Distribusi Quantity (nilai ≤ 0 tidak valid)"))
        c3, c4 = st.columns(2)
        with c3:
            fig = px.box(raw.dropna(subset=["Unit_Price"]), y="Unit_Price", points="outliers", color_discrete_sequence=["#DB7093"])
            fig.update_yaxes(type="log")
            show_fig(style_fig(fig, 340, "Boxplot Unit_Price (skala log) — outlier ekstrem terlihat"))
        with c4:
            tabs_inc = st.radio("Lihat variasi penulisan", ["Kategori", "Metode Pembayaran"], horizontal=True)
            col = "Category" if tabs_inc == "Kategori" else "Payment_Method"
            v = raw[col].fillna("(kosong)").value_counts().reset_index()
            v.columns = ["Nilai", "Jumlah"]
            show_df(v)
        with st.expander("🔍 Rincian pemeriksaan lainnya"):
            st.write(f"- Baris dengan Quantity ≤ 0 atau > {int(st.session_state.qty_max)}: **{fmt_int(a['qty_invalid'])}**")
            st.write(f"- Baris dengan Unit_Price ≤ 0 atau > 500.000: **{fmt_int(a['price_invalid'])}**")
            st.write(f"- Baris dengan Line_Total tidak sama dengan Quantity × Harga × (1 − Diskon): **{fmt_int(a['lt_mismatch'])}**")

    with t4:
        st.markdown("Visualisasi memakai **data yang sudah dibersihkan** (lihat Tahap 2).")
        c1, c2 = st.columns(2)
        with c1:
            metric = st.radio("Metrik tren", ["Pendapatan", "Jumlah Transaksi"], horizontal=True, key="eda_metric")
            if metric == "Pendapatan":
                m = clean.groupby("Month")["Line_Total"].sum().reset_index(name="Nilai")
            else:
                m = clean.groupby("Month")["Transaction_ID"].nunique().reset_index(name="Nilai")
            fig = px.line(m, x="Month", y="Nilai", markers=True, color_discrete_sequence=["#C71585"])
            fig.update_traces(line_width=3, fill="tozeroy", fillcolor="rgba(199,21,133,0.10)")
            show_fig(style_fig(fig, 340, "Tren Bulanan 2025"))
        with c2:
            cat = clean.groupby("Category")["Line_Total"].sum().sort_values().reset_index()
            fig = px.bar(cat, x="Line_Total", y="Category", orientation="h", color="Line_Total",
                         color_continuous_scale=["#FFC0CB", "#C71585"])
            fig.update_coloraxes(showscale=False)
            fig.update_layout(xaxis_title="Pendapatan (Rp)", yaxis_title="")
            show_fig(style_fig(fig, 340, "Pendapatan per Kategori"))
        c3, c4 = st.columns(2)
        with c3:
            trx = clean.groupby("Transaction_ID")["Line_Total"].sum()
            fig = px.histogram(trx, nbins=40, color_discrete_sequence=["#FF69B4"])
            fig.update_layout(showlegend=False, xaxis_title="Nilai transaksi (Rp)", yaxis_title="Jumlah transaksi")
            show_fig(style_fig(fig, 340, "Distribusi Nilai per Transaksi"))
        with c4:
            hr = clean.assign(Hari=clean["Transaction_Date"].dt.dayofweek, Jam=clean["Transaction_Date"].dt.hour)
            hm = hr.groupby(["Hari", "Jam"])["Transaction_ID"].nunique().unstack(fill_value=0)
            hm.index = [["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"][i] for i in hm.index]
            fig = px.imshow(hm, aspect="auto", color_continuous_scale=["#FFF0F5", "#C71585"])
            fig.update_layout(xaxis_title="Jam", yaxis_title="")
            show_fig(style_fig(fig, 340, "Kepadatan Transaksi: Hari × Jam"))

        # insight dinamis
        mon = clean.groupby("Month")["Line_Total"].sum()
        top_m, low_m = mon.idxmax(), mon.idxmin()
        cat_s = clean.groupby("Category")["Line_Total"].sum().sort_values(ascending=False)
        share = cat_s.iloc[0] / cat_s.sum() * 100
        ct = clean.groupby("Customer_Type")["Line_Total"].sum()
        trx_c = clean.groupby("Customer_Type")["Transaction_ID"].nunique()
        avg_t = (ct / trx_c).round(0)
        hour_peak = int(clean.groupby(clean["Transaction_Date"].dt.hour)["Transaction_ID"].nunique().idxmax())
        st.markdown("#### 💡 Insight Awal")
        insight(f"<b>Musiman:</b> pendapatan tertinggi terjadi pada <b>{top_m}</b> ({fmt_short(mon.max())}) dan terendah pada <b>{low_m}</b> ({fmt_short(mon.min())}).", "fact")
        insight(f"<b>Kategori:</b> <b>{cat_s.index[0]}</b> menyumbang {share:.1f}% pendapatan — kategori terbesar, diikuti {cat_s.index[1]} dan {cat_s.index[2]}.", "fact")
        insight(f"<b>Jam sibuk:</b> transaksi paling banyak terjadi sekitar pukul <b>{hour_peak:02d}.00</b> — jadwalkan kasir dan stok pada jam ini.", "fact")
        if len(avg_t) > 1:
            txt = " vs ".join(f"{k}: {fmt_idr(v)}" for k, v in avg_t.items())
            insight(f"<b>Nilai per transaksi:</b> rata-rata {txt}.", "fact")
        insight(f"<b>Kualitas data:</b> {fmt_int(len(raw) - len(clean))} dari {fmt_int(len(raw))} baris mentah ({(len(raw) - len(clean)) / len(raw) * 100:.1f}%) dibuang saat pembersihan, "
                f"sehingga angka mentah akan menggelembungkan penjualan.", "fact")
        st.button("Lanjut ke Tahap 2 ➜", on_click=goto, args=(MENU[2],), key="n1")


# =====================================================================
# TAHAP 2 — PREPROCESSING
# =====================================================================
def build_export(ctx, rules, rc, prof):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        ctx["clean"].to_excel(xw, "Data_Clean", index=False)
        ctx["log"].to_excel(xw, "Log_Preprocessing", index=False)
        ctx["basket"].reset_index().to_excel(xw, "Basket_Size", index=False)
        ctx["rfm"].reset_index().to_excel(xw, "RFM", index=False)
        if rules is not None and not rules.empty:
            rules[["Aturan", "support", "confidence", "lift", "Jumlah Transaksi"]].to_excel(xw, "Aturan_Apriori", index=False)
        rc.reset_index().to_excel(xw, "Segmen_KMeans", index=False)
        prof.to_excel(xw, "Profil_Segmen", index=False)
    return buf.getvalue()


def page_prep(ctx):
    raw, clean, log = ctx["raw"], ctx["clean"], ctx["log"]
    hero("🧹 Tahap 2: Data Preprocessing", "Pembersihan transparan & dapat direproduksi: Data_Raw tidak ditimpa.", "Bobot 25%")
    t1, t2, t3, t4, t5 = st.tabs(["🧾 Log Penanganan Data", "🛍️ Basket Size & Matriks", "👥 RFM Pelanggan", "⚖️ Scaling & Data Besar", "📥 Unduh Hasil"])

    with t1:
        cols = st.columns(4)
        cols[0].metric("Baris Data Mentah", fmt_int(len(raw)))
        cols[1].metric("Baris Data Bersih", fmt_int(len(clean)))
        cols[2].metric("Baris Dibuang", fmt_int(len(raw) - len(clean)))
        cols[3].metric("% Dibuang", f"{(len(raw) - len(clean)) / len(raw) * 100:.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)
        st.number_input("Batas maksimum Quantity wajar (di atas ini dianggap outlier)", 10, 1000, key="qty_max", step=10,
                        help="Mengubah nilai ini akan menjalankan ulang seluruh pembersihan dan analisis.")
        st.markdown("#### 🧾 Keputusan untuk setiap masalah (sebelum → sesudah)")
        show_df(log)
        issues = log[log["Jumlah Terdeteksi"] > 0].sort_values("Jumlah Terdeteksi")
        fig = px.bar(issues, x="Jumlah Terdeteksi", y="Masalah", orientation="h", color="Keputusan",
                     color_discrete_sequence=PINK, text="Jumlah Terdeteksi")
        show_fig(style_fig(fig, 420, "Jumlah masalah terdeteksi per jenis"))
        st.markdown("#### 🔄 Bandingkan Data Mentah vs Data Bersih")
        which = st.radio("Tampilkan", ["Data Bersih", "Data Mentah"], horizontal=True)
        tid = st.text_input("Cari Transaction_ID (opsional, contoh: T000012)")
        src = clean if which == "Data Bersih" else raw
        if tid:
            src = src[src["Transaction_ID"].astype(str).str.contains(tid.strip(), case=False)]
        show_df(src.head(100))
        st.caption(f"Menampilkan maksimal 100 baris dari {fmt_int(len(src))} baris.")

    with t2:
        basket = ctx["basket"]
        cols = st.columns(3)
        cols[0].metric("Jumlah Transaksi (basket)", fmt_int(len(basket)))
        cols[1].metric("Rata-rata basket_size", f"{basket.mean():.2f}")
        cols[2].metric("Basket terbesar", int(basket.max()))
        card("<b>basket_size</b> = jumlah produk <b>unik</b> per Transaction_ID. Baris item <b>bukan</b> dihitung sebagai transaksi terpisah.")
        c1, c2 = st.columns(2)
        with c1:
            d = basket.value_counts().sort_index().reset_index()
            d.columns = ["basket_size", "Jumlah Transaksi"]
            fig = px.bar(d, x="basket_size", y="Jumlah Transaksi", text="Jumlah Transaksi", color_discrete_sequence=["#C71585"])
            show_fig(style_fig(fig, 340, "Distribusi basket_size"))
        with c2:
            sup = ctx["matrix"].mean().sort_values().reset_index()
            sup.columns = ["Produk", "Support"]
            fig = px.bar(sup.tail(15), x="Support", y="Produk", orientation="h", color="Support", color_continuous_scale=["#FFC0CB", "#C71585"])
            fig.update_coloraxes(showscale=False)
            show_fig(style_fig(fig, 340, "15 produk dengan support tertinggi"))
        st.markdown("#### 🧮 Tabel transaksi × produk (1 = dibeli, 0 = tidak)")
        n = st.slider("Jumlah transaksi ditampilkan", 5, 100, 10, key="mx_n")
        show_df(ctx["matrix"].astype(int).head(n).reset_index(), hide_index=True)

    with t3:
        rfm, ref = ctx["rfm"], ctx["ref_date"]
        card(f"<b>Tanggal acuan RFM:</b> {ref:%d %B %Y} (satu hari setelah transaksi maksimum data bersih). "
             f"<b>Recency</b> = hari sejak transaksi terakhir; <b>Frequency</b> = jumlah transaksi unik; <b>Monetary</b> = total penjualan bersih. "
             f"Baris dengan Customer_ID 'UNKNOWN' dikeluarkan dari RFM.")
        cols = st.columns(4)
        cols[0].metric("Pelanggan (RFM)", fmt_int(len(rfm)))
        cols[1].metric("Recency rata-rata", f"{rfm['Recency'].mean():.0f} hari")
        cols[2].metric("Frequency rata-rata", f"{rfm['Frequency'].mean():.2f}")
        cols[3].metric("Monetary rata-rata", fmt_short(rfm["Monetary"].mean()))
        sort_by = st.selectbox("Urutkan tabel RFM berdasarkan", ["Monetary", "Frequency", "Recency"])
        asc = st.checkbox("Urut naik", value=False)
        show_df(rfm.sort_values(sort_by, ascending=asc).reset_index().head(100).round(1))
        c1, c2, c3 = st.columns(3)
        for c, col, color in zip([c1, c2, c3], ["Recency", "Frequency", "Monetary"], PINK[:3]):
            fig = px.histogram(rfm, x=col, nbins=30, color_discrete_sequence=[color])
            fig.update_layout(showlegend=False)
            with c:
                show_fig(style_fig(fig, 280, f"Distribusi {col}"))

    with t4:
        rfm = ctx["rfm"]
        card("<b>Mengapa perlu standardisasi?</b> K-Means memakai jarak Euclidean. Monetary berskala jutaan Rupiah, Recency ratusan hari, "
             "dan Frequency hanya satuan — tanpa scaling, Monetary akan mendominasi jarak. <b>StandardScaler</b> (rata-rata 0, simpangan baku 1) menyamakan bobot ketiga fitur.")
        X = rfm[["Recency", "Frequency", "Monetary"]].astype(float)
        Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sebelum scaling**")
            show_df(X.describe().round(2).reset_index())
        with c2:
            st.markdown("**Sesudah scaling**")
            show_df(Z.describe().round(2).reset_index())
        long = pd.concat([X.melt(var_name="Fitur", value_name="Nilai").assign(Tahap="Sebelum"),
                          Z.melt(var_name="Fitur", value_name="Nilai").assign(Tahap="Sesudah")])
        fig = px.box(long, x="Fitur", y="Nilai", color="Tahap", facet_col="Tahap", color_discrete_sequence=PINK)
        fig.update_yaxes(matches=None)
        show_fig(style_fig(fig, 360, "Skala fitur sebelum vs sesudah standardisasi"))
        st.markdown("#### 🐘 Strategi untuk Data Besar")
        c1, c2, c3 = st.columns(3)
        with c1:
            card("<b>🎲 Sampling</b><br>Ambil sampel acak/stratified (mis. 10–20%) untuk eksplorasi cepat atau penentuan k, lalu terapkan model ke seluruh data.")
        with c2:
            card("<b>🧱 Chunking</b><br>Baca dan proses data per potongan (<code>chunksize</code>) lalu gabungkan hasilnya; cocok bila data tidak muat di memori.")
        with c3:
            card(f"<b>🧮 Agregasi</b><br>Pada proyek ini {fmt_int(len(clean))} baris item diringkas menjadi {fmt_int(len(rfm))} baris pelanggan (RFM) — K-Means hanya memproses ringkasan ini. Untuk skala lebih besar gunakan <code>MiniBatchKMeans</code>.")

    with t5:
        st.markdown("#### 📥 Unduh file hasil preprocessing & dokumentasi perubahan")
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("⬇️ Data_Clean (CSV)", csv_bytes(clean), "Data_Clean.csv", "text/csv")
        c2.download_button("⬇️ Log Preprocessing", csv_bytes(log), "Log_Preprocessing.csv", "text/csv")
        c3.download_button("⬇️ Tabel RFM", csv_bytes(ctx["rfm"], index=True), "RFM_Pelanggan.csv", "text/csv")
        c4.download_button("⬇️ basket_size", csv_bytes(ctx["basket"].reset_index()), "Basket_Size.csv", "text/csv")
        st.markdown("---")
        st.markdown("**📦 Paket Excel lengkap** (Data_Clean, log, basket, RFM, aturan Apriori, segmen K-Means)")
        if st.button("⚙️ Siapkan paket Excel"):
            with st.spinner("Menyusun workbook..."):
                _, rules = get_rules(ctx)
                rc, prof, *_ = get_segments(ctx)
                st.session_state["export_bytes"] = build_export(ctx, rules, rc, prof)
            st.success("Paket siap diunduh!")
        if st.session_state.get("export_bytes"):
            st.download_button("⬇️ Unduh Hasil_UTS_BI.xlsx", st.session_state["export_bytes"], "Hasil_UTS_BI.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.button("Lanjut ke Tahap 3 ➜", on_click=goto, args=(MENU[3],), key="n2")


# =====================================================================
# TAHAP 3 — APRIORI
# =====================================================================
def set_ap(sup, conf, lift):
    st.session_state.ap_sup, st.session_state.ap_conf, st.session_state.ap_lift = sup, conf, lift


def network_fig(r, n_rules):
    r = r[(r["n_ant"] == 1) & (r["n_con"] == 1)].drop_duplicates("Pasangan").head(n_rules)
    if r.empty:
        return None
    nodes = sorted(set(r["Jika (Antecedent)"]) | set(r["Maka (Consequent)"]))
    ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {nd: (float(np.cos(a)), float(np.sin(a))) for nd, a in zip(nodes, ang)}
    lmin, lmax = r["lift"].min(), r["lift"].max()
    fig = go.Figure()
    mx, my, mt = [], [], []
    for _, row in r.iterrows():
        x0, y0 = pos[row["Jika (Antecedent)"]]
        x1, y1 = pos[row["Maka (Consequent)"]]
        w = 1.5 + (0 if lmax == lmin else (row["lift"] - lmin) / (lmax - lmin) * 4.5)
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=w, arrowcolor="#DB7093", standoff=14)
        mx.append((x0 + x1) / 2); my.append((y0 + y1) / 2)
        mt.append(f"{row['Aturan']}<br>Support {row['support']:.3f} | Conf {row['confidence']:.2f} | Lift {row['lift']:.2f}")
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)"),
                             hovertext=mt, hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
                             text=nodes, textposition="top center", hoverinfo="text", hovertext=nodes,
                             marker=dict(size=22, color="#C71585", line=dict(color="white", width=2)), showlegend=False))
    fig.update_xaxes(visible=False, range=[-1.7, 1.7])
    fig.update_yaxes(visible=False, range=[-1.5, 1.5], scaleanchor="x")
    return style_fig(fig, 520, "Network Graph Aturan Asosiasi (tebal panah = Lift)")


def page_apriori(ctx):
    hero("🛒 Tahap 3: Apriori / Market Basket Analysis", "Menemukan produk yang sering dibeli bersama.", "Bobot 20%")
    if not MLXTEND_OK:
        st.error("Pustaka `mlxtend` belum terpasang. Jalankan: `pip install mlxtend`")
        return
    matrix = ctx["matrix"]
    N = len(matrix)
    item_sup = matrix.mean()

    st.markdown("#### 🎛️ Parameter")
    pc = st.columns(4)
    pc[0].button("🌸 Longgar", on_click=set_ap, args=(0.005, 0.15, 1.0), key="p1", help="Support 0,5% · Confidence 15% · Lift ≥ 1,0")
    pc[1].button("💖 Seimbang", on_click=set_ap, args=(0.01, 0.2, 1.2), key="p2", help="Support 1% · Confidence 20% · Lift ≥ 1,2")
    pc[2].button("👑 Ketat", on_click=set_ap, args=(0.02, 0.3, 1.5), key="p3", help="Support 2% · Confidence 30% · Lift ≥ 1,5")
    pc[3].button("🔄 Reset", on_click=set_ap, args=(0.01, 0.2, 1.2), key="p4")
    c1, c2, c3, c4 = st.columns(4)
    c1.slider("Minimum Support", 0.002, 0.05, step=0.001, key="ap_sup", format="%.3f")
    c2.slider("Minimum Confidence", 0.05, 0.9, step=0.05, key="ap_conf")
    c3.slider("Minimum Lift", 1.0, 4.0, step=0.1, key="ap_lift")
    c4.slider("Ukuran itemset maks.", 2, 4, key="ap_len")

    sup_s, conf_s, lift_s = st.session_state.ap_sup, st.session_state.ap_conf, st.session_state.ap_lift
    base = float(item_sup.mean())
    card(f"<b>Alasan pemilihan parameter.</b> Rata-rata support satu produk ≈ {base * 100:.1f}% sehingga bila dua produk saling bebas, "
         f"pasangannya muncul sekitar {base ** 2 * 100:.2f}% transaksi. <b>Min support {sup_s * 100:.1f}%</b> menjaga agar pasangan yang dipilih muncul cukup sering "
         f"(≥ {int(sup_s * N)} dari {fmt_int(N)} transaksi) dan bukan kebetulan. <b>Min confidence {conf_s * 100:.0f}%</b> = peluang membeli produk kedua "
         f"minimal {conf_s / base:.1f}× peluang dasarnya ({base * 100:.1f}%). <b>Lift ≥ {lift_s:.1f}</b> memastikan hubungan lebih kuat dari kebetulan (lift = 1 berarti independen).")

    fi, rules = get_rules(ctx)
    cols = st.columns(4)
    cols[0].metric("Transaksi (basket)", fmt_int(N))
    cols[1].metric("Produk", matrix.shape[1])
    cols[2].metric("Frequent itemsets", fmt_int(len(fi)))
    cols[3].metric("Aturan asosiasi", fmt_int(len(rules)))
    if rules.empty:
        st.warning("Tidak ada aturan yang memenuhi parameter. Coba turunkan Minimum Support/Confidence atau tekan tombol 🌸 Longgar.")
        return

    t1, t2, t3, t4, t5 = st.tabs(["🧩 Frequent Itemsets", "🏆 Top 10 Aturan", "🕸️ Visualisasi", "🎁 Rekomendasi Cross-Selling", "🔍 Cari Produk"])

    with t1:
        size = st.radio("Ukuran itemset", ["Semua", 1, 2, 3, 4], horizontal=True)
        d = fi if size == "Semua" else fi[fi["Ukuran"] == size]
        d = d.sort_values("support", ascending=False)[["Itemset", "Ukuran", "support", "Jumlah Transaksi"]].rename(columns={"support": "Support"})
        show_df(d)
        st.download_button("⬇️ Unduh frequent itemsets", csv_bytes(d), "frequent_itemsets.csv", "text/csv")

    with t2:
        crit = st.selectbox("Kriteria pemeringkatan",
                            ["Lift → Confidence → Support", "Confidence → Lift → Support",
                             "Support → Lift → Confidence", "Skor gabungan (Lift × Confidence)"])
        one_way = st.checkbox("Hilangkan aturan simetris (A→B dan B→A dihitung satu pasangan)", value=True)
        top = rank_rules(rules, crit, one_way, 10)
        show_df(rules_table(top), column_config={
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
            "Support": st.column_config.NumberColumn("Support", format="%.4f"),
            "Lift": st.column_config.NumberColumn("Lift", format="%.2f"),
        })
        insight("<b>Cara membaca:</b> <b>Support</b> = proporsi transaksi yang memuat kedua produk; <b>Confidence</b> = peluang membeli produk kanan bila produk kiri sudah dibeli; "
                "<b>Lift</b> = seberapa kuat hubungan dibanding kebetulan (&gt; 1 = saling menguatkan). Peringkat memakai Lift terlebih dahulu karena Confidence tinggi bisa "
                "sekadar karena produk itu populer. <i>Aturan asosiasi bukan bukti sebab-akibat.</i>")
        st.download_button("⬇️ Unduh seluruh aturan", csv_bytes(rules_table(rank_rules(rules, crit, False, len(rules)))), "aturan_apriori.csv", "text/csv")

    with t3:
        view = st.radio("Jenis visualisasi", ["🕸️ Network Graph", "🔥 Heatmap Lift", "✨ Scatter Support–Confidence"], horizontal=True)
        top20 = rank_rules(rules, "Lift → Confidence → Support", True, 40)
        if view.startswith("🕸️"):
            n_r = st.slider("Jumlah aturan pada graph", 3, 20, 10)
            fig = network_fig(top20, n_r)
            if fig is None:
                st.info("Tidak ada aturan satu-produk → satu-produk pada parameter saat ini.")
            else:
                show_fig(fig)
        elif view.startswith("🔥"):
            pr = rules[(rules["n_ant"] == 1) & (rules["n_con"] == 1)]
            if pr.empty:
                st.info("Belum ada aturan pasangan untuk heatmap.")
            else:
                pv = pr.pivot_table(index="Jika (Antecedent)", columns="Maka (Consequent)", values="lift")
                fig = px.imshow(pv, aspect="auto", color_continuous_scale=["#FFF0F5", "#FF69B4", "#8B008B"], labels=dict(color="Lift"))
                show_fig(style_fig(fig, 560, "Heatmap Lift antar produk (Jika → Maka)"))
        else:
            fig = px.scatter(rules, x="support", y="confidence", color="lift", size="lift", hover_name="Aturan",
                             color_continuous_scale=["#FFC0CB", "#C71585", "#8B008B"])
            fig.update_layout(xaxis_title="Support", yaxis_title="Confidence")
            show_fig(style_fig(fig, 480, "Sebaran aturan: Support vs Confidence (warna & ukuran = Lift)"))

    with t4:
        pairs = rank_rules(rules[(rules["n_ant"] == 1) & (rules["n_con"] == 1)], "Lift → Confidence → Support", True, 3)
        if pairs.empty:
            st.info("Belum ada pasangan produk yang memenuhi parameter.")
        for i, r in pairs.iterrows():
            a, b = r["Jika (Antecedent)"], r["Maka (Consequent)"]
            card(f"<h4>🎁 Rekomendasi {i + 1}: Bundling “{a} + {b}”</h4>"
                 f"<p><b>Bukti:</b> support {r['support'] * 100:.2f}% ({r['Jumlah Transaksi']} transaksi), confidence {r['confidence'] * 100:.1f}%, lift {r['lift']:.2f}.</p>"
                 f"<ul><li><b>Cross-selling:</b> saat kasir/pelanggan mengambil <i>{a}</i>, tawarkan <i>{b}</i>.</li>"
                 f"<li><b>Bundling:</b> paket hemat {a} + {b} dengan diskon kecil (mis. 3–5%).</li>"
                 f"<li><b>Penempatan rak:</b> letakkan <i>{a}</i> dan <i>{b}</i> berdampingan atau pada gondola yang sama.</li></ul>")
        insight("Asosiasi menunjukkan produk cenderung dibeli bersama, <b>bukan</b> bahwa satu produk menyebabkan pembelian produk lain. Uji dengan eksperimen kecil sebelum diperluas.")

    with t5:
        prod = st.selectbox("Pilih produk", list(matrix.columns))
        sub = rules[(rules["n_ant"] == 1) & (rules["Jika (Antecedent)"] == prod)].sort_values(["confidence", "lift"], ascending=False)
        st.metric(f"Support {prod}", f"{item_sup[prod] * 100:.1f}% transaksi")
        if sub.empty:
            st.info(f"Tidak ada aturan dengan “{prod}” sebagai pemicu pada parameter ini.")
        else:
            st.markdown(f"**Pelanggan yang membeli _{prod}_ cenderung juga membeli:**")
            show_df(rules_table(sub.head(10)))
    st.button("Lanjut ke Tahap 4 ➜", on_click=goto, args=(MENU[4],), key="n3")


# =====================================================================
# TAHAP 4 — K-MEANS
# =====================================================================
def use_best_k(best):
    st.session_state.km_k = best


def page_kmeans(ctx):
    hero("🎯 Tahap 4: K-Means (Segmentasi Pelanggan)", "Mengelompokkan pelanggan berdasarkan perilaku RFM.", "Bobot 20%")
    rfm = ctx["rfm"]
    st.checkbox("Terapkan transformasi log1p sebelum scaling (data Frequency/Monetary cenderung miring)", key="km_log")
    ev = kmeans_eval(rfm, bool(st.session_state.km_log))
    best = int(ev.loc[ev["Silhouette"].idxmax(), "k"])
    if "km_k" not in st.session_state:
        st.session_state.km_k = best
    c1, c2 = st.columns([3, 1])
    c1.slider("Jumlah klaster (k)", 2, 8, key="km_k")
    c2.markdown("<br>", unsafe_allow_html=True)
    c2.button(f"✨ Pakai k rekomendasi ({best})", on_click=use_best_k, args=(best,))
    rc, prof, ev, best, k = get_segments(ctx)
    cmap = seg_color_map(prof["Segment"])

    t1, t2, t3, t4, t5 = st.tabs(["🧬 Fitur", "📉 Elbow & Silhouette", "👥 Profil Klaster", "📣 Strategi Pemasaran", "🔎 Cari Pelanggan"])

    with t1:
        card("<b>Fitur K-Means:</b> Recency, Frequency, Monetary — satu baris mewakili satu pelanggan. Customer_ID <b>tidak</b> dimasukkan sebagai variabel numerik. "
             "Fitur distandardisasi dengan StandardScaler" + (" setelah log1p." if st.session_state.km_log else "."))
        c1, c2 = st.columns(2)
        c1.metric("Pelanggan dianalisis", fmt_int(len(rfm)))
        c2.metric("Tanggal acuan", f"{ctx['ref_date']:%d %b %Y}")
        show_df(rfm.reset_index().head(50).round(1))

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(ev, x="k", y="Inertia (WCSS)", markers=True, color_discrete_sequence=["#C71585"])
            fig.add_vline(x=k, line_dash="dash", line_color="#8B008B")
            show_fig(style_fig(fig, 340, "Elbow Method"))
        with c2:
            fig = px.line(ev, x="k", y="Silhouette", markers=True, color_discrete_sequence=["#FF69B4"])
            fig.add_vline(x=k, line_dash="dash", line_color="#8B008B")
            show_fig(style_fig(fig, 340, "Silhouette Score"))
        show_df(ev.round(3))
        sil_k = float(ev.loc[ev["k"] == k, "Silhouette"].iloc[0])
        sil_b = float(ev["Silhouette"].max())
        insight(f"<b>Pemilihan k:</b> silhouette tertinggi ada pada <b>k = {best}</b> ({sil_b:.3f}). k terpilih saat ini = <b>{k}</b> (silhouette {sil_k:.3f}). "
                f"Pilih k yang menggabungkan metrik dengan interpretasi bisnis: bila selisih silhouette kecil, k yang lebih besar dapat dipakai asalkan tiap klaster "
                f"memiliki strategi pemasaran yang jelas dan berbeda.", "fact")

    with t3:
        show = prof.copy()
        tbl = pd.DataFrame({
            "Segmen": show["Segment"], "Pelanggan": show["Pelanggan"], "% Pelanggan": show["% Pelanggan"].round(1),
            "Recency (mean)": show["Recency_Mean"].round(1), "Recency (median)": show["Recency_Median"].round(1),
            "Frequency (mean)": show["Frequency_Mean"].round(2), "Frequency (median)": show["Frequency_Median"].round(1),
            "Monetary (mean)": show["Monetary_Mean"].round(0), "Monetary (median)": show["Monetary_Median"].round(0),
            "% Pendapatan": show["% Pendapatan"].round(1),
        })
        show_df(tbl)
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = px.pie(prof, names="Segment", values="Pelanggan", hole=0.45, color="Segment", color_discrete_map=cmap)
            show_fig(style_fig(fig, 380, "Komposisi pelanggan"))
        with c2:
            axes = ["Recency", "Frequency", "Monetary"]
            ax1, ax2 = st.columns(2)
            xa = ax1.selectbox("Sumbu X", axes, index=0)
            ya = ax2.selectbox("Sumbu Y", axes, index=2)
            three_d = st.checkbox("Tampilan 3D (R–F–M)")
            rr = rc.reset_index()
            if three_d:
                fig = px.scatter_3d(rr, x="Recency", y="Frequency", z="Monetary", color="Segment", color_discrete_map=cmap,
                                    hover_name="Customer_ID", opacity=0.75)
            else:
                fig = px.scatter(rr, x=xa, y=ya, color="Segment", color_discrete_map=cmap, hover_name="Customer_ID", opacity=0.75)
            show_fig(style_fig(fig, 380, "Scatter plot klaster pelanggan"))
        cm = rc.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
        z = (cm - rc[["Recency", "Frequency", "Monetary"]].mean()) / rc[["Recency", "Frequency", "Monetary"]].std()
        fig = px.imshow(z, aspect="auto", color_continuous_scale=["#FFF0F5", "#FF69B4", "#8B008B"], text_auto=".2f")
        show_fig(style_fig(fig, 280, "Profil relatif tiap segmen (z-score terhadap rata-rata seluruh pelanggan)"))
        insight("Label segmen ditentukan otomatis dari profil data (mis. Recency tinggi → <b>Churn Risk</b>; Frequency & Monetary tinggi dengan Recency rendah → <b>Loyal Customers</b>), "
                "sehingga hanya muncul bila didukung hasil klaster.")

    with t4:
        for _, r in prof.iterrows():
            seg = r["Segment"]
            desc, acts = STRATEGY.get(base_label(seg), ("Segmen hasil klaster.", ["Tentukan strategi berdasarkan profil RFM."]))
            with st.expander(f"{seg} — {int(r['Pelanggan'])} pelanggan ({r['% Pendapatan']:.1f}% pendapatan)", expanded=True):
                st.markdown(f"**Profil:** R = {r['Recency_Mean']:.0f} hari · F = {r['Frequency_Mean']:.1f} transaksi · M = {fmt_idr(r['Monetary_Mean'])}")
                st.markdown(f"_{desc}_")
                for a in acts:
                    st.markdown(f"- {a}")

    with t5:
        cid = st.selectbox("Pilih / ketik Customer_ID", list(rc.index))
        row = rc.loc[cid]
        cols = st.columns(4)
        cols[0].metric("Segmen", row["Segment"])
        cols[1].metric("Recency", f"{int(row['Recency'])} hari")
        cols[2].metric("Frequency", int(row["Frequency"]))
        cols[3].metric("Monetary", fmt_short(row["Monetary"]))
        st.download_button("⬇️ Unduh segmentasi semua pelanggan", csv_bytes(rc, index=True), "segmentasi_pelanggan.csv", "text/csv")
    st.button("Lanjut ke Tahap 5 ➜", on_click=goto, args=(MENU[5],), key="n4")


# =====================================================================
# TAHAP 5 — DASHBOARD
# =====================================================================
def page_dashboard(ctx):
    hero("📈 Tahap 5: Business Intelligence Dashboard", "Pantau KPI, tren, produk unggulan, aturan Apriori, dan segmen pelanggan.", "Bobot 15%")
    clean = ctx["clean"]
    rc, prof, *_ = get_segments(ctx)
    cmap = seg_color_map(prof["Segment"])
    df = clean.copy()
    df["Segment"] = df["Customer_ID"].map(rc["Segment"]).fillna("Tidak Terklasifikasi")

    with st.expander("🎛️ Filter Dashboard", expanded=True):
        f1, f2, f3 = st.columns(3)
        dmin, dmax = df["Transaction_Date"].min().date(), df["Transaction_Date"].max().date()
        dr = f1.date_input("Rentang tanggal", (dmin, dmax), min_value=dmin, max_value=dmax)
        d0, d1 = (dr[0], dr[1]) if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dmin, dmax)
        cats = f2.multiselect("Kategori", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
        regs = f3.multiselect("Wilayah", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
        f4, f5 = st.columns(2)
        ctypes = f4.multiselect("Tipe pelanggan", sorted(df["Customer_Type"].unique()), default=sorted(df["Customer_Type"].unique()))
        segs = f5.multiselect("Segmen pelanggan", sorted(df["Segment"].unique()), default=sorted(df["Segment"].unique()))

    mask = (df["Transaction_Date"].dt.date.between(d0, d1) & df["Category"].isin(cats) & df["Region"].isin(regs)
            & df["Customer_Type"].isin(ctypes) & df["Segment"].isin(segs))
    fdf = df[mask]
    if fdf.empty:
        st.warning("Tidak ada data untuk kombinasi filter ini. Longgarkan filter.")
        return

    total_rev = fdf["Line_Total"].sum()
    total_trx = fdf["Transaction_ID"].nunique()
    total_cust = fdf.loc[~fdf["Customer_Unknown"], "Customer_ID"].nunique()
    avg_basket = fdf.groupby("Transaction_ID")["Product_ID"].nunique().mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("💰 Total Penjualan", fmt_short(total_rev), fmt_idr(total_rev)), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("🧾 Jumlah Transaksi", fmt_int(total_trx)), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("👥 Pelanggan Unik", fmt_int(total_cust)), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("🛍️ Rata-rata Basket Size", f"{avg_basket:.2f}"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["📈 Tren & Top 10", "🛒 Hasil Apriori", "🎯 Profil K-Means", "💡 Insight Manajemen"])

    with t1:
        a, b = st.columns(2)
        with a:
            mt = st.selectbox("Metrik tren", ["Pendapatan", "Jumlah Transaksi", "Pelanggan Unik"])
            ct = st.radio("Bentuk grafik", ["Garis", "Batang"], horizontal=True, key="dash_ct")
            g = fdf.groupby("Month")
            m = (g["Line_Total"].sum() if mt == "Pendapatan" else g["Transaction_ID"].nunique() if mt == "Jumlah Transaksi"
                 else g["Customer_ID"].nunique()).reset_index(name="Nilai")
            if ct == "Garis":
                fig = px.line(m, x="Month", y="Nilai", markers=True, color_discrete_sequence=["#C71585"])
                fig.update_traces(line_width=3, fill="tozeroy", fillcolor="rgba(199,21,133,0.10)")
            else:
                fig = px.bar(m, x="Month", y="Nilai", color_discrete_sequence=["#C71585"])
            show_fig(style_fig(fig, 360, f"📈 Tren Bulanan — {mt}"))
        with b:
            lvl = st.selectbox("Tingkat", ["Produk", "Kategori"])
            mtr = st.radio("Diurutkan berdasarkan", ["Pendapatan", "Kuantitas"], horizontal=True, key="dash_mtr")
            col = "Product_Name" if lvl == "Produk" else "Category"
            vcol = "Line_Total" if mtr == "Pendapatan" else "Quantity"
            top = fdf.groupby(col)[vcol].sum().sort_values(ascending=False).head(10).sort_values().reset_index()
            fig = px.bar(top, x=vcol, y=col, orientation="h", color=vcol, color_continuous_scale=["#FFC0CB", "#C71585"])
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis_title="", xaxis_title=mtr)
            show_fig(style_fig(fig, 360, f"🏆 Top 10 {lvl}"))
        a, b = st.columns(2)
        with a:
            cat_sales = fdf.groupby("Category")["Line_Total"].sum().reset_index()
            fig = px.pie(cat_sales, names="Category", values="Line_Total", hole=0.4, color_discrete_sequence=PINK + ["#E75480", "#F4A6C0"])
            fig.update_traces(textinfo="percent", textposition="inside")
            show_fig(style_fig(fig, 360, "🥧 Distribusi Penjualan per Kategori"))
        with b:
            pay = fdf.drop_duplicates("Transaction_ID")["Payment_Method"].value_counts().reset_index()
            pay.columns = ["Metode", "Jumlah"]
            fig = px.bar(pay, x="Jumlah", y="Metode", orientation="h", color_discrete_sequence=["#FF69B4"], text="Jumlah")
            show_fig(style_fig(fig, 360, "💳 Metode Pembayaran (per baris item)"))

    with t2:
        if not MLXTEND_OK:
            st.error("Pustaka `mlxtend` belum terpasang.")
        else:
            mx = make_basket_matrix(fdf)
            if len(mx) < 200:
                st.info("Transaksi terlalu sedikit pada filter ini untuk Apriori yang stabil. Longgarkan filter.")
            else:
                _, rl = run_apriori(mx, float(st.session_state.ap_sup), float(st.session_state.ap_conf),
                                    float(st.session_state.ap_lift), int(st.session_state.ap_len))
                if rl.empty:
                    st.info("Tidak ada aturan pada filter & parameter ini (atur di Tahap 3).")
                else:
                    tp = rank_rules(rl, "Lift → Confidence → Support", True, 5)
                    show_df(rules_table(tp))
                    fig = network_fig(rank_rules(rl, "Lift → Confidence → Support", True, 30), 8)
                    if fig:
                        show_fig(fig)

    with t3:
        sp = fdf.groupby("Segment").agg(Pelanggan=("Customer_ID", "nunique"), Pendapatan=("Line_Total", "sum")).reset_index()
        a, b = st.columns(2)
        with a:
            fig = px.bar(sp, x="Segment", y="Pelanggan", color="Segment", color_discrete_map=cmap, text="Pelanggan")
            fig.update_layout(showlegend=False)
            show_fig(style_fig(fig, 340, "Pelanggan per segmen (sesuai filter)"))
        with b:
            fig = px.pie(sp, names="Segment", values="Pendapatan", hole=0.45, color="Segment", color_discrete_map=cmap)
            show_fig(style_fig(fig, 340, "Kontribusi pendapatan per segmen"))
        show_df(prof[["Segment", "Pelanggan", "Recency_Mean", "Frequency_Mean", "Monetary_Mean"]].round(1)
                .rename(columns={"Recency_Mean": "Recency (mean)", "Frequency_Mean": "Frequency (mean)", "Monetary_Mean": "Monetary (mean)"}))
        st.markdown("#### ⭐ Bonus: pola produk per segmen pelanggan")
        pv = fdf.pivot_table(index="Category", columns="Segment", values="Line_Total", aggfunc="sum", fill_value=0)
        pv = pv / pv.sum() * 100
        fig = px.imshow(pv.round(1), aspect="auto", text_auto=".1f", color_continuous_scale=["#FFF0F5", "#FF69B4", "#8B008B"],
                        labels=dict(color="% pendapatan segmen"))
        show_fig(style_fig(fig, 420, "Porsi pendapatan tiap kategori dalam setiap segmen (%)"))

    with t4:
        mon = fdf.groupby("Month")["Line_Total"].sum()
        cat_s = fdf.groupby("Category")["Line_Total"].sum().sort_values(ascending=False)
        prd_s = fdf.groupby("Product_Name")["Line_Total"].sum().sort_values(ascending=False)
        seg_rev = fdf.groupby("Segment")["Line_Total"].sum().sort_values(ascending=False)
        insight(f"<b>Waktu puncak:</b> penjualan tertinggi pada <b>{mon.idxmax()}</b> ({fmt_short(mon.max())}); terendah <b>{mon.idxmin()}</b> ({fmt_short(mon.min())}). Siapkan stok dan promo mengikuti pola ini.", "fact")
        insight(f"<b>Kategori:</b> {cat_s.index[0]} memimpin dengan {cat_s.iloc[0] / cat_s.sum() * 100:.1f}% penjualan; produk terlaris berdasarkan pendapatan adalah <b>{prd_s.index[0]}</b>.", "fact")
        insight(f"<b>Segmen:</b> <b>{seg_rev.index[0]}</b> memberi {seg_rev.iloc[0] / seg_rev.sum() * 100:.1f}% pendapatan pada filter ini.", "fact")
        insight(f"<b>Keranjang:</b> rata-rata {avg_basket:.2f} produk unik per transaksi — peluang menaikkan nilai keranjang lewat bundling.", "fact")
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("📸 Untuk laporan, ambil tangkapan layar dashboard ini (Windows: Win+Shift+S) setelah memilih filter yang diinginkan.")
        st.download_button("⬇️ Unduh data sesuai filter (CSV)", csv_bytes(fdf), "data_terfilter.csv", "text/csv")
    st.button("Lanjut ke Tahap 6 ➜", on_click=goto, args=(MENU[6],), key="n5")


# =====================================================================
# TAHAP 6 — REKOMENDASI
# =====================================================================
def rec_table(bukti, tindakan, kpi_txt, biaya, asumsi, evaluasi):
    t = pd.DataFrame({"Komponen": ["📌 Bukti data", "🛠️ Tindakan", "🎯 KPI keberhasilan", "💰 Estimasi biaya/manfaat", "📝 Asumsi", "🧪 Evaluasi dampak"],
                      "Uraian": [bukti, tindakan, kpi_txt, biaya, asumsi, evaluasi]})
    show_df(t)


def page_rekomendasi(ctx):
    hero("💡 Tahap 6: Rekomendasi Keputusan Bisnis", "Tiga rekomendasi prioritas: penempatan/bundling, retensi pelanggan, dan tindakan pemasaran.", "Bobot 10%")
    clean, matrix = ctx["clean"], ctx["matrix"]
    rc, prof, *_ = get_segments(ctx)
    _, rules = get_rules(ctx) if MLXTEND_OK else (None, pd.DataFrame())
    N = len(matrix)
    results = {}
    summary_txt = ["# Ringkasan Rekomendasi Keputusan Bisnis\n"]

    card("<b>Cara membaca halaman ini.</b> Setiap rekomendasi memisahkan tiga hal: "
         "<b>Temuan faktual</b> (dihitung langsung dari data), <b>Estimasi berbasis asumsi</b> (dapat diubah dengan slider) dan "
         "<b>Dampak aktual</b> (belum ada — baru bisa diukur setelah implementasi diuji).")

    t1, t2, t3 = st.tabs(["🥇 Prioritas 1: Bundling & Penempatan Rak", "🥈 Prioritas 2: Retensi Pelanggan", "🥉 Prioritas 3: Program Loyalitas & Promo Musiman"])

    # --- Rekomendasi 1
    with t1:
        pairs = rank_rules(rules[(rules["n_ant"] == 1) & (rules["n_con"] == 1)], "Lift → Confidence → Support", True, 3) if not rules.empty else pd.DataFrame()
        if pairs.empty:
            st.info("Belum ada aturan pasangan produk. Longgarkan parameter di Tahap 3.")
            results["Bundling & penempatan rak"] = 0.0
        else:
            r = pairs.iloc[0]
            a, b = r["Jika (Antecedent)"], r["Maka (Consequent)"]
            n_a = int(matrix[a].sum())
            avg_b = float(clean.loc[clean["Product_Name"] == b, "Line_Total"].mean())
            top3 = ", ".join(f"{x['Jika (Antecedent)']}–{x['Maka (Consequent)']} (lift {x['lift']:.2f})" for _, x in pairs.iterrows())
            bukti = (f"{a} dan {b} dibeli bersama pada {r['Jumlah Transaksi']} transaksi (support {r['support'] * 100:.2f}%), confidence {r['confidence'] * 100:.1f}%, "
                     f"lift {r['lift']:.2f}. Tiga pasangan terkuat: {top3}.")
            insight(f"<b>Temuan faktual:</b> {bukti}", "fact")
            c1, c2, c3 = st.columns(3)
            up = c1.slider("Asumsi kenaikan attach rate (poin persen)", 1, 20, 10, key="r1_up")
            mg = c2.slider("Asumsi margin keuntungan (%)", 5, 40, 20, key="r1_mg")
            cost = c3.number_input("Biaya implementasi (Rp)", 0, 20_000_000, 100_000, step=50_000, key="r1_cost")
            extra_rev = n_a * up / 100 * avg_b
            net = extra_rev * mg / 100 - cost
            results["Bundling & penempatan rak"] = net
            insight(f"<b>Estimasi (berbasis asumsi):</b> {fmt_int(n_a)} transaksi memuat {a}. Bila {up} poin persen di antaranya ikut membeli {b} "
                    f"(rata-rata {fmt_idr(avg_b)}/baris), tambahan pendapatan ≈ <b>{fmt_idr(extra_rev)}</b> dan manfaat bersih ≈ <b>{fmt_idr(net)}</b> setelah margin {mg}% dan biaya.", "estimate")
            insight("<b>Dampak aktual:</b> belum diketahui — perlu diuji lebih dulu (lihat evaluasi).", "actual")
            rec_table(bukti, f"Letakkan {a} dan {b} berdampingan, buat paket hemat, dan minta kasir menawarkan {b} saat {a} dibeli.",
                      f"Attach rate {b} saat {a} dibeli (saat ini {r['confidence'] * 100:.1f}%), rata-rata basket_size, pendapatan per transaksi.",
                      f"Biaya {fmt_idr(cost)}; manfaat bersih estimasi {fmt_idr(net)} (bukan fakta).",
                      "Pelanggan bereaksi pada penataan; margin & rata-rata harga stabil; pola 2025 berlanjut; asosiasi ≠ sebab-akibat.",
                      "Uji A/B antar rak/toko atau perbandingan sebelum–sesudah 4–8 minggu; bandingkan attach rate dan basket_size dengan periode/toko kontrol.")
            summary_txt.append(f"## 1. Bundling & penempatan rak\n- Fakta: {bukti}\n- Estimasi manfaat bersih: {fmt_idr(net)} (asumsi: +{up} poin, margin {mg}%, biaya {fmt_idr(cost)})\n- Dampak aktual: belum diukur\n")

    # --- Rekomendasi 2
    with t2:
        churn = prof[prof["Segment"].apply(base_label) == "Churn Risk"]
        row = churn.sort_values("Pelanggan", ascending=False).iloc[0] if not churn.empty else prof.sort_values("Recency_Mean", ascending=False).iloc[0]
        seg = row["Segment"]
        n_c = int(row["Pelanggan"])
        trx_val = row["Total_Monetary"] / (rc.loc[rc["Segment"] == seg, "Frequency"].sum())
        bukti = (f"Segmen '{seg}': {n_c} pelanggan ({row['% Pelanggan']:.1f}%), Recency rata-rata {row['Recency_Mean']:.0f} hari, "
                 f"menyumbang {row['% Pendapatan']:.1f}% pendapatan; nilai rata-rata per transaksi {fmt_idr(trx_val)}.")
        insight(f"<b>Temuan faktual:</b> {bukti}", "fact")
        c1, c2, c3 = st.columns(3)
        rate = c1.slider("Asumsi tingkat pelanggan kembali (%)", 1, 40, 10, key="r2_rate")
        ntrx = c2.slider("Asumsi transaksi setelah kembali", 1, 6, 2, key="r2_trx")
        vc = c3.slider("Biaya voucher (% dari pendapatan tambahan)", 0, 30, 10, key="r2_vc")
        c4, c5 = st.columns(2)
        mg2 = c4.slider("Asumsi margin keuntungan (%)", 5, 40, 20, key="r2_mg")
        fixed = c5.number_input("Biaya kampanye tetap (Rp)", 0, 20_000_000, 400_000, step=100_000, key="r2_fix")
        extra = n_c * rate / 100 * ntrx * trx_val
        net2 = extra * (mg2 - vc) / 100 - fixed
        results["Retensi pelanggan berisiko"] = net2
        insight(f"<b>Estimasi (berbasis asumsi):</b> bila {rate}% dari {n_c} pelanggan kembali dan berbelanja {ntrx}× lagi, tambahan pendapatan ≈ <b>{fmt_idr(extra)}</b>; "
                f"manfaat bersih ≈ <b>{fmt_idr(net2)}</b> setelah margin {mg2}%, voucher {vc}% dan biaya tetap.", "estimate")
        insight("<b>Dampak aktual:</b> belum diketahui — baru dapat diukur setelah kampanye dijalankan pada kelompok uji.", "actual")
        rec_table(bukti, f"Kampanye win-back (voucher terbatas waktu + pesan personal) untuk segmen '{seg}'; survei singkat alasan berhenti.",
                  "Reactivation rate (% pelanggan segmen yang bertransaksi lagi dalam 60 hari), Recency rata-rata segmen, pendapatan dari pelanggan yang kembali.",
                  f"Biaya tetap {fmt_idr(fixed)} + voucher {vc}%; manfaat bersih estimasi {fmt_idr(net2)} (bukan fakta).",
                  "Pelanggan berisiko masih dapat dijangkau; tingkat kembali & jumlah transaksi sesuai slider; margin stabil.",
                  "Bagi acak segmen menjadi kelompok kampanye vs kontrol (tanpa kampanye); bandingkan reactivation rate dan pendapatan 60–90 hari.")
        summary_txt.append(f"## 2. Retensi pelanggan berisiko\n- Fakta: {bukti}\n- Estimasi manfaat bersih: {fmt_idr(net2)} (asumsi: kembali {rate}%, {ntrx} transaksi, voucher {vc}%, margin {mg2}%)\n- Dampak aktual: belum diukur\n")

    # --- Rekomendasi 3
    with t3:
        loyal = prof[prof["Segment"].apply(base_label) == "Loyal Customers"]
        lrow = loyal.iloc[0] if not loyal.empty else prof.sort_values("Monetary_Mean", ascending=False).iloc[0]
        mon = clean.groupby("Month")["Line_Total"].sum()
        peak_m, peak_v, avg_m = mon.idxmax(), mon.max(), mon.mean()
        bukti = (f"Segmen '{lrow['Segment']}' = {lrow['% Pelanggan']:.1f}% pelanggan tetapi {lrow['% Pendapatan']:.1f}% pendapatan "
                 f"(Monetary rata-rata {fmt_idr(lrow['Monetary_Mean'])}). Bulan puncak {peak_m} ({fmt_short(peak_v)}) = {peak_v / avg_m * 100 - 100:+.1f}% dibanding rata-rata bulanan.")
        insight(f"<b>Temuan faktual:</b> {bukti}", "fact")
        c1, c2, c3 = st.columns(3)
        upl = c1.slider("Asumsi kenaikan belanja segmen loyal (%)", 1, 20, 5, key="r3_up")
        mg3 = c2.slider("Asumsi margin keuntungan (%)", 5, 40, 20, key="r3_mg")
        rew = c3.slider("Biaya reward (% dari kenaikan belanja)", 0, 50, 10, key="r3_rw")
        fixed3 = st.number_input("Biaya program & promosi tetap (Rp)", 0, 30_000_000, 700_000, step=100_000, key="r3_fix")
        extra3 = lrow["Total_Monetary"] * upl / 100
        net3 = extra3 * (mg3 - rew) / 100 - fixed3
        results["Program loyalitas & promo musiman"] = net3
        insight(f"<b>Estimasi (berbasis asumsi):</b> kenaikan belanja {upl}% pada segmen ini ≈ <b>{fmt_idr(extra3)}</b> per tahun; "
                f"manfaat bersih ≈ <b>{fmt_idr(net3)}</b> setelah margin {mg3}%, reward {rew}% dan biaya tetap.", "estimate")
        insight("<b>Dampak aktual:</b> belum diketahui — butuh periode uji dan pembanding.", "actual")
        rec_table(bukti, f"Program poin/tier untuk '{lrow['Segment']}', ditambah promo musiman menjelang {peak_m} (stok & iklan lebih awal).",
                  "Frekuensi belanja segmen loyal, retensi 90 hari, pendapatan per pelanggan loyal, penjualan bulan puncak.",
                  f"Biaya tetap {fmt_idr(fixed3)} + reward {rew}%; manfaat bersih estimasi {fmt_idr(net3)} (bukan fakta).",
                  "Pelanggan loyal merespons reward; pola musiman 2025 berulang; margin stabil.",
                  "Bandingkan pelanggan loyal yang ikut program vs yang belum (atau periode sebelum–sesudah) dan hitung selisih pendapatan.")
        summary_txt.append(f"## 3. Program loyalitas & promo musiman\n- Fakta: {bukti}\n- Estimasi manfaat bersih: {fmt_idr(net3)} (asumsi: +{upl}%, margin {mg3}%, reward {rew}%)\n- Dampak aktual: belum diukur\n")

    st.markdown("### 🧭 Ringkasan Prioritas")
    res = pd.DataFrame({"Rekomendasi": list(results.keys()), "Estimasi manfaat bersih (Rp)": list(results.values())})
    res = res.sort_values("Estimasi manfaat bersih (Rp)", ascending=False).reset_index(drop=True)
    res.insert(0, "Prioritas (berdasarkan estimasi)", range(1, len(res) + 1))
    c1, c2 = st.columns([1, 1])
    with c1:
        show_df(res)
        st.caption("Estimasi bergantung penuh pada asumsi di slider; bukan fakta kenaikan penjualan.")
    with c2:
        fig = px.bar(res, x="Estimasi manfaat bersih (Rp)", y="Rekomendasi", orientation="h", color_discrete_sequence=["#C71585"])
        show_fig(style_fig(fig, 260, "Estimasi manfaat bersih (berbasis asumsi)"))
    st.download_button("⬇️ Unduh ringkasan rekomendasi (.md)", "\n".join(summary_txt).encode("utf-8"), "ringkasan_rekomendasi.md", "text/markdown")
    if st.button("🎉 Selesai — rayakan!"):
        st.balloons()


# =====================================================================
# ROUTER HALAMAN
# =====================================================================
if menu == MENU[0]:
    page_home()
elif st.session_state.df_raw is None:
    st.warning("⚠️ Silakan kembali ke menu **Beranda & Upload Dataset** terlebih dahulu untuk meng-upload file Excel UTS Anda!")
    st.button("🏠 Ke Beranda", on_click=goto, args=(MENU[0],))
    st.stop()
else:
    _ctx = get_ctx()
    if menu == MENU[1]:
        page_eda(_ctx)
    elif menu == MENU[2]:
        page_prep(_ctx)
    elif menu == MENU[3]:
        page_apriori(_ctx)
    elif menu == MENU[4]:
        page_kmeans(_ctx)
    elif menu == MENU[5]:
        page_dashboard(_ctx)
    elif menu == MENU[6]:
        page_rekomendasi(_ctx)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #C71585;'>💖 Dibuat dengan penuh ketelitian untuk UTS Business Intelligence — {NAMA} ({NPM}) | Universitas Timor 💖</p>", unsafe_allow_html=True)