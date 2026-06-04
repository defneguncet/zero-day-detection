"""
ZeroSentinel — Zero-Day Tehdit Tespit Dashboard
Backend: Flask
Modeller: xgboost_hybrid_zeroday.json (72 feat) + isolation_forest.pkl (71 feat)
Yükleme: joblib (Python 3.14 uyumluluğu için)
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import shap
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Model yolları ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# Render'da repo kökü BASE_DIR ile aynı olur, lokalda bir üst klasör
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

XGB_PATH      = os.path.join(MODELS_DIR, 'xgboost_hybrid_zeroday.json')
IF_PATH       = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'final_hybrid_features.json')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

# ── Modelleri yükle ──────────────────────────────────────────────────────────
print("=" * 55)
print("  ZeroSentinel — Model Yükleme Başlıyor")
print("=" * 55)

# XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)
print(f"  ✅ XGBoost yüklendi  ({xgb_model.n_features_in_} özellik)")

# Isolation Forest — joblib gerekiyor (Python 3.14 uyumluluğu)
iso_forest = joblib.load(IF_PATH)
print(f"  ✅ Isolation Forest yüklendi  ({iso_forest.n_features_in_} özellik)")

# Feature listesi (72 adet — IF_Anomaly_Score dahil)
with open(FEATURES_PATH, 'r') as f:
    ALL_FEATURES = json.load(f)

# IF için kullanılacak 71 özellik (IF_Anomaly_Score hariç)
IF_FEATURES = [c for c in ALL_FEATURES if c != 'IF_Anomaly_Score']
print(f"  ✅ {len(ALL_FEATURES)} özellik yüklendi  (IF girdisi: {len(IF_FEATURES)})")

# Metadata — eşik ve diğer config
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)
THRESHOLD = float(metadata.get('best_threshold', 0.5))
print(f"  ✅ Dinamik eşik: {THRESHOLD:.6f}")

# Leak sütunları (08_feature_engineering'de çıkarılmıştı)
LEAK_COLS = ['Init Fwd Win Byts', 'Init Bwd Win Byts']

# SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
print("  ✅ SHAP explainer hazır")
print("=" * 55)
print("  🚀 http://localhost:5000")
print("=" * 55 + "\n")


# ── Ön işlem fonksiyonları ───────────────────────────────────────────────────

def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    08_feature_engineering.ipynb ile birebir aynı mantık.
    Leak sütunlarını düşür + 4 davranışsal özellik üret.
    """
    df = df.drop(columns=LEAK_COLS, errors='ignore')
    df['Pkt_Size_Ratio']      = df['Fwd Pkt Len Mean'] / (df['Bwd Pkt Len Mean'] + 1)
    total_pkts                = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
    df['Duration_per_Packet'] = df['Flow Duration'] / (total_pkts + 1)
    df['Fwd_IAT_Ratio']       = df['Fwd IAT Mean']  / (df['Flow IAT Mean'] + 1)
    df['Fwd_Bwd_Pkt_Ratio']   = df['Tot Fwd Pkts']  / (df['Tot Bwd Pkts'] + 1)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham CSV → model girdisine hazır DataFrame.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Ham CSV fazladan sütunları düşür (03_preprocessing ile silindiler)
    RAW_EXTRA = [
        "Dst Port", "Timestamp",
        "Bwd PSH Flags", "Bwd URG Flags",
        "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
        "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    ]
    df.drop(columns=RAW_EXTRA, errors="ignore", inplace=True)

    # Tüm sütunları sayısala zorla — karışık tip içeren CSV'ler için
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

    # FE (leak temizliği burada)
    df = add_behavioral_features(df)

    # Adım 4 — Isolation Forest (71 IF_FEATURES üzerinden)
    if_input_cols = [c for c in IF_FEATURES if c in df.columns]
    if_input      = df[if_input_cols].copy()
    if_input.fillna(if_input.median(numeric_only=True), inplace=True)

    # IF, eğitimde tam 71 sütun gördü; eksik sütunları 0 ile tamamla
    for col in IF_FEATURES:
        if col not in if_input.columns:
            if_input[col] = 0.0
    if_input = if_input[IF_FEATURES]

    df['IF_Anomaly_Score'] = iso_forest.decision_function(if_input)

    # Adım 5 — Sütunları hizala (ALL_FEATURES sırasına göre)
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df = df[ALL_FEATURES]

    # Adım 6 — NaN temizle
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


# ── Grafik üretici fonksiyonlar ──────────────────────────────────────────────

def shap_bar_to_b64(shap_values, feature_names, top_n=15):
    mean_abs  = np.abs(shap_values).mean(axis=0)
    idx       = np.argsort(mean_abs)[-top_n:][::-1]
    top_feats = [feature_names[i] for i in idx]
    top_vals  = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ['#ef4444' if v > top_vals.mean() else '#f97316' for v in top_vals]
    ax.barh(top_feats[::-1], top_vals[::-1], color=colors[::-1], edgecolor='none')
    ax.set_xlabel('mean(|SHAP value|)', fontsize=10, color='#94a3b8')
    ax.set_title('Karar Mekanizması — Özellik Etkileri', fontsize=12,
                 fontweight='bold', color='#f1f5f9', pad=12)
    ax.tick_params(colors='#94a3b8')
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#334155')
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def cm_to_b64(tn, fp, fn, tp):
    cm  = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Normal', 'Saldırı'],
                yticklabels=['Normal', 'Saldırı'],
                ax=ax, linewidths=0.5, linecolor='#1e293b',
                annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
    ax.set_ylabel('Gerçek', color='#94a3b8')
    ax.set_xlabel('Tahmin', color='#94a3b8')
    ax.set_title('Confusion Matrix', color='#f1f5f9', fontweight='bold')
    ax.tick_params(colors='#94a3b8')
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── Route'lar ────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZeroSentinel</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#020817;--sur:#0f172a;--s2:#1e293b;--brd:#1e293b;--red:#ef4444;--org:#f97316;--grn:#22c55e;--yel:#eab308;--txt:#f1f5f9;--mut:#64748b;--mono:'Share Tech Mono',monospace;--sans:'Syne',sans-serif}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);min-height:100vh}
body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;background-image:linear-gradient(rgba(239,68,68,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(239,68,68,.025) 1px,transparent 1px);background-size:40px 40px}
header{position:relative;z-index:10;height:62px;border-bottom:1px solid var(--brd);padding:0 2rem;display:flex;align-items:center;gap:1.25rem;background:rgba(15,23,42,.92);backdrop-filter:blur(12px)}
.lhex{width:34px;height:34px;background:var(--red);clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);display:grid;place-items:center;font-family:var(--mono);font-size:.6rem;color:#fff;font-weight:700;animation:hp 3s ease-in-out infinite}
@keyframes hp{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.5)}50%{box-shadow:0 0 0 10px rgba(239,68,68,0)}}
h1{font-size:1.15rem;font-weight:800;letter-spacing:.08em;background:linear-gradient(135deg,#ef4444,#f97316);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hr{margin-left:auto;display:flex;align-items:center;gap:1rem}
.badge{font-family:var(--mono);font-size:.68rem;padding:.2rem .6rem;border-radius:4px;border:1px solid var(--brd);color:var(--mut)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--grn);box-shadow:0 0 6px var(--grn);animation:bk 2s ease-in-out infinite}
@keyframes bk{0%,100%{opacity:1}50%{opacity:.25}}
main{position:relative;z-index:1;max-width:1380px;margin:0 auto;padding:2rem;display:grid;gap:1.5rem}
.zdbanner{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.25);border-radius:10px;padding:1rem 1.4rem;font-family:var(--mono);font-size:.75rem;color:var(--mut);line-height:1.8}
.zdbanner strong{color:var(--red)}
/* Upload */
.upwrap{position:relative}
.upzone{border:2px dashed var(--brd);border-radius:12px;padding:3rem 2rem;text-align:center;background:var(--sur);transition:all .3s;position:relative;overflow:hidden;cursor:pointer}
.upzone::before{content:'';position:absolute;inset:0;opacity:0;transition:opacity .3s;background:radial-gradient(ellipse at center,rgba(239,68,68,.06) 0%,transparent 70%)}
.upzone:hover{border-color:var(--red)}
.upzone:hover::before{opacity:1}
.upzone.drag{border-color:var(--red)}
.upzone.drag::before{opacity:1}
.upicon{font-size:2.8rem;display:block;margin-bottom:.75rem}
.upzone h2{font-size:1rem;font-weight:700;margin-bottom:.4rem;pointer-events:none}
.upzone p{font-size:.8rem;color:var(--mut);font-family:var(--mono);pointer-events:none}
#fname{margin-top:.65rem;color:var(--red);font-family:var(--mono);font-size:.78rem;pointer-events:none}
.btnrun{margin-top:1.4rem;padding:.7rem 2.2rem;background:linear-gradient(135deg,var(--red),var(--org));border:none;border-radius:8px;color:#fff;font-family:var(--sans);font-size:.88rem;font-weight:700;cursor:pointer;letter-spacing:.05em;transition:all .2s;display:none;position:relative;z-index:5}
.btnrun:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(239,68,68,.3)}
.btnrun.show{display:inline-block}
.btnrun:disabled{opacity:.5;cursor:not-allowed;transform:none}
/* Loader */
.loader{display:none;text-align:center;margin-top:1rem}
.loader.show{display:block}
.lbar{width:220px;height:3px;background:var(--s2);border-radius:2px;margin:.5rem auto;overflow:hidden}
.lbar::after{content:'';display:block;width:40%;height:100%;background:linear-gradient(90deg,var(--red),var(--org));animation:sw 1.3s ease-in-out infinite}
@keyframes sw{0%{transform:translateX(-100%)}100%{transform:translateX(350%)}}
.loader p{font-family:var(--mono);font-size:.78rem;color:var(--mut)}
/* Stats */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:1rem}
.scard{background:var(--sur);border:1px solid var(--brd);border-radius:10px;padding:1.2rem 1.4rem;position:relative;overflow:hidden;opacity:0;transform:translateY(10px);transition:all .4s ease}
.scard.show{opacity:1;transform:translateY(0)}
.scard::after{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.scard.r::after{background:linear-gradient(90deg,var(--red),var(--org))}
.scard.g::after{background:var(--grn)}
.scard.y::after{background:var(--yel)}
.scard.b::after{background:#38bdf8}
.slabel{font-size:.68rem;font-family:var(--mono);color:var(--mut);letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem}
.sval{font-size:1.9rem;font-weight:800;line-height:1.1}
.sval.r{color:var(--red)}.sval.g{color:var(--grn)}.sval.y{color:var(--yel)}.sval.b{color:#38bdf8}
.ssub{font-size:.72rem;color:var(--mut);font-family:var(--mono);margin-top:.3rem}
/* Panel */
.panel{background:var(--sur);border:1px solid var(--brd);border-radius:12px;overflow:hidden}
.ph{padding:.85rem 1.4rem;border-bottom:1px solid var(--brd);display:flex;align-items:center;gap:.65rem}
.pdot{width:7px;height:7px;border-radius:50%;background:var(--red)}
.ph h3{font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--mut)}
.pb{padding:1.4rem}
/* Risk */
.rw{margin-bottom:.7rem}
.rl{display:flex;justify-content:space-between;font-size:.72rem;font-family:var(--mono);color:var(--mut);margin-bottom:.28rem}
.rt{height:7px;background:var(--s2);border-radius:4px;overflow:hidden}
.rf{height:100%;border-radius:4px;width:0%;transition:width 1.1s cubic-bezier(.34,1.56,.64,1)}
.rf.low{background:var(--grn)}.rf.mid{background:var(--yel)}.rf.high{background:var(--red)}
/* Two col */
.two{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
@media(max-width:860px){.two{grid-template-columns:1fr}}
.cimg{width:100%;border-radius:8px;display:block}
/* Metrics */
.mt{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.78rem;margin-top:1rem}
.mt td{padding:.55rem .7rem;border-bottom:1px solid var(--brd)}
.mt td:first-child{color:var(--mut)}.mt td:last-child{text-align:right;font-weight:700}
/* Top5 */
.t5{width:100%;border-collapse:collapse;font-size:.78rem}
.t5 th{font-family:var(--mono);font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);padding:.45rem .7rem;text-align:left;border-bottom:1px solid var(--brd)}
.t5 td{padding:.55rem .7rem;border-bottom:1px solid rgba(30,41,59,.5);font-family:var(--mono)}
.t5 tr:last-child td{border-bottom:none}
.tag{display:inline-block;padding:.12rem .45rem;border-radius:4px;font-size:.68rem;font-weight:700}
.tag.atk{background:rgba(239,68,68,.15);color:var(--red)}.tag.nrm{background:rgba(34,197,94,.15);color:var(--grn)}
.sp{display:inline-block;width:56px;height:5px;border-radius:3px;background:var(--s2);vertical-align:middle;margin-right:.4rem;overflow:hidden}
.spf{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--org),var(--red))}
#results{display:none}
#results.show{display:grid;gap:1.5rem}
.toast{position:fixed;bottom:1.5rem;right:1.5rem;z-index:200;background:var(--s2);border:1px solid var(--red);border-radius:8px;padding:.9rem 1.3rem;font-family:var(--mono);font-size:.8rem;color:var(--red);opacity:0;transform:translateY(6px);transition:all .3s;pointer-events:none}
.toast.show{opacity:1;transform:translateY(0)}
footer{position:relative;z-index:1;text-align:center;padding:1.5rem 2rem;font-family:var(--mono);font-size:.65rem;color:var(--mut);border-top:1px solid var(--brd)}
</style>
</head>
<body>
<header>
  <div style="display:flex;align-items:center;gap:.75rem">
    <div class="lhex">ZS</div>
    <h1>ZEROSENTINEL</h1>
  </div>
  <div class="hr">
    <span class="badge">CIC-IDS 2018</span>
    <span class="badge">XGBoost + IF</span>
    <span class="badge" id="thr-badge">Eşik: THRESHOLD_VAL</span>
    <span class="badge" id="feat-badge">FEAT_COUNT Özellik</span>
    <div style="display:flex;align-items:center;gap:.4rem">
      <div class="dot"></div>
      <span style="font-family:var(--mono);font-size:.68rem;color:var(--grn)">HAZIR</span>
    </div>
  </div>
</header>

<main>
  <div class="zdbanner">
    <strong>Zero-Day Simülasyonu:</strong> Eğitimde görülen → DoS, DDoS, SSH BruteForce, Web Attack &nbsp;|&nbsp;
    <strong>Gizlenen Zero-Day:</strong> Bot &amp; Infilteration &nbsp;|&nbsp;
    <strong>Leak giderildi:</strong> Init Fwd/Bwd Win Byts çıkarıldı &nbsp;|&nbsp;
    <strong>Davranışsal özellikler:</strong> Pkt_Size_Ratio, Duration_per_Packet, Fwd_IAT_Ratio, Fwd_Bwd_Pkt_Ratio
  </div>

  <!-- Gizli file input -->
  <input type="file" id="fileInput" accept=".csv" style="display:none">

  <div class="upzone" id="dropzone">
    <span class="upicon">⬡</span>
    <h2>CSV Dosyası Yükle</h2>
    <p>NetFlow / CIC-IDS 2018 formatında ağ akış verisi</p>
    <p>Sürükle-bırak veya tıkla &nbsp;·&nbsp; Label sütunu varsa Confusion Matrix otomatik</p>
    <p id="fname"></p>
    <button class="btnrun" id="runBtn">⬡ ANALİZ BAŞLAT</button>
  </div>

  <div class="loader" id="loader">
    <div class="lbar"></div>
    <p id="ltxt">Model çalışıyor...</p>
  </div>

  <div id="results">
    <div class="stats">
      <div class="scard r" id="sc0"><div class="slabel">Tespit Edilen Saldırı</div><div class="sval r" id="vatk">—</div><div class="ssub" id="satk">paket</div></div>
      <div class="scard g" id="sc1"><div class="slabel">Normal Trafik</div><div class="sval g" id="vnrm">—</div><div class="ssub">paket</div></div>
      <div class="scard y" id="sc2"><div class="slabel">Ort. Tehdit Olasılığı</div><div class="sval y" id="vprb">—</div><div class="ssub">0.0 güvenli → 1.0 tehdit</div></div>
      <div class="scard b" id="sc3"><div class="slabel">Toplam Analiz</div><div class="sval b" id="vtot">—</div><div class="ssub">paket satırı</div></div>
    </div>
    <div class="panel">
      <div class="ph"><div class="pdot"></div><h3>Risk Dağılımı</h3></div>
      <div class="pb">
        <div class="rw"><div class="rl"><span>🟢 Düşük Risk (&lt;0.3)</span><span id="rll">—</span></div><div class="rt"><div class="rf low" id="rlb"></div></div></div>
        <div class="rw"><div class="rl"><span>🟡 Orta Risk (0.3–0.7)</span><span id="rml">—</span></div><div class="rt"><div class="rf mid" id="rmb"></div></div></div>
        <div class="rw"><div class="rl"><span>🔴 Yüksek Risk (&gt;0.7)</span><span id="rhl">—</span></div><div class="rt"><div class="rf high" id="rhb"></div></div></div>
      </div>
    </div>
    <div class="two">
      <div class="panel">
        <div class="ph"><div class="pdot"></div><h3>SHAP — Karar Açıklaması (XAI)</h3></div>
        <div class="pb"><img id="shapimg" class="cimg" src="" alt="SHAP"></div>
      </div>
      <div class="panel" id="cmpanel" style="display:none">
        <div class="ph"><div class="pdot"></div><h3>Confusion Matrix + Metrikler</h3></div>
        <div class="pb">
          <img id="cmimg" class="cimg" src="" alt="CM">
          <table class="mt">
            <tr><td>Accuracy</td><td id="macc">—</td></tr>
            <tr><td>Precision</td><td id="mpre">—</td></tr>
            <tr><td>Recall</td><td id="mrec">—</td></tr>
            <tr><td>F1-Score</td><td id="mf1">—</td></tr>
          </table>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="ph"><div class="pdot"></div><h3>En Yüksek Riskli 5 Paket</h3></div>
      <div class="pb" style="padding:0">
        <table class="t5">
          <thead><tr><th>Satır #</th><th>Tehdit Skoru</th><th>Tahmin</th></tr></thead>
          <tbody id="t5body"></tbody>
        </table>
      </div>
    </div>
  </div>
</main>

<footer>ZeroSentinel &nbsp;·&nbsp; CIC-IDS 2018 &nbsp;·&nbsp; XGBoost + Isolation Forest &nbsp;·&nbsp; Zero-Day: Bot &amp; Infilteration</footer>
<div class="toast" id="toast"></div>

<script>
// ── Sabit değerleri yerleştir ──
document.getElementById('thr-badge').textContent  = 'Eşik: THRESHOLD_VAL';
document.getElementById('feat-badge').textContent = 'FEAT_COUNT Özellik';

// ── Değişkenler ──
const fileInput = document.getElementById('fileInput');
const runBtn    = document.getElementById('runBtn');
const dropzone  = document.getElementById('dropzone');
let   selFile   = null;

// ── Dropzone tıklama: buton hariç her yere tıklayınca dosya seç ──
dropzone.addEventListener('click', function(e) {
  if (e.target === runBtn || runBtn.contains(e.target)) return;
  fileInput.click();
});

// ── Buton ayrı listener ──
runBtn.addEventListener('click', function(e) {
  e.stopPropagation();
  analyze();
});

// ── Dosya seçilince ──
fileInput.addEventListener('change', function(e) {
  const f = e.target.files[0];
  if (!f) return;
  selFile = f;
  document.getElementById('fname').textContent = '⬡ ' + f.name;
  runBtn.classList.add('show');
});

// ── Drag & Drop ──
dropzone.addEventListener('dragover',  function(e){ e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', function()  { dropzone.classList.remove('drag'); });
dropzone.addEventListener('drop', function(e) {
  e.preventDefault(); dropzone.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f && f.name.endsWith('.csv')) {
    selFile = f;
    document.getElementById('fname').textContent = '⬡ ' + f.name;
    runBtn.classList.add('show');
  } else { showToast('Sadece .csv dosyaları destekleniyor.'); }
});

// ── Analiz ──
async function analyze() {
  if (!selFile) { showToast('Önce bir CSV dosyası seç.'); return; }
  runBtn.disabled = true;
  document.getElementById('loader').classList.add('show');
  document.getElementById('results').classList.remove('show');

  const msgs = [
    'Veriler okunuyor...',
    'Leak sütunları temizleniyor, özellikler üretiliyor...',
    'Isolation Forest anomali skorları hesaplanıyor...',
    'XGBoost tahminleri yapılıyor...',
    'SHAP değerleri hesaplanıyor...'
  ];
  let mi = 0;
  const iv = setInterval(() => {
    document.getElementById('ltxt').textContent = msgs[Math.min(++mi, msgs.length-1)];
  }, 2200);

  const fd = new FormData();
  fd.append('file', selFile);

  try {
    const res  = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    clearInterval(iv);
    if (!res.ok || data.error) { showToast('Hata: ' + (data.error || 'Bilinmeyen')); return; }
    render(data);
  } catch(err) {
    clearInterval(iv);
    showToast('Sunucu hatası: ' + err.message);
  } finally {
    document.getElementById('loader').classList.remove('show');
    runBtn.disabled = false;
  }
}

function render(d) {
  g('vatk').textContent = d.n_attack.toLocaleString('tr-TR');
  g('vnrm').textContent = d.n_normal.toLocaleString('tr-TR');
  g('vprb').textContent = d.avg_proba.toFixed(4);
  g('vtot').textContent = d.total_rows.toLocaleString('tr-TR');
  g('satk').textContent = 'toplam %' + ((d.n_attack/d.total_rows)*100).toFixed(1);

  ['sc0','sc1','sc2','sc3'].forEach((id,i) =>
    setTimeout(() => g(id).classList.add('show'), i*80));

  rbar('l', d.risk_low,    d.total_rows);
  rbar('m', d.risk_medium, d.total_rows);
  rbar('h', d.risk_high,   d.total_rows);

  g('shapimg').src = 'data:image/png;base64,' + d.shap_img;

  if (d.has_label && d.cm_img) {
    g('cmimg').src = 'data:image/png;base64,' + d.cm_img;
    g('cmpanel').style.display = 'block';
    g('macc').textContent = pct(d.accuracy);
    g('mpre').textContent = pct(d.precision);
    g('mrec').textContent = pct(d.recall);
    g('mf1').textContent  = pct(d.f1);
  }

  const tb = g('t5body');
  tb.innerHTML = '';
  d.top5.forEach(r => {
    const w = (r.score * 100).toFixed(1);
    tb.innerHTML += `<tr>
      <td>#${r.row}</td>
      <td><span class="sp"><span class="spf" style="width:${w}%"></span></span>${r.score}</td>
      <td><span class="tag ${r.label==='Saldırı'?'atk':'nrm'}">${r.label}</span></td>
    </tr>`;
  });

  g('results').classList.add('show');
  g('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function rbar(k, count, total) {
  const p = total > 0 ? count/total*100 : 0;
  g('r'+k+'l').textContent = count.toLocaleString('tr-TR') + ' (%' + p.toFixed(1) + ')';
  setTimeout(() => { g('r'+k+'b').style.width = p+'%'; }, 80);
}
function g(id) { return document.getElementById(id); }
function pct(v) { return v != null ? '%' + (v*100).toFixed(2) : '—'; }
function showToast(msg) {
  const t = g('toast'); t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 4500);
}
</script>
</body>
</html>"""


@app.route('/')
def index():
    thr_str  = '{:.6f}'.format(float(THRESHOLD))
    feat_str = '{:d}'.format(int(len(ALL_FEATURES)))
    page = HTML_PAGE.replace('THRESHOLD_VAL', thr_str)
    page = page.replace('FEAT_COUNT', feat_str)
    return page


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı.'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Sadece .csv dosyaları destekleniyor.'}), 400

    try:
        df_raw    = pd.read_csv(file)
        total_rows = len(df_raw)

        # Label sütunu varsa ayır
        df_raw.columns = df_raw.columns.str.strip()
        has_label = 'Label' in df_raw.columns
        y_true    = None

        if has_label:
            y_raw   = df_raw['Label'].values
            df_feat = df_raw.drop(columns=['Label'])
            # Her durumda string'e çevir → binary yap (karışık tip sorununu önler)
            y_raw_str = [str(v).strip() for v in y_raw]
            y_true    = np.array([0 if v == 'Benign' else 1 for v in y_raw_str], dtype=int)
        else:
            df_feat = df_raw.copy()

        # Ön işlem
        X = preprocess(df_feat)

        # Tahmin
        y_proba = xgb_model.predict_proba(X)[:, 1]
        y_pred  = (y_proba >= THRESHOLD).astype(int)

        n_attack  = int(y_pred.sum())
        n_normal  = int((y_pred == 0).sum())
        avg_proba = float(y_proba.mean())

        # Risk dağılımı
        risk_low    = int((y_proba < 0.3).sum())
        risk_medium = int(((y_proba >= 0.3) & (y_proba < 0.7)).sum())
        risk_high   = int((y_proba >= 0.7).sum())

        # SHAP (max 300 satır)
        sample_n   = min(300, len(X))
        X_samp     = X.iloc[:sample_n]
        shap_vals  = explainer.shap_values(X_samp)
        shap_img   = shap_bar_to_b64(shap_vals, list(X.columns))

        # Confusion matrix + metrikler (label varsa)
        cm_img    = None
        accuracy = precision = recall = f1 = None

        if y_true is not None:
            from sklearn.metrics import confusion_matrix, classification_report
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cm_img   = cm_to_b64(int(tn), int(fp), int(fn), int(tp))
            report   = classification_report(y_true, y_pred, output_dict=True,
                                              zero_division=0)
            accuracy  = round(report['accuracy'], 4)
            # Saldırı sınıfı = '1'
            precision = round(report.get('1', report.get(1, {})).get('precision', 0), 4)
            recall    = round(report.get('1', report.get(1, {})).get('recall', 0), 4)
            f1        = round(report.get('1', report.get(1, {})).get('f1-score', 0), 4)

        # En riskli 5 paket
        top5_idx = np.argsort(y_proba)[-5:][::-1]
        top5     = [{'row': int(i),
                     'score': round(float(y_proba[i]), 4),
                     'label': 'Saldırı' if y_pred[i] == 1 else 'Normal'}
                    for i in top5_idx]

        return jsonify({
            'success'    : True,
            'total_rows' : total_rows,
            'n_attack'   : n_attack,
            'n_normal'   : n_normal,
            'avg_proba'  : round(avg_proba, 4),
            'risk_low'   : risk_low,
            'risk_medium': risk_medium,
            'risk_high'  : risk_high,
            'threshold'  : round(THRESHOLD, 6),
            'shap_img'   : shap_img,
            'cm_img'     : cm_img,
            'accuracy'   : accuracy,
            'precision'  : precision,
            'recall'     : recall,
            'f1'         : f1,
            'top5'       : top5,
            'has_label'  : has_label,
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/metadata')
def get_metadata():
    return jsonify({
        'threshold'         : round(THRESHOLD, 6),
        'feature_count'     : len(ALL_FEATURES),
        'if_feature_count'  : len(IF_FEATURES),
        'model'             : 'XGBoost + Isolation Forest Hibrit',
        'dataset'           : 'CIC-IDS 2018',
        'zero_day_classes'  : ['Bot', 'Infilteration'],
        'leak_cols_removed' : LEAK_COLS,
        'status'            : metadata.get('status', ''),
    })


if __name__ == '__main__':
    app.run(debug=False, port=5000)
