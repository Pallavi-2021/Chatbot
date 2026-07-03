"""
Sales Visit Intelligence — Streamlit Cloud Edition
====================================================
Single file. No utils/ folder. No heavy dependencies.
All errors shown in the UI (never a blank crash page).

Requirements: streamlit, pandas, numpy, plotly, openpyxl, google-genai
"""

import gc
import io
import os
import re
import time
import hashlib
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = "your-gemini-api-key-here"  # ← paste key here, OR use Streamlit secrets
GEMINI_MODEL   = "gemini-2.5-flash"

def get_key() -> Optional[str]:
    # 1. hardcoded above
    if GEMINI_API_KEY and "your-gemini" not in GEMINI_API_KEY:
        return GEMINI_API_KEY
    # 2. Streamlit secrets
    try:
        v = st.secrets.get("GEMINI_API_KEY")
        if v: return v
    except Exception:
        pass
    # 3. environment variable
    return os.environ.get("GEMINI_API_KEY")

def get_model() -> str:
    try:
        v = st.secrets.get("GEMINI_MODEL")
        if v: return str(v).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_MODEL", GEMINI_MODEL)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sales Visit Intelligence",
    layout="wide", page_icon="🤖",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#f7f8fa}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e8eaed}
.kpi-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.kpi-card{background:#fff;border-radius:10px;padding:14px 18px;flex:1;
          min-width:130px;box-shadow:0 1px 4px rgba(0,0,0,.07);
          border-left:4px solid #4f8ef7}
.kpi-card.red{border-left-color:#e53935}.kpi-card.amber{border-left-color:#fb8c00}
.kpi-card.green{border-left-color:#43a047}
.kpi-label{font-size:11px;color:#888;font-weight:700;
           text-transform:uppercase;letter-spacing:.05em}
.kpi-value{font-size:24px;font-weight:800;color:#1a1a2e;line-height:1.2}
.kpi-sub{font-size:11px;color:#bbb;margin-top:2px}
.sr-row{display:flex;justify-content:space-between;align-items:center;
        padding:6px 0;border-bottom:1px solid #f0f0f0;font-size:13px}
.sr-row:last-child{border-bottom:none}
</style>""", unsafe_allow_html=True)

# session state defaults
for k, v in [("result", None), ("store", None), ("chat", None),
              ("history", []), ("model_used", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

PLAN_COLS   = ["DistributorCode","SRCode","Date","StoreID","PlannedCalls"]
ACTUAL_COLS = ["DistributorCode","SRCode","Date","StoreID","VisitStatus",
               "VisitLatitude","VisitLongitude","StoreLatitude","StoreLongitude"]
_SKIP = {"DistributorCode","DistributorName","SRCode","Date","TimeIn","TimeOut",
         "CallDur","StoreIDREF","StoreID","StoreName","VisitStatus","TotalCalls",
         "VisitLatitude","VisitLongitude","StoreLatitude","StoreLongitude",
         "STOREGPSUPDATED","VisitType","MonthPeriod"}

def load_file(b: bytes, name: str) -> pd.DataFrame:
    buf = io.BytesIO(b)
    return pd.read_excel(buf) if name.lower().endswith((".xlsx",".xls")) else pd.read_csv(buf)

def missing_cols(df, required):
    return [c for c in required if c not in df.columns]

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA DETECTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Schema:
    route_source:    str        = "planned_match"
    role_cols:       list       = field(default_factory=list)
    asm_col:         str        = ""
    supervisor_col:  str        = ""
    has_gps_flag:    bool       = False
    has_call_dur:    bool       = False

def detect(df: pd.DataFrame) -> Schema:
    s = Schema()
    cols = set(df.columns)
    if "VisitType"    in cols: s.route_source = "VisitType"
    elif "StoreIDREF" in cols: s.route_source = "StoreIDREF"
    for c in df.columns:
        cl = c.lower()
        if c.startswith("Role_") or c.startswith("role_"):
            s.role_cols.append(c)
            if "assistant" in cl or "asm" in cl: s.asm_col = c
            elif "supervisor" in cl:             s.supervisor_col = c
    s.has_gps_flag = "STOREGPSUPDATED" in cols
    s.has_call_dur = "CallDur" in cols
    return s

# ══════════════════════════════════════════════════════════════════════════════
# COERCION & GPS
# ══════════════════════════════════════════════════════════════════════════════

def prep(p: pd.DataFrame, a: pd.DataFrame):
    for df in (p, a):
        df["Date"]            = pd.to_datetime(df["Date"], errors="coerce")
        df["SRCode"]          = df["SRCode"].astype(str).str.strip()
        df["DistributorCode"] = df["DistributorCode"].astype(str).str.strip()
        df["StoreID"]         = df["StoreID"].astype(str).str.strip()
        df["MonthPeriod"]     = df["Date"].dt.to_period("M").astype(str)
    p["PlannedCalls"] = pd.to_numeric(p["PlannedCalls"], errors="coerce")
    for c in ["CallDur","TotalCalls","VisitLatitude","VisitLongitude",
              "StoreLatitude","StoreLongitude"]:
        if c in a.columns:
            a[c] = pd.to_numeric(a[c], errors="coerce")
    return p, a

def hav(la1, lo1, la2, lo2):
    R=6371000.0; r=np.radians
    a=(np.sin((r(la2)-r(la1))/2)**2
       +np.cos(r(la1))*np.cos(r(la2))*np.sin((r(lo2)-r(lo1))/2)**2)
    return R*2*np.arcsin(np.minimum(1.0,np.sqrt(a)))

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def calc_compliance(p, a):
    pg = p.groupby(["SRCode","DistributorCode","MonthPeriod"],as_index=False).agg(
        Planned=("StoreID","nunique"), PlanCalls=("PlannedCalls","sum"))
    done = a[a["VisitStatus"].astype(str).str.lower()=="completed"]
    ag = done.groupby(["SRCode","DistributorCode","MonthPeriod"],as_index=False).agg(
        Completed=("StoreID","nunique"))
    m = pg.merge(ag, on=["SRCode","DistributorCode","MonthPeriod"], how="outer")
    m[["Planned","Completed"]] = m[["Planned","Completed"]].fillna(0)
    m["CompliancePct"] = np.where(m["Planned"]>0, m["Completed"]/m["Planned"]*100, np.nan)
    m["Missed"]        = (m["Planned"]-m["Completed"]).clip(lower=0)
    return m.rename(columns={"Planned":"PlannedVisits","Completed":"CompletedVisits",
                              "Missed":"MissedVisits"})

def calc_route(p, a, schema: Schema):
    a2 = a.copy()
    if schema.route_source=="VisitType":
        a2["Off"] = a2["VisitType"].astype(str).str.lower().isin(
            ["offroute","off route","off_route"])
    elif schema.route_source=="StoreIDREF":
        a2["Off"] = a2["StoreID"] != a2["StoreIDREF"].astype(str).str.strip()
    else:
        keys = set(p["SRCode"]+"|"+p["Date"].dt.strftime("%Y-%m-%d")+"|"+p["StoreID"])
        a2["Off"] = ~(a2["SRCode"]+"|"+a2["Date"].dt.strftime("%Y-%m-%d")+"|"+a2["StoreID"]).isin(keys)
    g = a2.groupby(["SRCode","DistributorCode","MonthPeriod"],as_index=False).agg(
        TotalVisits=("StoreID","count"), OffRouteVisits=("Off","sum"))
    g["OffRoutePct"] = np.where(g["TotalVisits"]>0,
        g["OffRouteVisits"]/g["TotalVisits"]*100, np.nan)
    return g

def calc_gps(a, thresh):
    a2 = a.copy()
    a2["Dist"] = hav(a2["VisitLatitude"],a2["VisitLongitude"],
                     a2["StoreLatitude"],a2["StoreLongitude"])
    a2["Miss"] = a2["Dist"]>thresh
    g = a2.groupby(["SRCode","DistributorCode","MonthPeriod"],as_index=False).agg(
        TotalVisits=("StoreID","count"), GPSMismatches=("Miss","sum"),
        AvgDist=("Dist","mean"), MaxDist=("Dist","max"))
    g["GPSMismatchPct"] = np.where(g["TotalVisits"]>0,
        g["GPSMismatches"]/g["TotalVisits"]*100, np.nan)
    return g

def calc_miss(p, a):
    pg = p.groupby(["StoreID","DistributorCode"],as_index=False).agg(
        TimesPlanned=("Date","nunique"))
    done = a[a["VisitStatus"].astype(str).str.lower()=="completed"]
    ag = done.groupby(["StoreID","DistributorCode"],as_index=False).agg(
        TimesCompleted=("Date","nunique"))
    m = pg.merge(ag, on=["StoreID","DistributorCode"], how="left")
    m["TimesCompleted"] = m["TimesCompleted"].fillna(0)
    m["TimesMissed"]    = (m["TimesPlanned"]-m["TimesCompleted"]).clip(lower=0)
    m["MissRate"]       = np.where(m["TimesPlanned"]>0,
        m["TimesMissed"]/m["TimesPlanned"]*100, np.nan)
    return m.sort_values("TimesMissed", ascending=False)

def norm(s):
    lo,hi = s.min(),s.max()
    if hi==lo or s.dropna().empty: return pd.Series(50.0,index=s.index)
    return (s-lo)/(hi-lo)*100

def calc_scores(comp, route, gps):
    df = comp.merge(gps,on=["SRCode","DistributorCode","MonthPeriod"],how="outer")
    has_r = route is not None
    if has_r:
        df = df.merge(route,on=["SRCode","DistributorCode","MonthPeriod"],how="outer")
        df["RN"] = norm(100-df.get("OffRoutePct", pd.Series(0,index=df.index)))
    df["CN"] = norm(df["CompliancePct"])
    df["GN"] = norm(100-df["GPSMismatchPct"])
    df["XN"] = 100 - norm(df.groupby("SRCode")["CompliancePct"]
                          .transform(lambda x: x.std(ddof=0)).fillna(0))
    if has_r:
        df["BehaviourScore"] = (df["CN"]*0.35+df["RN"]*0.25+df["GN"]*0.25+df["XN"]*0.15).round(1)
    else:
        df["BehaviourScore"] = (df["CN"]*0.40+df["GN"]*0.35+df["XN"]*0.25).round(1)
    df["RiskCategory"] = pd.cut(df["BehaviourScore"],bins=[-np.inf,40,70,np.inf],
                                labels=["High Risk","Medium Risk","Low Risk"])
    keep = ["SRCode","DistributorCode","MonthPeriod","CompliancePct",
            "GPSMismatchPct","BehaviourScore","RiskCategory"]
    if has_r: keep.insert(4,"OffRoutePct")
    return df[[c for c in keep if c in df.columns]]

def calc_trend(scores):
    rows=[]
    for sr,g in scores.groupby("SRCode"):
        y=g.sort_values("MonthPeriod")["BehaviourScore"].dropna().values
        if len(y)<2: t,sl="N/A",np.nan
        else:
            sl=np.polyfit(np.arange(len(y)),y,1)[0]
            t="Improving" if sl>1 else ("Declining" if sl<-1 else "Stable")
        rows.append({"SRCode":sr,"Trend":t,"Slope":sl})
    return pd.DataFrame(rows)

def calc_anomalies(scores):
    out=[]
    for _,g in scores.groupby("SRCode"):
        g=g.sort_values("MonthPeriod").reset_index(drop=True)
        d=g["BehaviourScore"].diff()
        std=d.std(ddof=0); mean=d.mean()
        z=(d-mean)/std if std and not np.isnan(std) else pd.Series(0,index=d.index)
        g["Delta"]=d; g["ZScore"]=z; g["IsAnomaly"]=z<-1.5
        out.append(g)
    if not out: return pd.DataFrame()
    r=pd.concat(out,ignore_index=True)
    return r[r["IsAnomaly"]==True]

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT GENERATION  (concise — keeps memory low)
# ══════════════════════════════════════════════════════════════════════════════

def pf(v,d=1):
    if v is None or (isinstance(v,float) and np.isnan(float(v))): return "N/A"
    return f"{round(float(v),d)}%"

def nf(v,d=1):
    if v is None or (isinstance(v,float) and np.isnan(float(v))): return "N/A"
    return str(round(float(v),d))

def make_docs(p, a, scores, comp, route, gps, miss, schema, trend_df, anom_df) -> list[dict]:
    docs = []

    # --- Executive overview (single short doc)
    hr  = scores[scores["RiskCategory"]=="High Risk"]["SRCode"].unique().tolist()
    imp = trend_df[trend_df["Trend"]=="Improving"]["SRCode"].tolist() if not trend_df.empty else []
    dec = trend_df[trend_df["Trend"]=="Declining"]["SRCode"].tolist() if not trend_df.empty else []
    lines = [
        "EXECUTIVE OVERVIEW",
        f"SRs: {scores['SRCode'].nunique()} | Latest: {scores['MonthPeriod'].max()}",
        f"Avg Compliance: {pf(comp['CompliancePct'].mean())}",
        f"Avg GPS Mismatch: {pf(gps['GPSMismatchPct'].mean())}",
        f"High-Risk SRs ({len(hr)}): {', '.join(hr) or 'None'}",
        f"Improving: {', '.join(imp) or 'None'}",
        f"Declining: {', '.join(dec) or 'None'}",
    ]
    if route is not None:
        lines.append(f"Avg Off-Route: {pf(route['OffRoutePct'].mean())}")
    if anom_df is not None and not anom_df.empty:
        lines.append("Anomalies: "+", ".join(
            f"{r['SRCode']} ({r['MonthPeriod']},Δ={nf(r.get('Delta'))})"
            for _,r in anom_df.head(5).iterrows()))
    docs.append({"id":"exec","title":"Executive Overview","text":"\n".join(lines)})

    # --- One doc per SR
    for sr in sorted(scores["SRCode"].unique()):
        c=comp[comp["SRCode"]==sr].sort_values("MonthPeriod")
        g=gps[gps["SRCode"]==sr].sort_values("MonthPeriod")
        s=scores[scores["SRCode"]==sr].sort_values("MonthPeriod")
        r=route[route["SRCode"]==sr].sort_values("MonthPeriod") if route is not None else None
        dist=c["DistributorCode"].iloc[0] if not c.empty else "?"
        lsc=s["BehaviourScore"].iloc[-1] if not s.empty else None
        lrk=str(s["RiskCategory"].iloc[-1]) if not s.empty else "?"
        trd=trend_df[trend_df["SRCode"]==sr]["Trend"].values
        trd=trd[0] if len(trd) else "?"
        # role info
        ri=""
        if schema.role_cols:
            row=a[a["SRCode"]==sr]
            if not row.empty:
                ri=" | ".join(f"{rc.replace('Role_','')}: {row.iloc[0].get(rc,'')}"
                              for rc in schema.role_cols
                              if pd.notna(row.iloc[0].get(rc,"")))
        lines=[f"SR {sr} | Distributor:{dist} | Score:{nf(lsc)}/100 | Risk:{lrk} | Trend:{trd}"]
        if ri: lines.append(f"Hierarchy: {ri}")
        lines.append(f"Avg Compliance:{pf(c['CompliancePct'].mean())} | Avg GPS Mismatch:{pf(g['GPSMismatchPct'].mean())}")
        lines.append("Monthly compliance: "+
            " | ".join(f"{row['MonthPeriod']}:{pf(row.get('CompliancePct'))}"
                       f"(pl={int(row.get('PlannedVisits',0))},"
                       f"cp={int(row.get('CompletedVisits',0))},"
                       f"ms={int(row.get('MissedVisits',0))})"
                       for _,row in c.iterrows()))
        lines.append("GPS by month: "+
            " | ".join(f"{row['MonthPeriod']}:{pf(row.get('GPSMismatchPct'))}"
                       f"({int(row.get('GPSMismatches',0))}/{int(row.get('TotalVisits',0))},"
                       f"avg={nf(row.get('AvgDist'),0)}m)"
                       for _,row in g.iterrows()))
        if r is not None:
            lines.append("Route by month: "+
                " | ".join(f"{row['MonthPeriod']}:{pf(row.get('OffRoutePct'))}"
                           f"({int(row.get('OffRouteVisits',0))}/{int(row.get('TotalVisits',0))})"
                           for _,row in r.iterrows()))
        # observations
        obs=[]
        if len(c)>=2:
            drop=c["CompliancePct"].iloc[0]-c["CompliancePct"].iloc[-1]
            if drop>10: obs.append(f"Compliance fell {drop:.1f}pp — declining trend.")
        if len(g)>=2:
            rise=g["GPSMismatchPct"].iloc[-1]-g["GPSMismatchPct"].iloc[0]
            if rise>10: obs.append(f"GPS mismatch rose {rise:.1f}pp — possible spoofing.")
        if lrk=="High Risk": obs.append("HIGH RISK — immediate manager review needed.")
        if obs: lines.append("Observations: "+" | ".join(obs))
        docs.append({"id":f"sr_{sr}","title":f"SR {sr} Performance","text":"\n".join(lines)})

    # --- One doc per distributor
    for dist in sorted(comp["DistributorCode"].unique()):
        c=comp[comp["DistributorCode"]==dist]
        g=gps[gps["DistributorCode"]==dist]
        sr_rank=c.groupby("SRCode")["CompliancePct"].mean().sort_values()
        months=sorted(c["MonthPeriod"].unique())
        lines=[f"DISTRIBUTOR {dist}",
               f"SRs: {', '.join(sorted(c['SRCode'].unique()))}",
               f"Avg Compliance:{pf(c['CompliancePct'].mean())} | Avg GPS:{pf(g['GPSMismatchPct'].mean())}",
               f"Total: planned={int(c['PlannedVisits'].sum())} completed={int(c['CompletedVisits'].sum())} missed={int(c['MissedVisits'].sum())}",
               "SR ranking: "+", ".join(f"{sr}:{pf(v)}" for sr,v in sr_rank.items()),
               "Monthly: "+
               " | ".join(f"{m}: comp={pf(c[c['MonthPeriod']==m]['CompliancePct'].mean())}"
                          f" gps={pf(g[g['MonthPeriod']==m]['GPSMismatchPct'].mean())}"
                          for m in months)]
        docs.append({"id":f"dist_{dist}","title":f"Distributor {dist}","text":"\n".join(lines)})

    # --- One doc per month
    for month in sorted(comp["MonthPeriod"].unique()):
        c=comp[comp["MonthPeriod"]==month]
        g=gps[gps["MonthPeriod"]==month]
        s=scores[scores["MonthPeriod"]==month]
        sr_rank=c.groupby("SRCode")["CompliancePct"].mean().sort_values()
        hr=s[s["RiskCategory"]=="High Risk"]["SRCode"].tolist()
        lines=[f"MONTH {month}",
               f"Avg Compliance:{pf(c['CompliancePct'].mean())} | Avg GPS:{pf(g['GPSMismatchPct'].mean())}",
               f"High-Risk SRs: {', '.join(hr) or 'None'}",
               f"Planned:{int(c['PlannedVisits'].sum())} Completed:{int(c['CompletedVisits'].sum())} Missed:{int(c['MissedVisits'].sum())}",
               "SR ranking: "+", ".join(f"{sr}:{pf(v)}" for sr,v in sr_rank.items()),
               f"Top performers: {', '.join(sr_rank.tail(3).index.tolist())}",
               f"Needs attention: {', '.join(sr_rank.head(3).index.tolist())}"]
        if route is not None:
            rt2=route[route["MonthPeriod"]==month]
            lines.append(f"Avg Off-Route:{pf(rt2['OffRoutePct'].mean())}")
        docs.append({"id":f"month_{month}","title":f"Month {month}","text":"\n".join(lines)})

    # --- Role docs
    for rc in schema.role_cols:
        if rc not in a.columns: continue
        lbl=rc.replace("Role_","").replace("_"," ")
        for val in sorted(a[rc].dropna().astype(str).str.strip().unique()):
            if not val or val.lower() in ("nan","none",""): continue
            my_srs=a[a[rc].astype(str).str.strip()==val]["SRCode"].unique().tolist()
            c2=comp[comp["SRCode"].isin(my_srs)]
            s2=scores[scores["SRCode"].isin(my_srs)]
            hr2=s2[s2["RiskCategory"]=="High Risk"]["SRCode"].tolist()
            lines=[f"{lbl.upper()}: {val}",
                   f"SRs: {', '.join(my_srs)}",
                   f"Avg Compliance:{pf(c2['CompliancePct'].mean())} | High-Risk:{len(hr2)}/{len(my_srs)}",
                   f"Planned:{int(c2['PlannedVisits'].sum())} Completed:{int(c2['CompletedVisits'].sum())}"]
            if hr2: lines.append(f"High-Risk SRs needing attention: {', '.join(hr2)}")
            docs.append({"id":f"role_{rc}_{val}","title":f"{lbl} '{val}'",
                         "text":"\n".join(lines)})

    # --- Store miss doc
    lines=["STORE MISS ANALYSIS — stores planned but frequently missed"]
    for _,row in miss.head(15).iterrows():
        lines.append(f"Store {row['StoreID']} (Dist:{row['DistributorCode']}): "
                     f"planned {int(row.get('TimesPlanned',0))}x "
                     f"missed {int(row.get('TimesMissed',0))}x "
                     f"rate {pf(row.get('MissRate'))}")
    docs.append({"id":"store_miss","title":"Store Miss Analysis","text":"\n".join(lines)})

    # --- GPS doc
    latest_m = gps["MonthPeriod"].max()
    lg=gps[gps["MonthPeriod"]==latest_m].sort_values("GPSMismatchPct",ascending=False)
    lines=["GPS VERIFICATION INTELLIGENCE",
           f"Latest month: {latest_m}",
           "Worst GPS offenders: "+
           ", ".join(f"{row['SRCode']}:{pf(row.get('GPSMismatchPct'))}"
                     f"({int(row.get('GPSMismatches',0))}/{int(row.get('TotalVisits',0))},"
                     f"avg={nf(row.get('AvgDist'),0)}m)"
                     for _,row in lg.head(8).iterrows())]
    docs.append({"id":"gps","title":"GPS Intelligence","text":"\n".join(lines)})

    return docs

# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE  (pure numpy — no chromadb)
# ══════════════════════════════════════════════════════════════════════════════

class KWEmbed:
    """Hash-based keyword embedder. Zero dependencies beyond numpy."""
    DIM = 256  # smaller = less RAM
    def __init__(self): self._c:dict={}
    def _w(self,w):
        if w not in self._c:
            h=hashlib.md5(w.encode()).digest()
            seed=int.from_bytes(h[:4],"little")
            v=np.random.RandomState(seed).randn(self.DIM).astype(np.float32)
            n=np.linalg.norm(v); self._c[w]=v/n if n>0 else v
        return self._c[w]
    def embed(self,text:str)->np.ndarray:
        ws=text.lower().split()
        if not ws: return np.zeros(self.DIM,dtype=np.float32)
        v=np.stack([self._w(w) for w in ws[:200]]).mean(0)  # cap at 200 words
        n=np.linalg.norm(v); return v/n if n>0 else v

class VecStore:
    def __init__(self):
        self._E:Optional[np.ndarray]=None
        self._T:list=[]; self._M:list=[]
    def clear(self): self._E=None; self._T.clear(); self._M.clear(); gc.collect()
    def add(self,texts,embs,metas):
        e=embs.astype(np.float32)
        self._E=e if self._E is None else np.vstack([self._E,e])
        self._T.extend(texts); self._M.extend(metas)
    def query(self,qv,k):
        if self._E is None: return []
        q=qv.astype(np.float32); n=np.linalg.norm(q)
        if n>0: q/=n
        sims=self._E@q; k=min(k,len(self._T))
        idx=np.argpartition(sims,-k)[-k:]
        idx=idx[np.argsort(sims[idx])[::-1]]
        return [(self._T[i],self._M[i],float(sims[i])) for i in idx]
    @property
    def size(self): return len(self._T)

def chunk_text(text:str,sz=400,ov=50)->list:
    if not text or not text.strip(): return []
    if len(text)<=sz: return [text.strip()]
    out,start=[],0; n=len(text)
    while start<n:
        end=min(start+sz,n)
        if end<n:
            for sep in ["\n",". "," "]:
                i=text.rfind(sep,start,end)
                if i>start: end=i+len(sep); break
        c=text[start:end].strip()
        if c: out.append(c)
        start=end-ov
        if start>=n: break
    return out

def build_store(docs:list, emb:KWEmbed) -> VecStore:
    store=VecStore()
    for doc in docs:
        chunks=chunk_text(doc["text"])
        if not chunks: continue
        embs=np.stack([emb.embed(c) for c in chunks])
        metas=[{"id":doc["id"],"title":doc["title"]} for _ in chunks]
        store.add(chunks,embs,metas)
        gc.collect()
    return store

def retrieve(store:VecStore, emb:KWEmbed, query:str, k=5)->str:
    if store.size==0: return "(No knowledge base.)"
    results=store.query(emb.embed(query),k)
    if not results: return "(No results.)"
    return "\n\n---\n\n".join(
        f"[{m.get('title','?')} | score:{s:.2f}]\n{t}"
        for t,m,s in results)

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

SYS = """You are an expert AI Sales Performance Analyst.
Answer questions using ONLY the retrieved context provided before each question.
Always cite specific SR codes, months, and numbers from the context.
Explain root causes, prioritise by business impact, end with 1-3 recommendations.
If data is not in the context, say so — never invent information."""

def _retry(fn, n=3):
    for i in range(n+1):
        try: return fn()
        except Exception as e:
            msg=str(e)
            if ("429" in msg or "quota" in msg.lower()) and i<n:
                m=re.search(r"seconds:\s*(\d+)",msg)
                time.sleep(int(m.group(1))+2 if m else 30*(2**i))
            else: raise

class Chat:
    CHAIN=["gemini-2.5-flash","gemini-2.0-flash","gemini-2.0-flash-lite","gemini-1.5-flash-latest"]
    def __init__(self,store,emb,schema):
        key=get_key()
        if not key: raise RuntimeError("No Gemini API key found.")
        from google import genai
        from google.genai import types
        self._cl=genai.Client(api_key=key)
        self._ty=types; self._chat=None
        self._store=store; self._emb=emb
        note=("" if "VisitType" in (schema.route_source or "")
              else f"Route data derived from {schema.route_source}. ")
        role_note=(f"Role hierarchy: {', '.join(schema.role_cols)}. "
                   if schema.role_cols else "")
        self._sys=SYS+"\n\n"+note+role_note
        self.model=""
    def start(self):
        model=get_model()
        chain=[model]+[m for m in self.CHAIN if m!=model]
        for m in chain:
            try:
                self._chat=self._cl.chats.create(
                    model=m,
                    config=self._ty.GenerateContentConfig(
                        system_instruction=self._sys,
                        max_output_tokens=2048, temperature=0.3))
                self.model=m; return
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower(): continue
                raise
        raise RuntimeError("No Gemini model available.")
    def ask(self,q:str)->str:
        if not self._chat: raise RuntimeError("Not started.")
        ctx=retrieve(self._store,self._emb,q)
        prompt=f"CONTEXT\n{'='*7}\n{ctx}\n\n{'='*7}\nQUESTION: {q}\n\nAnswer from context only."
        return _retry(lambda:self._chat.send_message(prompt)).text

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 Sales Visit Intelligence")
    st.caption("RAG-powered AI analyst — Streamlit Cloud Edition")
    st.divider()
    pf_up=st.file_uploader("Planned Visits (csv/xlsx)", type=["csv","xlsx","xls"])
    af_up=st.file_uploader("Actual Visits (csv/xlsx)",  type=["csv","xlsx","xls"])
    gps_t=st.slider("GPS threshold (m)",20,1000,100,10)
    go   =st.button("Analyse Data", type="primary", use_container_width=True)
    st.divider()
    if st.session_state.store:
        st.success(f"✅ {st.session_state.store.size} chunks indexed")
    api_ok=bool(get_key())
    st.success(f"Gemini ✓ {st.session_state.model_used}" if api_ok and st.session_state.model_used
               else ("Gemini key detected ✓" if api_ok else "⚠️ Add GEMINI_API_KEY to secrets"))
    if st.session_state.result:
        if st.button("🔄 Reset", use_container_width=True):
            for k in ["result","store","chat","history","model_used"]:
                st.session_state[k]=[] if k=="history" else None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING  — every step wrapped in try/except, shows real errors
# ══════════════════════════════════════════════════════════════════════════════

if go:
    if not pf_up or not af_up:
        st.error("Please upload both files first."); st.stop()

    # --- load files
    try:
        p_raw=load_file(pf_up.getvalue(), pf_up.name)
        a_raw=load_file(af_up.getvalue(), af_up.name)
    except Exception:
        st.error(f"Could not read uploaded files:\n\n```\n{traceback.format_exc()}\n```"); st.stop()

    m1=missing_cols(p_raw, PLAN_COLS)
    m2=missing_cols(a_raw, ACTUAL_COLS)
    if m1: st.error(f"Planned Visits missing columns: {m1}"); st.stop()
    if m2: st.error(f"Actual Visits missing columns: {m2}");  st.stop()

    # --- analytics
    try:
        with st.spinner("Step 1/3 — Computing analytics…"):
            p,a  = prep(p_raw.copy(), a_raw.copy())
            schema = detect(a_raw)
            comp   = calc_compliance(p,a)
            route  = calc_route(p,a,schema)
            gps    = calc_gps(a,float(gps_t))
            miss   = calc_miss(p,a)
            scores = calc_scores(comp,route,gps)
            tdf    = calc_trend(scores)
            adf    = calc_anomalies(scores)
    except Exception:
        st.error(f"Error during analytics:\n\n```\n{traceback.format_exc()}\n```"); st.stop()

    # --- documents
    try:
        with st.spinner("Step 2/3 — Generating knowledge documents…"):
            docs=make_docs(p,a,scores,comp,route,gps,miss,schema,tdf,adf)
    except Exception:
        st.error(f"Error generating documents:\n\n```\n{traceback.format_exc()}\n```"); st.stop()

    # --- RAG index
    try:
        with st.spinner(f"Step 3/3 — Indexing {len(docs)} documents…"):
            emb=KWEmbed()
            store=build_store(docs,emb)
    except Exception:
        st.error(f"Error building index:\n\n```\n{traceback.format_exc()}\n```"); st.stop()

    st.session_state.result={"scores":scores,"comp":comp,"route":route,"gps":gps,
        "miss":miss,"schema":schema,"tdf":tdf,"adf":adf,"gps_t":gps_t,"actual":a}
    st.session_state.store=store
    st.session_state.emb  =emb
    st.session_state.chat =None
    st.session_state.history=[]
    st.session_state.model_used=""
    st.success(f"✅ {len(docs)} documents · {store.size} chunks ready"); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# LANDING
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.result:
    st.markdown("""
    <div style='text-align:center;padding:90px 20px'>
      <div style='font-size:72px'>🤖</div>
      <h2 style='color:#1a1a2e'>AI Sales Performance Analyst</h2>
      <p style='color:#666;max-width:520px;margin:0 auto;font-size:15px;line-height:1.6'>
        Upload your <strong>Planned Visits</strong> and <strong>Actual Visits</strong>
        files, then click <strong>Analyse Data</strong>.
      </p>
    </div>""", unsafe_allow_html=True); st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# INIT CHAT
# ══════════════════════════════════════════════════════════════════════════════

d=st.session_state.result
if api_ok and not st.session_state.chat and st.session_state.store:
    try:
        chat=Chat(st.session_state.store, st.session_state.emb, d["schema"])
        chat.start()
        st.session_state.chat=chat
        st.session_state.model_used=chat.model
    except Exception:
        st.warning(f"AI analyst error:\n\n```\n{traceback.format_exc()}\n```")

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════

scores=d["scores"]; comp=d["comp"]; gps=d["gps"]
route=d["route"]; miss=d["miss"]; schema=d["schema"]
tdf=d["tdf"]; adf=d["adf"]

def kpi(label,val,sub="",cls=""):
    return (f'<div class="kpi-card {cls}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>')
def pp(v):
    return f"{round(float(v),1)}%" if v is not None and not(isinstance(v,float)and np.isnan(v)) else "—"

ac=comp["CompliancePct"].mean(); ag=gps["GPSMismatchPct"].mean()
hr=int((scores["RiskCategory"]=="High Risk").sum())
tot=scores["SRCode"].nunique()
nd=int((tdf["Trend"]=="Declining").sum()) if not tdf.empty else 0
ni=int((tdf["Trend"]=="Improving").sum()) if not tdf.empty else 0

cards=[
    kpi("Compliance",pp(ac),"planned→completed",
        "green" if ac>=80 else "amber" if ac>=60 else "red"),
    kpi("GPS Mismatches",pp(ag),f">{d['gps_t']}m",
        "red" if ag>=20 else "amber" if ag>=10 else "green"),
    kpi("High-Risk SRs",str(hr),f"of {tot}","red" if hr else "green"),
    kpi("Declining",str(nd),"SRs","red" if nd else "green"),
    kpi("Improving",str(ni),"SRs","green" if ni else ""),
    kpi("Latest",scores["MonthPeriod"].max(),"period"),
]
if route is not None:
    ao=route["OffRoutePct"].mean()
    cards.insert(1,kpi("Off-Route",pp(ao),schema.route_source,
                       "red" if ao>=30 else "amber" if ao>=15 else "green"))

st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>',unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

left,right=st.columns([1,2.2],gap="large")

with left:
    with st.expander("📋 Seller Risk Snapshot",expanded=True):
        lm=scores["MonthPeriod"].max()
        snap=(scores[scores["MonthPeriod"]==lm]
              .sort_values("BehaviourScore")
              .merge(tdf[["SRCode","Trend"]],on="SRCode",how="left"))
        RI={"High Risk":"🔴","Medium Risk":"🟡","Low Risk":"🟢"}
        TR={"Improving":"📈","Declining":"📉","Stable":"➡️"}
        for _,r in snap.iterrows():
            ri=RI.get(str(r["RiskCategory"]),"⚪"); ti=TR.get(str(r.get("Trend","")),"❓")
            sc=f"{r['BehaviourScore']:.0f}" if pd.notna(r["BehaviourScore"]) else "—"
            st.markdown(f'<div class="sr-row"><span>{ri} <strong>{r["SRCode"]}</strong></span>'
                        f'<span style="color:#888">Score <strong style="color:#1a1a2e">{sc}</strong> {ti}</span></div>',
                        unsafe_allow_html=True)

    for rc in schema.role_cols:
        if rc in d["actual"].columns:
            lbl=rc.replace("Role_","").replace("_"," ")
            with st.expander(f"👥 {lbl}",expanded=False):
                role_map=d["actual"][["SRCode",rc]].drop_duplicates("SRCode")
                cg=comp.merge(role_map,on="SRCode",how="left")
                gg=gps.merge(role_map,on="SRCode",how="left")
                c2=cg.groupby(rc,as_index=False).agg(
                    AvgComp=("CompliancePct","mean"),SRs=("SRCode","nunique"))
                g2=gg.groupby(rc,as_index=False).agg(AvgGPS=("GPSMismatchPct","mean"))
                st.dataframe(c2.merge(g2,on=rc).rename(columns={
                    rc:lbl,"AvgComp":"Compliance %","AvgGPS":"GPS Mismatch %"}),
                    hide_index=True,use_container_width=True)

    with st.expander("📅 Monthly Trend",expanded=False):
        ms=d["comp"].groupby("MonthPeriod",as_index=False).agg(
            Comp=("CompliancePct","mean"))
        gs=d["gps"].groupby("MonthPeriod",as_index=False).agg(
            GPS=("GPSMismatchPct","mean"))
        ms2=ms.merge(gs,on="MonthPeriod")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=ms2["MonthPeriod"],y=ms2["Comp"],
            name="Compliance %",mode="lines+markers",line=dict(color="#4f8ef7",width=2)))
        fig.add_trace(go.Scatter(x=ms2["MonthPeriod"],y=ms2["GPS"],
            name="GPS Mismatch %",mode="lines+markers",line=dict(color="#e53935",width=2)))
        if route is not None:
            rs=route.groupby("MonthPeriod",as_index=False).agg(OR=("OffRoutePct","mean"))
            ms2=ms2.merge(rs,on="MonthPeriod",how="left")
            fig.add_trace(go.Scatter(x=ms2["MonthPeriod"],y=ms2["OR"],
                name="Off-Route %",mode="lines+markers",line=dict(color="#fb8c00",width=2)))
        fig.update_layout(height=210,margin=dict(l=0,r=0,t=4,b=0),
            legend=dict(orientation="h",y=-0.45,font=dict(size=10)),
            plot_bgcolor="#fff",paper_bgcolor="#fff")
        st.plotly_chart(fig,use_container_width=True)

    with st.expander("🏪 Top Missed Stores",expanded=False):
        m2=miss.head(8)[["StoreID","TimesPlanned","TimesMissed","MissRate"]].copy()
        m2["MissRate"]=m2["MissRate"].map(lambda x:f"{x:.0f}%" if pd.notna(x) else "—")
        st.dataframe(m2,hide_index=True,use_container_width=True)

    if adf is not None and not adf.empty:
        with st.expander(f"⚠️ Anomalies ({len(adf)})",expanded=False):
            st.dataframe(adf[["SRCode","MonthPeriod","BehaviourScore","Delta"]]
                .rename(columns={"Delta":"Score Drop"}),hide_index=True,use_container_width=True)

with right:
    st.markdown('<div style="font-size:11px;font-weight:700;color:#888;'
                'text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">'
                '💬 AI Sales Analyst (RAG-powered)</div>',unsafe_allow_html=True)

    if not api_ok:
        st.info("Add GEMINI_API_KEY to Streamlit Cloud secrets to activate the AI analyst."); st.stop()
    if not st.session_state.chat:
        st.warning("AI analyst not connected. Check the error message above."); st.stop()

    chat_obj=st.session_state.chat
    box=st.container(height=480)
    with box:
        if not st.session_state.history:
            st.markdown("""<div style='text-align:center;padding:40px 20px;color:#888'>
              <div style='font-size:40px'>🔍</div>
              <div style='font-size:15px;font-weight:600;color:#444;margin:8px 0 4px'>
                Analyst ready. Click a question or type below.</div>
            </div>""",unsafe_allow_html=True)
        else:
            for role,msg in st.session_state.history:
                with st.chat_message(role): st.markdown(msg)

    user_turns=[r for r,_ in st.session_state.history if r=="user"]
    if not user_turns:
        SUGG=[
            ("🔎 Full analysis","Comprehensive analysis: top issues, trends, anomalies, action plan."),
            ("🚨 Needs attention","Which SRs need immediate managerial attention and why?"),
            ("📉 Compliance drop","Which SRs have declining compliance? Root causes and months."),
            ("🛰️ GPS check","Signs of GPS manipulation or fake check-ins? Evidence please."),
            ("🏆 Best performers","Which SRs are performing well or improving? Specific metrics."),
            ("🏪 Skipped stores","Stores repeatedly planned but missed. What to do?"),
        ]
        if schema.supervisor_col:
            SUGG.append(("👥 Supervisors","Compare supervisor teams. Best and worst performing?"))
        if schema.asm_col:
            SUGG.append(("🏢 ASMs","Which ASMs oversee best and worst performing teams?"))
        cols=st.columns(2)
        for i,(label,q) in enumerate(SUGG):
            if cols[i%2].button(label,use_container_width=True,key=f"s{i}"):
                st.session_state._pq=q; st.rerun()

    user_q=st.chat_input("Ask anything about your sales data…")
    pending=getattr(st.session_state,"_pq",None)
    question=pending or user_q
    if question and chat_obj:
        if pending and hasattr(st.session_state,"_pq"): del st.session_state._pq
        st.session_state.history.append(("user",question))
        with st.spinner("Searching → generating answer…"):
            try: answer=chat_obj.ask(question)
            except Exception as e:
                err=str(e)
                if "429" in err or "quota" in err.lower():
                    answer="⏳ **Rate limit.** Wait ~30s and retry."
                else: answer=f"⚠️ Error: {traceback.format_exc()}"
        st.session_state.history.append(("assistant",answer))
        st.rerun()
