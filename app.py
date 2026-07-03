"""
Sales Visit Intelligence — Single-File Deployment
===================================================
Everything in one file: data processing, scoring, document generation,
RAG pipeline, Gemini chatbot, and Streamlit UI.
No utils/ folder needed. No module import errors possible.

To deploy: upload only this file + requirements.txt to Streamlit Cloud.
"""

import gc
import io
import os
import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  put your Gemini API key here
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = "your-gemini-api-key-here"   # ← paste your key
GEMINI_MODEL   = "gemini-2.5-flash"

def _get_api_key() -> Optional[str]:
    if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
        return GEMINI_API_KEY
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PLANNED_REQUIRED = ["DistributorCode", "SRCode", "Date", "StoreID", "PlannedCalls"]
ACTUAL_REQUIRED  = ["DistributorCode", "SRCode", "Date", "StoreID",
                    "VisitStatus", "VisitLatitude", "VisitLongitude",
                    "StoreLatitude", "StoreLongitude"]

_KNOWN_COLS = {
    "DistributorCode","DistributorName","SRCode","Date","TimeIn","TimeOut",
    "CallDur","StoreIDREF","StoreID","StoreName","VisitStatus","TotalCalls",
    "VisitLatitude","VisitLongitude","StoreLatitude","StoreLongitude",
    "STOREGPSUPDATED","VisitType","MonthPeriod","GPSDistanceMeters",
    "GPSMismatch","WasPlanned","IsOffRoute",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def read_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    return pd.read_csv(buf)

def validate(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA DETECTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Schema:
    route_source:       Optional[str]  = None
    role_cols:          list[str]      = field(default_factory=list)
    asm_col:            Optional[str]  = None
    supervisor_col:     Optional[str]  = None
    has_gps_updated:    bool           = False
    has_call_duration:  bool           = False
    product_cols:       list[str]      = field(default_factory=list)

def detect_schema(df: pd.DataFrame) -> Schema:
    cols = set(df.columns)
    s    = Schema()
    if "VisitType"   in cols: s.route_source = "VisitType"
    elif "StoreIDREF" in cols: s.route_source = "StoreIDREF"
    else:                      s.route_source = "planned_match"
    for col in df.columns:
        cl = col.lower()
        if col.startswith("Role_") or col.startswith("role_"):
            s.role_cols.append(col)
            if "assistant" in cl or "asm" in cl: s.asm_col = col
            elif "supervisor" in cl:             s.supervisor_col = col
    s.has_gps_updated   = "STOREGPSUPDATED" in cols
    s.has_call_duration = "CallDur" in cols
    s.product_cols      = [c for c in df.columns
                           if c not in _KNOWN_COLS and c not in s.role_cols
                           and pd.api.types.is_numeric_dtype(df[c])]
    return s

# ══════════════════════════════════════════════════════════════════════════════
# COERCION & GPS
# ══════════════════════════════════════════════════════════════════════════════

def coerce(planned: pd.DataFrame, actual: pd.DataFrame):
    for df in (planned, actual):
        df["Date"]            = pd.to_datetime(df["Date"], errors="coerce")
        df["SRCode"]          = df["SRCode"].astype(str).str.strip()
        df["DistributorCode"] = df["DistributorCode"].astype(str).str.strip()
        df["StoreID"]         = df["StoreID"].astype(str).str.strip()
        df["MonthPeriod"]     = df["Date"].dt.to_period("M").astype(str)
    planned["PlannedCalls"] = pd.to_numeric(planned["PlannedCalls"], errors="coerce")
    for col in ["CallDur","TotalCalls","VisitLatitude","VisitLongitude",
                "StoreLatitude","StoreLongitude"]:
        if col in actual.columns:
            actual[col] = pd.to_numeric(actual[col], errors="coerce")
    return planned, actual

def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6_371_000.0
    rl = np.radians
    a = (np.sin((rl(lat2)-rl(lat1))/2)**2
         + np.cos(rl(lat1))*np.cos(rl(lat2))*np.sin((rl(lon2)-rl(lon1))/2)**2)
    return R * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))

def add_gps_dist(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["GPSDistanceMeters"] = haversine(
        out["VisitLatitude"], out["VisitLongitude"],
        out["StoreLatitude"], out["StoreLongitude"])
    return out

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def compliance_table(planned, actual):
    pg = planned.groupby(["SRCode","DistributorCode","MonthPeriod"], as_index=False).agg(
        PlannedVisits=("StoreID","nunique"), PlannedCallsSum=("PlannedCalls","sum"))
    done = actual[actual["VisitStatus"].astype(str).str.lower() == "completed"]
    ag = done.groupby(["SRCode","DistributorCode","MonthPeriod"], as_index=False).agg(
        CompletedVisits=("StoreID","nunique"))
    m = pg.merge(ag, on=["SRCode","DistributorCode","MonthPeriod"], how="outer")
    m[["PlannedVisits","CompletedVisits"]] = m[["PlannedVisits","CompletedVisits"]].fillna(0)
    m["CompliancePct"]  = np.where(m["PlannedVisits"]>0,
        (m["CompletedVisits"]/m["PlannedVisits"])*100, np.nan)
    m["MissedVisits"]   = (m["PlannedVisits"]-m["CompletedVisits"]).clip(lower=0)
    return m.sort_values(["SRCode","MonthPeriod"])

def route_table(planned, actual, schema: Schema) -> Optional[pd.DataFrame]:
    a = actual.copy()
    if schema.route_source == "VisitType":
        vt = a["VisitType"].astype(str).str.lower()
        a["IsOffRoute"] = vt.isin(["offroute","off route","off_route"])
    elif schema.route_source == "StoreIDREF":
        a["StoreIDREF"] = a["StoreIDREF"].astype(str).str.strip()
        a["IsOffRoute"] = a["StoreID"] != a["StoreIDREF"]
    else:
        pairs = set(planned["SRCode"]+"|"+planned["Date"].dt.strftime("%Y-%m-%d")+"|"+planned["StoreID"])
        a["IsOffRoute"] = ~(a["SRCode"]+"|"+a["Date"].dt.strftime("%Y-%m-%d")+"|"+a["StoreID"]).isin(pairs)
    g = a.groupby(["SRCode","DistributorCode","MonthPeriod"], as_index=False).agg(
        TotalVisits=("StoreID","count"), OffRouteVisits=("IsOffRoute","sum"))
    g["OffRoutePct"] = np.where(g["TotalVisits"]>0,
        (g["OffRouteVisits"]/g["TotalVisits"])*100, np.nan)
    return g.sort_values(["SRCode","MonthPeriod"])

def gps_table(actual, threshold: float):
    a = add_gps_dist(actual)
    a["GPSMismatch"] = a["GPSDistanceMeters"] > threshold
    g = a.groupby(["SRCode","DistributorCode","MonthPeriod"], as_index=False).agg(
        TotalVisits=("StoreID","count"), GPSMismatches=("GPSMismatch","sum"),
        AvgGPSDistance=("GPSDistanceMeters","mean"),
        MaxGPSDistance=("GPSDistanceMeters","max"))
    g["GPSMismatchPct"] = np.where(g["TotalVisits"]>0,
        (g["GPSMismatches"]/g["TotalVisits"])*100, np.nan)
    return g

def store_miss_table(planned, actual):
    pg = planned.groupby(["StoreID","DistributorCode"], as_index=False).agg(
        TimesPlanned=("Date","nunique"))
    done = actual[actual["VisitStatus"].astype(str).str.lower()=="completed"]
    ag = done.groupby(["StoreID","DistributorCode"], as_index=False).agg(
        TimesCompleted=("Date","nunique"))
    m = pg.merge(ag, on=["StoreID","DistributorCode"], how="left")
    m["TimesCompleted"] = m["TimesCompleted"].fillna(0)
    m["TimesMissed"]    = (m["TimesPlanned"]-m["TimesCompleted"]).clip(lower=0)
    m["MissRate"]       = np.where(m["TimesPlanned"]>0,
        (m["TimesMissed"]/m["TimesPlanned"])*100, np.nan)
    return m.sort_values("TimesMissed", ascending=False)

def monthly_summary(comp, route, gps):
    c = comp.groupby("MonthPeriod",as_index=False).agg(
        AvgCompliancePct=("CompliancePct","mean"),
        TotalPlanned=("PlannedVisits","sum"),
        TotalCompleted=("CompletedVisits","sum"))
    g = gps.groupby("MonthPeriod",as_index=False).agg(
        AvgGPSMismatchPct=("GPSMismatchPct","mean"))
    out = c.merge(g, on="MonthPeriod", how="outer")
    if route is not None:
        r = route.groupby("MonthPeriod",as_index=False).agg(
            AvgOffRoutePct=("OffRoutePct","mean"))
        out = out.merge(r, on="MonthPeriod", how="outer")
    return out.sort_values("MonthPeriod")

def distributor_summary(comp, route, gps):
    c = comp.groupby("DistributorCode",as_index=False).agg(
        AvgCompliancePct=("CompliancePct","mean"))
    g = gps.groupby("DistributorCode",as_index=False).agg(
        AvgGPSMismatchPct=("GPSMismatchPct","mean"))
    out = c.merge(g, on="DistributorCode", how="outer")
    if route is not None:
        r = route.groupby("DistributorCode",as_index=False).agg(
            AvgOffRoutePct=("OffRoutePct","mean"))
        out = out.merge(r, on="DistributorCode", how="outer")
    return out.sort_values("AvgCompliancePct")

def role_analytics(actual, comp, gps, schema: Schema) -> dict:
    res = {}
    if not schema.role_cols:
        return res
    role_map = actual[["SRCode"]+schema.role_cols].drop_duplicates("SRCode")
    for rc in schema.role_cols:
        cg = comp.merge(role_map[["SRCode",rc]], on="SRCode", how="left")
        gg = gps.merge(role_map[["SRCode",rc]], on="SRCode", how="left")
        c2 = cg.groupby(rc,as_index=False).agg(
            AvgCompliancePct=("CompliancePct","mean"),
            TotalPlanned=("PlannedVisits","sum"),
            TotalCompleted=("CompletedVisits","sum"),
            SRCount=("SRCode","nunique"))
        g2 = gg.groupby(rc,as_index=False).agg(
            AvgGPSMismatchPct=("GPSMismatchPct","mean"))
        res[rc] = c2.merge(g2, on=rc, how="outer").sort_values("AvgCompliancePct")
    return res

# ══════════════════════════════════════════════════════════════════════════════
# BEHAVIOUR SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _norm(s: pd.Series) -> pd.Series:
    lo,hi = s.min(),s.max()
    if hi==lo or s.dropna().empty:
        return pd.Series(50.0, index=s.index)
    return (s-lo)/(hi-lo)*100

def behaviour_scores(comp, route, gps) -> pd.DataFrame:
    df = comp.merge(gps, on=["SRCode","DistributorCode","MonthPeriod"],
                    how="outer", suffixes=("","_g"))
    has_route = route is not None
    if has_route:
        df = df.merge(route, on=["SRCode","DistributorCode","MonthPeriod"],
                      how="outer", suffixes=("","_r"))
        df["RouteNorm"] = _norm(100-df["OffRoutePct"])
    df["CN"] = _norm(df["CompliancePct"])
    df["GN"] = _norm(100-df["GPSMismatchPct"])
    vol      = df.groupby("SRCode")["CompliancePct"].transform(
        lambda x: x.std(ddof=0)).fillna(0)
    df["XN"] = 100-_norm(vol)
    if has_route:
        df["BehaviourScore"] = (df["CN"]*0.35+df["RouteNorm"]*0.25
                                +df["GN"]*0.25+df["XN"]*0.15).round(1)
    else:
        df["BehaviourScore"] = (df["CN"]*0.40+df["GN"]*0.35+df["XN"]*0.25).round(1)
    df["RiskCategory"] = pd.cut(df["BehaviourScore"],
        bins=[-np.inf,40,70,np.inf], labels=["High Risk","Medium Risk","Low Risk"])
    keep = ["SRCode","DistributorCode","MonthPeriod","CompliancePct",
            "GPSMismatchPct","BehaviourScore","RiskCategory"]
    if has_route: keep.insert(4,"OffRoutePct")
    return df[[c for c in keep if c in df.columns]].sort_values(["SRCode","MonthPeriod"])

def trend_direction(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sr, g in scores.groupby("SRCode"):
        y = g.sort_values("MonthPeriod")["BehaviourScore"].dropna().values
        if len(y)<2: trend,slope="Insufficient Data",np.nan
        else:
            slope=np.polyfit(np.arange(len(y)),y,1)[0]
            trend="Improving" if slope>1 else ("Declining" if slope<-1 else "Stable")
        rows.append({"SRCode":sr,"Slope":slope,"Trend":trend})
    return pd.DataFrame(rows)

def anomalies(scores: pd.DataFrame, z=1.5) -> pd.DataFrame:
    out=[]
    for _,g in scores.groupby("SRCode"):
        g=g.sort_values("MonthPeriod").reset_index(drop=True)
        d=g["BehaviourScore"].diff(); std=d.std(ddof=0); mean=d.mean()
        z_=(d-mean)/std if std and not np.isnan(std) else pd.Series(0,index=d.index)
        g["Delta"]=d; g["ZScore"]=z_; g["IsAnomaly"]=z_<-z
        out.append(g)
    if not out: return pd.DataFrame()
    r=pd.concat(out,ignore_index=True)
    return r[r["IsAnomaly"]==True]

def exec_stats(scores,comp,gps,route,schema:Schema) -> dict:
    latest=sorted(scores["MonthPeriod"].dropna().unique())[-1] if not scores.empty else None
    d={"latest_month":latest,
       "avg_compliance":round(comp["CompliancePct"].mean(),1) if not comp.empty else None,
       "avg_gps_mismatch_pct":round(gps["GPSMismatchPct"].mean(),1) if not gps.empty else None,
       "high_risk_count":int((scores["RiskCategory"]=="High Risk").sum()) if not scores.empty else 0,
       "total_srs":scores["SRCode"].nunique() if not scores.empty else 0,
       "has_route_data":route is not None,
       "route_source":schema.route_source,
       "role_cols":schema.role_cols}
    d["avg_offroute_pct"]=round(route["OffRoutePct"].mean(),1) if route is not None and not route.empty else None
    return d

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _p(v,d=1):
    if v is None or (isinstance(v,float) and np.isnan(float(v))): return "N/A"
    return f"{round(float(v),d)}%"
def _n(v,d=1):
    if v is None or (isinstance(v,float) and np.isnan(float(v))): return "N/A"
    return str(round(float(v),d))

def make_documents(planned, actual, scores, comp, route, gps,
                   miss, dist_sum, month_sum, role_ana,
                   trend_df, anom_df, stats, schema:Schema) -> list[dict]:
    docs = []

    # ── SR documents
    for sr in sorted(scores["SRCode"].unique()):
        c=comp[comp["SRCode"]==sr].sort_values("MonthPeriod")
        g=gps[gps["SRCode"]==sr].sort_values("MonthPeriod")
        s=scores[scores["SRCode"]==sr].sort_values("MonthPeriod")
        r=route[route["SRCode"]==sr].sort_values("MonthPeriod") if route is not None else None
        dist=c["DistributorCode"].iloc[0] if not c.empty else "?"
        months=sorted(s["MonthPeriod"].unique())
        latest_s=s["BehaviourScore"].iloc[-1] if not s.empty else None
        latest_r=s["RiskCategory"].iloc[-1] if not s.empty else "?"
        role_info=""
        if schema.role_cols:
            row=actual[actual["SRCode"]==sr]
            if not row.empty:
                parts=[f"{rc.replace('Role_','')}: {row.iloc[0].get(rc,'')}"
                       for rc in schema.role_cols if pd.notna(row.iloc[0].get(rc,""))]
                role_info=" | ".join(parts)
        lines=[f"SR PERFORMANCE SUMMARY","="*22,
               f"SR: {sr} | Distributor: {dist}"]
        if role_info: lines.append(f"Hierarchy: {role_info}")
        lines+=[f"Period: {months[0]} to {months[-1]}" if months else "",
               f"Behaviour Score: {_n(latest_s)}/100 | Risk: {latest_r}",
               f"Avg Compliance: {_p(c['CompliancePct'].mean())} | Avg GPS Mismatch: {_p(g['GPSMismatchPct'].mean())}",
               "","MONTHLY COMPLIANCE"]
        for _,row in c.iterrows():
            lines.append(f"  {row['MonthPeriod']}: {_p(row.get('CompliancePct'))} "
                         f"(planned={int(row.get('PlannedVisits',0))}, "
                         f"completed={int(row.get('CompletedVisits',0))}, "
                         f"missed={int(row.get('MissedVisits',0))})")
        lines+=["","GPS VERIFICATION"]
        for _,row in g.iterrows():
            lines.append(f"  {row['MonthPeriod']}: {_p(row.get('GPSMismatchPct'))} mismatch "
                         f"({int(row.get('GPSMismatches',0))}/{int(row.get('TotalVisits',0))}, "
                         f"avg dist={_n(row.get('AvgGPSDistance'),0)}m)")
        if r is not None:
            lines+=["",f"ROUTE ADHERENCE (source:{schema.route_source})"]
            for _,row in r.iterrows():
                lines.append(f"  {row['MonthPeriod']}: {_p(row.get('OffRoutePct'))} off-route "
                             f"({int(row.get('OffRouteVisits',0))}/{int(row.get('TotalVisits',0))})")
        lines+=["","SCORE HISTORY"]
        for _,row in s.iterrows():
            lines.append(f"  {row['MonthPeriod']}: score={_n(row.get('BehaviourScore'))} ({row.get('RiskCategory','')})")
        # observations
        obs=[]
        if len(c)>=2:
            drop=c["CompliancePct"].iloc[0]-c["CompliancePct"].iloc[-1]
            if drop>10: obs.append(f"Compliance fell {drop:.1f}pp — significant decline.")
        if len(g)>=2:
            rise=g["GPSMismatchPct"].iloc[-1]-g["GPSMismatchPct"].iloc[0]
            if rise>10: obs.append(f"GPS mismatch rose {rise:.1f}pp — possible spoofing, needs investigation.")
        if not s.empty and str(s["RiskCategory"].iloc[-1])=="High Risk":
            obs.append("HIGH RISK — immediate manager review recommended.")
        if obs: lines+=["","OBSERVATIONS"]+[f"  • {o}" for o in obs]
        docs.append({"id":f"sr_{sr}","title":f"SR {sr} Summary","text":"\n".join(lines)})

    # ── Distributor documents
    for dist in sorted(comp["DistributorCode"].unique()):
        c=comp[comp["DistributorCode"]==dist]
        g=gps[gps["DistributorCode"]==dist]
        months=sorted(c["MonthPeriod"].unique())
        sr_rank=c.groupby("SRCode")["CompliancePct"].mean().sort_values()
        lines=[f"DISTRIBUTOR SUMMARY","="*18,
               f"Distributor: {dist}",
               f"SRs: {', '.join(sorted(c['SRCode'].unique()))}",
               f"Avg Compliance: {_p(c['CompliancePct'].mean())} | Avg GPS Mismatch: {_p(g['GPSMismatchPct'].mean())}",
               f"Total Planned: {int(c['PlannedVisits'].sum())} | Completed: {int(c['CompletedVisits'].sum())} | Missed: {int(c['MissedVisits'].sum())}",
               "","SR COMPLIANCE RANKING"]
        for sr_c,val in sr_rank.items(): lines.append(f"  {sr_c}: {_p(val)}")
        lines+=["","MONTHLY TREND"]
        for m in months:
            mc=c[c["MonthPeriod"]==m]["CompliancePct"].mean()
            mg=g[g["MonthPeriod"]==m]["GPSMismatchPct"].mean()
            lines.append(f"  {m}: compliance={_p(mc)} gps-mismatch={_p(mg)}")
        docs.append({"id":f"dist_{dist}","title":f"Distributor {dist} Summary","text":"\n".join(lines)})

    # ── Monthly documents
    months_all=sorted(comp["MonthPeriod"].unique())
    for i,month in enumerate(months_all):
        c=comp[comp["MonthPeriod"]==month]
        g=gps[gps["MonthPeriod"]==month]
        s=scores[scores["MonthPeriod"]==month]
        sr_rank=c.groupby("SRCode")["CompliancePct"].mean().sort_values()
        hr=int((s["RiskCategory"]=="High Risk").sum())
        prev=""
        if i>0:
            pc=comp[comp["MonthPeriod"]==months_all[i-1]]["CompliancePct"].mean()
            diff=c["CompliancePct"].mean()-pc
            prev=f"vs {months_all[i-1]}: compliance {'improved' if diff>0 else 'declined'} by {abs(diff):.1f}pp."
        lines=[f"MONTHLY SUMMARY: {month}","="*20,
               f"Avg Compliance: {_p(c['CompliancePct'].mean())} | Avg GPS Mismatch: {_p(g['GPSMismatchPct'].mean())}",
               f"High-Risk SRs: {hr} | Planned: {int(c['PlannedVisits'].sum())} | Completed: {int(c['CompletedVisits'].sum())}"]
        if prev: lines.append(f"MOM: {prev}")
        lines+=["","SR RANKING"]
        for sr_c,val in sr_rank.items():
            rk=s[s["SRCode"]==sr_c]["RiskCategory"].values
            lines.append(f"  {sr_c}: {_p(val)}"+(f" [{rk[0]}]" if len(rk) else ""))
        lines+=[f"","TOP: {', '.join(sr_rank.tail(3).index.tolist())}",
                f"NEEDS ATTENTION: {', '.join(sr_rank.head(3).index.tolist())}"]
        docs.append({"id":f"month_{month}","title":f"Monthly Summary {month}","text":"\n".join(lines)})

    # ── Role documents
    for rc,df_r in role_ana.items():
        label=rc.replace("Role_","").replace("_"," ")
        for _,row in df_r.iterrows():
            val=str(row.get(rc,""))
            if not val or val.lower() in ("nan","none",""): continue
            my_srs=actual[actual[rc].astype(str).str.strip()==val]["SRCode"].unique().tolist()
            c2=comp[comp["SRCode"].isin(my_srs)]
            s2=scores[scores["SRCode"].isin(my_srs)]
            hr=int((s2["RiskCategory"]=="High Risk").sum())
            lines=[f"{label.upper()} SUMMARY","="*(len(label)+8),
                   f"{label}: {val}",
                   f"SRs: {', '.join(my_srs)}",
                   f"Avg Compliance: {_p(row.get('AvgCompliancePct'))} | Avg GPS Mismatch: {_p(row.get('AvgGPSMismatchPct'))}",
                   f"High-Risk SRs: {hr}/{len(my_srs)}",
                   f"Planned: {int(c2['PlannedVisits'].sum())} | Completed: {int(c2['CompletedVisits'].sum())}"]
            if hr>0:
                hr_list=s2[s2["RiskCategory"]=="High Risk"]["SRCode"].unique().tolist()
                lines+=["",f"HIGH RISK SRs under {label} {val}: {', '.join(hr_list)} — manager review needed."]
            docs.append({"id":f"role_{rc}_{val}","title":f"{label} '{val}' Summary","text":"\n".join(lines)})

    # ── Store miss document
    lines=["STORE MISS ANALYSIS","="*18,"Stores repeatedly planned but missed.",""]
    for _,row in miss.head(20).iterrows():
        lines.append(f"  Store {row['StoreID']} (Dist:{row['DistributorCode']}): "
                     f"planned {int(row.get('TimesPlanned',0))}x | "
                     f"missed {int(row.get('TimesMissed',0))}x | "
                     f"miss rate {_p(row.get('MissRate'))}")
    docs.append({"id":"store_miss","title":"Store Miss Analysis","text":"\n".join(lines)})

    # ── GPS document
    a2=add_gps_dist(actual)
    a2["GPSMismatch"]=a2["GPSDistanceMeters"]>100
    susp=a2[a2["GPSMismatch"]].sort_values("GPSDistanceMeters",ascending=False)
    latest_m=gps["MonthPeriod"].max()
    lg=gps[gps["MonthPeriod"]==latest_m].sort_values("GPSMismatchPct",ascending=False)
    lines=["GPS VERIFICATION INTELLIGENCE","="*28,
           "","WORST OFFENDERS (latest month)"]
    for _,row in lg.head(8).iterrows():
        lines.append(f"  {row['SRCode']}: {_p(row.get('GPSMismatchPct'))} "
                     f"({int(row.get('GPSMismatches',0))}/{int(row.get('TotalVisits',0))}, avg={_n(row.get('AvgGPSDistance'),0)}m)")
    lines+=["","TOP OUTLIER VISITS"]
    for _,row in susp.head(8).iterrows():
        lines.append(f"  SR={row['SRCode']} Store={row['StoreID']}: dist={_n(row['GPSDistanceMeters'],0)}m")
    docs.append({"id":"gps_intel","title":"GPS Verification Intelligence","text":"\n".join(lines)})

    # ── Executive overview
    hr=scores[scores["RiskCategory"]=="High Risk"]["SRCode"].unique().tolist()
    mr=scores[scores["RiskCategory"]=="Medium Risk"]["SRCode"].unique().tolist()
    imp=trend_df[trend_df["Trend"]=="Improving"]["SRCode"].tolist() if trend_df is not None and not trend_df.empty else []
    dec=trend_df[trend_df["Trend"]=="Declining"]["SRCode"].tolist() if trend_df is not None and not trend_df.empty else []
    lines=["EXECUTIVE OVERVIEW","="*17,
           f"SRs: {stats.get('total_srs','?')} | Latest: {stats.get('latest_month','?')}",
           f"Avg Compliance: {stats.get('avg_compliance','N/A')}% | Avg GPS Mismatch: {stats.get('avg_gps_mismatch_pct','N/A')}%",
           f"High-Risk SRs: {stats.get('high_risk_count',0)}",
           "","RISK BREAKDOWN",
           f"  HIGH RISK ({len(hr)}): {', '.join(hr) or 'None'}",
           f"  MEDIUM RISK ({len(mr)}): {', '.join(mr) or 'None'}",
           "","TRENDS",
           f"  Improving: {', '.join(imp) or 'None'}",
           f"  Declining: {', '.join(dec) or 'None'}"]
    if anom_df is not None and not anom_df.empty:
        lines+=["","ANOMALIES"]
        for _,row in anom_df.head(5).iterrows():
            lines.append(f"  {row['SRCode']} {row['MonthPeriod']}: score={_n(row.get('BehaviourScore'))} drop={_n(row.get('Delta'))}")
    lines+=["","DISTRIBUTOR PERFORMANCE"]
    for _,row in dist_sum.iterrows():
        lines.append(f"  {row['DistributorCode']}: compliance={_p(row.get('AvgCompliancePct'))} gps={_p(row.get('AvgGPSMismatchPct'))}")
    docs.append({"id":"executive","title":"Executive Overview","text":"\n".join(lines)})

    return docs

# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE — numpy vector store, zero heavy dependencies
# ══════════════════════════════════════════════════════════════════════════════

class KeywordEmbedder:
    DIM = 384
    def __init__(self):
        self._cache: dict[str,np.ndarray] = {}
    def _vec(self,w):
        if w not in self._cache:
            h=hashlib.md5(w.encode()).digest()
            seed=int.from_bytes(h[:4],"little")
            v=np.random.RandomState(seed).randn(self.DIM).astype(np.float32)
            n=np.linalg.norm(v); self._cache[w]=v/n if n>0 else v
        return self._cache[w]
    def embed(self,text:str)->np.ndarray:
        words=text.lower().split()
        if not words: return np.zeros(self.DIM,dtype=np.float32)
        v=np.stack([self._vec(w) for w in words]).mean(axis=0)
        n=np.linalg.norm(v); return v/n if n>0 else v
    def embed_batch(self,texts:list[str])->np.ndarray:
        return np.stack([self.embed(t) for t in texts])

def _chunk(text:str,size=500,overlap=60)->list[str]:
    if not text.strip(): return []
    if len(text)<=size: return [text.strip()]
    chunks,start=[],0
    while start<len(text):
        end=min(start+size,len(text))
        if end<len(text):
            for sep in ["\n\n","\n",". "," "]:
                i=text.rfind(sep,start,end)
                if i>start: end=i+len(sep); break
        c=text[start:end].strip()
        if c: chunks.append(c)
        start=end-overlap
        if start>=len(text): break
    return chunks

@st.cache_resource(show_spinner="Preparing embedding engine…")
def _get_embedder():
    api_key=_get_api_key()
    if api_key:
        try:
            from google import genai
            client=genai.Client(api_key=api_key)
            client.models.embed_content(model="text-embedding-004",content="test")
            class GeminiEmb:
                def __init__(self,c): self._c=c
                def embed(self,text):
                    try:
                        r=self._c.models.embed_content(model="text-embedding-004",content=text[:2000])
                        return np.array(r.embeddings[0].values,dtype=np.float32)
                    except: return KeywordEmbedder().embed(text)
                def embed_batch(self,texts):
                    out=[]
                    for t in texts:
                        out.append(self.embed(t)); time.sleep(0.04)
                    return np.stack(out)
            return GeminiEmb(client)
        except Exception:
            pass
    return KeywordEmbedder()

class VectorStore:
    def __init__(self): self._embs=None; self._texts=[]; self._meta=[]
    def clear(self): self._embs=None; self._texts.clear(); self._meta.clear(); gc.collect()
    def add(self,texts,embs,metas):
        self._embs=embs.astype(np.float32) if self._embs is None else np.vstack([self._embs,embs.astype(np.float32)])
        self._texts.extend(texts); self._meta.extend(metas)
    def query(self,qv,k):
        if self._embs is None: return []
        q=qv.astype(np.float32); n=np.linalg.norm(q)
        if n>0: q=q/n
        sims=self._embs@q; k=min(k,len(self._texts))
        idxs=np.argpartition(sims,-k)[-k:]
        idxs=idxs[np.argsort(sims[idxs])[::-1]]
        return [(self._texts[i],self._meta[i],float(sims[i])) for i in idxs]
    @property
    def size(self): return len(self._texts)

def build_rag(docs:list[dict]) -> VectorStore:
    emb=_get_embedder()
    store=VectorStore()
    BATCH=15
    buf_t,buf_m=[],[]
    def flush():
        if not buf_t: return
        e=emb.embed_batch(buf_t)
        store.add(buf_t[:],e,buf_m[:])
        buf_t.clear(); buf_m.clear(); gc.collect()
    for doc in docs:
        for i,chunk in enumerate(_chunk(doc["text"])):
            buf_t.append(chunk)
            buf_m.append({"doc_id":doc["id"],"doc_title":doc["title"]})
            if len(buf_t)>=BATCH: flush()
    flush()
    return store

def retrieve(store:VectorStore,query:str,k=5)->str:
    if store.size==0: return "(No knowledge base built yet.)"
    emb=_get_embedder()
    qv=emb.embed(query)
    results=store.query(qv,k)
    if not results: return "(No relevant context found.)"
    return "\n\n---\n\n".join(
        f"[Source: {m.get('doc_title','?')} | score: {s:.2f}]\n{t}"
        for t,m,s in results)

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert AI Sales Performance Analyst. You answer questions
by reasoning over retrieved business intelligence documents from a field-sales dataset.
Always cite specific SR codes, distributor codes, months, and numbers from the context.
Explain root causes, not just numbers. Prioritise by business impact.
End every answer with 1-3 concrete, actionable recommendations.
If something is not in the retrieved context, say so — never invent data."""

def _retry(fn,retries=3):
    for attempt in range(retries+1):
        try: return fn()
        except Exception as e:
            msg=str(e)
            if ("429" in msg or "quota" in msg.lower()) and attempt<retries:
                m=re.search(r"seconds:\s*(\d+)",msg)
                time.sleep(int(m.group(1))+3 if m else 30*(2**attempt))
            else: raise

class GeminiChat:
    FALLBACK=["gemini-2.5-flash","gemini-2.0-flash","gemini-2.0-flash-lite","gemini-1.5-flash-latest"]
    def __init__(self,store:VectorStore,stats:dict):
        key=_get_api_key()
        if not key: raise RuntimeError("No Gemini API key configured.")
        from google import genai
        from google.genai import types
        self._genai=genai; self._types=types
        self._client=genai.Client(api_key=key)
        self._store=store; self._chat=None
        route_note=("" if stats.get("has_route_data")
                    else "NOTE: No VisitType column — route data derived from store matching.")
        role_note=(f"Role hierarchy detected: {', '.join(stats.get('role_cols',[]))}."
                   if stats.get("role_cols") else "")
        self._sys=SYSTEM_PROMPT+("\n\n"+route_note if route_note else "")+("\n"+role_note if role_note else "")
        self.model=GEMINI_MODEL

    def start(self):
        chain=[self.model]+[m for m in self.FALLBACK if m!=self.model]
        for m in chain:
            try:
                self._chat=self._client.chats.create(model=m,config=self._types.GenerateContentConfig(
                    system_instruction=self._sys,max_output_tokens=2048,temperature=0.3))
                self.model=m; return
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower(): continue
                raise
        raise RuntimeError("No working Gemini model found.")

    def ask(self,question:str)->str:
        if self._chat is None: raise RuntimeError("Call start() first.")
        ctx=retrieve(self._store,question)
        prompt=(f"RETRIEVED CONTEXT\n{'='*16}\n{ctx}\n\n{'='*16}\nQUESTION: {question}\n\n"
                "Answer based on the retrieved context only.")
        return _retry(lambda:self._chat.send_message(prompt)).text

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Sales Visit Intelligence",layout="wide",
                   page_icon="🤖",initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#f7f8fa}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e8eaed}
.kpi-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.kpi-card{background:#fff;border-radius:10px;padding:14px 18px;flex:1;min-width:130px;
          box-shadow:0 1px 4px rgba(0,0,0,.07);border-left:4px solid #4f8ef7}
.kpi-card.red{border-left-color:#e53935}.kpi-card.amber{border-left-color:#fb8c00}
.kpi-card.green{border-left-color:#43a047}
.kpi-label{font-size:11px;color:#888;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
.kpi-value{font-size:24px;font-weight:800;color:#1a1a2e;line-height:1.2}
.kpi-sub{font-size:11px;color:#bbb;margin-top:2px}
.sr-row{display:flex;justify-content:space-between;align-items:center;
        padding:6px 0;border-bottom:1px solid #f0f0f0;font-size:13px}
.sr-row:last-child{border-bottom:none}
.rag-badge{display:inline-block;background:#e8f5e9;color:#2e7d32;
           border-radius:20px;padding:3px 12px;font-size:12px;font-weight:700}
</style>""",unsafe_allow_html=True)

# session state
for k,v in [("data",None),("store",None),("chat",None),("history",[]),("active_model","")]:
    if k not in st.session_state: st.session_state[k]=v

# ── Sidebar
with st.sidebar:
    st.markdown("## 🤖 Sales Visit Intelligence")
    st.caption("RAG-powered AI analyst")
    st.divider()
    planned_f=st.file_uploader("Planned Visits (csv/xlsx)",type=["csv","xlsx","xls"])
    actual_f =st.file_uploader("Actual Visits (csv/xlsx)", type=["csv","xlsx","xls"])
    gps_thresh=st.slider("GPS threshold (m)",20,1000,100,10)
    go=st.button("Analyse Data",type="primary",use_container_width=True)
    st.divider()
    if st.session_state.store:
        st.markdown(f'<div class="rag-badge">✅ {st.session_state.store.size} chunks indexed</div>',
                    unsafe_allow_html=True)
    api_ok=bool(_get_api_key())
    if api_ok:
        st.success(f"Gemini ✓ {st.session_state.active_model}",icon="✅")
    else:
        st.warning("Set GEMINI_API_KEY in Streamlit secrets")
    if st.session_state.data:
        if st.button("🔄 Reset",use_container_width=True):
            for k in ["data","store","chat","active_model"]:
                st.session_state[k]=None
            st.session_state.history=[]
            st.rerun()

# ── Process
if go:
    if not planned_f or not actual_f:
        st.error("Upload both files first.")
    else:
        try:
            p_raw=read_file(planned_f.getvalue(),planned_f.name)
            a_raw=read_file(actual_f.getvalue(),actual_f.name)
        except Exception as e:
            st.error(f"Cannot read files: {e}"); st.stop()
        m1=validate(p_raw,PLANNED_REQUIRED)
        m2=validate(a_raw,ACTUAL_REQUIRED)
        if m1: st.error(f"Planned Visits missing: {m1}"); st.stop()
        if m2: st.error(f"Actual Visits missing: {m2}"); st.stop()

        with st.spinner("Step 1/3 — Computing analytics…"):
            p,a=coerce(p_raw.copy(),a_raw.copy())
            schema=detect_schema(a_raw)
            comp  =compliance_table(p,a)
            rt    =route_table(p,a,schema)
            gps   =gps_table(a,float(gps_thresh))
            miss  =store_miss_table(p,a)
            scores=behaviour_scores(comp,rt,gps)
            trend_df=trend_direction(scores)
            anom_df =anomalies(scores)
            month_s =monthly_summary(comp,rt,gps)
            dist_s  =distributor_summary(comp,rt,gps)
            role_a  =role_analytics(a,comp,gps,schema)
            stats   =exec_stats(scores,comp,gps,rt,schema)

        with st.spinner("Step 2/3 — Generating knowledge documents…"):
            docs=make_documents(p,a,scores,comp,rt,gps,miss,dist_s,
                                month_s,role_a,trend_df,anom_df,stats,schema)

        with st.spinner(f"Step 3/3 — Building vector index from {len(docs)} documents…"):
            store=build_rag(docs)

        st.session_state.data={"scores":scores,"comp":comp,"route":rt,"gps":gps,
            "miss":miss,"month_s":month_s,"schema":schema,"stats":stats,
            "trend_df":trend_df,"anom_df":anom_df,"gps_thresh":gps_thresh,"actual":a,
            "role_a":role_a}
        st.session_state.store=store
        st.session_state.chat=None
        st.session_state.history=[]
        st.session_state.active_model=""
        st.success(f"✅ Ready — {len(docs)} documents · {store.size} chunks indexed")
        st.rerun()

# ── Landing
if st.session_state.data is None:
    st.markdown("""
    <div style='text-align:center;padding:90px 20px'>
      <div style='font-size:72px'>🤖</div>
      <h2 style='color:#1a1a2e'>AI Sales Performance Analyst</h2>
      <p style='color:#666;max-width:520px;margin:0 auto;font-size:15px;line-height:1.6'>
        Upload your <strong>Planned Visits</strong> and <strong>Actual Visits</strong>
        files, then click <strong>Analyse Data</strong>.<br><br>
        The system computes analytics, generates AI knowledge documents,
        and builds a semantic search index — all in your browser.
      </p>
    </div>""",unsafe_allow_html=True)
    st.stop()

# ── Unpack
d=st.session_state.data
scores=d["scores"]; comp=d["comp"]; gps=d["gps"]; route=d["route"]
miss=d["miss"]; month_s=d["month_s"]; schema=d["schema"]; stats=d["stats"]
trend_df=d["trend_df"]; anom_df=d["anom_df"]; store=st.session_state.store
gps_thresh=d["gps_thresh"]; role_a=d["role_a"]

# ── Init chat
if api_ok and st.session_state.chat is None and store:
    try:
        chat=GeminiChat(store,stats)
        chat.start()
        st.session_state.chat=chat
        st.session_state.active_model=chat.model
    except Exception as e:
        err=str(e)
        if "404" in err or "not found" in err.lower():
            st.error("Gemini model not found. Check GEMINI_MODEL in secrets.")
        elif "429" in err or "quota" in err.lower():
            st.error("Quota exceeded. Wait or enable billing at aistudio.google.com")
        else:
            st.error(f"Could not start AI analyst: {e}")

# ── KPI strip
def _kpi(label,value,sub="",cls=""):
    return (f'<div class="kpi-card {cls}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')
def _pp(v,d=1):
    return f"{round(float(v),d)}%" if v is not None and not(isinstance(v,float)and np.isnan(v)) else "—"

avg_c=stats.get("avg_compliance"); avg_gp=stats.get("avg_gps_mismatch_pct")
hr=stats.get("high_risk_count",0); total=stats.get("total_srs",0)
n_dec=int((trend_df["Trend"]=="Declining").sum()) if trend_df is not None else 0
n_imp=int((trend_df["Trend"]=="Improving").sum()) if trend_df is not None else 0

cards=[
    _kpi("Visit Compliance",_pp(avg_c),"planned→completed",
         "green" if avg_c and avg_c>=80 else "amber" if avg_c and avg_c>=60 else "red"),
    _kpi("GPS Mismatches",_pp(avg_gp),f">{gps_thresh}m away",
         "red" if avg_gp and avg_gp>=20 else "amber" if avg_gp and avg_gp>=10 else "green"),
    _kpi("High-Risk SRs",str(hr),f"of {total}","red" if hr else "green"),
    _kpi("Declining SRs",str(n_dec),"month-over-month","red" if n_dec else "green"),
    _kpi("Improving SRs",str(n_imp),"month-over-month","green" if n_imp else ""),
    _kpi("Latest Period",stats.get("latest_month",""),"in dataset"),
]
if stats.get("avg_offroute_pct") is not None:
    avg_or=stats["avg_offroute_pct"]
    cards.insert(1,_kpi("Off-Route Rate",_pp(avg_or),f"source:{schema.route_source}",
                         "red" if avg_or>=30 else "amber" if avg_or>=15 else "green"))
st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>',unsafe_allow_html=True)
st.divider()

left,right=st.columns([1,2.2],gap="large")

# ── Left panel
with left:
    with st.expander("📋 Seller Risk Snapshot",expanded=True):
        lm=scores["MonthPeriod"].max()
        snap=(scores[scores["MonthPeriod"]==lm].sort_values("BehaviourScore")
              .merge(trend_df[["SRCode","Trend"]],on="SRCode",how="left"))
        RISK={"High Risk":"🔴","Medium Risk":"🟡","Low Risk":"🟢"}
        TREND={"Improving":"📈","Declining":"📉","Stable":"➡️"}
        for _,r in snap.iterrows():
            ri=RISK.get(str(r["RiskCategory"]),"⚪"); ti=TREND.get(str(r.get("Trend","")),"❓")
            sc=f"{r['BehaviourScore']:.0f}" if pd.notna(r["BehaviourScore"]) else "—"
            st.markdown(f'<div class="sr-row"><span>{ri} <strong>{r["SRCode"]}</strong></span>'
                        f'<span style="color:#888">Score <strong style="color:#1a1a2e">{sc}</strong> {ti}</span></div>',
                        unsafe_allow_html=True)

    for rc in schema.role_cols:
        if rc in role_a:
            lbl=rc.replace("Role_","").replace("_"," ")
            with st.expander(f"👥 {lbl} Performance",expanded=False):
                st.dataframe(role_a[rc].rename(columns={rc:lbl,
                    "AvgCompliancePct":"Compliance %","AvgGPSMismatchPct":"GPS Mismatch %",
                    "SRCount":"SRs"}),hide_index=True,use_container_width=True)

    with st.expander("📅 Monthly Trend",expanded=False):
        if not month_s.empty:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=month_s["MonthPeriod"],y=month_s["AvgCompliancePct"],
                name="Compliance %",mode="lines+markers",line=dict(color="#4f8ef7",width=2)))
            fig.add_trace(go.Scatter(x=month_s["MonthPeriod"],y=month_s["AvgGPSMismatchPct"],
                name="GPS Mismatch %",mode="lines+markers",line=dict(color="#e53935",width=2)))
            if "AvgOffRoutePct" in month_s.columns:
                fig.add_trace(go.Scatter(x=month_s["MonthPeriod"],y=month_s["AvgOffRoutePct"],
                    name="Off-Route %",mode="lines+markers",line=dict(color="#fb8c00",width=2)))
            fig.update_layout(height=210,margin=dict(l=0,r=0,t=4,b=0),
                legend=dict(orientation="h",y=-0.45,font=dict(size=10)),
                plot_bgcolor="#fff",paper_bgcolor="#fff")
            st.plotly_chart(fig,use_container_width=True)

    with st.expander("🏪 Top Missed Stores",expanded=False):
        m2=miss.head(8)[["StoreID","TimesPlanned","TimesMissed","MissRate"]].copy()
        m2["MissRate"]=m2["MissRate"].map(lambda x:f"{x:.0f}%" if pd.notna(x) else "—")
        st.dataframe(m2,hide_index=True,use_container_width=True)

    if anom_df is not None and not anom_df.empty:
        with st.expander(f"⚠️ Anomalies ({len(anom_df)})",expanded=False):
            st.dataframe(anom_df[["SRCode","MonthPeriod","BehaviourScore","Delta"]]
                .rename(columns={"Delta":"Score Drop"}),hide_index=True,use_container_width=True)

# ── Chatbot
with right:
    st.markdown('<div style="font-size:11px;font-weight:700;color:#888;'
                'text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">'
                '💬 AI Sales Performance Analyst (RAG)</div>',unsafe_allow_html=True)
    if not api_ok:
        st.info("Set GEMINI_API_KEY in Streamlit Cloud secrets to activate the AI analyst.")
        st.stop()
    if not st.session_state.chat:
        st.warning("AI analyst not connected. Check the error above.")
        st.stop()

    chat_obj=st.session_state.chat
    box=st.container(height=490)
    with box:
        if not st.session_state.history:
            route_note=f"Route data source: {schema.route_source}." if schema.route_source else "No route data detected."
            role_note=(f" Role hierarchy detected: {', '.join(schema.role_cols)}." if schema.role_cols else "")
            st.markdown(f"""<div style='text-align:center;padding:40px 20px;color:#888'>
              <div style='font-size:40px'>🔍</div>
              <div style='font-size:15px;font-weight:600;color:#444;margin:8px 0 4px'>
                RAG-powered analyst ready.</div>
              <div style='font-size:13px'>{route_note}{role_note}</div>
            </div>""",unsafe_allow_html=True)
        else:
            for role,msg in st.session_state.history:
                with st.chat_message(role): st.markdown(msg)

    user_turns=[r for r,_ in st.session_state.history if r=="user"]
    if not user_turns:
        SUGG=[
            ("🔎 Full analysis","Give a comprehensive analysis: top issues, trends, anomalies, and a prioritised action plan."),
            ("🚨 Who needs attention?","Which SRs need immediate managerial attention and why?"),
            ("📉 Compliance decline","Which SRs have declining compliance? Name specific months and root causes."),
            ("🛰️ GPS integrity","Are there signs of GPS manipulation or fake check-ins? Provide evidence."),
            ("🏆 Best performers","Which SRs are performing well or improving? Cite specific metrics."),
            ("🏪 Skipped stores","Which stores are repeatedly planned but missed? What should be done?"),
        ]
        if schema.supervisor_col:
            SUGG.append(("👥 Supervisor performance","Compare supervisor performance. Who oversees the best and worst teams?"))
        if schema.asm_col:
            SUGG.append(("🏢 ASM overview","Which ASMs oversee the highest and lowest performing teams?"))
        cols=st.columns(2)
        for i,(label,question) in enumerate(SUGG):
            if cols[i%2].button(label,use_container_width=True,key=f"s{i}"):
                st.session_state._pq=question; st.rerun()

    user_q=st.chat_input("Ask anything about your sales visit data…")
    pending=getattr(st.session_state,"_pq",None)
    question=pending or user_q
    if question and chat_obj:
        if pending and hasattr(st.session_state,"_pq"): del st.session_state._pq
        st.session_state.history.append(("user",question))
        with st.spinner("Searching knowledge base → generating answer…"):
            try: answer=chat_obj.ask(question)
            except Exception as e:
                err=str(e)
                if "429" in err or "quota" in err.lower():
                    answer="⏳ **Rate limit.** Wait ~30s and retry, or enable billing at aistudio.google.com"
                else: answer=f"⚠️ Error: {e}"
        st.session_state.history.append(("assistant",answer))
        st.rerun()
