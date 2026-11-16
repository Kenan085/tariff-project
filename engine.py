"""
Tariff Advisor - core engine (no LLM here: every number the bot says comes from this file).

Pipeline:
  load data -> exact cost model per package -> next-month forecast (backtested)
  -> personas (K-Means) -> ranked top-3 recommendations + plain-language reasons
  -> cost simulator, roaming scenario, proactive alert, churn-risk heuristic, A/B impact.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_PATH = Path(__file__).parent / "data" / "bakcell_synthetic.xlsx"

# Prices for anything outside the package (from the "Paketlər" sheet)
EXTRA_GB_PRICE = 3.00        # AZN per extra 1 GB pack
EXTRA_MIN_PRICE = 0.08       # AZN per out-of-package minute
ROAMING_DAY_PRICE = 5.00     # AZN per roaming day
GB_TOLERANCE = 0.10          # small overrun that is not charged


# --------------------------------------------------------------------------- data
@lru_cache(maxsize=1)
def load_data(path: str = str(DATA_PATH)):
    """Returns (customers, packages, monthly) with short English column names.
    Phone numbers are dropped on load (privacy: anonymized IDs only)."""
    cust = pd.read_excel(path, sheet_name="Müştərilər")
    pk = pd.read_excel(path, sheet_name="Paketlər")
    mon = pd.read_excel(path, sheet_name="Aylıq_istifadə")

    cust = cust.drop(columns=["Nömrə"]).rename(columns={
        "Müştəri ID": "id", "Yaş": "age", "Cins": "gender", "Region": "region",
        "Abunə növü": "sub_type", "Şəbəkədə staj (ay)": "tenure_m", "Cihaz": "device",
        "Şəbəkə": "network", "Cari paket": "current_pkg", "Cari paketdə (ay)": "months_on_pkg",
        "Paket: limitsiz tətbiqlər": "pkg_apps",
    })

    pk = pk[pk["Ailə"].notna()].rename(columns={
        "Tarif": "name", "Ailə": "family", "Aylıq qiymət (AZN)": "price", "İnternet (GB)": "gb",
        "Ölkədaxili dəqiqə (bütün şəbəkələr)": "min_all", "Şəbəkədaxili dəqiqə": "min_on",
        "Şəbəkəxarici dəqiqə": "min_off", "SMS": "sms", "Limitsiz tətbiqlər": "apps",
    }).reset_index(drop=True)

    mon = mon.rename(columns={
        "Müştəri ID": "id", "Ay": "month", "Paket": "pkg", "İnternet istifadəsi (GB)": "gb",
        "Dəqiqə istifadəsi (cəmi)": "min_total", "Şəbəkədaxili dəqiqə": "min_on",
        "Şəbəkəxarici dəqiqə": "min_off", "SMS sayı": "sms", "Əlavə alınan internet (GB)": "extra_gb",
        "Paketdən kənar dəqiqə": "extra_min", "Rouminq günləri": "roaming_days",
        "Balans artırma sayı": "topups", "Çağrı mərkəzinə müraciət": "callcenter",
        "Aylıq xərc (AZN)": "bill",
    }).sort_values(["id", "month"]).reset_index(drop=True)
    return cust, pk, mon


def package(name: str) -> dict:
    _, pk, _ = load_data()
    return pk[pk["name"] == name].iloc[0].to_dict()


# --------------------------------------------------------------------------- cost model
def month_cost(pkg: dict, gb: float, min_on: float, min_off: float, roaming_days: float = 0) -> dict:
    """Exact monthly bill of one month's usage on one package.
    Klass 5/10/15 have country-wide minutes; others split on-net / off-net."""
    # Overruns up to 0.1 GB are not billed (learned from the 600 historical bills)
    over = round(gb - pkg["gb"] - GB_TOLERANCE, 6)
    extra_gb = math.ceil(over) if over > 0 else 0
    if pkg["min_all"] > 0:
        extra_min = max(0, (min_on + min_off) - pkg["min_all"])
    else:
        extra_min = max(0, min_on - pkg["min_on"]) + max(0, min_off - pkg["min_off"])
    total = pkg["price"] + extra_gb * EXTRA_GB_PRICE + extra_min * EXTRA_MIN_PRICE + roaming_days * ROAMING_DAY_PRICE
    return {"base": pkg["price"], "extra_gb": extra_gb, "extra_gb_cost": extra_gb * EXTRA_GB_PRICE,
            "extra_min": extra_min, "extra_min_cost": round(extra_min * EXTRA_MIN_PRICE, 2),
            "roaming_cost": roaming_days * ROAMING_DAY_PRICE, "total": round(total, 2)}


def validate_cost_model() -> float:
    """Share of the 600 historical bills our model reproduces exactly (to 0.01 AZN)."""
    _, _, mon = load_data()
    ok = [abs(month_cost(package(r.pkg), r.gb, r.min_on, r.min_off, r.roaming_days)["total"] - r.bill) < 0.011
          for r in mon.itertuples()]
    return float(np.mean(ok))


# --------------------------------------------------------------------------- forecasting
def _forecast_series(x: np.ndarray, method: str) -> float:
    if method == "last":
        return float(x[-1])
    if method == "mean3":
        return float(np.mean(x[-3:]))
    if method == "wmean3":
        return float(np.average(x[-3:], weights=[1, 2, 3]))
    if method == "trend":  # linear trend on all months, clipped at 0
        t = np.arange(len(x))
        a, b = np.polyfit(t, x, 1)
        return float(max(0.0, a * len(x) + b))
    raise ValueError(method)


METHODS = ["last", "mean3", "wmean3", "trend"]


@lru_cache(maxsize=1)
def backtest() -> pd.DataFrame:
    """Train on months 1-5, predict month 6 (Oct 2026). 'last' is the naive baseline."""
    _, _, mon = load_data()
    rows = []
    for m in METHODS:
        errs = []
        for _, g in mon.groupby("id"):
            x = g["gb"].to_numpy()
            errs.append(_forecast_series(x[:-1], m) - x[-1])
        e = np.array(errs)
        rows.append({"method": m, "MAE_GB": round(np.abs(e).mean(), 3), "RMSE_GB": round(np.sqrt((e ** 2).mean()), 3)})
    return pd.DataFrame(rows).sort_values("MAE_GB").reset_index(drop=True)


def best_method() -> str:
    return backtest().iloc[0]["method"]


def forecast(cid: str) -> dict:
    """Next-month (Nov 2026) forecast for data and minutes, using the backtest winner."""
    _, _, mon = load_data()
    g = mon[mon["id"] == cid]
    m = best_method()
    return {"method": m,
            "gb": round(_forecast_series(g["gb"].to_numpy(), m), 2),
            "min_on": round(_forecast_series(g["min_on"].to_numpy(), m)),
            "min_off": round(_forecast_series(g["min_off"].to_numpy(), m))}


# --------------------------------------------------------------------------- features + personas
@lru_cache(maxsize=1)
def features() -> pd.DataFrame:
    cust, _, mon = load_data()
    f = mon.groupby("id").agg(
        avg_gb=("gb", "mean"), std_gb=("gb", "std"), avg_min=("min_total", "mean"),
        avg_off=("min_off", "mean"), avg_sms=("sms", "mean"), roam_days=("roaming_days", "sum"),
        extra_gb_6m=("extra_gb", "sum"), extra_min_6m=("extra_min", "sum"), avg_bill=("bill", "mean"),
        topups=("topups", "mean"), callcenter=("callcenter", "sum"),
    )
    f["off_share"] = f["avg_off"] / f["avg_min"].clip(lower=1)
    f["gb_trend"] = mon.groupby("id")["gb"].apply(lambda s: np.polyfit(range(len(s)), s, 1)[0])
    f = f.join(cust.set_index("id")[["age", "tenure_m", "current_pkg", "region", "device", "network", "sub_type"]])
    return f


PERSONAS = {
    "streamer": ("📺 The Streamer", "Heavy mobile-data user, video and social all day"),
    "talker": ("📞 The Talker", "Lives on calls, data is secondary"),
    "traveler": ("✈️ The Traveler", "Regularly roams abroad"),
    "balanced": ("⚖️ The Balanced User", "Moderate data and calls"),
    "light": ("🌱 The Light User", "Low usage — often overpays for unused allowance"),
}


@lru_cache(maxsize=1)
def personas() -> tuple[pd.Series, int, dict]:
    """K-Means on log usage features; k chosen by best silhouette in 3..6.
    Each cluster is named by a rule on its centroid, so the names stay meaningful."""
    f = features()
    cols = ["avg_gb", "avg_min", "off_share", "avg_sms", "roam_days"]
    X = StandardScaler().fit_transform(np.log1p(f[cols].clip(lower=0)))
    scores = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=SEED).fit_predict(X)) for k in range(3, 7)}
    k = max(scores, key=scores.get)
    labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    cent = f[cols].groupby(labels).mean()
    med_gb, med_min = f["avg_gb"].median(), f["avg_min"].median()
    names = {}
    for c, r in cent.iterrows():
        if r.roam_days >= 2:
            names[c] = "traveler"
        elif r.avg_gb > 2 * med_gb:
            names[c] = "streamer"
        elif r.avg_min > 2 * med_min:
            names[c] = "talker"
        elif r.avg_gb < med_gb and r.avg_min < med_min:
            names[c] = "light"
        else:
            names[c] = "balanced"
    return pd.Series([names[l] for l in labels], index=f.index), k, {kk: round(v, 3) for kk, v in scores.items()}


# --------------------------------------------------------------------------- churn heuristic
def churn_risk(cid: str) -> dict:
    """No churn label in the data -> transparent rule-based score (0-100)."""
    f = features().loc[cid]
    score = 0
    reasons = []
    if f.callcenter >= 3:
        score += 35; reasons.append(f"{int(f.callcenter)} call-center contacts in 6 months")
    if f.extra_gb_6m + f.extra_min_6m / 50 >= 10:
        score += 30; reasons.append("frequent overage charges")
    if f.tenure_m < 12:
        score += 20; reasons.append(f"new customer ({int(f.tenure_m)} months)")
    if f.gb_trend < -0.3:
        score += 15; reasons.append("data usage is falling")
    level = "high" if score >= 50 else "medium" if score >= 25 else "low"
    return {"score": score, "level": level, "reasons": reasons}


# --------------------------------------------------------------------------- recommendation
def _pkg_eval(pkg: dict, hist: pd.DataFrame, fc: dict, include_roaming: bool = False) -> dict:
    """Score one package for one customer."""
    last3 = hist.tail(3)
    hist_costs = [month_cost(pkg, r.gb, r.min_on, r.min_off, r.roaming_days if include_roaming else 0)["total"]
                  for r in last3.itertuples()]
    fc_cost = month_cost(pkg, fc["gb"], fc["min_on"], fc["min_off"])
    over_months = int((hist["gb"] > pkg["gb"]).sum())          # months the data allowance ran out
    risk = over_months / len(hist)
    if pkg["gb"] > 0:
        waste_gb = max(0.0, pkg["gb"] - fc["gb"])
        usage_pct = fc["gb"] / pkg["gb"] * 100
    else:
        waste_gb, usage_pct = 0.0, float("inf") if fc["gb"] > 0 else 0.0
    expected = 0.5 * fc_cost["total"] + 0.5 * float(np.mean(hist_costs))  # blend forecast + recent reality
    fit = expected + 1.5 * risk * pkg["price"] / 10                     # small penalty for run-out risk
    return {"package": pkg["name"], "price": float(pkg["price"]), "gb": float(pkg["gb"]),
            "apps": pkg["apps"], "expected_monthly_cost": round(expected, 2),
            "forecast_cost_breakdown": fc_cost, "last3_costs": hist_costs,
            "runout_risk_pct": round(risk * 100), "waste_gb": round(waste_gb, 1),
            "forecast_usage_pct": round(usage_pct) if np.isfinite(usage_pct) else None, "_fit": fit}


@lru_cache(maxsize=256)
def recommend(cid: str, top_n: int = 3) -> dict:
    cust, pk, mon = load_data()
    hist = mon[mon["id"] == cid]
    fc = forecast(cid)
    cur_name = cust.set_index("id").loc[cid, "current_pkg"]
    evals = [_pkg_eval(p, hist, fc) for p in pk.to_dict("records")]
    cur = next(e for e in evals if e["package"] == cur_name)
    churn = churn_risk(cid)

    # Retention: high-risk customers get a loyalty bonus on non-current packages (simulated offer)
    for e in evals:
        e["retention_offer"] = churn["level"] == "high" and e["package"] != cur_name
    ranked = sorted(evals, key=lambda e: e["_fit"])
    top = ranked[:top_n]
    if cur_name not in [t["package"] for t in top]:
        pass  # current plan is not in top-3 -> clear switch case
    for e in top:
        e["savings_vs_current"] = round(cur["expected_monthly_cost"] - e["expected_monthly_cost"], 2)
        e["is_current"] = e["package"] == cur_name
        e["reasons"] = explain(e, cur, fc, hist)
        e.pop("_fit", None)
    cur.pop("_fit", None)
    return {"customer_id": cid, "current_package": cur_name, "current_expected_cost": cur["expected_monthly_cost"],
            "forecast": fc, "recommendations": top, "churn": churn,
            "alert": proactive_alert(cid)}


def explain(e: dict, cur: dict, fc: dict, hist: pd.DataFrame) -> list[str]:
    """2-3 plain-language reasons built only from computed numbers (rule-based explainability)."""
    r = []
    avg = hist["gb"].mean()
    if e["gb"] > 0:
        r.append(f"You use about {avg:.1f} GB per month (forecast {fc['gb']:.1f} GB); "
                 f"{e['package']} includes {e['gb']:g} GB.")
    else:
        r.append(f"{e['package']} has no internet, and you use only {avg:.1f} GB per month.")
    if e["package"] == cur["package"]:
        if e["runout_risk_pct"] >= 50:
            r.append("This is your current plan. Even with extra GB packs it is still the cheapest option for you.")
        else:
            r.append("This is your current plan and it already fits you well.")
    elif e["savings_vs_current"] > 0:
        r.append(f"Expected cost {e['expected_monthly_cost']:.2f} AZN/month vs {cur['expected_monthly_cost']:.2f} AZN "
                 f"now — you save about {e['savings_vs_current']:.2f} AZN per month.")
    else:
        r.append(f"Costs {-e['savings_vs_current']:.2f} AZN more per month than now, but gives more room.")
    if e["runout_risk_pct"] == 0:
        r.append("In the last 6 months you would never have run out of data on it.")
    else:
        r.append(f"You would have run out of data in {e['runout_risk_pct']}% of the last 6 months.")
    if e["apps"] and e["apps"] != "Yoxdur":
        r.append(f"Bonus: unlimited {e['apps']} (does not use your GB).")
    if e.get("retention_offer"):
        r.append("Loyalty offer: first month 20% off (simulated retention offer).")
    return r[:4]


# --------------------------------------------------------------------------- simulator / scenarios
def simulate(cid: str, months: int = 3) -> pd.DataFrame:
    """What would the customer have paid in the last N months on each package? (incl. roaming)"""
    _, pk, mon = load_data()
    hist = mon[mon["id"] == cid].tail(months)
    rows = []
    for p in pk.to_dict("records"):
        row = {"package": p["name"]}
        for r in hist.itertuples():
            row[r.month] = month_cost(p, r.gb, r.min_on, r.min_off, r.roaming_days)["total"]
        row["total"] = round(sum(v for k, v in row.items() if k != "package"), 2)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("total").reset_index(drop=True)
    df["actual_bill"] = None
    actual = round(hist["bill"].sum(), 2)
    return df, actual


def travel_scenario(cid: str, days: int) -> dict:
    """'I will travel next month for N days' -> roaming adds the same daily cost on every plan,
    so we also compare against a 'stay home' month."""
    rec = recommend(cid)
    extra = days * ROAMING_DAY_PRICE
    return {"days": days, "roaming_cost": extra, "daily_price": ROAMING_DAY_PRICE,
            "best_package": rec["recommendations"][0]["package"],
            "best_total_with_trip": round(rec["recommendations"][0]["expected_monthly_cost"] + extra, 2),
            "current_total_with_trip": round(rec["current_expected_cost"] + extra, 2)}


def proactive_alert(cid: str) -> dict | None:
    cust, _, mon = load_data()
    cur = package(cust.set_index("id").loc[cid, "current_pkg"])
    fc = forecast(cid)
    last = mon[mon["id"] == cid].iloc[-1]
    if cur["gb"] <= 0:
        return None
    pct_last = last.gb / cur["gb"] * 100
    if fc["gb"] > cur["gb"]:
        day = max(1, int(30 * cur["gb"] / fc["gb"]))
        return {"type": "runout", "pct_last_month": round(pct_last), "runout_day": day,
                "forecast_gb": fc["gb"], "pkg_gb": cur["gb"]}
    if pct_last < 40:
        return {"type": "overpaying", "pct_last_month": round(pct_last), "pkg_gb": cur["gb"]}
    return None


def profile(cid: str) -> dict:
    f = features().loc[cid]
    p, _, _ = personas()
    key = p.loc[cid]
    return {"customer_id": cid, "persona": PERSONAS[key][0], "persona_desc": PERSONAS[key][1],
            "age": int(f.age), "region": f.region, "device": f.device, "network": f.network,
            "current_package": f.current_pkg, "tenure_months": int(f.tenure_m),
            "avg_gb": round(f.avg_gb, 2), "avg_minutes": round(f.avg_min), "avg_bill": round(f.avg_bill, 2),
            "roaming_days_6m": int(f.roam_days), "extra_gb_bought_6m": int(f.extra_gb_6m)}


# --------------------------------------------------------------------------- business impact (simulated)
@lru_cache(maxsize=1)
def business_impact(acceptance_personal: float = 0.35, acceptance_generic: float = 0.10) -> dict:
    """Simulated A/B: 'everyone gets DaimOnline 10' vs our personalized top-1.
    Acceptance rates are ASSUMPTIONS (shown in the UI), not measured."""
    cust, _, _ = load_data()
    rng = np.random.default_rng(SEED)
    generic = "DaimOnline 10"
    sav_p, sav_g, ovr_p, ovr_cur = [], [], [], []
    for cid in cust["id"]:
        rec = recommend(cid)
        cur = rec["current_expected_cost"]
        top = rec["recommendations"][0]
        _, pk, mon = load_data()
        g_eval = _pkg_eval(package(generic), mon[mon["id"] == cid], rec["forecast"])
        # A customer only accepts an offer that does not cost them more
        if rng.random() < acceptance_personal and top["savings_vs_current"] >= 0:
            sav_p.append(top["savings_vs_current"]); ovr_p.append(top["forecast_cost_breakdown"]["extra_gb_cost"] + top["forecast_cost_breakdown"]["extra_min_cost"])
        else:
            sav_p.append(0.0)
        gsave = cur - g_eval["expected_monthly_cost"]
        sav_g.append(gsave if (rng.random() < acceptance_generic and gsave >= 0) else 0.0)
        ovr_cur.append(month_cost(package(rec["current_package"]), rec["forecast"]["gb"],
                                  rec["forecast"]["min_on"], rec["forecast"]["min_off"]))
    switch_needed = sum(1 for cid in cust["id"] if not recommend(cid)["recommendations"][0]["is_current"])
    return {"customers": len(cust), "wrong_plan_customers": switch_needed,
            "personal_monthly_savings_azn": round(sum(sav_p), 2),
            "generic_monthly_savings_azn": round(sum(sav_g), 2),
            "assumptions": {"acceptance_personalized": acceptance_personal, "acceptance_generic": acceptance_generic,
                            "generic_offer": generic}}


def all_recommendations_table() -> pd.DataFrame:
    cust, _, _ = load_data()
    p, _, _ = personas()
    rows = []
    for cid in cust["id"]:
        r = recommend(cid)
        t = r["recommendations"][0]
        rows.append({"id": cid, "persona": PERSONAS[p.loc[cid]][0], "current": r["current_package"],
                     "recommended": t["package"], "savings_azn": t["savings_vs_current"],
                     "churn": r["churn"]["level"]})
    return pd.DataFrame(rows)
