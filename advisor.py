"""
Chat layer: intent -> engine tool call -> facts -> LLM wording -> NUMBER GUARD -> reply.

Guardrails (for the judges):
 1. The LLM never calculates. Python calls the engine first and passes the result as FACTS.
 2. System prompt: use only numbers from FACTS, answer in the user's language, keep it short.
 3. Number guard: every number in the LLM reply is checked against FACTS.
    If any number is unknown, the reply is replaced by a safe template answer.
 4. No API key / API error -> the same template answers, so the demo never breaks.
"""
from __future__ import annotations

import json
import os
import re

import engine

LANG_NAMES = {"az": "Azerbaijani", "en": "English", "ru": "Russian"}

# ------------------------------------------------------------------ intents
INTENTS = [
    ("activate", r"activat|aktiv|qoşul|qoş |подключ|активир"),
    ("travel", r"travel|trip|abroad|roaming|səyah|xaric|rouminq|ezamiyy|поездк|путешеств|роуминг|за границ"),
    ("cheaper", r"cheap|ucuz|niyə .*deyil|why not|дешевл|почему не"),
    ("simulate", r"would have paid|have paid|simulat|last 3|ödəyərdim|ödəmiş|nə qədər ödə|son 3|заплатил|сколько бы|симул"),
    ("usage", r"usage|left|run out|how much data|istifadə|qalıb|bitəcək|трафик|остал|законч"),
    ("recommend", r"recommend|best|package|tariff|plan|paket|tarif|tövsiyə|məsləhət|тариф|пакет|посовет|рекоменд"),
]


def detect_intent(text: str) -> str:
    t = text.lower()
    for name, pattern in INTENTS:
        if re.search(pattern, t):
            return name
    return "general"


def detect_days(text: str, default: int = 7) -> int:
    m = re.search(r"(\d{1,2})\s*(day|gün|дн|день|дня)", text.lower())
    if m:
        return int(m.group(1))
    if re.search(r"week|həftə|недел", text.lower()):
        return 7
    return default


# ------------------------------------------------------------------ tools
def build_facts(cid: str, intent: str, text: str, chosen: str | None) -> dict:
    """Calls the engine 'tools' needed for this intent. Output = the only allowed source of numbers."""
    rec = engine.recommend(cid)
    prof = engine.profile(cid)
    facts = {"profile": prof,
             "current_package": rec["current_package"],
             "current_expected_cost_azn": rec["current_expected_cost"],
             "forecast_next_month": rec["forecast"],
             "top3": [{k: r[k] for k in ("package", "price", "gb", "apps", "expected_monthly_cost",
                                        "savings_vs_current", "runout_risk_pct", "is_current", "reasons")}
                      for r in rec["recommendations"]],
             "alert": rec["alert"]}
    if intent == "cheaper":
        _, pk, _ = engine.load_data()
        top = rec["recommendations"][0]
        cheaper = [p for p in pk.to_dict("records") if p["price"] < top["price"]]
        if cheaper:
            c = max(cheaper, key=lambda p: p["price"])
            ev = engine._pkg_eval(c, engine.load_data()[2].query("id == @cid"), rec["forecast"])
            facts["cheaper_option"] = {"package": c["name"], "price": c["price"], "gb": c["gb"],
                                       "expected_monthly_cost": ev["expected_monthly_cost"],
                                       "extra_gb_packs_needed": ev["forecast_cost_breakdown"]["extra_gb"],
                                       "runout_risk_pct": ev["runout_risk_pct"]}
    if intent == "travel":
        facts["travel"] = engine.travel_scenario(cid, detect_days(text))
    if intent == "simulate":
        df, actual = engine.simulate(cid)
        facts["simulation_last3_months"] = df.drop(columns=["actual_bill"]).head(5).to_dict("records")
        facts["actual_paid_last3_months"] = actual
    if intent == "activate":
        facts["activation"] = {"package": chosen or rec["recommendations"][0]["package"], "status": "simulated_success"}
    return facts


# ------------------------------------------------------------------ number guard
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _norm(x: str) -> set[str]:
    x = x.replace(",", ".")
    try:
        v = float(x)
    except ValueError:
        return {x}
    out = {f"{v:g}", f"{round(v):d}", f"{v:.1f}", f"{v:.2f}"}
    return {o.rstrip("0").rstrip(".") if "." in o else o for o in out}


def allowed_numbers(facts: dict) -> set[str]:
    s = set()
    for n in _NUM.findall(json.dumps(facts, ensure_ascii=False, default=str)):
        s |= _norm(n)
    return s | {"1", "2", "3", "4", "5", "4G", "5G"}  # list positions / network names


def number_guard(reply: str, facts: dict) -> list[str]:
    """Returns numbers in the reply that are NOT supported by the facts (empty = safe)."""
    ok = allowed_numbers(facts)  # fact numbers + their rounded forms (24.37 -> 24.4, 24)
    bad = []
    for n in _NUM.findall(reply):
        v = float(n.replace(",", "."))
        if f"{v:g}" not in ok:      # the reply number itself is NOT rounded -> 99.9 stays 99.9
            bad.append(n)
    return bad


# ------------------------------------------------------------------ LLM
SYSTEM_PROMPT = """You are Bakcell's friendly Tariff Advisor inside the Bakcell chatbot.
Rules (strict):
- Use ONLY numbers that appear in FACTS. Never calculate, estimate or invent prices, GB, minutes, dates or savings.
- If FACTS do not contain what the user asks, say you don't have that data and offer what you can show.
- Answer in {lang}. Plain, warm, short: at most {max_sent} sentences. No tables, no markdown headers.
- The reply may be read aloud to a blind or low-vision user: no emojis, no symbols like '→', write 'AZN' as 'manat' in Azerbaijani.
- Explain WHY with 2-3 concrete reasons from FACTS (usage vs package GB, savings, run-out risk, app bonuses).
- If FACTS contain 'activation', confirm the package switch is done (this is a demo, simulated).
- If FACTS contain 'travel', explain that roaming costs per day and the total for the trip.
- Never ask for passwords, card numbers or personal data."""


def llm_reply(user_text: str, facts: dict, lang: str, simple: bool, history: list[dict]) -> str | None:
    """Returns LLM text, or None if no key / error (caller falls back to templates)."""
    system = SYSTEM_PROMPT.format(lang=LANG_NAMES[lang], max_sent=3 if simple else 5)
    content = f"FACTS (from the recommendation engine):\n{json.dumps(facts, ensure_ascii=False, default=str)}\n\nUser: {user_text}"
    msgs = [m for m in history[-6:] if m["role"] in ("user", "assistant")] + [{"role": "user", "content": content}]
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            r = anthropic.Anthropic().messages.create(
                model=os.getenv("LLM_MODEL", "claude-haiku-5-5"), max_tokens=400, system=system, messages=msgs)
            return r.content[0].text.strip()
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"), max_tokens=400,
                messages=[{"role": "system", "content": system}] + msgs)
            return r.choices[0].message.content.strip()
    except Exception as exc:  # network / quota / wrong model -> safe fallback
        print("LLM error:", exc)
    return None


# ------------------------------------------------------------------ template answers (fallback + guard)
T = {
    "en": {
        "rec": "Best plan for you: {p} at {price:g} AZN. {r0} {r1}",
        "keep": "Your current plan {p} is already the best choice. {r0} {r2}",
        "cheaper": "{c} costs {cp:g} AZN, but with your usage you would need about {packs} extra GB packs, so the real cost is about {ce:.2f} AZN per month. {p} comes out cheaper overall at {pe:.2f} AZN.",
        "cheaper_none": "{p} is already the cheapest plan that fits you.",
        "travel": "For a {d}-day trip, roaming costs {dp:g} AZN per day, so {rc:g} AZN in total. With {p} your month would cost about {bt:.2f} AZN.",
        "sim": "In the last 3 months you actually paid {a:.2f} AZN. On {p} you would have paid {t:.2f} AZN.",
        "act": "Done! {p} is activated (demo). It starts from your next billing month.",
        "runout": "Heads up: at your current pace you will likely run out of data around day {d} of the month. Want me to show a better plan?",
        "over": "You used only {pct}% of your data last month, so you may be overpaying. Want to see a cheaper plan?",
        "ok": "Your usage fits your plan well this month.",
        "general": "I can find the best tariff for you, explain why, show what you would have paid, or plan for travel. Just ask.",
    },
    "az": {
        "rec": "Sizin üçün ən uyğun paket: {p}, qiyməti {price:g} manat. {r0} {r1}",
        "keep": "Hazırkı paketiniz {p} artıq ən yaxşı seçimdir. {r0} {r2}",
        "cheaper": "{c} paketi {cp:g} manatdır, amma istifadənizə görə təxminən {packs} əlavə GB paketi lazım olacaq, real xərc ayda {ce:.2f} manat olar. {p} isə ümumilikdə daha ucuzdur: {pe:.2f} manat.",
        "cheaper_none": "{p} artıq sizə uyğun olan ən ucuz paketdir.",
        "travel": "{d} günlük səfər üçün rouminq gündə {dp:g} manat, cəmi {rc:g} manat edir. {p} paketi ilə ayınız təxminən {bt:.2f} manat olar.",
        "sim": "Son 3 ayda faktiki {a:.2f} manat ödəmisiniz. {p} paketində {t:.2f} manat ödəyərdiniz.",
        "act": "Hazırdır! {p} paketi aktivləşdirildi (demo). Növbəti hesab ayından qüvvəyə minir.",
        "runout": "Diqqət: bu templə internetiniz təxminən ayın {d}-{suf} günü bitəcək. Daha uyğun paket göstərim?",
        "over": "Keçən ay internetinizin yalnız {pct}%-ni istifadə etmisiniz, ola bilər ki, artıq ödəyirsiniz. Daha ucuz paket göstərim?",
        "ok": "Bu ay istifadəniz paketinizə yaxşı uyğun gəlir.",
        "general": "Sizə ən uyğun tarifi tapa, səbəbini izah edə, keçmiş xərcləri müqayisə edə və ya səfər üçün plan qura bilərəm. Sadəcə soruşun.",
    },
    "ru": {
        "rec": "Лучший тариф для вас: {p} за {price:g} AZN. {r0} {r1}",
        "keep": "Ваш текущий тариф {p} уже лучший выбор. {r0} {r2}",
        "cheaper": "{c} стоит {cp:g} AZN, но при вашем расходе понадобится около {packs} доп. пакетов GB, реально около {ce:.2f} AZN в месяц. {p} в итоге дешевле: {pe:.2f} AZN.",
        "cheaper_none": "{p} уже самый дешёвый подходящий тариф.",
        "travel": "Поездка на {d} дней: роуминг {dp:g} AZN в день, всего {rc:g} AZN. С {p} месяц обойдётся примерно в {bt:.2f} AZN.",
        "sim": "За последние 3 месяца вы заплатили {a:.2f} AZN. На {p} вы бы заплатили {t:.2f} AZN.",
        "act": "Готово! Тариф {p} подключён (демо). Он начнёт действовать со следующего расчётного месяца.",
        "runout": "Внимание: при текущем расходе интернет закончится примерно на {d}-й день месяца. Показать тариф получше?",
        "over": "В прошлом месяце вы использовали только {pct}% интернета, возможно, вы переплачиваете. Показать тариф дешевле?",
        "ok": "В этом месяце ваш расход хорошо подходит тарифу.",
        "general": "Я могу подобрать тариф, объяснить почему, показать, сколько бы вы заплатили, или помочь с поездкой. Просто спросите.",
    },
}

# Reasons are generated in English by the engine; short translations for the template path.
def _reasons(top: dict, lang: str) -> list[str]:
    if lang == "en":
        return top["reasons"] + ["", "", ""]
    sav, risk = top["savings_vs_current"], top["runout_risk_pct"]
    if lang == "az":
        r = [f"Paketdə {top['gb']:g} GB internet var.",
             f"Ayda təxminən {sav:.2f} manat qənaət edirsiniz." if sav > 0 else "Hazırkı xərcinizə yaxın qiymətdir.",
             "Son 6 ayda internetiniz heç vaxt bitməzdi." if risk == 0 else f"Son 6 ayın {risk}%-ində internet bitərdi."]
    else:
        r = [f"В пакете {top['gb']:g} GB интернета.",
             f"Экономия около {sav:.2f} AZN в месяц." if sav > 0 else "Цена близка к вашим текущим расходам.",
             "За последние 6 месяцев интернет бы ни разу не закончился." if risk == 0 else f"Интернет закончился бы в {risk}% месяцев."]
    return r


def template_reply(intent: str, facts: dict, lang: str) -> str:
    t = T[lang]
    top = facts["top3"][0]
    rs = _reasons(top, lang)
    if intent == "activate":
        return t["act"].format(p=facts["activation"]["package"])
    if intent == "cheaper":
        c = facts.get("cheaper_option")
        if not c:
            return t["cheaper_none"].format(p=top["package"])
        return t["cheaper"].format(c=c["package"], cp=c["price"], packs=c["extra_gb_packs_needed"],
                                   ce=c["expected_monthly_cost"], p=top["package"], pe=top["expected_monthly_cost"])
    if intent == "travel":
        tr = facts["travel"]
        return t["travel"].format(d=tr["days"], dp=tr["daily_price"], rc=tr["roaming_cost"],
                                  p=tr["best_package"], bt=tr["best_total_with_trip"])
    if intent == "simulate":
        best = facts["simulation_last3_months"][0]
        return t["sim"].format(a=facts["actual_paid_last3_months"], p=best["package"], t=best["total"])
    if intent == "usage":
        return alert_text(facts["alert"], lang)
    if intent == "recommend":
        if top["is_current"]:
            return t["keep"].format(p=top["package"], r0=rs[0], r2=rs[2])
        return t["rec"].format(p=top["package"], price=top["price"], r0=rs[0], r1=rs[1])
    return t["general"]


def az_ordinal(n: int) -> str:
    """Azerbaijani ordinal suffix (vowel harmony on the last number word): 1-ci, 3-cü, 6-cı, 9-cu, 10-cu."""
    if n % 10 == 0:
        return {10: "cu", 20: "ci", 30: "cu", 40: "cı", 50: "ci", 60: "cı", 70: "ci", 80: "ci", 90: "cı"}.get(n % 100, "cü")
    return {1: "ci", 2: "ci", 3: "cü", 4: "cü", 5: "ci", 6: "cı", 7: "ci", 8: "ci", 9: "cu"}[n % 10]


def alert_text(alert: dict | None, lang: str) -> str:
    t = T[lang]
    if not alert:
        return t["ok"]
    if alert["type"] == "runout":
        return t["runout"].format(d=alert["runout_day"], suf=az_ordinal(alert["runout_day"]))
    return t["over"].format(pct=alert["pct_last_month"])


# ------------------------------------------------------------------ main entry (call this from any chatbot)
def answer(cid: str, user_text: str, lang: str = "az", simple: bool = False,
           history: list[dict] | None = None, chosen: str | None = None) -> dict:
    intent = detect_intent(user_text)
    facts = build_facts(cid, intent, user_text, chosen)
    text, source, blocked = None, "template", []
    llm = llm_reply(user_text, facts, lang, simple, history or [])
    if llm:
        blocked = number_guard(llm, facts)
        if blocked:
            source = "template (guard blocked numbers: " + ", ".join(blocked) + ")"
        else:
            text, source = llm, "llm"
    if text is None:
        text = template_reply(intent, facts, lang)
    return {"text": text, "intent": intent, "source": source, "facts": facts}
