"""
Bakcell Tariff Advisor — Streamlit demo.   Run:  streamlit run app.py
Tabs: Chat (text + voice) | My usage | Cost simulator | Bakcell analytics
"""
import hashlib
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import advisor
import engine
import voice

load_dotenv()
st.set_page_config(page_title="Bakcell Tariff Advisor", page_icon="📶", layout="wide")

UI = {  # interface labels (short) in 3 languages
    "en": dict(chat="💬 Chat", usage="📊 My usage", sim="🧮 Cost simulator", biz="🏢 Bakcell analytics",
               speak="🎤 Press and speak", ask="Ask about your tariff...", rec="Best plan for me",
               cheap="Why not the cheaper one?", travel="I will travel 7 days next month",
               paid="What would I have paid?", act="Activate this package", read="🔊 Read aloud",
               heard="I heard", noheard="Sorry, I could not hear that. Please try again."),
    "az": dict(chat="💬 Söhbət", usage="📊 İstifadəm", sim="🧮 Xərc simulyatoru", biz="🏢 Bakcell analitika",
               speak="🎤 Basın və danışın", ask="Tarifiniz haqqında soruşun...", rec="Mənə ən uyğun paket",
               cheap="Niyə daha ucuz paket deyil?", travel="Gələn ay 7 günlük səyahətə gedirəm",
               paid="Son 3 ayda nə qədər ödəyərdim?", act="Bu paketi aktiv et", read="🔊 Səslə oxu",
               heard="Eşitdim", noheard="Bağışlayın, eşidə bilmədim. Yenidən cəhd edin."),
    "ru": dict(chat="💬 Чат", usage="📊 Мой расход", sim="🧮 Симулятор", biz="🏢 Аналитика Bakcell",
               speak="🎤 Нажмите и говорите", ask="Спросите о тарифе...", rec="Какой тариф мне подходит?",
               cheap="Почему не дешевле?", travel="Поездка на 7 дней в следующем месяце",
               paid="Сколько бы я заплатил?", act="Подключить этот тариф", read="🔊 Озвучить",
               heard="Я услышал", noheard="Извините, не расслышал. Попробуйте ещё раз."),
}

# ----------------------------------------------------------------- sidebar: customer + accessibility
cust, pk, mon = engine.load_data()
with st.sidebar:
    st.title("📶 Tariff Advisor")
    lang = st.radio("Language / Dil / Язык", ["az", "en", "ru"], horizontal=True,
                    format_func=lambda x: {"az": "AZ", "en": "EN", "ru": "RU"}[x])
    cid = st.selectbox("Demo customer (anonymized ID)", cust["id"].tolist(), index=6)
    prof = engine.profile(cid)
    st.markdown(f"### {prof['persona']}\n{prof['persona_desc']}")

    st.divider()
    st.subheader("♿ Accessibility")
    voice_mode = st.toggle("Voice mode (speak + listen)", value=True,
                           help="Mic input and every answer is read aloud — for blind/low-vision users")
    big_text = st.toggle("Large text", value=False)
    contrast = st.toggle("High contrast", value=False)
    simple = st.toggle("Simple language (short answers)", value=False,
                       help="Shorter sentences for cognitive accessibility / elderly users")
    speed = st.select_slider("Speech speed", ["-25%", "-10%", "+0%", "+15%"], value="+0%")
    st.caption("LLM: " + ("connected" if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
                          else "no key → safe template answers"))

t = UI[lang]
css = ""
if big_text:
    css += "html, body, [class*='st-'], p, li, label, .stMarkdown {font-size: 1.35rem !important;} button p {font-size:1.3rem !important;}"
if contrast:
    css += (".stApp, [data-testid='stSidebar'] {background:#000 !important; color:#fff !important;} "
            "p, span, label, h1, h2, h3, li {color:#fff !important;} button {border:3px solid #ffd400 !important;} "
            "[data-testid='stChatMessage'] {border:2px solid #ffd400;}")
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ----------------------------------------------------------------- session state
ss = st.session_state
ss.setdefault("chats", {})
ss.setdefault("audio_hash", None)
ss.setdefault("tts", None)
ss.setdefault("activated", {})


def say(text: str):
    """Queue an answer to be spoken after the page renders."""
    if voice_mode:
        ss.tts = text


def bot_turn(user_text: str):
    msgs = ss.chats[cid]
    msgs.append({"role": "user", "content": user_text})
    res = advisor.answer(cid, user_text, lang=lang, simple=simple, history=msgs[:-1])
    if res["intent"] == "activate":
        ss.activated[cid] = res["facts"]["activation"]["package"]
    msgs.append({"role": "assistant", "content": res["text"], "source": res["source"], "intent": res["intent"]})
    say(res["text"])


# Proactive opener: first message for each customer is an alert from real usage
if cid not in ss.chats:
    opener = advisor.alert_text(engine.recommend(cid)["alert"], lang)
    ss.chats[cid] = [{"role": "assistant", "content": opener, "source": "proactive alert"}]
    say(opener)

rec = engine.recommend(cid)
tab_chat, tab_use, tab_sim, tab_biz = st.tabs([t["chat"], t["usage"], t["sim"], t["biz"]])

# ----------------------------------------------------------------- CHAT
with tab_chat:
    if voice_mode:
        audio = st.audio_input(t["speak"], key=f"mic_{cid}")
        if audio is not None:
            raw = audio.getvalue()
            h = hashlib.md5(raw).hexdigest()
            if h != ss.audio_hash:                    # process each recording once
                ss.audio_hash = h
                with st.spinner("..."):
                    heard = voice.speech_to_text(raw, lang)
                if heard:
                    bot_turn(heard)
                else:
                    ss.chats[cid].append({"role": "assistant", "content": t["noheard"], "source": "voice"})
                    say(t["noheard"])

    # Big one-tap buttons: usable without typing (motor impairments, elderly)
    cols = st.columns(5)
    for col, key in zip(cols, ["rec", "cheap", "travel", "paid", "act"]):
        if col.button(t[key], use_container_width=True, key=f"q_{key}"):
            bot_turn(t[key])

    for m in ss.chats[cid]:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["role"] == "assistant" and m.get("source"):
                st.caption(f"source: {m['source']}")

    typed = st.chat_input(t["ask"])
    if typed:
        bot_turn(typed)
        st.rerun()

# ----------------------------------------------------------------- MY USAGE
with tab_use:
    c = st.columns(5)
    c[0].metric("Persona", prof["persona"])
    c[1].metric("Current plan", prof["current_package"])
    c[2].metric("Avg data / month", f"{prof['avg_gb']} GB")
    c[3].metric("Avg bill / month", f"{prof['avg_bill']} AZN")
    c[4].metric("Churn risk", rec["churn"]["level"])
    if cid in ss.activated:
        st.success(f"✅ Activated (demo): {ss.activated[cid]}")

    hist = mon[mon["id"] == cid]
    cur_gb = engine.package(rec["current_package"])["gb"]
    fig = go.Figure()
    fig.add_bar(x=hist["month"], y=hist["gb"], name="Used GB", marker_color="#7a2ff7")
    fig.add_bar(x=["2026-11 (forecast)"], y=[rec["forecast"]["gb"]], name="Forecast", marker_color="#f7b32f")
    fig.add_hline(y=cur_gb, line_dash="dash", annotation_text=f"Plan limit {cur_gb:g} GB")
    fig.update_layout(height=320, margin=dict(t=30, b=10), title="Mobile data: last 6 months + next-month forecast")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 3 recommendations")
    cards = st.columns(3)
    for col, r in zip(cards, rec["recommendations"]):
        with col.container(border=True):
            st.markdown(f"### {'⭐ ' if r is rec['recommendations'][0] else ''}{r['package']}"
                        f"{' (current)' if r['is_current'] else ''}")
            st.markdown(f"**{r['price']:g} AZN** · {r['gb']:g} GB · expected **{r['expected_monthly_cost']:.2f} AZN**/month")
            st.markdown(f"Savings vs now: **{r['savings_vs_current']:+.2f} AZN** · run-out risk **{r['runout_risk_pct']}%**")
            for reason in r["reasons"]:
                st.markdown(f"- {reason}")
            if not r["is_current"] and st.button(f"Activate {r['package']}", key=f"act_{r['package']}"):
                ss.activated[cid] = r["package"]
                msg = advisor.T[lang]["act"].format(p=r["package"])
                ss.chats[cid].append({"role": "assistant", "content": msg, "source": "activation"})
                say(msg)
                st.rerun()

    top = rec["recommendations"][0]
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=max(0, top["savings_vs_current"]) * 12,
                                   number={"suffix": " AZN / year"}, title={"text": "Savings meter (best plan)"},
                                   gauge={"axis": {"range": [0, max(60, top["savings_vs_current"] * 24)]},
                                          "bar": {"color": "#22a06b"}}))
    gauge.update_layout(height=250, margin=dict(t=50, b=0))
    st.plotly_chart(gauge, use_container_width=True)

    if st.button(t["read"], key="read_cards"):
        ss.tts = advisor.template_reply("recommend", advisor.build_facts(cid, "recommend", "", None), lang)

# ----------------------------------------------------------------- SIMULATOR
with tab_sim:
    df, actual = engine.simulate(cid, 3)
    st.markdown(f"**What would you have paid in the last 3 months on each package?** "
                f"(includes extra GB, extra minutes and roaming) — you actually paid **{actual:.2f} AZN**.")
    show = df.drop(columns=["actual_bill"]).copy()
    show["vs actual"] = (show["total"] - actual).round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(show, x="package", y="total", color="vs actual", color_continuous_scale="RdYlGn_r",
                           title="3-month total by package (AZN)").update_layout(height=320),
                    use_container_width=True)

    days = st.slider("Travel scenario: roaming days next month", 0, 15, 0)
    if days:
        tr = engine.travel_scenario(cid, days)
        st.info(f"{days} roaming days × {tr['daily_price']:g} AZN = {tr['roaming_cost']:g} AZN. "
                f"Month on {tr['best_package']}: about {tr['best_total_with_trip']:.2f} AZN "
                f"(on your current plan: {tr['current_total_with_trip']:.2f} AZN).")

# ----------------------------------------------------------------- BAKCELL ANALYTICS
with tab_biz:
    bi = engine.business_impact()
    tab = engine.all_recommendations_table()
    c = st.columns(4)
    c[0].metric("Customers", bi["customers"])
    c[1].metric("On a non-optimal plan", f"{bi['wrong_plan_customers'] / bi['customers']:.0%}")
    c[2].metric("Customer savings / month — personalized", f"{bi['personal_monthly_savings_azn']:.2f} AZN")
    c[3].metric("Same, 'one offer for all' baseline", f"{bi['generic_monthly_savings_azn']:.2f} AZN")
    a = bi["assumptions"]
    st.caption(f"⚠️ SIMULATED A/B — assumptions: personalized offer acceptance {a['acceptance_personalized']:.0%}, "
               f"generic offer ({a['generic_offer']}) acceptance {a['acceptance_generic']:.0%}; a customer only accepts "
               f"an offer that does not raise their bill. Not real results.")

    l, r = st.columns(2)
    l.plotly_chart(px.histogram(tab, x="persona", title="Segment sizes (K-Means personas)").update_layout(height=300),
                   use_container_width=True)
    moves = tab[tab["current"] != tab["recommended"]].groupby(["current", "recommended"]).size().reset_index(name="customers")
    r.plotly_chart(px.bar(moves, x="current", y="customers", color="recommended",
                          title="Recommended switches (from → to)").update_layout(height=300),
                   use_container_width=True)

    _, k, sil = engine.personas()
    st.markdown("**Model quality**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Cost model reproduces real bills", f"{engine.validate_cost_model():.1%}")
    m2.metric("Clusters (best silhouette)", f"k = {k}", f"{sil[k]:.2f}")
    m3.metric("Forecast method (backtest winner)", engine.best_method())
    st.dataframe(engine.backtest(), hide_index=True)
    st.caption("Backtest: train on May–Sep 2026, predict Oct 2026. 'last' = naive baseline (same as last month).")
    st.dataframe(tab, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- speak queued answer (end of page)
if ss.tts:
    text, ss.tts = ss.tts, None
    mp3 = voice.text_to_speech(text, lang, rate=speed)
    if mp3:
        st.audio(mp3, format="audio/mp3", autoplay=True)
    else:
        components.html(voice.browser_tts_html(text, lang), height=0)
