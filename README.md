# 📶 Bakcell Tariff Advisor — with voice for people with disabilities

An AI advisor that looks at each customer's real usage, finds the plan that is cheapest *for them*, explains why with real numbers, and can be used entirely by voice (Azerbaijani, English, Russian).

## Run (2 commands)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Optional: copy `.env.example` to `.env` and add an `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. Without a key everything works; the bot answers with safe template sentences instead of LLM wording.
Voice needs internet (Google speech recognition + Microsoft Edge neural voices, both free, no key). Use Chrome and allow the microphone.

## Files
| File | What it does |
|---|---|
| `engine.py` | All numbers: exact cost model, forecast, personas, top-3 ranking, explanations, simulator, alerts, churn heuristic, A/B impact |
| `advisor.py` | Chat brain: intent → engine "tool" → FACTS → LLM wording → **number guard** → reply. `advisor.answer(cid, text, lang)` can be called from any existing chatbot |
| `voice.py` | Speech-to-text (Whisper or Google) and text-to-speech (edge-tts with the Azerbaijani voice `az-AZ-BanuNeural`, browser voice as fallback) |
| `app.py` | Streamlit UI: Chat · My usage · Cost simulator · Bakcell analytics |

## How it works
1. **Cost model:** price + 3 AZN per extra GB + 0.08 AZN per extra minute + 5 AZN per roaming day. Klass 5/10/15 have country-wide minutes, the other plans have on-net/off-net minutes. Overruns up to 0.1 GB are not billed. **It reproduces 96.5% of the 600 real bills exactly**; in the other 3.5% the customer bought an extra pack before running out.
2. **Forecast:** we backtested 4 methods (train May–Sep, predict Oct). With only 6 months × 100 customers, "same as last month" won (MAE 0.81 GB), so we use it and say so. A bigger model on this little data would overfit.
3. **Personas:** K-Means on log usage features, with k picked by silhouette (k = 5, score 0.41). Clusters are named by rules on their centres: Streamer, Traveler, Balanced, Light.
4. **Ranking:** each of the 9 plans gets an expected cost (50% next-month forecast + 50% replay of the last 3 real months), plus a small penalty for the risk of running out of data. The output is the top 3 with savings, run-out risk and 2–4 reasons.
5. **Guardrails:** the LLM never calculates. Every number in its reply is checked against the engine output, and any reply containing an unknown number is replaced by a template answer (the UI shows `source: template (guard blocked …)`).

## ♿ Accessibility features
- **Voice mode:** press the mic, speak in AZ/EN/RU, and every answer is read aloud with a natural Azerbaijani voice. The text sent to the voice says "manat" instead of "AZN" and drops emojis.
- **Five large one-tap buttons:** the main questions can be asked without typing (motor impairments, elderly users).
- **Large text**, **high contrast** (black/yellow) and **simple-language** toggles, plus an adjustable speech speed.
- **"Read aloud"** button on the recommendation cards.
- Activation works by voice too ("Bu paketi aktiv et").

## 5-minute demo script
1. **(30 s) Problem:** 29 of 100 customers are on a non-optimal plan. Open *Bakcell analytics* to show this.
2. **(60 s) BKC-0007 (Streamer, runs out):** the bot opens proactively: "your data will likely run out around day 21". Open *My usage* to show the forecast bar above the plan limit. Ask "Why not the cheaper one?": a cheaper plan would need 23 extra GB packs, so it is not cheaper.
3. **(90 s) Voice, for a blind user.** Turn on high contrast and large text. Choose AZ, press the mic and say *"Mənə hansı paket uyğundur?"*. The answer is spoken. Then say *"Bu paketi aktiv et"* to activate.
4. **(45 s) BKC-0066 (Streamer, overpaying on Klass 50):** recommended Klass 25, saving 25 AZN/month. Show the 3-month table in the *Cost simulator*.
5. **(30 s) BKC-0045 (Traveler):** ask "I will travel 7 days next month" to get the roaming cost for the trip.
6. **(45 s) Trust:** show the cost-model match rate, the backtest table and the guardrail caption.

Other good customers: BKC-0083 (Balanced, Klass 25 → DaimOnline 10, saves 12 AZN), BKC-0055 (Light, saves 9.5 AZN).

## Slide outline (6 slides)
1. **Problem:** customers overpay or run out of data, and choosing among plans is confusing. 29% of customers are on the wrong plan.
2. **Solution:** a personal advisor inside the Bakcell chatbot that you can use by text **or voice**.
3. **Architecture:** Data → Engine (cost model, forecast, personas, ranking) → Advisor (intent → FACTS → LLM → number guard) → Voice → UI.
4. **Demo**
5. **Impact:** customer savings, fewer overage complaints, accessibility for blind, elderly and motor-impaired users (state the assumptions).
6. **Limits & next steps:** synthetic data, 6 months of history, simulated A/B. Next: real A/B test, learning from accepted offers, a Telegram/WhatsApp voice channel.

## Likely judge questions
- **How do you stop the LLM inventing prices?** It never calculates. Python computes everything, the LLM only words it, and a number guard rejects any number that isn't in the engine output, falling back to a template answer.
- **Why not LightGBM?** Six data points per customer is too little. We backtested and the naive method won, and we'd rather show an honest baseline than an overfitted model.
- **How accurate is the cost model?** It reproduces 96.5% of the real bills to the qəpik (0.01 AZN), and we can explain the remaining 3.5%.
- **How does it scale to millions of users?** Scoring is vectorizable arithmetic over 9 plans, so nightly batch scoring would be cheap. The LLM is only called when a customer is chatting.
- **What about privacy?** Phone numbers are dropped on load, only anonymized IDs are used, and no personal data is sent to the LLM except usage numbers.
- **Why voice?** Blind, low-vision, elderly and motor-impaired customers can't easily compare tariff tables. Azerbaijani neural voices make the advisor usable for them.
- **What's simulated?** The A/B acceptance rates (35% vs 10%), activation and the retention discount. They're labelled in the UI.
