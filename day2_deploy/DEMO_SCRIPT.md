# RespiraAI demo script

A tight sequence to record, hitting all four components and the two "graceful failure" moments
that are genuinely worth showing off — those are the parts that make this look like a deployed
system and not just a notebook. Should run 3-5 minutes end to end.

## 0. Before recording

- Deployed instance: hit `/health` about a minute beforehand so the free-tier cold start doesn't
  eat the first 30-60 seconds on camera.
- Local instead: have `uvicorn app.main:app --reload` already running.
- Have one real chest X-ray image ready to upload (`data/chest_xray_sample/PNEUMONIA/...` works,
  or any real chest X-ray you find).

## 1. Open the app (10 sec)

Load `/`. Point out the status row at the top — say what it's showing (which of the four
components are actually ready right now), not just that it exists.

## 2. Lung cancer risk — ML vs DL (60 sec)

- Fill in a **high-risk** profile: older age, smoker, several symptoms checked.
- Submit. Both cards populate side by side.
- **The actual point to make on camera:** this one form submission just called two different
  models — a Random Forest and a small neural net — on the identical input, and you can see them
  agree (or disagree). That side-by-side comparison is the payoff of the whole ML-vs-DL exercise
  from earlier, not just a UI nicety.
- Clear the form, submit a **low-risk** profile (young, no symptoms checked), show both cards
  flip to low risk.

## 3. Chest X-ray (45 sec)

- Upload the real X-ray image, show the preview appear.
- Submit. Show the result.
- **Say the honest thing out loud**, don't skip it: this model's classification head hasn't been
  trained on the full dataset yet, so the warning banner is doing exactly what it should — telling
  you not to trust this number, rather than confidently returning a wrong one. That's the
  deployment lesson here, not a bug to explain away.

## 4. Ask about a condition — RAG (45 sec)

- Ask something the knowledge base actually covers, e.g. *"What's the difference between asthma
  and COPD?"* Show the answer plus the source chips underneath it — say that those sources are
  what the retrieval step actually pulled, not something the model invented.
- Ask something it doesn't cover, e.g. *"Can vitamin C cure a cold?"* Show that it says so
  honestly rather than answering anyway. This is the single most important behavior to demonstrate
  in any RAG system, so don't cut it for time.

## 5. Close (10 sec)

Back to `/health` or `/docs` for a second — one line on the stack: FastAPI backend, four routers,
one shared model-loading step at startup, plain-JS frontend, deployed without a container.

## If something's degraded during the actual recording

- `/ask` failing with a friendly error instead of crashing → that's the intended behavior on an
  LLM provider outage, not a bug. Worth narrating rather than re-recording around it.
- Free-tier cold start caught you anyway → cut to `/health` returning while narrating what's
  happening, then continue once it's warm.
