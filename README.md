# Top11_player_predicting
This project predicts the *top‐performing 11 players* **before a match** using historical player statistics.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (creates `artifact/model.pkl`)
python src/train.py --data_path data/cricket_top11_multinational.csv

# 3. Launch the Streamlit app
streamlit run application.py
```

## Repository Layout

```
artifact/               # saved models + preprocessors
logs/                   # training & inference logs
notebook/               # exploratory analysis (optional)
penv/                   # *optional* virtual‑env directory
src/                    # core Python package
templates/              # custom HTML templates for Streamlit (optional)
data/
    cricket_top11_multinational.csv
application.py          # Streamlit UI entry‑point
requirements.txt
README.md
```

## How it works

* **Model** – Gradient Boosting (XGBoost) classifier predicts whether a player is in
the top‑performing cohort (`top_performer` = 1).  
* **Features** – Batting, bowling & fielding stats, match metadata
(team, opposition, venue, format, role, etc.).  
* **Inference** – Sort the 22 predicted probabilities for both squads and pick the top 11.

See `src/train.py` & `src/predict.py` for full details.

## Deployment

1. **Local** – Run the Streamlit app as above.  
2. **Cloud** – Deploy to **Streamlit Community Cloud** (free) or
   containerize with Docker and host on Render/Fly.io/AWS App Runner.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "application.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
