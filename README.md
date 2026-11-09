# F1 ELO Engine

ELO ratings for every Formula 1 driver and constructor since 1950, with an
interactive viewer.

- `f1data/` + `cli.py`: downloads the data from the Jolpica-F1 API into SQLite
- `f1_elo_app/`: the ELO engine and the Streamlit app

## Setup

```powershell
pip install -r requirements.txt
pip install -r f1_elo_app/requirements.txt

py cli.py build                               # ~1 hour, first time only
py cli.py export --out .\f1_elo_app\data
cd f1_elo_app
py run_elo.py
py -m streamlit run app.py
```

After that, use "Check for new races" in the app to stay up to date.