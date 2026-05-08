# Data - Smart Meters in London

## Source

Kaggle: <https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london>

Original collection: Low Carbon London Project (UK Power Networks, 2012-2014). The published Kaggle bundle re-shares the cleaned, anonymised half-hourly readings from 5,567 households participating in the trial between November 2011 and February 2014, with weather and ACORN demographic covariates merged in.

## Status: not downloaded (judgment call)

The Kaggle bundle is approximately 1.25 GB compressed (a few GB unzipped). This places it in the 500 MB - 2 GB judgment-call zone defined in the agent rules; default is to document only and have the main session pull it on demand. Phase 1 deliverables (model scripts, manuscript, presentation) reference the schema and cohort sizes from the dataset documentation, not from a local file.

## How to download (run from main session, not from this scaffolding agent)

Kaggle CLI auth is already at `~/.kaggle/kaggle.json` (project rule).

```bash
# from /root/AI/liora_projects/16_smart_meters/data
kaggle datasets download -d jeanmidev/smart-meters-in-london
unzip -q smart-meters-in-london.zip
rm smart-meters-in-london.zip
```

Expected files after unzip:

| File | Rows / size | Description |
|------|-------------|-------------|
| `halfhourly_dataset/halfhourly_dataset/block_0.csv` ... `block_111.csv` | 167M rows total | half-hourly kWh per household, sharded by Acorn block |
| `daily_dataset/daily_dataset/block_0.csv` ... `block_111.csv` | aggregated | daily mean / sum per household |
| `informations_households.csv` | 5,567 rows | LCLid, stdorToU (Std vs dToU tariff), Acorn, Acorn_grouped, file (block) |
| `acorn_details.csv` | 18 + sub-segments | ACORN demographic descriptors (A-Q + U) |
| `weather_hourly_darksky.csv` | hourly | London weather: temp, humidity, wind, cloud cover, precipitation |
| `weather_daily_darksky.csv` | daily | aggregated weather |
| `uk_bank_holidays.csv` | calendar | UK bank holidays for the trial window |

Approximate sizes after unzip: roughly 2.5-3 GB. Keep `halfhourly_dataset/` for the TFT model; daily_dataset is enough for the SARIMA/Prophet baseline.

## Schema notes

- Half-hourly file columns: `LCLid` (household ID), `tstp` (timestamp, half-hour resolution), `energy(kWh/hh)`. The energy column is the energy delivered in that 30-minute interval (not power); a typical UK household consumes around 0.1 to 0.5 kWh per half-hour.
- Daily file columns: `LCLid`, `day`, plus mean/median/sum/min/max/std of half-hourly readings.
- ACORN groups: 18 main classes (A-Q + U, "unclassified"), aggregated into 6 supergroups in `Acorn_grouped` (Affluent, Comfortable, Adversity, etc.). The dataset bundles a small share of households on a dynamic Time-of-Use tariff (dToU); the rest are on the standard flat tariff (Std). The dToU subset is a natural experiment for tariff-response modelling.

## Pre-processing tasks (handled in `notebooks/01_EDA.ipynb` and `src/`)

1. Concatenate daily blocks for the baseline (~5 GB on disk; held as `pyarrow` partitions for efficient row-group reads).
2. Filter households with less than 3 months of continuous history (cold-start, dropped from baseline; kept for advanced model with a "short_history" indicator).
3. Join weather on date/hour and ACORN on LCLid.
4. Standardise timestamps to UTC and resample to 30-minute regular grid; mark gap-filled half-hours with a missingness mask.

## Citation

Jean-Michel D. (2019). *Smart meters in London* (Kaggle dataset). https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london. Re-share of the Low Carbon London project data: Schofield, J., Carmichael, R., Tindemans, S. H., et al. (2015). *Low Carbon London Project: Data from the Dynamic Time-of-Use Electricity Pricing Trial*. UK Data Service.
