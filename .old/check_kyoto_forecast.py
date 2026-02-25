import pandas as pd
import numpy as np

df = pd.read_csv('data/model_outputs/projected_climate_2026.csv', parse_dates=['date'])
kyoto = df[df['location'] == 'kyoto'].copy()
kyoto = kyoto.sort_values('date')

start = pd.to_datetime('2025-10-01')
end = pd.to_datetime('2026-05-31')
winter_window = kyoto[(kyoto['date'] >= start) & (kyoto['date'] <= end)].copy()

chill_days = (winter_window['tmean_c'] <= 5.0).sum()
print(f'Kyoto forecast Oct 2025-May 2026:')
print(f'Total days: {len(winter_window)}')
print(f'Chill days (tmean<=5C): {chill_days}')
print(f'Required chill: 115')
print(f'tmean range: {winter_window["tmean_c"].min():.1f} to {winter_window["tmean_c"].max():.1f}°C')
print(f'\nDec-Jan chill days: {((winter_window["date"] >= "2025-12-01") & (winter_window["date"] <= "2026-01-31") & (winter_window["tmean_c"] <= 5.0)).sum()}')
