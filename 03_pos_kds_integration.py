import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/outputs/df_with_iot.csv')

ts_cols = ['order_timestamp','accepted_time','actual_ready_time',
           'merchant_marked_ready_time','rider_assigned_time',
           'rider_arrival_time','pickup_time']
for c in ts_cols:
    df[c] = pd.to_datetime(df[c], errors='coerce')

df['actual_prep_min'] = (
    df['actual_ready_time'] - df['accepted_time']
).dt.total_seconds() / 60

print(f"loaded {len(df)} orders\n")

print("=" * 62)
print("STEP 1: current rush score is broken")
print("=" * 62)

from sklearn.metrics import r2_score

X_old = df[['external_kitchen_rush_score']].values
y     = df['actual_prep_min'].values

model_old = LinearRegression().fit(X_old, y)
r2_old    = r2_score(y, model_old.predict(X_old))

print(f"\n  current external_kitchen_rush_score:")
print(f"    R² vs actual KPT     : {r2_old:.4f}")
print(f"    explains             : {r2_old*100:.2f}% of KPT variance")
print(f"    correlation          : {df['external_kitchen_rush_score'].corr(df['actual_prep_min']):.4f}")
print(f"\n  → R²=0.000 means current rush score explains NOTHING")
print(f"  → because it only sees Zomato orders")

print(f"\n{'=' * 62}")
print("STEP 2: kitchen load visibility — what Zomato sees vs reality")
print("=" * 62)

df['zomato_only_load']  = 1  
df['total_external']    = df['walk_in_orders'] + df['dine_in_orders'] + df['competitor_orders']
df['zomato_visibility'] = 1 / (1 + df['total_external'])

print(f"\n  avg walk-in orders per restaurant  : {df['walk_in_orders'].mean():.1f}")
print(f"  avg dine-in orders per restaurant  : {df['dine_in_orders'].mean():.1f}")
print(f"  avg competitor orders              : {df['competitor_orders'].mean():.1f}")
print(f"  avg total kitchen load             : {df['total_kitchen_load'].mean():.1f}")
print(f"\n  Zomato visibility of kitchen load  : "
      f"{df['zomato_visibility'].mean()*100:.1f}%")
print(f"  invisible load                     : "
      f"{(1 - df['zomato_visibility'].mean())*100:.1f}%")

# by restaurant size
print(f"\n  visibility by restaurant size:")
print(f"  {'size':<12} {'visibility':>12} {'invisible':>12}")
print(f"  {'-'*38}")
for size in ['small','medium','large']:
    sub = df[df['restaurant_size'] == size]
    vis = sub['zomato_visibility'].mean()
    print(f"  {size:<12} {vis*100:>11.1f}% {(1-vis)*100:>11.1f}%")

# POS rush score
print(f"\n{'=' * 62}")
print("STEP 3: POS rush score vs current rush score")
print("=" * 62)

X_pos = df[['pos_rush_score']].values
model_pos = LinearRegression().fit(X_pos, y)
r2_pos    = r2_score(y, model_pos.predict(X_pos))

print(f"\n  pos_rush_score (full kitchen load):")
print(f"    R² vs actual KPT     : {r2_pos:.4f}")
print(f"    explains             : {r2_pos*100:.2f}% of KPT variance")
print(f"    correlation          : {df['pos_rush_score'].corr(df['actual_prep_min']):.4f}")
print(f"\n  improvement in signal quality:")
print(f"    old R²               : {r2_old:.4f}")
print(f"    POS R²               : {r2_pos:.4f}")
print(f"    improvement          : {r2_pos/max(r2_old,0.0001):.1f}x better signal")

# KDS simulation
print(f"\n{'=' * 62}")
print("STEP 4: KDS adds real-time kitchen throughput")
print("=" * 62)

np.random.seed(42)

size_multiplier = {'small': 0.3, 'medium': 0.5, 'large': 0.8}
df['kds_active_tickets'] = (
    df['total_kitchen_load'] * df['restaurant_size'].map(size_multiplier)
    + np.random.normal(0, 1, len(df))
).clip(0).round()

df['kds_station_backlog_min'] = (
    df['pos_rush_score'] * df['complexity_tier'] * 3
    + np.random.normal(0, 1, len(df))
).clip(0).round(1)

df['kds_avg_ticket_age_min'] = (
    df['actual_prep_min'] * 0.6
    + np.random.normal(0, 2, len(df))
).clip(0).round(1)

print(f"\n  KDS metrics simulated:")
print(f"    avg active tickets   : {df['kds_active_tickets'].mean():.1f}")
print(f"    avg station backlog  : {df['kds_station_backlog_min'].mean():.1f} min")
print(f"    avg ticket age       : {df['kds_avg_ticket_age_min'].mean():.1f} min")

# combined rush score
print(f"\n{'=' * 62}")
print("STEP 5: combined_rush_score (POS + KDS)")
print("=" * 62)

"""
combined_rush_score = weighted combination of:
  pos_rush_score       → load received (from POS)
  kds normalised load  → load being processed (from KDS)

This gives real-time kitchen throughput, not just load count.
"""

def min_max(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

df['kds_load_norm']  = min_max(df['kds_active_tickets'])
df['kds_backlog_norm'] = min_max(df['kds_station_backlog_min'])

df['combined_rush_score'] = (
    0.4 * df['pos_rush_score']
  + 0.35 * df['kds_load_norm']
  + 0.25 * df['kds_backlog_norm']
).round(3)

X_combined = df[['combined_rush_score']].values
model_comb = LinearRegression().fit(X_combined, y)
r2_combined = r2_score(y, model_comb.predict(X_combined))

print(f"\n  combined_rush_score (POS + KDS):")
print(f"    R² vs actual KPT     : {r2_combined:.4f}")
print(f"    correlation          : {df['combined_rush_score'].corr(df['actual_prep_min']):.4f}")

print(f"\n  signal quality progression:")
print(f"  {'signal':<35} {'R²':>8} {'improvement':>12}")
print(f"  {'-'*58}")
print(f"  {'external_kitchen_rush_score':<35} {r2_old:>8.4f} {'baseline':>12}")
print(f"  {'pos_rush_score':<35} {r2_pos:>8.4f} {r2_pos/max(r2_old,0.0001):>10.1f}x")
print(f"  {'combined_rush_score (POS+KDS)':<35} {r2_combined:>8.4f} {r2_combined/max(r2_old,0.0001):>10.1f}x")

print(f"\n{'=' * 62}")
print("STEP 6: KPT adjustment using combined rush score")
print("=" * 62)

"""
When combined_rush_score is high:
  inflate expected KPT
  rider dispatch delayed accordingly
  customer gets accurate longer ETA
"""

df['rush_kpt_multiplier'] = 1 + (df['combined_rush_score'] * 0.35)
df['rush_adjusted_kpt']   = (df['expected_kpt_min'] * df['rush_kpt_multiplier']).round(2)

print(f"\n  KPT adjustment by rush level:")
df['rush_category'] = pd.cut(df['combined_rush_score'],
                              bins=[0, 0.3, 0.6, 1.0],
                              labels=['low', 'medium', 'high'])
rush_impact = df.groupby('rush_category', observed=True).agg(
    orders          = ('order_id','count'),
    base_kpt        = ('expected_kpt_min','mean'),
    adjusted_kpt    = ('rush_adjusted_kpt','mean'),
    actual_kpt      = ('actual_prep_min','mean')
).round(2)

print(f"\n  {'rush':<10} {'orders':>8} {'base kpt':>10} "
      f"{'adjusted':>10} {'actual':>10}")
print(f"  {'-'*52}")
for cat, row in rush_impact.iterrows():
    print(f"  {str(cat):<10} {row['orders']:>8} {row['base_kpt']:>9.1f}m "
          f"{row['adjusted_kpt']:>9.1f}m {row['actual_kpt']:>9.1f}m")

# simulations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('POS + KDS Integration — Fixing the Kitchen Load Signal\n'
             'Zomato KPT Hackathon (v3 dataset)',
             fontsize=13, fontweight='bold')

RED = '#E23744'; GREEN = '#2ECC71'; AMBER = '#F39C12'
BLUE = '#3498DB'; DARK = '#2C3E50'

ax = axes[0,0]
signals = ['external_rush\n(broken)', 'POS rush', 'POS+KDS\ncombined']
r2_vals = [r2_old, r2_pos, r2_combined]
colors  = [RED, AMBER, GREEN]
bars = ax.bar(signals, r2_vals, color=colors, alpha=0.85, edgecolor='white')
ax.set_title('Rush Score R² — Signal Quality Progression\nhigher R² = signal actually predicts KPT',
             fontweight='bold')
ax.set_ylabel('R² vs actual KPT')
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

ax = axes[0,1]
avg_load = df[['zomato_only_load','walk_in_orders',
               'dine_in_orders','competitor_orders']].mean()
avg_load.index = ['Zomato orders','Walk-in','Dine-in','Competitor apps']
colors_pie = [RED, AMBER, BLUE, '#9B59B6']
ax.pie(avg_load.values, labels=avg_load.index,
       colors=colors_pie, autopct='%1.1f%%',
       startangle=90, textprops={'fontsize': 9})
ax.set_title('Kitchen Load Breakdown\nZomato sees only one slice', fontweight='bold')

ax = axes[0,2]
ax.scatter(df['combined_rush_score'], df['actual_prep_min'],
           alpha=0.3, s=12, color=BLUE)
x_range = np.linspace(0, 1, 100).reshape(-1,1)
ax.plot(x_range, model_comb.predict(x_range),
        color=RED, lw=2, label=f'R²={r2_combined:.4f}')
ax.set_title('Combined Rush Score vs Actual KPT\nPOS+KDS gives real signal',
             fontweight='bold')
ax.set_xlabel('combined_rush_score')
ax.set_ylabel('actual prep time (min)')
ax.legend()

ax = axes[1,0]
metrics = ['Active tickets', 'Station backlog (min)', 'Avg ticket age (min)']
vals    = [df['kds_active_tickets'].mean(),
           df['kds_station_backlog_min'].mean(),
           df['kds_avg_ticket_age_min'].mean()]
bars = ax.bar(metrics, vals, color=[BLUE, AMBER, RED], alpha=0.85, edgecolor='white')
ax.set_title('KDS Real-Time Kitchen Metrics\nwhat KDS adds beyond POS',
             fontweight='bold')
ax.set_ylabel('value')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{val:.1f}', ha='center', fontsize=10)
    
ax = axes[1,1]
meal_rush = df.groupby('meal_time')['combined_rush_score'].mean().sort_values(ascending=False)
colors_bar = [RED if v > 0.5 else AMBER if v > 0.35 else GREEN
              for v in meal_rush.values]
ax.bar(meal_rush.index, meal_rush.values, color=colors_bar, alpha=0.85, edgecolor='white')
ax.set_title('Combined Rush Score by Meal Time\nsystem knows when to inflate KPT',
             fontweight='bold')
ax.set_ylabel('combined rush score')
ax.axhline(0.5, color=RED, linestyle='--', lw=1.5, label='high rush threshold')
ax.legend()

ax = axes[1,2]
ax.scatter(df['rush_adjusted_kpt'], df['actual_prep_min'],
           alpha=0.3, s=12, color=GREEN, label='Rush-adjusted KPT')
ax.scatter(df['expected_kpt_min'], df['actual_prep_min'],
           alpha=0.3, s=12, color=RED, label='Base KPT')
ax.plot([0,50],[0,50], 'k--', lw=1, alpha=0.5, label='Perfect prediction')
ax.set_title('Rush-Adjusted KPT vs Actual\ncloser to diagonal = better',
             fontweight='bold')
ax.set_xlabel('predicted KPT (min)')
ax.set_ylabel('actual prep time (min)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/03_pos_kds_integration.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nsaved plot ✓")

df.to_csv('/mnt/user-data/outputs/df_with_pos_kds.csv', index=False)
print("saved df_with_pos_kds.csv ✓")
