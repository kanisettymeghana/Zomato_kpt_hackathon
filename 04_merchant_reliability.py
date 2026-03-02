import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/outputs/df_with_pos_kds.csv')

ts_cols = ['order_timestamp','accepted_time','actual_ready_time',
           'merchant_marked_ready_time','rider_assigned_time',
           'rider_arrival_time','pickup_time']
for c in ts_cols:
    df[c] = pd.to_datetime(df[c], errors='coerce')

print(f"loaded {len(df)} orders | {df['restaurant_id'].nunique()} restaurants\n")

df['actual_prep_min'] = (
    df['actual_ready_time'] - df['accepted_time']
).dt.total_seconds() / 60

df['merchant_for_error_min'] = (
    df['merchant_marked_ready_time'] - df['actual_ready_time']
).dt.total_seconds() / 60

df['rider_wait_min'] = (
    df['pickup_time'] - df['rider_arrival_time']
).dt.total_seconds() / 60

df['sync_bias_flag'] = df['merchant_for_error_min'] < -3

print("=" * 62)
print("STEP 1: computing per-merchant signal reliability metrics")
print("=" * 62)

merchant_metrics = df.groupby('restaurant_id').agg(
    order_count       = ('order_id', 'count'),
    #FOR accuracy
    avg_for_error     = ('merchant_for_error_min', lambda x: x.abs().mean()),
    #sync bias
    sync_bias_rate    = ('sync_bias_flag', 'mean'),
    #consistency (prep time std)
    prep_variability  = ('actual_prep_min', 'std'),
    #rider wait
    avg_rider_wait    = ('rider_wait_min', 'mean'),
    #bias rate
    bias_rate         = ('is_for_biased', 'mean'),
).reset_index().fillna(0).round(4)

print(f"\n  merchants analysed         : {len(merchant_metrics)}")
print(f"  avg FOR error per merchant : {merchant_metrics['avg_for_error'].mean():.2f} min")
print(f"  avg sync bias rate         : {merchant_metrics['sync_bias_rate'].mean()*100:.1f}%")
print(f"  avg prep variability       : {merchant_metrics['prep_variability'].mean():.2f} min")
print(f"  avg rider wait             : {merchant_metrics['avg_rider_wait'].mean():.2f} min")
print(f"  avg bias rate              : {merchant_metrics['bias_rate'].mean()*100:.1f}%")

print(f"\n{'=' * 62}")
print("STEP 2: computing Merchant Signal Reliability Index (MSRI)")
print("=" * 62)

def min_max_scale(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

merchant_metrics['for_norm']    = min_max_scale(merchant_metrics['avg_for_error'])
merchant_metrics['sync_norm']   = min_max_scale(merchant_metrics['sync_bias_rate'])
merchant_metrics['var_norm']    = min_max_scale(merchant_metrics['prep_variability'])
merchant_metrics['wait_norm']   = min_max_scale(merchant_metrics['avg_rider_wait'])
merchant_metrics['bias_norm']   = min_max_scale(merchant_metrics['bias_rate'])

merchant_metrics['MSRI'] = (
    1 - (
        0.30 * merchant_metrics['for_norm']
      + 0.25 * merchant_metrics['sync_norm']
      + 0.20 * merchant_metrics['var_norm']
      + 0.15 * merchant_metrics['wait_norm']
      + 0.10 * merchant_metrics['bias_norm']
    )
).clip(0, 1).round(4)

print(f"\n  MSRI distribution:")
print(f"    mean MSRI  : {merchant_metrics['MSRI'].mean():.3f}")
print(f"    min MSRI   : {merchant_metrics['MSRI'].min():.3f}")
print(f"    max MSRI   : {merchant_metrics['MSRI'].max():.3f}")
print(f"    P25        : {merchant_metrics['MSRI'].quantile(0.25):.3f}")
print(f"    P75        : {merchant_metrics['MSRI'].quantile(0.75):.3f}")

# reliability tiers
merchant_metrics['reliability_tier'] = pd.cut(
    merchant_metrics['MSRI'],
    bins=[0, 0.4, 0.6, 0.8, 1.0],
    labels=['low', 'medium', 'high', 'excellent']
)

tier_dist = merchant_metrics['reliability_tier'].value_counts().sort_index()
print(f"\n  reliability tier distribution:")
for tier, count in tier_dist.items():
    pct = count / len(merchant_metrics) * 100
    bar = '█' * int(pct/3)
    print(f"    {str(tier):<12} {count:>4} merchants ({pct:.1f}%) {bar}")

print(f"\n{'=' * 62}")
print("STEP 3: most and least reliable merchants")
print("=" * 62)

print(f"\n  TOP 5 most reliable:")
print(f"  {'restaurant':<12} {'MSRI':>6} {'FOR err':>9} {'bias%':>8} {'wait':>8}")
print(f"  {'-'*48}")
top5 = merchant_metrics.nlargest(5, 'MSRI')
for _, row in top5.iterrows():
    print(f"  {row['restaurant_id']:<12} {row['MSRI']:>6.3f} "
          f"{row['avg_for_error']:>8.2f}m {row['bias_rate']*100:>7.1f}% "
          f"{row['avg_rider_wait']:>7.2f}m")

print(f"\n  BOTTOM 5 least reliable:")
print(f"  {'restaurant':<12} {'MSRI':>6} {'FOR err':>9} {'bias%':>8} {'wait':>8}")
print(f"  {'-'*48}")
bot5 = merchant_metrics.nsmallest(5, 'MSRI')
for _, row in bot5.iterrows():
    print(f"  {row['restaurant_id']:<12} {row['MSRI']:>6.3f} "
          f"{row['avg_for_error']:>8.2f}m {row['bias_rate']*100:>7.1f}% "
          f"{row['avg_rider_wait']:>7.2f}m")

print(f"\n{'=' * 62}")
print("STEP 4: how MSRI changes system behaviour")
print("=" * 62)

print(f"""
  MSRI > 0.8  (excellent)
    → trust FOR signal fully
    → minimal bias correction applied
    → tight ETA window shown to customer
    → standard dispatch timing

  MSRI 0.6-0.8 (high)
    → trust FOR signal mostly
    → light bias correction
    → normal ETA window

  MSRI 0.4-0.6 (medium)
    → moderate distrust
    → stronger bias correction
    → slightly wider ETA window
    → small dispatch buffer added

  MSRI < 0.4  (low)
    → high distrust
    → maximum bias correction
    → wide ETA window (honest uncertainty)
    → larger dispatch buffer
    → flagged for IoT deployment priority
""")

print("=" * 62)
print("STEP 5: MSRI vs dataset merchant_reliability_score")
print("=" * 62)

df = df.merge(
    merchant_metrics[['restaurant_id','MSRI','reliability_tier']],
    on='restaurant_id', how='left'
)

corr = df['MSRI'].corr(df['merchant_reliability_score'])
print(f"\n  correlation with existing score : {corr:.3f}")
print(f"  our MSRI mean                   : {df['MSRI'].mean():.3f}")
print(f"  dataset score mean              : {df['merchant_reliability_score'].mean():.3f}")
print(f"\n  → our MSRI is built from objective components")
print(f"  → not just a single number but a traceable breakdown")
print(f"  → each component can be improved independently")

print(f"\n{'=' * 62}")
print("STEP 6: MSRI-adjusted KPT prediction")
print("=" * 62)

tier_baseline = {1: 5, 2: 10, 3: 18, 4: 28}
df['tier_baseline_kpt'] = df['complexity_tier'].map(tier_baseline)

df['msri_adjusted_kpt'] = (
    df['expected_kpt_min']    * df['MSRI']
  + df['tier_baseline_kpt']   * (1 - df['MSRI'])
).round(2)

df['actual_prep_min'] = (
    df['actual_ready_time'] - df['accepted_time']
).dt.total_seconds() / 60

base_error  = (df['expected_kpt_min'] - df['actual_prep_min']).abs().mean()
msri_error  = (df['msri_adjusted_kpt'] - df['actual_prep_min']).abs().mean()

print(f"\n  KPT prediction error:")
print(f"    without MSRI adjustment : {base_error:.2f} min")
print(f"    with MSRI adjustment    : {msri_error:.2f} min")
print(f"    improvement             : {(base_error-msri_error)/base_error*100:.1f}%")

#simulations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Merchant Signal Reliability Index (MSRI)\n'
             'Zomato KPT Hackathon (v3 dataset)',
             fontsize=13, fontweight='bold')

RED = '#E23744'; GREEN = '#2ECC71'; AMBER = '#F39C12'
BLUE = '#3498DB'; DARK = '#2C3E50'

ax = axes[0,0]
ax.hist(merchant_metrics['MSRI'], bins=30, color=BLUE, alpha=0.75, edgecolor='white')
ax.axvline(merchant_metrics['MSRI'].mean(), color=RED, lw=2,
           label=f"mean={merchant_metrics['MSRI'].mean():.3f}")
for thresh, col, lbl in [(0.4,'red','low'), (0.6,'orange','med'), (0.8,'green','high')]:
    ax.axvline(thresh, color=col, lw=1.2, linestyle='--', alpha=0.7)
ax.set_title('MSRI Distribution across 290 Restaurants', fontweight='bold')
ax.set_xlabel('MSRI score')
ax.set_ylabel('number of restaurants')
ax.legend()

ax = axes[0,1]
tier_vals = tier_dist.values
tier_lbls = [str(l) for l in tier_dist.index]
colors_p = [RED, AMBER, BLUE, GREEN]
ax.pie(tier_vals, labels=tier_lbls,
       colors=colors_p, autopct='%1.1f%%',
       startangle=90, textprops={'fontsize': 10})
ax.set_title('Merchant Reliability Tiers', fontweight='bold')

ax = axes[0,2]
components = ['FOR\naccuracy', 'Sync\nbias', 'Consistency', 'Rider\nwait', 'Bias\nrate']
top_vals = [1 - top5['for_norm'].mean(),
            1 - top5['sync_norm'].mean(),
            1 - top5['var_norm'].mean(),
            1 - top5['wait_norm'].mean(),
            1 - top5['bias_norm'].mean()]
bot_vals = [1 - bot5['for_norm'].mean(),
            1 - bot5['sync_norm'].mean(),
            1 - bot5['var_norm'].mean(),
            1 - bot5['wait_norm'].mean(),
            1 - bot5['bias_norm'].mean()]
x = np.arange(len(components))
ax.bar(x - 0.2, top_vals, 0.35, color=GREEN, alpha=0.8, label='Top 5 merchants')
ax.bar(x + 0.2, bot_vals, 0.35, color=RED,   alpha=0.8, label='Bottom 5 merchants')
ax.set_xticks(x); ax.set_xticklabels(components, fontsize=8)
ax.set_title('MSRI Components:\nTop vs Bottom Merchants', fontweight='bold')
ax.set_ylabel('component score (higher=better)')
ax.legend(fontsize=8)

ax = axes[1,0]
ax.scatter(merchant_metrics['MSRI'], merchant_metrics['avg_for_error'],
           alpha=0.5, s=30, color=BLUE)
ax.set_title('MSRI vs FOR Error per Merchant\nlow MSRI = high error', fontweight='bold')
ax.set_xlabel('MSRI score')
ax.set_ylabel('avg FOR error (min)')

ax = axes[1,1]
ax.scatter(merchant_metrics['MSRI'], merchant_metrics['avg_rider_wait'],
           alpha=0.5, s=30, color=AMBER)
ax.set_title('MSRI vs Rider Wait Time\nlow MSRI = longer rider wait', fontweight='bold')
ax.set_xlabel('MSRI score')
ax.set_ylabel('avg rider wait (min)')

ax = axes[1,2]
ax.bar(['Without MSRI\nadjustment', 'With MSRI\nadjustment'],
       [base_error, msri_error],
       color=[RED, GREEN], alpha=0.85, edgecolor='white')
ax.set_title(f'KPT Prediction Error\n'
             f'{(base_error-msri_error)/base_error*100:.1f}% improvement with MSRI',
             fontweight='bold')
ax.set_ylabel('mean absolute error (min)')
for i, val in enumerate([base_error, msri_error]):
    ax.text(i, val + 0.05, f'{val:.2f}m', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/04_merchant_reliability.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nsaved plot ✓")

df.to_csv('/mnt/user-data/outputs/df_with_msri.csv', index=False)
merchant_metrics.to_csv('/mnt/user-data/outputs/merchant_msri_scores.csv', index=False)
print("saved df_with_msri.csv ✓")
print("saved merchant_msri_scores.csv ✓")
