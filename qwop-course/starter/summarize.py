"""python summarize.py runs/某次评估：汇总全部回合，不只看最好的一次。"""
import argparse
import json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('evaluation', type=Path)
p.add_argument('--target', type=float, default=10.0)
args = p.parse_args()
data = pd.read_csv(args.evaluation / 'episodes.csv')
result = {
    'episodes': len(data),
    'mean_distance_m': float(data.distance.mean()),
    'median_distance_m': float(data.distance.median()),
    'min_distance_m': float(data.distance.min()),
    'max_distance_m': float(data.distance.max()),
    'target_distance_m': args.target,
    'target_reached_episodes': int((data.distance >= args.target).sum()),
    'target_reached_fraction': float((data.distance >= args.target).mean()),
    'mean_steps': float(data.steps.mean()),
    'game_end_episodes': int((data.reason == 'game_end').sum()),
    'limit_episodes': int((data.reason == 'limit').sum()),
    'note': 'Distance threshold measures forward progress, not upright walking; inspect the game images.'
}
if 'walking_success' in data:
    result.update({
        'walking_success_episodes': int(data.walking_success.sum()),
        'mean_nonfoot_fraction': float(data.nonfoot_fraction.mean()),
        'mean_knee_low_fraction': float(data.knee_low_fraction.mean()),
        'mean_upright_fraction': float(data.upright_fraction.mean()),
        'mean_support_switches': float(data.support_switches.mean()),
    })
text = json.dumps(result, ensure_ascii=False, indent=2)
(args.evaluation / 'summary.json').write_text(text, encoding='utf-8')
print(text)
