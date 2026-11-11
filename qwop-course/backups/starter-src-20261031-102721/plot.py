"""把 CSV 变成学生可直接看的曲线图；不需要运行网页服务。"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
args = p.parse_args()
monitor = args.run / 'episodes.monitor.csv'
if monitor.exists():
    data = pd.read_csv(monitor, comment='#')
    if data.empty:
        raise SystemExit('尚无完整回合，请继续训练后再画图')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(data['r'], alpha=.4, label='episode return')
    axes[0].plot(data['r'].rolling(10, min_periods=1).mean(), label='10-episode mean')
    axes[1].plot(data['distance'], label='distance (m)')
    axes[1].set_xlabel('Episode')
else:
    data = pd.read_csv(args.run / 'steps.csv')
    first = data[data.episode == data.episode.min()]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].plot(first['step'], first['distance'], label='distance (m)')
    cols = [c for c in first if c.startswith('p_')]
    im = axes[1].imshow(first[cols].T, aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_yticks(range(len(cols)), [c[2:] for c in cols])
    axes[1].set_xlabel('Step (first episode)')
    fig.colorbar(im, ax=axes[1], label='Action probability')
    axes[2].step(first['step'], first['action'], label='selected action')
    axes[2].set_yticks(range(len(cols)), [c[2:] for c in cols])
    if 'nonfoot_contact' in first:
        gait, ga = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        ga[0].plot(first.step, first.torso_y, label='Torso y (lower is higher body)')
        ga[0].axhline(0, color='red', ls='--', label='upright check: y <= 0')
        ga[1].plot(first.step, first.left_knee_y, label='left knee y')
        ga[1].plot(first.step, first.right_knee_y, label='right knee y')
        ga[1].axhline(6.5, color='red', ls='--', label='low-knee threshold')
        ga[2].step(first.step, first.nonfoot_contact, color='red', label='non-foot ground contact')
        ga[2].step(first.step, first.left_foot_contact, alpha=.6, label='left foot support')
        ga[2].step(first.step, first.right_foot_contact, alpha=.6, label='right foot support')
        ga[2].set_xlabel('Decision step (first episode)')
        for ax in ga:
            ax.legend(loc='upper right')
        gait.savefig(args.run / 'gait.png', dpi=150)
        plt.close(gait)
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
fig.savefig(args.run / 'curves.png', dpi=150)
print(args.run / 'curves.png')
