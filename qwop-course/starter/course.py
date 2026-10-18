"""面向课堂的简短命令；底层 train/evaluate 脚本仍可直接使用。"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description='QWOP：训练、评估和录制自己的策略')
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--steps', type=int, default=32768)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--reward', type=Path, default=ROOT / 'configs/distance_only.json')
    for command in ('evaluate', 'record'):
        e = sub.add_parser(command)
        e.add_argument('--run', type=Path, required=True, help='含 model.zip 的训练文件夹')
        e.add_argument('--out', type=Path, required=True, help='尚不存在的新文件夹')
        e.add_argument('--seed', type=int, default=100)
        e.add_argument('--max-steps', type=int, default=3000)
        if command == 'evaluate':
            e.add_argument('--episodes', type=int, default=10)
            e.add_argument('--start-jitter', type=int, default=3)
    a = p.parse_args()
    # Resolve before changing cwd so commands also work from a different folder.
    a.out = a.out.resolve()
    if a.out.exists():
        p.error('输出文件夹已存在，请换一个新名称，避免覆盖实验。')
    if a.command == 'train':
        if a.steps <= 0:
            p.error('steps 必须为正整数')
        run('train.py', '--out', a.out, '--steps', a.steps, '--seed', a.seed,
            '--reward-config', a.reward.resolve(), '--fast', '--frame-skip', 3,
            '--normalize', '--normalize-reward', '--direct-cdp', '--headless',
            '--rollout-steps', 1024, '--net-width', 128, '--entropy', .02,
            '--gamma', .995, '--max-steps', 1500)
        run('plot.py', a.out)
    else:
        model = a.run.resolve() / 'model.zip'
        if not model.exists():
            p.error(f'找不到自己训练的模型：{model}')
        flags = ['--model', model, '--out', a.out, '--seed', a.seed,
                 '--max-steps', a.max_steps, '--headless', '--direct-cdp']
        if a.command == 'evaluate':
            run('evaluate.py', *flags, '--episodes', a.episodes, '--start-jitter', a.start_jitter)
            run('summarize.py', a.out)
            run('plot.py', a.out)
        else:
            run('evaluate.py', *flags, '--episodes', 1, '--screenshots',
                '--canvas-only', '--frame-every', 2)
            run('make_video.py', a.out, '--out', a.out / 'final_demo.mp4',
                '--title', '我的 QWOP 训练结果', '--question', '观察身体姿态、距离变化和策略按键')
    print(f'完成，结果保存在：{a.out}')


if __name__ == '__main__':
    main()
