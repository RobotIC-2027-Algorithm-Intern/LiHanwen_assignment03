"""回放模型或随机策略，并导出每一步的按键概率。"""
import argparse
import base64
from io import BytesIO
import time
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import runtime  # noqa: F401
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from qwop_env import ACTIONS, QWOPEnv
from gait_metrics import walking_pass


def save_canvas_frame(env, path):
    for _ in range(8):
        encoded = env.js("window.__qwop.core.render(); return document.querySelector('canvas').toDataURL('image/png').split(',')[1]")
        data = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(data)) as frame:
            if not all(high-low < 5 for low, high in frame.convert('RGB').getextrema()):
                Path(path).write_bytes(data)
                return
        time.sleep(.03)
    raise RuntimeError('The rendered game canvas is blank; recording stopped without fabricating frames')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, help='不提供时使用随机策略作为基线')
    p.add_argument('--periodic-policy', type=Path, help='评估通过进化搜索学得的周期按键策略')
    p.add_argument('--out', type=Path, default=Path('runs/random'))
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--fast', action='store_true')
    p.add_argument('--realtime', action='store_true', help='强制使用真实时间模式，检查加速训练的策略能否迁移')
    p.add_argument('--stats', type=Path, help='指定与模型同步保存的观测标准化统计文件')
    p.add_argument('--stochastic', action='store_true', help='按策略概率采样，而非选择最大概率动作')
    p.add_argument('--max-steps', type=int, default=300)
    p.add_argument('--screenshots', action='store_true', help='每10步保存游戏截图；会降低速度')
    p.add_argument('--canvas-only', action='store_true', help='截图只记录原游戏画布，减少留白和文件体积')
    p.add_argument('--frame-every', type=int, default=10)
    p.add_argument('--start-jitter', type=int, default=0, help='每回合先随机等待0到指定步数，检查初始姿态变化下的表现')
    p.add_argument('--frame-skip', type=int)
    p.add_argument('--direct-cdp', action='store_true')
    p.add_argument('--reward-config', type=Path)
    p.add_argument('--target-distance', type=float, default=10.0)
    args = p.parse_args()
    if args.model and args.periodic_policy:
        p.error('Choose one policy type')
    phases = None
    if args.periodic_policy:
        policy = json.loads(args.periodic_policy.read_text(encoding='utf-8'))
        phases = policy['phases']
        if not phases or any(not (0 <= a < len(ACTIONS) and isinstance(n, int) and n > 0) for a,n in phases):
            p.error('Invalid periodic policy')
        args.frame_skip = args.frame_skip or policy['frame_skip']
        args.fast = True
    if args.episodes <= 0 or args.max_steps <= 0 or args.frame_every <= 0 or args.start_jitter < 0:
        p.error('回合数和步数必须为正数')
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    config = {}
    if args.model and (args.model.parent / 'config.json').exists():
        config = json.loads((args.model.parent / 'config.json').read_text(encoding='utf-8'))
        args.fast = args.fast or config.get('fast', False)
    args.frame_skip = args.frame_skip or config.get('frame_skip', 6)
    args.direct_cdp = args.direct_cdp or config.get('direct_cdp', False)
    if args.model and not args.reward_config and (args.model.parent / 'reward.json').exists():
        args.reward_config = args.model.parent / 'reward.json'
    stats = args.stats or (args.model.parent / 'normalize.pkl' if args.model else None)
    if (args.stats or config.get('normalize')) and (not stats or not stats.exists()):
        p.error('缺少匹配的 normalize.pkl；不能用未标准化的观测评估此模型')
    if args.realtime:
        args.fast = False
    (args.out / 'evaluation.json').write_text(json.dumps(vars(args), default=str, indent=2), encoding='utf-8')
    model = PPO.load(args.model, device='cpu') if args.model else None
    torch.set_num_threads(1)
    env = QWOPEnv(headless=args.headless, max_steps=args.max_steps, fast=args.fast,
                  frame_skip=args.frame_skip, reward_config=args.reward_config, direct_cdp=args.direct_cdp)
    normalizer = None
    if stats and stats.exists():
        normalizer = VecNormalize.load(stats, DummyVecEnv([lambda: env]))
        normalizer.training = False
    try:
        with (args.out / 'steps.csv').open('w', newline='', encoding='utf-8') as f, \
             (args.out / 'episodes.csv').open('w', newline='', encoding='utf-8') as g:
            steps = csv.writer(f)
            episodes = csv.writer(g)
            reward_names = ['forward', 'posture', 'knee_low', 'nonfoot', 'alive', 'fall', 'finish', 'step_cost']
            steps.writerow(['episode', 'step', 'action', 'keys', 'reward', 'distance', 'torso_angle', 'torso_y',
                            'left_knee_y', 'right_knee_y', 'nonfoot_contact', 'left_foot_contact', 'right_foot_contact',
                            'entropy'] + ['reward_' + k for k in reward_names] + ['p_' + (k or 'none') for k in ACTIONS])
            episodes.writerow(['episode', 'return', 'distance', 'steps', 'reason', 'nonfoot_fraction',
                               'knee_low_fraction', 'upright_fraction', 'support_switches', 'walking_success'])
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode)
                for _ in range(int(rng.integers(args.start_jitter + 1))):
                    obs, _, term, trunc, _ = env.step(len(ACTIONS) - 1)
                    if term or trunc:
                        raise RuntimeError('初始等待已导致回合结束，请减少 --start-jitter')
                total = 0.0
                nonfoot = low_knee = upright = switches = 0
                last_support = 0
                for step in range(args.max_steps):
                    if model:
                        with torch.no_grad():
                            inputs = normalizer.normalize_obs(obs) if normalizer else obs
                            tensor, _ = model.policy.obs_to_tensor(inputs)
                            probs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0]
                        action = int(rng.choice(len(ACTIONS), p=probs / probs.sum())) if args.stochastic else int(np.argmax(probs))
                    elif phases:
                        tick = step % sum(n for _, n in phases)
                        for selected, duration in phases:
                            if tick < duration:
                                action = selected
                                break
                            tick -= duration
                        probs = np.zeros(len(ACTIONS))
                        probs[action] = 1.0
                    else:
                        probs = np.full(len(ACTIONS), 1 / len(ACTIONS))
                        action = int(rng.choice(len(ACTIONS)))
                    obs, reward, term, trunc, info = env.step(action)
                    total += reward
                    nonfoot += info['nonfoot_contact']
                    low_knee += max(info['left_knee_y'], info['right_knee_y']) > 6.5
                    upright += info['torso_y'] <= 0.0
                    support = info['left_foot_contact'] - info['right_foot_contact']
                    if support:
                        switches += bool(last_support and support != last_support)
                        last_support = support
                    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
                    steps.writerow([episode, step, action, ACTIONS[action] or 'none', reward,
                                    info['distance'], info['torso_angle'], info['torso_y'],
                                    info['left_knee_y'], info['right_knee_y'], info['nonfoot_contact'],
                                    info['left_foot_contact'], info['right_foot_contact'], entropy,
                                    *[info.get('reward_terms', {'forward': reward}).get(k, 0.0) for k in reward_names], *probs])
                    if args.screenshots and (step % args.frame_every == 0 or term or trunc):
                        env.render()
                        path = str(args.out / f'episode-{episode}-step-{step}.png')
                        if args.canvas_only:
                            # Read the real rendered canvas directly, avoiding repeated
                            # WebDriver element scrolling/layout calls for every frame.
                            # WebGL clears its drawing buffer after presentation.
                            # Render and read in ONE JS task so a browser paint
                            # between two CDP calls cannot produce a black frame.
                            save_canvas_frame(env, path)
                        else:
                            env.driver.save_screenshot(path)
                    if term or trunc:
                        break
                n = step + 1
                walking = walking_pass(info['distance'], nonfoot / n, low_knee / n,
                                       upright / n, switches, args.target_distance)
                episodes.writerow([episode, total, info['distance'], n, info['reason'], nonfoot / n,
                                   low_knee / n, upright / n, switches, walking])
                f.flush()
                g.flush()
                print(f"回合 {episode}: 距离 {info['distance']:.2f} m，回报 {total:.2f}，{info['reason']}")
    finally:
        env.close()


if __name__ == '__main__':
    main()
