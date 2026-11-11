"""python train.py --steps 2048 --out runs/first"""
import argparse
import json
from pathlib import Path
import platform
from importlib.metadata import distributions
import runtime  # noqa: F401
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from qwop_env import QWOPEnv


class StopFileCallback(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self):
        return not self.path.exists()


def main():
    p = argparse.ArgumentParser(description='QWOP 教学训练，默认使用 CPU')
    p.add_argument('--steps', type=int, default=2048)
    p.add_argument('--out', type=Path, default=Path('runs/first'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--max-steps', type=int, default=300)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stats', type=Path, help='从 checkpoint 续训时指定匹配的标准化统计')
    p.add_argument('--fast', action='store_true', help='固定每动作6次游戏更新，加速训练')
    p.add_argument('--normalize', action='store_true', help='标准化身体观测；新模型训练使用')
    p.add_argument('--upright', action='store_true', help='加入低躯干惩罚与摔倒惩罚')
    p.add_argument('--reward-config', type=Path)
    p.add_argument('--frame-skip', type=int)
    p.add_argument('--direct-cdp', action='store_true', help='直连浏览器减少通信开销，物理与奖励不变')
    p.add_argument('--rollout-steps', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=64, help='PPO 小批量大小；加大可减少更新开销')
    p.add_argument('--epochs', type=int, default=10, help='PPO 每次采样的训练轮数；降低可提速')
    p.add_argument('--net-width', type=int, default=64)
    p.add_argument('--entropy', type=float, default=0.01)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--normalize-reward', action='store_true')
    p.add_argument('--idle-prior', type=float, default=0.0, help='新模型初始化时偏向不按键；不是预训练步态')
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--no-tensorboard', action='store_true', help='仅记录 CSV，跳过 TensorBoard')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu',
                   help='计算设备。本任务是小 MLP（71→128×2），实测 GPU 比 CPU 慢约 2 倍，默认 cpu；'
                        'auto 表示有 CUDA 就用 CUDA')
    args = p.parse_args()
    if args.frame_skip is None:
        config_path = args.resume.parent / 'config.json' if args.resume else None
        previous = json.loads(config_path.read_text(encoding='utf-8')) if config_path and config_path.exists() else {}
        args.frame_skip = previous.get('frame_skip', 6)
    if args.steps <= 0 or args.max_steps <= 0:
        p.error('步数必须为正数')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('检测不到可用的 CUDA 设备；可改用 --device cpu')
    args.device = device
    print(f'计算设备：{device}' + (f'（{torch.cuda.get_device_name(0)}）' if device == 'cuda' else ''))
    args.out.mkdir(parents=True, exist_ok=False)
    if args.reward_config:
        (args.out / 'reward.json').write_text(args.reward_config.read_text(encoding='utf-8'), encoding='utf-8')
    (args.out / 'config.json').write_text(json.dumps({**vars(args), 'python': platform.python_version(),
        'algorithm': 'PPO', 'reward': 'configured walking reward' if args.reward_config else 'signed torso displacement minus low torso and fall penalties' if args.upright else 'signed torso displacement', 'action_seconds': args.frame_skip / 60 if args.fast else 0.1},
        default=str, indent=2), encoding='utf-8')
    (args.out / 'packages.txt').write_text('\n'.join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in distributions())), encoding='utf-8')
    set_random_seed(args.seed)
    if device == 'cpu':
        torch.set_num_threads(1)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
    tb = None if args.no_tensorboard else str(args.out / 'tb')
    env = Monitor(QWOPEnv(headless=args.headless, max_steps=args.max_steps, fast=args.fast, upright=args.upright,
                        frame_skip=args.frame_skip, reward_config=args.reward_config, direct_cdp=args.direct_cdp,
                        curriculum=True),
                  str(args.out / 'episodes'), info_keywords=('distance', 'game_time', 'reason'))
    if args.normalize:
        vector = DummyVecEnv([lambda: env])
        if args.resume:
            stats = args.stats or args.resume.parent / 'normalize.pkl'
            env = VecNormalize.load(stats, vector)
        else:
            env = VecNormalize(vector, norm_obs=True, norm_reward=args.normalize_reward, clip_obs=10, gamma=args.gamma)
    model = None
    try:
        if args.resume:
            overrides = {'learning_rate': args.learning_rate, 'ent_coef': args.entropy} if args.learning_rate else {}
            model = PPO.load(args.resume, env=env, device=device, tensorboard_log=tb, **overrides)
        else:
            model = PPO('MlpPolicy', env, device=device, seed=args.seed, verbose=1,
                        n_steps=args.rollout_steps, batch_size=args.batch_size, n_epochs=args.epochs,
                        learning_rate=args.learning_rate or 3e-4, ent_coef=args.entropy, gamma=args.gamma,
                        policy_kwargs={'net_arch': dict(pi=[args.net_width]*2, vf=[args.net_width]*2)},
                        tensorboard_log=tb)
            if args.idle_prior:
                with torch.no_grad():
                    model.policy.action_net.bias[-1] += args.idle_prior
        callback = CheckpointCallback(2048, str(args.out / 'checkpoints'), name_prefix='ppo', save_vecnormalize=True)
        model.learn(total_timesteps=args.steps, callback=CallbackList([callback, StopFileCallback(args.out / 'STOP')]),
                    reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print('训练已停止，保存当前参数；当前尚未更新的采样不包含在参数中。')
    finally:
        if model is not None:
            model.save(args.out / 'model')
            if isinstance(env, VecNormalize):
                env.save(args.out / 'normalize.pkl')
        env.close()


if __name__ == '__main__':
    main()
