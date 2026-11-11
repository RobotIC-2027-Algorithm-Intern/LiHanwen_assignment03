"""学生主要修改这里或 configs/student.json；不必改 PPO 的实现。"""
import math


def walking_reward(dx, torso_y, state, config, failed=False, finished=False):
    """dx 为游戏内部位移，y 向下增大。返回总奖励及每项分解。"""
    height = max(0.0, min(1.0, (2.0 - torso_y) / 3.0))
    knee_y = max(state['left_knee_y'], state['right_knee_y'])
    pose_scale = state.get('pose_curriculum_scale', 1.0)
    if 'displacement_clip' in config:
        limit = config['displacement_clip']
        if limit <= 0:
            raise ValueError('displacement_clip must be positive')
        dx = max(-limit, min(limit, dx))
    terms = {
        'forward': config['forward_weight'] * dx * (1.0 - config.get('height_gate', 0.9) * (1.0 - height)),
        'posture': -pose_scale * config['posture_weight'] * max(torso_y - config['target_torso_y'], 0.0),
        'knee_low': -pose_scale * config['knee_low_penalty'] * max(knee_y - config['target_knee_y'], 0.0),
        'nonfoot': -config['nonfoot_contact_penalty'] * state['nonfoot_contact'],
        'alive': config['alive_bonus'] * height,
        'fall': -config['fall_penalty'] if failed else 0.0,
        'finish': config['success_bonus'] if finished else 0.0,
        'step_cost': -config.get('step_cost', 0.0),
    }
    total = sum(terms.values())
    if not math.isfinite(total):
        raise ValueError('奖励不是有限数值，请检查配置')
    return total, terms
