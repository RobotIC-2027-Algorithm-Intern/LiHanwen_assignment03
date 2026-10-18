# 课堂实现相对于原仓库的变化

本目录为单独实现，原仓库原样保留在 ../upstream/QWOP-RL。没有将原仓库替换后冒称为原版。

- 依赖：原版 Python 3.7、TensorFlow 1.14、Gym、stable_baselines；课堂版 Python 3.12 为 Ubuntu 目标、PyTorch、Gymnasium、Stable-Baselines3。
- 算法：课堂默认 PPO 两层 128 单元策略/价值网络，具体设置由 course.py 传给 train.py。原版提供的历史算法保留在原目录中。
- 接口：本地 HTTP 提供原游戏；仅在响应中暴露状态对象，读取身体状态并发送按键。快速模式每个策略动作推进固定 3 次原游戏更新；没有更改重力、碰撞和原游戏素材。
- 观察与动作：身体状态数值观测，躯干相对位置；11 个按键组合，见 qwop_env.py 的 ACTIONS。VecNormalize 标准化统计和模型一起保存。
- 奖励：新手基线只计算有符号躯干位移；rewards.py 支持姿态、膝部高度、触地、存活等分项，初始 configs/distance_only.json 不启用这些附加项。训练包含姿态课程缩放逻辑；修改相关奖励时，应同时阅读 qwop_env.py 的 pose_curriculum_scale。
- 记录：新增回合/逐步 CSV、动作概率、奖励分解、步态辅助指标、图表与实际游戏画面 MP4。
- 包装：DEFAULT_GAME 指向本包 ../upstream/QWOP-RL/game，或环境变量 QWOP_GAME_DIR；setup_ubuntu.sh 直接校验随包游戏而不重新下载。course.py 将常用命令封装为 train/evaluate/record；底层脚本仍可使用。

这份清单是课堂起点的实现说明。学生自己的修改必须另外记录，不应把以上适配当作自己的贡献。Ubuntu 安装脚本尚未在真实 Ubuntu 机器运行验证；已有浏览器端到端训练/录像验证来自 Windows。
