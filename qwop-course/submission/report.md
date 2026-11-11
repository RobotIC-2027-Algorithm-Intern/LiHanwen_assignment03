# 我的 QWOP 训练实验报告

姓名/小组：lihanwen（单人）
实验日期：2026-10-31 训练与录像；2026-11-11 补同预算基线、采样对照与材料整理

---

## 目标与假设

**目标**：训练一个控制 Q/W/O/P 的 QWOP 策略，让它不只是"把距离刷到 100 米"，而是尽量保持直立姿态走到终点；并解释它为什么做出这种行为。

**先写假设，再看结果**：

- **H1（奖励偷懒）**：只奖励前向位移，最优解是"低姿态贴地滑行"。跪着爬单位动作能换更多距离，而且重心低更不容易摔。预测：纯位移奖励训练越久，跪行越明显。
- **H2（门控单独不够）**：给前进奖励加高度门控（`height_gate`）只是给低姿态时的位移奖励打折，不会改变"跪着更划算"的排序。预测：只加门控，距离可能更高，但姿态不会直立。
- **H3（膝部惩罚有效但有代价）**：再加膝部过低惩罚（`knee_low_penalty` 小权重），会把姿态抬起来，代价是距离下降。预测：距离 < 纯位移，`upright_fraction` 明显上升。
- **H4（惩罚过头的两种坏结局）**：膝部惩罚加太大，前进被压住（距离暴跌）；门控加太大，策略会选择"站着不动"（对应参考视频 03 讨论的失败模式）。预测：前者距离下降但仍前进，后者出现低距离、高 upright 的回合。

---

## 代码来源与训练设置

| 项 | 内容 |
| --- | --- |
| 采用路线 | A｜课堂 `starter/`（Gymnasium + Stable-Baselines3 PPO + PyTorch + 浏览器原游戏），在此基础上改训练脚本 |
| 仓库 | `github.com/RobotIC-2027-Algorithm-Intern/Assignment-3-Reinforcement-Learning`，本次代码提交 **`a937568`**（starter 原始提交 `bd8bac4`） |
| 预训练模型 | **没有**，全部从零训练；未使用老师参考策略或上游旧模型 |
| 系统 | Ubuntu 22.04.5 LTS，Intel Core Ultra 7 255HX，Selenium 驱动连接 Google Chrome 155.0.8059.84，浏览器无头模式 |
| Python 环境 | Conda `torch`（**不是**任务书建议的 `qwop-rl`）：Python 3.11.17，torch 2.14.1+cu130，stable-baselines3 2.7.1，gymnasium 1.2.3，selenium 4.52.0，pandas 3.1.0，numpy 2.4.6，tensorboard 2.21.0，完整清单见 `train_logs/<实验>/packages.txt` |
| 计算设备 | CPU（`--device cpu`）。本机 `torch.cuda.is_available()` 为 False（NVIDIA 驱动不可用），所以**实际没有用 GPU** |
| 算法 | PPO，MlpPolicy，策略/价值网络各两层 128 单元，n_steps 1024，batch_size 64，n_epochs 10，lr 3e-4，γ 0.995，ent_coef 0.02，seed 42 |
| 观测/动作 | 未修改：游戏身体状态数值向量；11 个按键组合（`qw,qo,qp,q,wo,wp,w,op,o,p,无`） |
| 标准化 | VecNormalize（观测 + 奖励），与模型一起保存为 `normalize.pkl` |
| 每动作游戏更新 | 3 次（`frame_skip=3`，约 0.05 秒/动作）；`--fast` 固定物理步长、不按真实时间 |
| 回合上限 | 训练 1500 动作/回合；评估 3000 动作/回合 |
| 实际采样步数 | 正式实验每次 **131072**（训练日志 `total_timesteps: 131072`，用时 195 s，约 669 fps）；起步基线 32768；流程检查 2048 |
| 评估条件 | 10 回合、seed 100、`start_jitter` 0~3、确定性策略（取 argmax）、`max_steps` 3000、headless + direct-cdp，**所有实验完全一致** |

训练命令（在 `starter/` 目录、激活 `torch` 环境后执行，完整清单见本文附录 A）：

```
python course.py train --out runs/<实验名> --steps 131072 --seed 42 --reward configs/<配置>.json
python course.py evaluate --run runs/<实验名> --out runs/<实验名>-eval
python course.py record --run runs/<实验名> --out videos/<实验名>
```

---

## 修改清单（未修改的项目明确写"未修改"）

| 文件及位置 | 修改前 | 修改后 | 为什么改 | 证据与效果 |
| --- | --- | --- | --- | --- |
| `starter/requirements.txt` 第 2 行 | `torch==2.7.1` | 删除该固定版本 | 实际安装的是 cu130 轮子，固定 CPU 版会冲突 | 依赖由 stable-baselines3 解析；`packages.txt` 记录 torch 2.14.1+cu130 |
| `starter/train.py` 第 43-44 行 | `batch_size=64` 写死在 PPO 构造里；`n_epochs` 用 SB3 默认值 | 新增 `--batch-size`（默认 64）、`--epochs`（默认 10）并传入 PPO | 便于调整更新开销 | 默认值与原来一致，因此各实验仍可比；`config.json` 记录两个值 |
| `starter/train.py` 第 52-54、62-68、78-81、98-103 行 | `PPO(..., device='auto')`、`PPO.load(..., device='auto')` | 新增 `--device {auto,cpu,cuda}`（默认 cpu），显式传给 PPO；CPU 时 `torch.set_num_threads(1)`，CUDA 时开 TF32；请求 cuda 但无 CUDA 时报错 | 显式控制设备，避免 `auto` 在不同机器上行为不同 | 五个 131072 步实验的 `config.json` 均为 `"device": "cpu"`（32k 起步基线与 2048 步检查在加 `--device` 之前运行，配置里没有该字段）；本机无可用 CUDA，故未启用 GPU |
| `starter/course.py` 第 22-23、45 行 | 无设备参数 | 新增 `--device` 并透传给 `train.py` | 让课堂封装也能选设备 | `python course.py train --device cuda` 可切换 |
| 观察 / 动作集合 | — | **未修改** | — | 仍是 11 个按键组合、身体状态观测 |
| 策略网络 / 算法 / 损失 | — | **未修改** | — | PPO + 两层 128 单元的 MLP，超参见上表 |
| 奖励公式 `rewards.py` | — | **未修改** | — | 只改奖励配置数值（见下） |
| 环境物理、起点、终止条件 | — | **未修改** | — | 未改重力、碰撞、起始姿态；距离只在相同条件下比较 |
| 评估流程 / 录像脚本 | — | **未修改** | — | `course.py evaluate/record` 原样使用 |

**奖励配置说明**：`configs/my_reward.json` 是任务书 §7 要求的副本，但本次实验**没有在它上面改**，而是每次消融新建一个 `exp*.json`，这样每个配置能和对应实验一一对应、不会互相覆盖。各配置只改一个（或明示的两个）参数：

| 配置 | 与纯位移基线的差异 |
| --- | --- |
| `distance_only.json` | 原始：`forward=1.0`，其余全 0（只算有符号位移） |
| `exp1_gate06.json` | `height_gate` 0 → 0.6：低姿态时前进奖励按 `1-0.6×(1-height)` 打折 |
| `exp2b_gate_knee01.json` | 同 exp1，再加 `knee_low_penalty=0.1`（`target_knee_y=6.0`） |
| `exp3a_knee05.json` | 同 exp1，膝部惩罚改成 0.5 |
| `exp4_gate09.json` | `height_gate=0.9` + `knee_low_penalty=0.1` |

未启用的配置（只有设计、没有结果，报告不当作实验）: `exp2_knee005.json`、`exp3_nonfoot02.json`、`exp4b_posture03.json`。

---

## 对照与结果

### 1. 主表（全部 10 回合、同评估条件，单位：米 / 比例）

| 实验 | 奖励改动 | 预算 | 距离均值 | 最小 / 最大 | 到 10 m | upright | nonfoot | knee_low | walking_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-32k | 纯位移 | 32768 | 54.37 | 1.4 / 70.4 | 8/10 | 0.153 | 0.202 | 0.781 | 0/10 |
| **baseline-131k** | 纯位移 | 131072 | **106.24** | 100.8 / 117.1 | 10/10 | **0.005** | 0.344 | **0.989** | 0/10 |
| gate06 | 门控 0.6 | 131072 | 100.89 | 100.3 / 101.9 | 10/10 | 0.010 | 0.283 | 0.962 | 0/10 |
| **exp2b** | 门控 0.6 + 膝部 0.1 | 131072 | 66.36 | 8.5 / 101.9 | 8/10 | **0.791** | **0.022** | 0.111 | 0/10 |
| exp3a | 门控 0.6 + 膝部 0.5 | 131072 | 44.31 | 13.5 / 102.4 | 10/10 | 0.761 | 0.020 | 0.096 | 0/10 |
| exp4 | 门控 0.9 + 膝部 0.1 | 131072 | 34.57 | 2.6 / 100.4 | 5/10 | 0.768 | 0.014 | 0.102 | 0/10 |

数据位置：`eval/<实验>-summary.json`、`eval/<实验>-episodes.csv`（每行一个回合）、训练曲线 `train_logs/<实验>/curves.png` 与 `tb/`。

### 2. 逐条对照说明

- **H1 成立，而且比预想更彻底**。起步预算 32768 的纯位移基线只跑到 54.37 m（因为还在早期训练），补到同预算 131072 后，距离升到 106.24 m，但 `knee_low=0.989`、`upright=0.005`——**几乎整段都是膝部贴地**。也就是说"奖励一高就学会跪着走"不是训练不足的假象，而是纯位移奖励在 4 倍预算下的稳定解。另外有 2 个回合达到 3000 动作上限仍未结束（`limit`），最远 117.1 m，说明到达 100 m 后游戏不会自动停下。
- **H2 成立**。只加门控（gate06）距离 100.89 m，但 `knee_low=0.962`、`upright=0.010`，和纯位移基线几乎一样跪。门控只是把低姿态的前进奖励按比例打折，跪着走仍然是最划算的动作。
- **H3 成立**。门控不变、只加膝部惩罚 0.1（exp2b），`upright` 从 0.010 抬到 0.791，`nonfoot` 从 0.283 降到 0.022，代价是距离从 100.89 掉到 66.36 m。这个"距离换姿态"的权衡是 H3 预测的。
- **H4 两半都成立**。膝部惩罚 0.5（exp3a）把距离压到 44.31 m，虽然 10/10 都过了 10 m，但前进被明显抑制；门控 0.9 + 膝部 0.1（exp4）出现 5 个回合只有 2.6~3.2 m，这些回合 `upright≈0.86`、`knee_low=0`——策略选择**站住不动**，正是参考视频 03 / 05 讨论的"奖励设计导致放弃前进"。

### 3. 任务书要求回答的三个问题

**① 距离和姿态是否同时改善？——部分，没有同时。**
- exp2b 相对 gate06：`upright` 0.010 → 0.791、`knee_low` 0.962 → 0.111，但距离 100.89 → 66.36。
- exp2b 相对 baseline-131k：姿态从 0.005 抬到 0.791，距离从 106.24 掉到 66.36。
- 也就是在所有能直立的实验里，距离都付出了代价；没有一个实验同时做到"满 100 m + 直立"。
- 证据：`eval/exp2b-...-episodes.csv` 第 2 回合 101.9 m、`upright=0.845`；`eval/gate06-...-episodes.csv` 第 2 回合 100.3 m、`upright=0.003`。

**② 有没有"奖励/距离变高但目标没实现"？——有，而且是这次实验最重要的发现。**
- baseline-131k 和 gate06 都超过 100 m，但 `knee_low` 0.99 / 0.96、`upright` 0.005 / 0.010，`walking_success` 仍然是 0/10。距离数字漂亮，本质是跪着滑过去。
- 视频 `final_demo_gate06_kneeling.mp4` 是同一现象的连续录像：1068 个动作走到 100.3 m，画面里躯干几乎贴地。
- 所以任务书说的"走得远"和"像人一样走"确实不是同一个指标，这次用数据复核了这一点。

**③ 最差回合为什么值得关注？**
- exp2b 的第 3、8 回合数值完全相同（8.5 m、`knee_low=0.205`、`upright=0.641`），说明这个失败是**确定性、可复现的**，不是随机噪声；换言之策略存在一个"起步就摔"的固定失败模式，换 10 次起点仍会遇到。
- exp4 的 2.6 m 回合（第 0/1/9 回合，`knee_low=0`）不是摔倒，而是"站着不动"——最差回合里混着两种完全不同的失败，只有看 `steps.csv` 的按键和 `gait.png` 才能区分。
- exp2b 的这两次早期摔倒也解释了为什么它的距离均值（66.36）低于中位数（73.65）：均值被尾部拉低。

### 4. 评估数据的两个必须说明的现象

**（a）确定性策略下 10 个回合有重复轨迹。**
exp2b 的 10 个回合实际只有 8 条不同轨迹（第 1 与第 9 完全一致、第 3 与第 8 完全一致）；exp4 只有 5 条（第 0/1/9 三个回合完全一致，第 2/4、第 3/8、第 5/6 各自成对）；baseline-131k 与 gate06 也各有两对重复（第 1/9、第 2/4）。原因是评估用确定性策略 + 只有 0~3 步起始等待——少量扰动不足以把策略推到另一条轨迹。**所以不能把这 10 个回合当成 10 种独立情况**，有效样本比列数少。

**（b）训练日志的距离比评估低一个量级（exp2b：16.2 m vs 66.36 m）。**
训练时 PPO 按策略概率**采样**动作，评估时取**最大概率**动作。QWOP 是典型的"失之毫厘"游戏，直立姿态对动作扰动比跪行敏感得多。做了对照验证：

| 模型 | 确定性评估（10 回合均值） | 采样评估（5 回合均值） | 变化 |
| --- | --- | --- | --- |
| exp2b（直立） | 66.36 m | 27.46 m（52.7/29.8/12.7/25.1/17.0） | **−59%** |
| gate06（跪行） | 100.89 m | 89.42 m（100.2/46.6/100.3/100.0/100.0） | −11% |

采样评估命令：
`python evaluate.py --model runs/<实验>/model.zip --out runs/<实验>-stochastic-eval --episodes 5 --max-steps 3000 --seed 100 --stochastic --headless --direct-cdp`

结论：训练-评估差距主要来自"采样 vs 确定性"，而且**越直立的策略越脆弱**。这既是 exp2b 的训练曲线看起来"学不会"的原因，也说明跪行策略的鲁棒性有一部分来自姿态本身。数据不是错的。

### 5. 尚不能确定的原因与下一步验证

- exp2b 的 `knee_low≈0.111` 仍高于验收阈值 0.05，`walking_success` 仍是 0/10：卡在膝部高度，不是距离、不是触地、也不是 upright（0.791 只差 0.009）。下一步试 `knee_low_penalty` 0.1→0.2、或加少量 `posture_weight`（`exp4b_posture03.json` 已备好），预算保持 131072 看是否单调改善。
- 10 回合样本少且有重复轨迹：下一步每个关键配置跑 3 个训练种子 × 10 回合，比较"姿态提升"是否稳定出现。
- 直立策略对采样扰动敏感，训练时如果用更小 entropy 或更大 batch，能否缩小训练-评估差距，本次没有做对照，不能下结论。

---

## 最终 demo 对应关系

| 项 | 内容 |
| --- | --- |
| 视频文件 | `final_demo.mp4`（主片，直立行走）；`final_demo_gate06_kneeling.mp4`（对照，跪行） |
| 对应实验 | 主片：`runs/exp2b-gate-knee01-1031-110114`（门控 0.6 + 膝部惩罚 0.1）；对照：`runs/gate06-1031-104659`（门控 0.6） |
| 模型 | `model/exp2b-...-model.zip` + `normalize.pkl`；对照模型同目录 |
| 随机种子 / 起点 | 训练 seed 42；录像 seed 100，`start_jitter=0`（无起始等待） |
| 最大动作数 / 是否截断 | 3000；主片实际 **747 步**，`game_end` 自然结束，**未截断**；对照片 1068 步，同样未截断 |
| 为什么选这一回合 | 规则：在"距离 ≥ 10 m 且姿态最接近直立"的模型里选 exp2b（`upright` 0.791，五个实验里最高；`nonfoot` 0.022，最低之一）。它的 101.9 m 是 10 回合评估中最好的一回合，**全部 10 回合数据已附**（`eval/exp2b-...-episodes.csv`），没有只挑最好的而隐藏差的 |
| 视频时刻 | 视频开头 3 秒是标题卡，之后每步约 0.05 秒。主片：第 102 步（约 8.1 s）到 10 m，第 377 步（约 21.9 s）到 50 m，第 735 步（约 39.8 s）到 100 m，约 40.4 s 结束。对照片：第 132 步（约 9.6 s）10 m，第 550 步（约 30.5 s）50 m，第 1067 步（约 56.4 s）100 m |
| 代码 / 配置 / 日志 / 评估位置 | 代码 `a937568`；配置 `configs/exp2b_gate_knee01.json`；训练日志 `train_logs/exp2b-.../`；评估 `eval/exp2b-...-*` |

---

## 我学到的

- **策略不是按键脚本**。它看当前身体状态，给 11 个按键组合各一个概率；训练只是根据回报小幅调整这些概率。所以同一个模型在训练（按概率抽）和评估（取最大）时表现可以差很多。
- **奖励是每步打分，总回报是折扣累计**。改奖励必须重新训练才能检验，因为奖励公式不是策略。
- **回报上升不能证明会走路**。这次 baseline 和 gate06 的回报都不低、距离都过 100 m，但全是跪行；不同奖励下的总回报也不能横比（gate06 的回报和 exp2b 的回报没有可比性）。
- **"分数高但不像人走"先查三件事**：距离 vs `upright`/`knee_low`/`nonfoot` 的分解、最差回合的具体原因、以及录像画面。`walking_success` 是启发式，卡在哪个阈值要自己看。
- **失败回合里要分清"摔了"和"没走"**：exp2b 的 8.5 m 是摔，exp4 的 2.6 m 是站着不动，处理方式完全不同。
- 目前的策略还不会"像人一样跑"，只是"能直着身子挪到终点附近"。这是这次实验真实的结果，不装作学会了。

---

## 附录 A：完整命令（在 `starter/`、激活 `torch` 环境后执行）

暂存目录按 `runs/<实验名>`：

1. 流程检查（2048 步）
   `python course.py train --out runs/check --steps 2048`
   `python course.py record --run runs/check --out runs/check-video --max-steps 300`
2. 起步基线（32768 步）
   `python course.py train --out runs/baseline --steps 32768 --seed 42`
   `python course.py evaluate --run runs/baseline --out runs/baseline-eval`
3. 同预算纯位移基线（131072 步）
   `python course.py train --out runs/baseline-131k --steps 131072 --seed 42 --reward configs/distance_only.json`
   `python course.py evaluate --run runs/baseline-131k --out runs/baseline-131k-eval`
4. 门控 0.6
   `python course.py train --out runs/gate06-1031-104659 --steps 131072 --seed 42 --reward configs/exp1_gate06.json`
   `python course.py evaluate --run runs/gate06-1031-104659 --out runs/gate06-1031-104659-eval`
   `python course.py record --run runs/gate06-1031-104659 --out videos/gate06-1031-104659`
5. 门控 0.6 + 膝部 0.1（主片）
   `python course.py train --out runs/exp2b-gate-knee01-1031-110114 --steps 131072 --seed 42 --reward configs/exp2b_gate_knee01.json`
   `python course.py evaluate --run runs/exp2b-gate-knee01-1031-110114 --out runs/exp2b-gate-knee01-1031-110114-eval`
   `python course.py record --run runs/exp2b-gate-knee01-1031-110114 --out videos/exp2b-gate-knee01-1031-110114`
6. 门控 0.6 + 膝部 0.5
   `python course.py train --out runs/exp3a-knee05-1031-112623 --steps 131072 --seed 42 --reward configs/exp3a_knee05.json`
   `python course.py evaluate --run runs/exp3a-knee05-1031-112623 --out runs/exp3a-knee05-1031-112623-eval`
   `python course.py record --run runs/exp3a-knee05-1031-112623 --out videos/exp3a-knee05-1031-112623`
7. 门控 0.9 + 膝部 0.1
   `python course.py train --out runs/exp4-gate09-1031-113636 --steps 131072 --seed 42 --reward configs/exp4_gate09.json`
   `python course.py evaluate --run runs/exp4-gate09-1031-113636 --out runs/exp4-gate09-1031-113636-eval`
   `python course.py record --run runs/exp4-gate09-1031-113636 --out videos/exp4-gate09-1031-113636`
8. 采样对照（验证训练-评估差异）
   `python evaluate.py --model runs/exp2b-gate-knee01-1031-110114/model.zip --out runs/exp2b-stochastic-eval --episodes 5 --max-steps 3000 --seed 100 --stochastic --headless --direct-cdp`
   `python evaluate.py --model runs/gate06-1031-104659/model.zip --out runs/gate06-stochastic-eval --episodes 5 --max-steps 3000 --seed 100 --stochastic --headless --direct-cdp`

查看训练曲线：`python -m tensorboard.main --logdir runs`，浏览器打开 `http://localhost:6006`；主实验日志已随包放在 `train_logs/exp2b-.../tb/`。
