# 提交说明（QWOP 强化学习作业）

姓名：lihanwen
日期：2026-11-21
代码提交：`b0b828c`（starter 原始提交 `bd8bac4`）

## 三项交付物在哪

| 交付物 | 文件 |
| --- | --- |
| 最终录像 | `final_demo.mp4`（主片，连续单回合，未截断）；`final_demo_gate06_kneeling.mp4`（对照片：跪行滑行 100 m） |
| 实验报告 | `report.md`（第一部分为作者自述实验过程，第二部分为环境设置、奖励公式、全部实验数据与结论） |
| 可复现材料 | `experiment_log.csv`、`eval/`、`train_logs/`、`model/`、`code/my_code_changes.patch` |

## 目录结构

```
submission/
├── final_demo.mp4                       主片：exp2b（门控 0.6 + 膝部惩罚 0.1），101.9 m
├── final_demo_gate06_kneeling.mp4       对照：gate06（只有门控 0.6），100.3 m，跪行
├── report.md                            实验报告（自述 + 技术附注）
├── experiment_log.csv                   一行一实验：种子、步数、配置、评估距离、视频
├── eval/                                各次评估的 episodes.csv + summary.json、录像逐步数据
├── train_logs/<实验>/                   训练日志：episodes.monitor.csv、curves.png、
│                                        config.json、reward.json、packages.txt、model.zip、
│                                        normalize.pkl、tb/
├── model/                               model.zip + normalize.pkl（主片与对照片各一份）
├── figures/                             报告引用的步态图与训练曲线
└── code/my_code_changes.patch           相对 starter 原始提交的源码与配置改动
```

`code/my_code_changes.patch` 的重建方式：`git diff bd8bac4 HEAD -- qwop-course/starter ':!qwop-course/starter/videos'`。录像与逐步数据放在 `starter/videos/` 里单独提交，不进 patch。

## 视频、模型、数据的对应关系

| 视频 | 训练目录 / 配置 | 模型 | 评估数据 | 说明 |
| --- | --- | --- | --- | --- |
| `final_demo.mp4` | `runs/exp2b-gate-knee01-1031-110114` / `configs/exp2b_gate_knee01.json` | `model/exp2b-...` | `eval/exp2b-...-*` | seed 42 训练；录像 seed 100、`start_jitter=0`、747 步、`game_end` 自然结束（未截断） |
| `final_demo_gate06_kneeling.mp4` | `runs/gate06-1031-104659` / `configs/exp1_gate06.json` | `model/gate06-...` | `eval/gate06-...-*` | 同上，1068 步；用于说明"距离满 100 m 但 upright≈0.01" |

录像对应的逐步数据在 `eval/exp2b-demo-steps.csv`、`eval/exp2b-demo-evaluation.json`（对照片同理），可逐帧核对视频里的按键、距离和前三项概率。

## 追加实验的对应关系

除主表两组录像外，其余实验只有评估数据与训练日志，没有录像。配置都在 `starter/configs/`，评估在 `eval/<训练目录>-eval-*`，训练日志在 `train_logs/<训练目录>/`。

| 实验 | 配置 | 训练目录 | 距离均值 | 结论摘要 |
| --- | --- | --- | --- | --- |
| exp5 抬膝 0.2（从零） | `exp5_knee_high.json` | `exp5-kneehigh02-1119-171651` | 59.13 m | 抬膝分布无变化，等于白加 |
| exp5b 抬膝 0.8（从零） | `exp5b_knee_high_strong.json` | `exp5b-kneehigh08-1119-172348` | 28.40 m | 退化成蹲行，抬膝中位 2.58→4.00 |
| 种子对照 | `exp2b_gate_knee01.json`、seed 7 | `ctrl-exp2b-seed7-1119-172822` | 43.39 m | 同配置换种子距离差 35%，用于估计种子波动 |
| exp6b 摔倒惩罚 50 | `exp6_fall50.json` | `exp6b-fall50-1120-114853` | 74.62 m | 距离回来了但退回贴地滑行（upright 0.267） |
| exp6 摔倒惩罚 200 | `exp6_fall200.json` | `exp6-fall200-1120-113833` | 20.37 m | 每步只走 0.017 m，10/10 仍旧摔倒 |
| A 续训-对照 | `exp2b_gate_knee01.json`，lr 1e-4 | `exp5d-ctrl32k-1121-000211` | 41.96 m | 只多训 32768 步会退化，说明提升不能归给"训练更久" |
| B 续训-抬膝 0.2 | `exp5_knee_high.json`，lr 1e-4 | `exp5c-warm32k-1120-235908` | 91.42 m | 全部实验里最快（0.134 m/步），8/10 跑满 100 m |
| C 续训-抬膝 0.8 | `exp5b_knee_high_strong.json`，lr 1e-4 | `exp5e-warm-strong-1121-000404` | 61.11 m | 抬过髋部 13.6%→0.7%，代价是速度 −16% |

三行续训实验（A/B/C）在 `runs/` 下没有 `tb/`，因为训练时加了 `--no-tensorboard`；其余日志字段齐全。

## 复现

环境（与任务书建议的 `qwop-rl` 不同，见报告第二部分 2.1）：

```
Ubuntu 22.04 + Conda 环境 torch
Python 3.11.17 / torch 2.14.1+cu130 / stable-baselines3 2.7.1
gymnasium 1.2.3 / selenium 4.52.0 / pandas 3.1.0 / numpy 2.4.6 / tensorboard 2.21.0
Google Chrome 155，浏览器无头模式，CPU 训练
```

在 `starter/` 目录下执行，例如复现主片：

```
python course.py train --out runs/exp2b-gate-knee01-1031-110114 --steps 131072 --seed 42 --reward configs/exp2b_gate_knee01.json
python course.py evaluate --run runs/exp2b-gate-knee01-1031-110114 --out runs/exp2b-gate-knee01-1031-110114-eval
python course.py record --run runs/exp2b-gate-knee01-1031-110114 --out videos/exp2b-gate-knee01-1031-110114
```

复现热启动续训（B 组，其余两组只改 `--reward-config` 与 `--learning-rate` 之外的配置名）：

```
python train.py --out runs/exp5c-warm32k-1120-235908 --resume runs/exp2b-gate-knee01-1031-110114/model.zip --stats runs/exp2b-gate-knee01-1031-110114/normalize.pkl --steps 32768 --learning-rate 1e-4 --reward-config configs/exp5_knee_high.json --headless --fast --direct-cdp --normalize --normalize-reward --no-tensorboard --max-steps 1500 --rollout-steps 1024 --batch-size 64 --epochs 10 --net-width 128 --entropy 0.02 --gamma 0.995 --seed 42
python summarize.py runs/exp5c-warm32k-1120-235908-eval --target 10
```

注意：`course.py` 与 `train.py` 都拒绝覆盖已存在的输出目录，复现时请用新目录名；每个 131072 步实验在 CPU 上约 3~4 分钟，每次评估约 1~2 分钟。

## 来源与边界

- 路线 A：课堂提供的 `starter/` 实现（Gymnasium + Stable-Baselines3 PPO + PyTorch + 浏览器原游戏），原始项目见 `qwop-course/docs/QWOP训练任务书.md` 第 11 节与 `SOURCE_MANIFEST.json`。
- 本次实验相对 starter 的改动：训练脚本的设备/超参开关、`requirements.txt` 的 torch 版本固定，以及新增的奖励项 `knee_high`（默认 0，不影响此前所有实验）。观测、动作集合、环境物理均**未修改**（见 `code/my_code_changes.patch` 与报告第二部分 2.2）。
- 主实验（baseline、gate06、exp2b、exp3a、exp4、exp5、exp5b、exp6、exp6b）都是**从零训练**；只有 11-20 的三行续训实验（A/B/C）从本项目的 exp2b 权重热启动，未使用老师参考策略、人手控制或外部旧模型。
- 视频为同一模型的一次连续回放，不是拼接。
