# 提交说明（QWOP 强化学习作业）

姓名：lihanwen
日期：2026-11-11
代码提交：`a937568`（starter 原始提交 `bd8bac4`）

## 三项交付物在哪

| 交付物 | 文件 |
| --- | --- |
| 最终录像 | `final_demo.mp4`（主片，连续单回合，未截断）；`final_demo_gate06_kneeling.mp4`（对照片：跪行滑行 100 m） |
| 实验报告 | `report.md` |
| 可复现材料 | `experiment_log.csv`、`eval/`、`train_logs/`、`model/`、`code/my_code_changes.patch` |

## 目录结构

```
submission/
├── final_demo.mp4                       主片：exp2b（门控 0.6 + 膝部惩罚 0.1），101.9 m
├── final_demo_gate06_kneeling.mp4       对照：gate06（只有门控 0.6），100.3 m，跪行
├── report.md                            实验报告（目标、设置、修改、对照、demo 对应关系）
├── experiment_log.csv                   一行一实验：种子、步数、配置、评估距离、视频
├── eval/                                5 次评估的 episodes.csv + summary.json
├── train_logs/<实验>/                   训练日志：episodes.monitor.csv、curves.png、
│                                        config.json、reward.json、packages.txt、tb/
├── model/                               model.zip + normalize.pkl（主片与对照片各一份）
└── code/my_code_changes.patch           相对 starter 原始提交的源码与配置改动
```

## 视频、模型、数据的对应关系

| 视频 | 训练目录 / 配置 | 模型 | 评估数据 | 说明 |
| --- | --- | --- | --- | --- |
| `final_demo.mp4` | `runs/exp2b-gate-knee01-1031-110114` / `configs/exp2b_gate_knee01.json` | `model/exp2b-...` | `eval/exp2b-...-*` | seed 42 训练；录像 seed 100、`start_jitter=0`、747 步、`game_end` 自然结束（未截断） |
| `final_demo_gate06_kneeling.mp4` | `runs/gate06-1031-104659` / `configs/exp1_gate06.json` | `model/gate06-...` | `eval/gate06-...-*` | 同上，1068 步；用于说明"距离满 100 m 但 upright≈0.01" |

另外三个实验（exp3a、exp4 与起步基线）的数据都在 `eval/` 里，视频在仓库 `starter/videos/` 下，报告中有对应说明。

## 复现

环境（与任务书建议的 `qwop-rl` 不同，见报告"代码来源与训练设置"）：

```
Ubuntu 22.04 + Conda 环境 torch
Python 3.11.17 / torch 2.14.1+cu130 / stable-baselines3 2.7.1
gymnasium 1.2.3 / selenium 4.52.0 / pandas 3.1.0 / numpy 2.4.6 / tensorboard 2.21.0
Google Chrome 155，浏览器无头模式，CPU 训练
```

在 `starter/` 目录下执行（完整命令清单见 `report.md` 附录 A），例如复现主片：

```
python course.py train --out runs/exp2b-gate-knee01-1031-110114 --steps 131072 --seed 42 --reward configs/exp2b_gate_knee01.json
python course.py evaluate --run runs/exp2b-gate-knee01-1031-110114 --out runs/exp2b-gate-knee01-1031-110114-eval
python course.py record --run runs/exp2b-gate-knee01-1031-110114 --out videos/exp2b-gate-knee01-1031-110114
```

注意：`course.py` 拒绝覆盖已存在的输出目录，复现时请用新目录名；每个 131072 步实验在 CPU 上约 3~4 分钟。

## 来源与边界

- 路线 A：课堂提供的 `starter/` 实现（Gymnasium + Stable-Baselines3 PPO + PyTorch + 浏览器原游戏），原始项目见 `qwop-course/docs/QWOP训练任务书.md` 第 11 节与 `SOURCE_MANIFEST.json`。
- 本次实验相对 starter 的改动只有训练脚本的设备/超参开关与 `requirements.txt` 的 torch 版本固定（见 `code/my_code_changes.patch` 与报告"修改清单"）；奖励公式、观测、动作集合、环境物理均**未修改**。
- 全部为从零训练，**未使用**老师参考策略、人手控制或上游旧模型；视频为同一模型的一次连续回放。
