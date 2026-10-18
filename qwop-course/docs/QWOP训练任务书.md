# QWOP 强化学习实验任务书

## 1. 你要完成什么
训练一个能控制 Q、W、O、P 按键的 QWOP 策略，录制它训练后的真实游戏 demo，并尝试解释它为什么出现这种行为。随包有原始开源仓库、课堂入门代码，你可以自己修改、编写奖励、观测项和训练配置，甚至修改地形以做出更高难度的成果。最终你需要提交自己的训练录像，并编写文档记录你所做的修改，要说明改了什么、为什么改、结果怎样。

百米成绩不是本任务的唯一考核标准。更多是要看训练过程，以及在其中展现的归因能力与想象力。倒着跑、跳着跑也是有价值的研究对象，只要你确实是想训这个且训出来了，我就没意见。

## 2. 先看懂五个词
**观测 (observation)** 是策略得到的身体位置、角度和速度等信息。**动作 (action)** 是一次选择的按键组合。**奖励 (reward)** 是训练时告诉算法“这一步得多少分”的数字。**策略 (policy)** 根据当前观察，给每种动作一个概率。**回合 (episode)** 是从重新开始到摔倒、到达终点或达到步数上限的一次尝试。

一次循环是：观察身体 → 策略选择按键 → 游戏向前模拟 → 得到新观察和奖励。训练算法用许多这样的尝试更新策略。奖励公式本身不是策略；修改奖励后，必须重新训练才能检验新策略。

入门代码使用 PPO。你暂时可以把它理解为：收集尝试，根据结果小幅调整动作概率，再继续尝试。完成实验不需要先会推导 PPO 公式。

## 3. 看参考视频，先提问题
videos/01-walking-100m.mp4：向前行走到终点。观察双脚怎样轮换支撑。

videos/02-kneeling-full.mp4：跪行完成全程。思考“走得远”和“像人一样走”为什么不是同一个指标。

videos/03-standing-still.mp4：站立不动。思考鼓励存活或保持姿态，会不会让策略放弃前进。

videos/04-backward-stable.mp4：从跑道中段开始，低姿态持续后退。看距离数字怎样变化；这是低姿态后退案例。

videos/05-kneeling-still.mp4：低姿态停留。思考奖励里是否缺少了持续前进的要求。

视频中的“前三项概率”只显示概率最高的三个动作，三者相加不一定等于 100%。概率 80% 不是“有 80% 的可能通关”，而是当前观察下选择这个动作的概率。

小练习：任选两段视频，写下你看见的行为、你猜测的奖励条件，以及还需要什么数据才能验证猜测。仅凭画面不能确定训练时使用了哪些奖励系数。

## 4. 选择实现路线
A｜推荐新手：使用 starter/。这是单独提供的现代课堂实现：Gymnasium + Stable-Baselines3 PPO + PyTorch，浏览器运行原游戏。已经补充 CSV、概率图、步态指标和录像工具。

B｜自主实现：可以更换算法、观察、按键集合、网络、奖励或训练流程，也可以使用其他自己修改的接口。须写清来源、依赖和启动命令，并在真实 QWOP 游戏中运行。若改变重力、碰撞、起始姿态等环境条件，必须说明；成绩只在相同条件下比较。


## 5. 第一次启动
### Ubuntu 第一次启动
建议 Ubuntu 22.04 桌面版、Miniconda 或 Anaconda、Python 3.12、Google Chrome，有网络；不需要显卡。先完整解压 ZIP，再在 starter 文件夹右键“在终端打开”。不要在压缩包浏览窗口里直接运行。

下面的 sudo 命令会要求输入你自己的 Ubuntu 登录密码，输入时屏幕不显示字符。Google Chrome 请先从其官网安装 Linux .deb 版；安装完成后能在终端运行 google-chrome --version。

```bash
sudo apt update
sudo apt install -y fonts-noto-cjk
conda create -n qwop-rl python=3.12 pip -y
conda activate qwop-rl
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python course.py --help
```

首次安装需要下载 Python 依赖及浏览器驱动。游戏源文件和参考视频已经在包内。上面的命令使用 Conda 创建名为 `qwop-rl` 的独立环境；关闭终端后再次使用时，先进入 starter，再执行 `conda activate qwop-rl`。Windows 和 Ubuntu 都可以运行同一套课堂代码；当前完整浏览器训练和录像验证已在 Windows 完成。

如果终端提示没有 `conda` 命令，先安装 Miniconda 或 Anaconda，并按安装器提示初始化当前 shell；Bash 通常执行一次 `conda init bash` 后重新打开终端。若没有 Chrome，先安装浏览器。驱动下载失败时，请老师安装与 Chrome 匹配的 chromedriver，并设置 `export QWOP_CHROMEDRIVER=/实际路径/chromedriver`。不要反复改奖励来解决安装错误。`setup_ubuntu.sh` 是原来的 venv 安装脚本，使用 Conda 时不需要运行它。

### Windows 第一次启动
建议 Windows 10/11、Miniconda 或 Anaconda、Python 3.12、Google Chrome，有网络；不需要显卡。先完整解压 ZIP，在文件资源管理器中打开 starter 文件夹，在地址栏输入 `powershell` 后回车，或右键选择“在终端中打开”。不要在压缩包浏览窗口里直接运行。

在 PowerShell 中执行以下命令：

```powershell
conda --version
conda create -n qwop-rl python=3.12 pip -y
conda activate qwop-rl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python course.py --help
```

如果 PowerShell 提示没有 `conda` 命令，先安装 Miniconda 或 Anaconda，执行 `conda init powershell` 后关闭并重新打开终端。使用命令提示符（cmd）时，执行 `conda init cmd.exe` 后重新打开窗口。Conda 环境不需要单独运行 `.venv\Scripts\Activate.ps1`。

首次安装需要下载 Python 依赖；首次运行浏览器任务时，Selenium 可能还会下载与 Chrome 匹配的驱动，因此应保持网络连接。通常不需要手动安装 chromedriver；如果自动下载失败，可从老师提供的驱动位置获取匹配版本，并在当前 PowerShell 窗口设置：`$env:QWOP_CHROMEDRIVER = 'C:\path\to\chromedriver.exe'`。游戏源文件和参考视频已经在包内。关闭终端后再次使用时，进入 starter 并执行 `conda activate qwop-rl`。

## 6. 跑通第一个训练，再做正式实验
### `course.py` 功能与用法
`course.py` 是面向课堂的命令封装，必须在 `starter` 文件夹中、激活 `qwop-rl` Conda 环境后运行。它会调用底层的 `train.py`、`evaluate.py`、`summarize.py`、`plot.py` 和 `make_video.py`，并自动使用 CPU、浏览器无头模式和课堂默认参数。输出目录必须是尚不存在的新目录，脚本会拒绝覆盖旧实验。

先查看总帮助或某个子命令的参数：

```powershell
python course.py --help
python course.py train --help
python course.py evaluate --help
python course.py record --help
```

| 子命令 | 作用 | 常用参数 | 主要输出 |
| --- | --- | --- | --- |
| `train` | 从零训练 PPO，并生成训练曲线 | `--out` 输出目录，`--steps` 训练步数，`--seed` 随机种子，`--reward` 奖励配置 | `model.zip`、`normalize.pkl`、`config.json`、`episodes.monitor.csv`、`tb/`、`curves.png` |
| `evaluate` | 对已有模型运行多个回合并汇总数据 | `--run` 含 `model.zip` 的训练目录，`--out` 新评估目录，`--episodes` 回合数，`--max-steps` 单回合上限，`--start-jitter` 起始扰动 | `episodes.csv`、`steps.csv`、`summary.json`、`curves.png`、`gait.png` |
| `record` | 对已有模型录制一段连续回放并生成 MP4 | `--run` 训练目录，`--out` 新录像目录，`--max-steps` 单回合上限 | `final_demo.mp4`、`steps.csv`、`evaluation.json`；视频生成后删除截图 |

三个子命令的典型关系是：先用 `train` 生成 `runs/baseline/model.zip`，再把同一个训练目录交给 `evaluate` 或 `record`。`evaluate` 和 `record` 都会自动读取训练目录中的 `config.json`、`reward.json` 和 `normalize.pkl`；不要把不同实验的模型、标准化统计和视频混用。需要修改底层 PPO 参数、浏览器模式或更细的评估选项时，可以直接运行对应的 `train.py` 或 `evaluate.py`，并在报告中记录完整命令。

先运行一个短训练，只用于检查安装和流程。2048 步不是“保证学会走路”的训练量。所有命令里的输出目录都必须是新名字，脚本会拒绝覆盖旧实验。

```bash
python course.py train --out runs/check --steps 2048
python course.py record --run runs/check --out runs/check-video --max-steps 300
```

打开 runs/check-video/final_demo.mp4，就能看到刚训练的策略。这个短检查允许表现很差；它的作用是确认你的训练、模型保存和录像都能完成。

然后做正式基线实验。下面 32768 步是起始预算，可与老师约定更大的预算。耗时取决于电脑；增加步数不保证一定变好。PPO 按整批采样更新，实际采样量可能向上取整到批次边界，报告以日志为准。

```bash
python course.py train --out runs/baseline --steps 32768 --seed 42
python course.py evaluate --run runs/baseline --out runs/baseline-eval
```

评估默认运行 10 个回合、每回合最多 3000 个策略动作，并在开始时随机等待 0～3 个动作时长，检查轻微起始变化下的表现。训练时每回合上限为 1500 个动作。重复确定性策略且完全不改变初始状态，可能只得到重复轨迹；不要把重复十次同一起点理解成十种不同情况。

想停止正在训练的实验，可以另开终端，在对应训练目录创建空文件 STOP，例如 touch runs/baseline/STOP。脚本会在下一次回调保存当前模型并退出。不要删除正在使用的输出目录。

## 7. 做一个有理由的修改
先写一句假设，例如：“只奖励位移可能会出现跪行；增加非脚部接触惩罚，可能减少这种接触，也可能让策略停止前进。”先预测效果，再做实验。

```bash
cp configs/distance_only.json configs/my_reward.json
```

用文本编辑器打开 configs/my_reward.json。例如只把 nonfoot_contact_penalty 从 0.0 改为 0.1；这是实验起点，不是最优答案。JSON 的逗号、双引号和括号都要保留。接触惩罚按每次策略动作计算，权重太大会压过位移奖励。只增加姿态奖励也可能得到站住不动。

```bash
python course.py train --out runs/changed --steps 32768 --seed 42 --reward configs/my_reward.json
python course.py evaluate --run runs/changed --out runs/changed-eval
```

比较时保持训练预算、种子、动作时长和评估条件一致。先只改一个因素，方便判断原因；之后再探索组合修改。若改变多个因素，不能断言某一个因素单独造成了结果。资源允许时增加训练种子，检查结论是否重复出现。

也可以修改 train.py 的算法参数、qwop_env.py 的观察/动作，或 rewards.py 的奖励逻辑。直接改代码后，要保留修改前后文件或补丁；配置副本不能记录 Python 源码变化。没有修改策略结构也可以完成任务，但必须明确写“未修改”，说明实际使用的算法、参数和训练过程。

## 8. 怎样读策略与数据
训练目录中的 episodes.monitor.csv 和 curves.png 显示训练回合回报、距离等变化。回报上升可能只是更擅长拿奖励，不能单独证明正常行走。不同奖励公式下的总回报通常不能直接横比。

训练默认还会把 Stable-Baselines3 的训练指标写入训练目录的 `tb/`。在 `starter` 且已激活 `qwop-rl` Conda 环境的终端中启动 TensorBoard：

```powershell
python -m tensorboard.main --logdir runs
```

然后在浏览器打开 `http://localhost:6006`。如果只想查看一次实验，可以把日志目录缩小到 `runs/baseline/tb`；如果同时查看多个实验，使用 `runs` 作为根目录，TensorBoard 会按子目录区分不同实验。训练仍在运行时可以保持 TensorBoard 开启，它会周期性读取新日志；训练结束后也可以再启动。

重点查看 `rollout`、`train` 和 `time` 分组中的曲线，例如回合回报、回合长度、策略损失、价值损失和已采样步数。曲线用于发现训练是否在变化、是否提前停滞或出现异常，不等于“学会了走路”的证明；仍要结合 `episodes.csv`、`steps.csv`、`summary.json`、`gait.png` 和实际录像。比较两个实验时，保持奖励配置、训练步数和随机种子等条件一致，并在报告中注明 TensorBoard 日志所在目录。

如果只想生成 CSV 而不写 TensorBoard，可直接运行底层脚本并增加 `--no-tensorboard`；`course.py train` 默认开启 TensorBoard。

评估目录中的 episodes.csv 一行是一个回合；summary.json 汇总距离均值、最小值、最大值和结束原因等。steps.csv 一行是一步动作：keys 是选中的按键，p_qw 等列是完整动作概率，distance 是游戏距离；reward_forward、reward_nonfoot 等列拆开显示奖励来源。

curves.png 还包含第一回合的动作概率热图；gait.png 展示躯干、膝部与接触变化。身体坐标 y 向下增大，因此不能把更大的 y 直接理解成站得更高。nonfoot_fraction 是非脚部触地的动作占比，越高越值得检查是否跪行。support_switches 是脚部支撑变化次数，单独次数高也不能证明步态自然。

walking_success 是课堂代码的启发式检查，包含距离、姿态和接触阈值；reason=limit 表示达到设置的动作数上限；game_end 表示游戏结束，要结合距离和画面区分失败与到达终点。

至少回答三个问题：①距离和姿态是否同时改善？②是否出现“奖励变高但目标没实现”？③最差回合为什么值得关注？给每个回答附一个视频时刻、CSV 数值或图中位置。对原因使用“假设—证据—进一步验证”的写法，避免仅凭一次摔倒编造原因。

## 9. 录制最后的 demo
```bash
python course.py record --run runs/changed --out runs/final-video
```

最终文件为 runs/final-video/final_demo.mp4。脚本重新载入保存好的模型和标准化统计，实际运行游戏并采集画面，视频显示距离、动作和前三项概率；视频成功生成后，作为中间产物的 PNG 截图会自动删除。默认最多 3000 个动作；可用 --max-steps 提高上限，报告中写明。若到上限仍没结束，应称为截断回合。

提交自己最终策略的一次连续回放，保留起始和结束，不拼接不同回合来伪造成功。可以另加失败或对照片段；如选的是最好的一次，写明选择规则，并同时提交全部评估结果。用其他实现时，可自行录屏，但必须能对应实际模型和实验条件。不要用老师视频、人手控制或直接加载上游旧模型代替自己的训练。

若结果仍早早摔倒，单独放一段摔倒录像不足以说明实验。请附训练记录、同条件多回合数据，并提出可检验的原因假设，至少完成一次针对该假设的对照尝试。承认暂时没解决的问题，比没有证据地声称“学会了”更有价值。

## 10. 最终提交什么
1. final_demo.mp4：训练完的真实策略录像，注明对应实验、模型和是否截断。
2. 实验报告：使用 submission_template/report_template.md，或改写为 Word/PDF。必须说明目标、来源、训练设置、结果和修改；没有修改的项目也明确写未修改。
3. 可复现材料：你使用的代码或仓库提交编号、依赖说明、启动命令、实际奖励配置，以及评估的 episodes.csv 和 summary.json；有修改的源文件一并提供。至少保留一次正式训练的日志。

建议附自己的 model.zip 与配套 normalize.pkl，便于老师复查；文件过大时说明保存位置。老师参考策略不在本包内。训练设置、源码和标准化统计要与提交视频对应，不能混用不同实验的文件。

修改说明至少写清：文件/参数位置；修改前；修改后；预期作用；结果证据。可以写“只改奖励，策略网络与 PPO 算法未修改”。如果用了原作者的模型再训练，写清模型来源和新增训练量。

## 11. 建议评价方式（100 分）
训练过程与复现材料 25 分；修改或未修改的准确说明 20 分；真实且可对应实验的最终录像 20 分；数据分析、对照与局限 30 分；来源标注和文件整理 5 分。距离不作为唯一评分依据，照抄参考视频不能获得训练与分析分。

课堂建议：第一次课认识概念、看视频、跑通短训练并提出假设；课间完成基线与修改训练；第二次课读数据、做对照、录制 demo 并互相解释。小组可分操作、记录和解释角色，每个人都应能说清“策略如何选择动作”和“奖励为何可能偏离目标”。

## 12. 来源与实现边界
原始项目：https://github.com/Wesleyliao/QWOP-RL ，随包固定提交见 SOURCE_MANIFEST.json。upstream/QWOP-RL 保留原文件，包括公开模型和历史资料。其根目录未发现 LICENSE；本包不替原作者授予新的许可，游戏、图像等第三方权利归各自权利人。

starter 为独立的课堂适配实现，不是仅替换原仓库奖励：改用 Gymnasium/PyTorch/PPO、浏览器本地接口、固定物理更新步长，增加观察标准化、可配置奖励、数据图和录像。接口在本地响应中暴露游戏状态，未修改游戏重力和碰撞规则。详见 starter/ADAPTATION.md；课堂代码自身采用 MIT，不能据此认为原仓库与游戏也采用 MIT。
