# 彩票选号与可验证评估工具

[![CI](https://github.com/Dongfangxiaoli/lottery-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Dongfangxiaoli/lottery-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

一个本地运行的中文 Python / Gradio 工具，支持双色球、大乐透、排列3、排列5和7星彩。提供合法号码生成、组合覆盖、历史评估、模型恢复，以及 LSTM、XGBoost、简单马尔可夫与均匀随机的对照实验。

**当前没有证据证明这些模型能提高真实一等奖中奖率。** 在开奖公平、独立、均匀的前提下，历史号码不能提高下一期单注中奖概率。项目用于学习、软件验证和统计研究，不提供保中、回本或盈利承诺，也不代购、不自动投注。

## 功能

| 功能 | 范围与限制 |
|---|---|
| 五彩种、七玩法 | 双色球单式、大乐透基本、排列3直选/组选3/组选6、排列5直选、7星彩单式 |
| 组合方案 | 可调注数、整票去重、最高奖理论概率；随机去重模式无需历史数据或训练 |
| 小奖覆盖 | 与独立模拟对照分开显示；不能替代一等奖指标或保证回本 |
| 训练与恢复 | LSTM、因果历史特征、训练前缀拟合缩放器、训练档案和权重校验 |
| 候选算法研究 | 固定 XGBoost / Markov，与 LSTM 和均匀随机同注数比较 |
| 证据管理 | 冻结配置、提前登记、开奖后核验；本地哈希不是第三方可信时间戳 |

## 快速开始：Windows 源码版

本仓库公开的是**源码版**，不是内置 Python 的便携包。建议 Windows 10/11 x64、Python **3.12**，使用 CPU 即可；其他 Python 版本未纳入当前支持范围。

1. 从 [Python 官方网站](https://www.python.org/downloads/windows/) 安装 Python 3.12 x64；已有 Python 3.12 可跳过。
2. 点击 GitHub 的 **Code → Download ZIP** 并完整解压，或使用 Git：

```powershell
git clone https://github.com/Dongfangxiaoli/lottery-prediction.git
cd lottery-prediction
```

3. 在项目目录打开 PowerShell，创建隔离环境并安装依赖。不需要激活环境或修改执行策略：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

4. 启动：

```powershell
$env:GRADIO_ANALYTICS_ENABLED = "False"
.\.venv\Scripts\python.exe -X utf8 -B gradio_app.py
```

保留终端窗口。浏览器通常自动打开 `http://127.0.0.1:7861`；端口占用时以终端实际地址为准。按 `Ctrl+C` 退出。默认只监听本机，不开启 Gradio 公共分享链接。

## 第一次怎么用

- **不训练先体验**：在“组合方案与中奖概率”选择彩种、玩法、注数，使用“均匀随机去重”生成合法号码和理论概率。
- **训练模型**：在“分步操作”获取数据，或将自己有权使用的历史 CSV 放到 `data/` 后点击“读取本地数据（不联网）”；再训练、选号。格式见[使用指南](使用说明.md)。
- **恢复模型**：仅加载自己生成或可信来源的训练档案和权重；点击“恢复已保存模型（不重训）”。历史或权重变动时需重新训练。
- **研究实验**：公开仓库不附私人研究目录。“国际算法对照”首次显示无完成实验是正常状态，运行前须准备基准研究和冻结协议，见[研究指南](docs/research/README.md)。

获取开奖数据需要网络；数据源是第三方网站，可能变更或不可用。离线训练需要事先准备合规取得的本地数据。数据抓取不等于官方真实性验证。

## 已有实验结论

v0.10.0 的历史探索采用每玩法300期、每种方法每期5组；21项模型比较经多重检验校正后均未发现显著的最高奖优势。5组只是统一实验口径，不限制普通组合页的可调注数。

| 彩种/玩法 | XGBoost 命中期 | Markov 命中期 | LSTM 命中期 | 随机命中期 |
|---|---:|---:|---:|---:|
| 双色球单式 | 0 | 0 | 0 | 0 |
| 大乐透基本 | 0 | 0 | 0 | 0 |
| 排列3直选 | 3 | 1 | 0 | 0 |
| 排列3组选3 | 5 | 4 | 4 | 3 |
| 排列3组选6 | 5 | 9 | 7 | 13 |
| 排列5直选 | 0 | 0 | 0 | 0 |
| 7星彩单式 | 0 | 0 | 0 | 0 |

历史已经被查看，不能当作独立未来验证；“3比0”也不能证明优势。大乐透早期训练数据存在频数异常，尚待独立来源核验，不能直接解释成可利用规律。见[完整方法与局限](docs/research/v0.10.0-results.md)。

## 测试与维护

```powershell
$env:GRADIO_ANALYTICS_ENABLED = "False"
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s . -p "test_*.py"
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s packaging -p "test_*.py"
```

测试使用临时目录和合成数据，不要求下载历史开奖、不要求 GPU，也不需要账户凭证。CI 结果以页面上的实际运行状态为准；测试通过不等于预测有效。

```text
gradio_app.py              本地中文界面
portfolio_engine.py       组合与概率评估
train.py / lstm_model.py   LSTM 训练
candidate_models.py       XGBoost 与 Markov
bias_diagnostics.py       偏差诊断
evidence_*.py              冻结协议与前瞻登记
test_*.py                 自动测试
docs/                     方法、限制、项目状态
packaging/                本地私用便携包构建工具（非公开发布流水线）
```

## 公开范围与安全

仓库不包含历史数据全集、训练权重、个人选号记录、前瞻记录、运行日志、虚拟环境、第三方安装程序或原个人便携包。它们由 `.gitignore` 排除，CI 会检查已跟踪文件中的越界内容。

**不要把现有个人便携包直接上传到 Releases。** 当前本地打包器会收集数据、模型及研究记录；公开二进制发布需要独立的隐私和第三方许可证审查。本次 GitHub Release 提供源码快照，不提供免安装运行包。

不支持把本地 Gradio 服务直接暴露到互联网；不提供认证、多用户隔离或生产服务器部署保证。问题反馈请删除个人路径、票号记录和凭证。

## 参与与许可

欢迎提交可复现 Bug、测试和文档改进；方法修改必须报告随机基准、数据切分和全部比较。请阅读 [贡献指南](CONTRIBUTING.md)、[安全说明](SECURITY.md)、[更新记录](CHANGELOG.md)。

代码采用 [MIT License](LICENSE)。保留原始项目元数据署名 `bairuchang`，当前维护：[Dongfangxiaoli](https://github.com/Dongfangxiaoli)。依赖与第三方数据不因本项目 MIT 许可而自动获得相同授权，见[第三方说明](THIRD_PARTY_NOTICES.md)。
