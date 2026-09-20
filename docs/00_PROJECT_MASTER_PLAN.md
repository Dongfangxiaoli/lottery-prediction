# 项目状态与边界

## 当前版本

0.10.0，首次公开源码整理日期2026-09-20，MIT许可。维护者Dongfangxiaoli；保留原始元数据作者bairuchang。公开源码位于`Dongfangxiaoli/lottery-prediction`，不代表已部署公网Web服务。

## 目标

提供本地中文、可测试、可追溯的五彩种合法选号和统计评估系统。真实一等奖预测优势必须有独立未见数据证据；当前没有。

## 已完成

- 五彩种七玩法、组合去重、最高奖理论概率、小奖独立模拟。
- LSTM训练、因果特征、训练边界控制、档案恢复与漂移拒绝。
- 隔离历史实验、冻结协议、未来登记与核验。
- XGBoost/Markov/偏差诊断、与LSTM和均匀随机固定对照。
- 公开数据边界、中文说明、许可证、合成测试样本和CI配置。

## 架构

界面层`gradio_app.py`与各`*_ui.py`；数据层`data_fetcher_*.py/local_history.py`；训练`feature_engineering.py/train.py/lstm_model.py/model_session.py`；选号及评估`predictor.py/portfolio_engine.py/jackpot_*.py`；证据`evidence_*.py`；候选研究`candidate_*.py/bias_diagnostics.py/run_candidate_experiment.py`。

## 公开与私有分离

公开源码、测试与汇总文档。不发布个人data/models/results、研究目录、日志、虚拟环境和旧便携包。私用打包工具仅作为源码保留，不能未经审查用于公开二进制发行。

## 已知问题与下一步

1. 独立核验大乐透早期历史来源；未完成前不把频数异常当选号信号。
2. 保持冻结研究边界；新候选未来多模型前瞻协议尚未实现。
3. 如需公开免安装版，另建清洁打包与隐私/依赖许可审查流程，不复用个人运行包。

第三方数据源可能失效；本地文件无独立时间可信性；没有跨进程写锁；未承诺全新Windows电脑兼容。`SOL_REVIEW_PENDING`：无新增事项。
