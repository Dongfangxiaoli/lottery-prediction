# 仅供本地私用的便携构建工具

本目录不是已构建的便携包。直接运行这里的Start.bat/Check.bat会因缺少runtime失败，公开源码用户应遵循根目录README。

打包器沿用项目原本的Windows Python3.12.14本地环境迁移流程：需要项目`.venv`、受支持的基础Python运行时、已单独从微软官方获取并验证数字签名的VC++安装程序，以及个人数据/模型目录。仅有这个GitHub源码快照不满足全部构建输入。

`build_portable.py --output <新目录>`会收集`data/models/results/evidence_runs/candidate_runs`及当前环境。`--seal <已构建目录>`会封装ZIP并校验CRC。它不会自动脱敏，也不能替代第三方许可审查。

**不要将其输出直接公开上传。** 当前发行仅公开源码快照；公开免安装版需要另行设计不含个人数据的清洁构建及再分发审查。

`verify_portable.py`和`verify_candidates.py`用于有对应私有工件的本地验收，不在普通CI内执行；`test_launcher.py`使用临时夹具，不要求这些工件。
