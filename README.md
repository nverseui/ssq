# ssq

双色球量化研究与前向回测系统（SSQ Master V3）。

## 审计优先规则

本仓库的输出只用于历史统计研究，不构成中奖概率承诺、收益承诺或购买彩票建议。

回测和前瞻流程必须遵守：

- 每一期只使用该期之前的开奖数据。
- 所有方案按相同实际注数比较。
- 固定奖金和浮动一等奖、二等奖用法必须在报告中明确区分。
- 候选不足时保留缺注，不通过放宽交集上限隐式补注。
- 多方法比较使用配对区块 bootstrap，并进行多重检验校正。

## 当前主要文件

- `ssq_master_v3.py`：生产候选生成与展示。
- `ssq_backtest_audit.py`：严格走向前审计与消融回测。
- `ssq_hardening.py`：尾数配额、概率权重、残差去中心和注单验证加固工具。
- `tests/test_ssq_backtest_audit.py`：审计指标与统计辅助测试。
- `tests/test_ssq_hardening.py`：加固模块测试。

## 加固模块示例

```python
from ssq_hardening import largest_remainder_tail_quota, exact_overlap_portfolio

quota = largest_remainder_tail_quota(
    {3: 0.05, 4: 0.33, 5: 0.50, 6: 0.12},
    n_sets=15,
)
assert sum(quota.values()) == 15

portfolio = exact_overlap_portfolio(candidates, n_sets=15, max_overlap=3)
assert len(portfolio) <= 15
```

配额使用最大余数法精确分配注数；交集筛选不会把硬上限静默放宽到更大的重合数。

## 回测审计

```bash
python ssq_backtest_audit.py --eval 500 --sets 5 --random-replicates 32 --workers 1
```

如需运行测试：

```bash
python -m pytest -q
```

详细回测口径见 `BACKTEST_AUDIT.md`。
