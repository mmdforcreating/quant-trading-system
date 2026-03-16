# Qlib Pandas 2.x 兼容性补丁

> 适用版本: qlib 0.9.8.dev27 + pandas 2.3.3
> 修改日期: 2026-03-12

如果 `git pull` 更新了 qlib，请检查以下改动是否仍然需要。
如果官方已修复，删除本文件即可。

---

## 补丁列表

### 补丁 1: `qlib/utils/paral.py` — resample(level=...) 已移除

**问题**: `DataFrame.resample(rule, level=...)` 的 `level` 参数在 pandas 2.0 中被移除。

**原始代码** (约第 63-66 行):

```python
if n_jobs != 1:
    dfs = ParallelExt(n_jobs=n_jobs)(
        delayed(_naive_group_apply)(sub_df) for idx, sub_df in df.resample(resample_rule, level=level)
    )
    return pd.concat(dfs, axis=axis).sort_index()
```

**修改后**:

```python
if n_jobs != 1:
    period_freq = resample_rule[:-1] if resample_rule.endswith(("E", "S")) else resample_rule
    datetime_vals = pd.DatetimeIndex(df.index.get_level_values(level))
    period_groups = datetime_vals.to_period(period_freq)
    dfs = ParallelExt(n_jobs=n_jobs)(
        delayed(_naive_group_apply)(sub_df) for _, sub_df in df.groupby(period_groups)
    )
    return pd.concat(dfs, axis=axis).sort_index()
```

**还原方法**: 将修改后的代码替换回原始代码。

---

### 补丁 2: `qlib/utils/paral.py` — groupby(axis=...) 已移除

**问题**: `DataFrame.groupby(axis=...)` 的 `axis` 参数在 pandas 2.1 中被移除。

**原始代码** (约第 60 行):

```python
return getattr(df.groupby(axis=axis, level=level, group_keys=False), apply_func)()
```

**修改后**:

```python
return getattr(df.groupby(level=level, group_keys=False), apply_func)()
```

**还原方法**: 在 `groupby(` 后加回 `axis=axis, `。

---

### 补丁 3: `qlib/contrib/report/analysis_model/analysis_model_performance.py` — freq 别名变更

**问题**: pandas 2.2 起，`freq="1M"` (月末) 应改为 `"1ME"`。

**原始代码** (约第 160 行):

```python
freq="1M",
```

**修改后**:

```python
freq="1ME",
```

**还原方法**: 将 `"1ME"` 改回 `"1M"`。

---

## 无需修改的项目 (扫描确认安全)

| API | 状态 |
|-----|------|
| `DataFrame.append()` | 未使用 (已用 list append 或 pd.concat) |
| `.iteritems()` | 未使用 (已用 .items()) |
| `.mad()` | 未使用 |
| `.is_monotonic` | 未使用 (已用 .is_monotonic_increasing) |
| `.swaplevel()` | pandas 2.x 仍支持，无需修改 |
| `sort_index(inplace=True)` | pandas 2.x 仍支持，无需修改 |
| `pd.Panel` | 未使用 |
| `.ftypes` / `.ftype` | 未使用 |
| `Index.format()` | 未使用 |
| `line_terminator` 参数 | 未使用 |
