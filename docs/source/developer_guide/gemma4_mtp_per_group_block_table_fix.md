# Gemma4 MTP 在 Ascend A2/A3 上的 KV Cache 串组 Bug 修复纪实

> 一份关于 Gemma4 MTP draft 模型 `full_attention` 层读到错误 KV cache 的根因排查与修复记录。
> 适用版本：vllm-ascend，torch_npu 2.10.0 / CANN，Ascend 910B4 (A2)。
> 关键词：speculative decoding、MTP、multi KV cache group、block_table、head_dim=512。

---

## 1. 背景

Gemma4 是一个**混合注意力**模型：浅层用 sliding window attention（局部），深层用 full/global attention（全局）。在 vLLM 的 KV cache 抽象里，不同注意力类型对应不同的 **KV cache group**，每个 group 有：

- 自己的 `kv_cache_spec`（block_size、head_dim 等）
- 自己的 **block_table**（物理 KV block 的地址映射）
- 自己的 attention metadata builder

目标模型（target）在 `model_runner` 里通过 `set_per_group_block_table(gid, block_table)` 为每个 group 分别捕获 block_table。

Gemma4 的 MTP（Multi-Token-Prediction）draft 模型与 target 共享同一套 KV cache，因此 draft 的 attention 层也跨越**多个 KV cache group**：

| Draft 层            | 注意力类型 | head_dim | 所属 group（target gid） |
| ------------------- | ---------- | -------- | ------------------------ |
| sliding 层          | 局部滑窗   | 256      | gid 4                    |
| full_attention 层   | 全局       | **512**  | gid 5                    |

问题的核心就在"draft 要像 target 一样，按 group 切换 block_table"。

## 2. 现象

在 Ascend A2（910B4）上跑 Gemma4 MTP，draft 输出与 L20（CUDA + FlashAttention）参考实现**数值发散**：

- A2 上 draft `full_attention` 层输出 std ≈ **0.912**，L20 参考 std ≈ **1.334**——量级就对不上。
- K=1（speculative 接受 1 个 token）的接受率只有 **50%–78%**，远低于预期。

接受率低 + 数值发散，第一反应是"是不是 head_dim=512 的 attention kernel 算错了？"

## 3. 排查过程（一个被排除的假说）

团队先怀疑是 Ascend 的 head_dim=512 attention kernel 本身有缺陷，写了一套**脱离 vLLM 框架的独立算子验证**（`test_pa_512.py` / `tools/test_pa_vs_fa.py`）：用 CPU 生成固定 seed 的 Q/K/V，`.to(device)` 后分别喂给各候选 kernel，再和 CPU SDPA 参考逐元素比对。

结论（见 `tools/test_pa_512.py`，A2 实测）：

| 候选 kernel                          | head_dim=512 表现                                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------------ |
| `npu_fusion_attention`（TND）       | ✅ **MATCH**，max_abs=0.000244（fp16 正常误差）                                            |
| `npu_fused_infer_attention_score`（FIA） | ❌ 不支持 512，报 `only headDim = 64/128/192 are supported, but got 512`                  |
| `_npu_paged_attention`（ATB PA）    | ❌ 当前 torch_npu 2.10.0 / CANN 下 ATB lib `setup failed`，无法独立跑                       |

**算子本身是对的**。这就排除了"kernel bug"假说，把方向拉回到"框架给 kernel 喂的输入（KV 地址）不对"——即 block_table 串组。

> **踩坑提示**：NPU 和 CUDA 的随机生成器不同，跨平台比对时**必须在 CPU 上生成输入再 `.to(device)`**，否则连输入都对不齐。

## 4. 根因

Bug 在 `vllm_ascend/spec_decode/llm_base_proposer.py` 的 `_propose`（真实 eager 推理路径）。原代码这样构建 draft 的 attention metadata：

```python
builder = self.draft_attn_groups[0].get_metadata_builder()   # 只取第 0 个 group 的 builder
...
attn_metadata = builder.build(0, common_attn_metadata, ...)  # 用 group[0] 的 block_table 建一份
for layer_name in self.attn_layer_names:
    per_layer_attn_metadata[layer_name] = attn_metadata       # 所有层共用这一份
```

问题：`common_attn_metadata.block_table_tensor` 属于 **gid 0（sliding，head_dim=256）**。这份 metadata 被赋给了**所有 draft 层**，包括 full_attention 层（head_dim=512，本该用 gid 5 的 block_table）。

后果链：

```
full_attention 层拿到 gid 0 的 block_table
  → 按 256-dim 的物理布局去寻址
  → 实际读到了 sliding 组的 KV cache（地址完全错位）
  → 512-dim 的 query 对错误的 K/V 做 attention
  → 输出发散（A2 std=0.912 vs L20 std=1.334）
  → draft token 质量差，K=1 接受率 50-78%
```

一句话：**draft 没有像 target 那样按 group 切换 block_table，导致 full_attention 层读到了 sliding 组的 KV cache。**

## 5. 修复方案

### 5.1 抽取共享 util

在 `llm_base_proposer.py` 模块级新增 `build_per_group_layer_attn_metadata()`，把"逐 group 浅拷贝 common metadata + 装上该 group 的 block_table"这个模式收口到一个地方（与 target `model_runner` 的 `_prepare_inputs` 同模式）：

```python
def build_per_group_layer_attn_metadata(
    draft_attn_groups, common_attn_metadata,
    per_group_block_tables, num_reqs, build_attn_metadata,
):
    per_layer_attn_metadata = {}
    for attn_group in draft_attn_groups:
        gid = attn_group.kv_cache_group_id
        if per_group_block_tables is not None and gid in per_group_block_tables:
            cm_group = copy.copy(common_attn_metadata)                        # 浅拷贝
            cm_group.block_table_tensor = per_group_block_tables[gid][:num_reqs]  # 换上本组 block_table
        else:
            cm_group = common_attn_metadata
        md = build_attn_metadata(cm_group, attn_group)
        for layer_name in attn_group.layer_names:                             # 只赋给本组的层
            per_layer_attn_metadata[layer_name] = md
    return per_layer_attn_metadata
```

`build_attn_metadata` 是回调，由各调用方决定具体用 `build` / `build_for_graph_capture` / `build_for_drafting`，以及 attn_state、SAS extra args 等差异——util 只管"group 切换"这件正交的事。

### 5.2 三处调用点全部接入

| 调用点                                                 | 路径                          | 说明                                       |
| ----------------------------------------------------- | ----------------------------- | ------------------------------------------ |
| `_propose`                                            | 真实 eager 推理（**唯一真正出 bug 的路径**） | 替换原"单 builder + 全层共用"写法          |
| `dummy_run`                                           | ACL graph 捕获路径            | 同样按 group 切换，保证图复用一致          |
| `AscendGemma4Proposer.build_per_group_and_layer_attn_metadata` | 上游 API override             | 复用同一 util                              |

### 5.3 兼容单组 draft（EAGLE/MLP）

EAGLE/MLP 这类 draft 只有一个 KV cache group，没有 `_per_group_block_tables` 属性。util 用 `getattr(self, "_per_group_block_tables", None)` 兜底，`None` 时直接复用 common block_table，行为与改前完全一致。

## 6. 踩坑记录（按踩中顺序）

1. **真正出 bug 的是 `_propose`（eager 路径），不是 `dummy_run` 也不是 `set_inputs_first_pass`。** `set_inputs_first_pass` 每次 draft step 都进，但它只准备输入、返回 cad，**不构建 attention metadata**。一开始改错地方，验证无变化。

2. **必须保留 `_propose` 里 `if self.method == "mtp": md.attn_state = SpecDecoding`。** 没有它，chunked-prefill 时 common metadata 继承 target 的 `ChunkedPrefill` 状态，会让 `forward_impl` 里 head_dim=512 的 PA gate 失败，退化到 dense-KV-gather 的 prefill fallback（`_gather_paged_kv_to_dense`），长序列直接 OOM。

3. **`_per_group_block_tables` 只在 Gemma4/step3.5 draft 上存在**，必须 `getattr` 兜底，否则 EAGLE/MLP 直接 `AttributeError`。

4. **验证手段**：env `ASCEND_MTP_PER_GROUP_DEBUG=1`。确认修复有效的特征——不同 group 的 block_table 指针不同：gid=4(sliding) sample=[5,…]，gid=5(full) sample=[12,13,…]。

5. **【重构引入的回归】多步更新循环的 `UnboundLocalError`。** 这是修复落地后 UT 抓到、真机 eager 验证没覆盖到的问题，值得单独说（见下节）。

## 7. 重构引入的回归：多步循环 `UnboundLocalError`

### 现象

修复提交前跑 UT `tests/ut/spec_decode/a2/test_eagle_proposer.py`，EAGLE K>1 路径直接报：

```
UnboundLocalError: cannot access local variable 'attn_metadata' where it is not associated with a value
```

### 根因

`_propose` 里有一段**多步 speculative 更新循环**（`for draft_index in range(1, num_speculative_tokens)`），用于 EAGLE 这种 K>1 的多 token 草稿。原代码靠一个在循环外预先建好的单一变量 `attn_metadata` 作为初值：

```python
attn_metadata = builder.build(0, ...)          # 旧：循环外的单一初值
for draft_index in range(1, num_spec):
    for attn_group in self.draft_attn_groups:
        common_attn_metadata, attn_metadata = self.attn_update_stack_num_spec_norm(
            draft_index, attn_metadata, ...)   # 把上一步的 attn_metadata 喂进去
```

per-group 重构删掉了循环外那个单一 `attn_metadata`（改成了 `per_layer_attn_metadata` 字典），却没同步改这个循环 → 首次迭代引用一个不存在的变量。

### 关键洞察：`old_attn_metadata` 是个死参数

深入 `attn_update_stack_num_spec_norm(old_attn_metadata, ...)` 函数体发现：**`old_attn_metadata` 这个形参从头到尾没被使用**——函数每次都用 `attn_group.get_metadata_builder().build_for_drafting(common_attn_metadata, draft_index)` **重建**一份新的 per-group metadata。所以循环里传进去的"上一步 attn_metadata"其实从未被读，纯粹是个占位。

### 修复

既然是死参数，循环只需要一个占位初值：

```python
attn_metadata = attn_metadata_i    # step-0 的 metadata，仅作占位
for draft_index in range(1, num_spec):
    for attn_group in self.draft_attn_groups:
        common_attn_metadata, attn_metadata = self.attn_update_stack_num_spec_norm(...)
        for layer_name in attn_group.layer_names:           # 顺带改为按组赋值
            per_layer_attn_metadata[layer_name] = attn_metadata
```

同时把内层 `for layer_name in self.attn_layer_names` 改成 `attn_group.layer_names`，消除"最后一组的 metadata 覆盖所有层"的潜在隐患（与 util 语义一致）。

> 注：Gemma4 MTP 是 K=1，`range(1,1)` 为空，**根本不进这个循环**，所以真机验证没暴露；只有 EAGLE K>1 的 UT 才会命中。这正是单测的价值。

## 8. 多步循环里的"同卵双胞胎" bug：pos1/pos2 接受率崩塌

第 5 节修复了 `_propose` 入口（初始 metadata，pos0 用）的 block_table 串组后，K=1 接受率回归正常。但 **K=3（num_speculative_tokens=3）时 pos1/pos2 仍严重低于 L20 基线**：

| | pos0 | pos1 | pos2 | 聚合 | 每轮 token |
| --- | --- | --- | --- | --- | --- |
| L20 (CUDA) 基线 | 98.2% | 95.5% | 90.0% | — | — |
| A2（仅修 pos0 后） | 95.0% | 76.8% | 61.2% | 77.7% | 2.33 |

pos0 基本对（差 3%），但 pos1/pos2 差距巨大且递增（-18.7 / -28.8）。这个"越往后越差"的特征直接指向**多步 draft 循环**：pos0 不进循环（它是循环前初始 forward 直接出的），pos1/pos2 才进 `for draft_index in range(1, K)` 循环。

### 根因

`_propose` 的多步循环通过 `attn_update_stack_num_spec_norm` 为每个 draft step（pos1、pos2）**重建** attention metadata。但这个函数里：

```python
common_attn_metadata = self.shallow_copy_metadata(old_common_metadata)
# ... block_table 全程是 old_common 那一份（sliding 组 gid 4），从没按 group 换
block_ids = old_common_metadata.block_table_tensor.gather(...)        # slot_mapping 用 sliding 组
attn_metadata = builder.build_for_drafting(common_attn_metadata, ...) # build 也用 sliding 组
```

`common_attn_metadata.block_table_tensor` 属于 sliding 组（gid 4，head_dim=256）。这份 metadata 被用于**所有 draft group**，包括 full_attention（gid 5，head_dim=512）。后果和第 4 节的初始 metadata bug **完全同构**——只是上次漏了多步循环这一路：

- **slot_mapping 算错**（用 sliding 组 block_table 算 block_ids）→ draft KV 写到 head_dim=256 布局的物理 block；
- **build_for_drafting 用错 block_table** → full_attention 层 attention 读 sliding 组的 KV。

pos0 用的是 `_propose` 入口已修复的初始 metadata（第 5 节的 util swap 过），所以不受影响；pos1/pos2 走多步循环重建的 metadata，没 swap → 崩。这正好解释了"pos0 好但 pos1/pos2 崩"的非对称现象。

### 修复

在 `attn_update_stack_num_spec_norm` 里，`shallow_copy_metadata` 之后、`build_for_drafting` / slot_mapping 之前，按当前 `attn_group.kv_cache_group_id` 换 block_table：

```python
common_attn_metadata = self.shallow_copy_metadata(old_common_metadata)
per_group_bts = getattr(self, "_per_group_block_tables", None)
_upd_gid = attn_group.kv_cache_group_id
if per_group_bts is not None and _upd_gid in per_group_bts:
    common_attn_metadata.block_table_tensor = per_group_bts[_upd_gid][:batch_size]
```

并把 slot_mapping 的 `old_common_metadata.block_table_tensor` 改成 swap 后的 `common_attn_metadata.block_table_tensor`。该函数本就 `for attn_group in self.draft_attn_groups` 逐组调用，每次 swap 成当前组的 block_table，正好和初始 metadata 的 util 同模式。

### 这个 bug 是怎么被遗漏的

第一轮（pos0）修复后，只测了 K=1（pos0 接受率），没测 K=3。pos0 回归正常就以为修好了，实际上多步循环这一路还坏着。**经验：多 group draft 的修复必须覆盖所有 metadata-(re)build 路径，并至少测到 K = num_speculative_tokens，不能只测 K=1。**

## 9. 关联修复：constant_draft_positions gate（非接受率根因）

排查 pos1/pos2 期间一度怀疑"多步循环里 position/seq_len 自动 +1"是根因（MTP 所有 draft step 应从同一位置预测）。上游 `gemma4.py` 注释明确："All draft steps predict from the same position"，并用 `if not self.constant_draft_positions:` gate 包住每次自增；vllm-ascend 内联了自增、丢了这个 gate。

补上 gate（3 处：forward loop 的 `positions += 1`、`attn_update_stack_num_spec_norm` 里的 `used_update_positions += 1` 和 `seq_lens[:bs] += 1`），MTP（`constant_draft_positions=True`）全跳过，EAGLE（`False`）照常自增——行为符合上游语义。

**但实测 K=3 接受率纹丝不动（76.8/61.2 → 76.8/61.2）。** 说明 position 自增不是 pos1/pos2 的根因，真正的根因是第 8 节的多步循环 block_table swap。这次修复保留（语义正确、符合上游、无副作用），但要诚实记录：**它不是治病的药。**

> 教训：当一个"修复"完全没动指标时，假设多半是错的。本次正是靠 L20 基线（95.5/90.0）推翻了"自回归固有衰减"的臆断，把方向逼回多步循环的 block_table。

## 10. 验证结果

- **数值**：A2 draft `full_attention` 输出 std 从 0.912 回到与 L20 对齐（~1.334 量级）。
- **接受率（K=3，coding prompts，greedy）**：

  | | pos0 | pos1 | pos2 | 聚合 | 每轮 token |
  | --- | --- | --- | --- | --- | --- |
  | L20 基线 | 98.2% | 95.5% | 90.0% | — | — |
  | A2 修复后 | **98.2%** | **95.4%** | **91.1%** | **94.9%** | **2.85** |

  pos1 从 76.8→95.4（+18.6），pos2 从 61.2→91.1（+29.9），**完全追平 L20**。
- **group 隔离**：初始 metadata + 多步循环两处都按 group swap，gid=4 与 gid=5 block_table 不同，full_attention 层全程读自己的 KV。
- **UT**：`tests/ut/spec_decode/a2/` → **69 passed / 13 skipped**。
- **Lint**：新增代码行宽 ≤120。

## 11. 启示与可复用经验

1. **数值发散先别急着怪 kernel。** 用 CPU-seed 输入做脱离框架的独立算子比对，能快速把"算子错"和"框架喂错"二分。本次正是这一步把方向从"kernel bug"扳回"block_table 串组"。

2. **多 KV cache group 模型（Gemma4、未来更多混合注意力模型）的 draft 必须按 group 切换 block_table**，机制要和 target `model_runner` 完全对齐。任何"取 group[0] 给所有层用"的写法都是潜在 bug。

3. **正交重构要找到真正的公共模式。** 把"group 切换"独立成 util、用回调注入差异（builder 类型、attn_state），三处调用点共享一份逻辑，避免改一处漏两处。

4. **真机验证 ≠ 全路径覆盖。** eager 真机路径过了不等于 UT 路径过——EAGLE K>1 的多步循环只有 UT 能覆盖。重构后务必跑 UT，哪怕改动看起来只服务于 MTP。

5. **警惕"死参数"掩盖的耦合。** `old_attn_metadata` 这种形参让人误以为有状态依赖，实则是每次重建。重构时认清哪些是真实状态线程（`common_attn_metadata`）、哪些是占位，能避免 `UnboundLocalError` 这类错误。

6. **多 group draft 的修复要覆盖所有 metadata-(re)build 路径，并测到 K=num_speculative_tokens。** 本 bug 的 pos0（初始 metadata）和 pos1/pos2（多步循环重建）是同卵双胞胎，只修前者、只测 K=1，会以为修好了——必须测到 K=3 才暴露。同理，"修复没动指标"时别急着归因为"固有衰减"，先拿 CUDA 基线（L20）对照。

---

## 涉及文件

分支 `MTP-A2A3A5`，拆成 3 个 commit：

1. **`fix(spec_decode): swap per-group block_table in Gemma4 MTP draft`**（核心，接受率 77.7% → 94.9%）
   - `vllm_ascend/spec_decode/llm_base_proposer.py`（util + 初始 metadata swap + 多步循环 swap + slot_mapping）
   - `vllm_ascend/spec_decode/gemma4_proposer.py`（override 接 util）
   - `tests/ut/spec_decode/a2/test_eagle_proposer.py`（mock 补 `layer_names`）
   - `docs/source/developer_guide/gemma4_mtp_per_group_block_table_fix.md`（本文）
2. **`fix(spec_decode): gate per-step advance with constant_draft_positions`**（3 处 gate，符合上游语义）
3. **`fix(spec_decode): seed multi-step loop attn_metadata (UnboundLocalError)`**（多步循环回归修复）

辅助诊断脚本（未跟踪，不随修复提交）：

- `tools/test_pa_vs_fa.py`、`tools/verify_pa_output.py`、`test_pa_512.py`、`test_pa1_512.py`、`dump_craft.py`
