# HYG LLaVA-SLoFo：Slurm 工作手册

更新日期：2026-08-13
账号：`202400390068@10.102.36.220`（本机 SSH 别名：`school-gpu`）

## 1. 已可继续工作的结论

项目已迁移到个人私有目录，LLaVA-1.5-7B、SLoFo 扫描定位、四阶段回答链路和 TextVQA/GQA/POPE 批处理入口均已在 Slurm 的 RTX 4090 上实际执行成功。

这里的“验收通过”表示：环境、权重、图片、manifest、模型推理、SLoFo 定位、Focus 剪枝、结果写入与 Slurm 调度链路完整。它不等于论文精度已经复现；例如最小 GQA 验收题的模型回答与标注不一致，这属于后续算法效果优化对象。

## 2. 只使用这些个人路径

- 项目根目录：`/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo`
- 代码：`/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/repo`
- Conda 环境：`/labmount/users/202400390068/envs/HYG_LLaVA_SLoFo/llava-slofo-py310`
- 模型：`/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/models`
- 固定 benchmark 子集：`/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/benchmarks`
- 结果：`/labmount/users/202400390068/results/HYG_LLaVA_SLoFo`
- 日志：`/labmount/users/202400390068/logs/HYG_LLaVA_SLoFo`

不要浏览、复制、改名或删除其他账号的项目。不要操作作业 `2611 / MSE-Adapter-train`。

## 3. VS Code 中打开项目

1. 本机先连接 aTrust。
2. VS Code 选择 “Remote-SSH: Connect to Host”，连接 `school-gpu`。
3. 选择 “Open Folder”，输入：
   `/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/repo`
4. VS Code 菜单选择“终端 → 新建终端”。
5. 在远程终端执行：

```bash
source /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/activate_slurm.sh
which python
python --version
```

期望 Python 路径位于个人环境中，版本为 3.10.20。

VS Code 不只是查看和改代码：它的远程终端也能执行 `sinfo`、`squeue`、`sbatch`、查看日志等命令。但 GPU 推理必须通过 `sbatch` 或 `srun` 申请计算节点，不能直接在登录节点运行。

## 4. 查看队列，不干扰他人

```bash
sinfo -p GPUNorm -N -o '%N %t %G'
squeue -p GPUNorm -o '%.18i %.18u %.24j %.9T %.10M %.6D %R'
squeue -u 202400390068 -o '%.18i %.24j %.9T %.10M %R'
```

节点显示 `idle` 只能说明 Slurm 视角可分配，不能完全证明具体 GPU 显存为空。因此所有项目作业模板都在模型加载前调用：

```bash
/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/preflight_empty_gpu.sh
```

默认要求所分配 GPU 的已用显存不超过 512 MiB，否则作业自动退出，不加载模型。该机制已由作业 `2686` 实测：gn8 / RTX 4090 / 1 MiB / 0%。

## 5. 最小验收命令

先做空卡预检：

```bash
sbatch /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/gpu_preflight_smoke.sbatch
```

基础 LLaVA：

```bash
sbatch /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/llava_smoke.sbatch
```

SLoFo + Focus：

```bash
sbatch /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/slofo_smoke.sbatch
```

单题 benchmark 链路：

```bash
sbatch /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/textvqa_batch_smoke.sbatch
sbatch /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/gqa_batch_smoke.sbatch
sbatch /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/pope_batch_smoke.sbatch
```

已验证作业：

| 作业 | 节点 | 状态 | 关键结果 |
|---|---|---|---|
| 2677 LLaVA | gn8 | COMPLETED 0:0 | 576 个视觉 Token，峰值约 14.84 GiB |
| 2678 SLoFo | gn8 | COMPLETED 0:0 | Scan-Locate + Focus，576→288→144→72，峰值约 17.36 GiB |
| 2679 TextVQA | gn8 | COMPLETED 0:0 | 题 34607 回答 22，与标注一致 |
| 2682 GQA | gn8 | COMPLETED 0:0 | 数据与推理链路通过；回答 No，标注 yes |
| 2683 POPE | gn8 | COMPLETED 0:0 | 回答 Yes，与标注一致 |
| 2686 空卡预检 | gn8 | COMPLETED 0:0 | 4090 初始 1 MiB、0% |

## 6. 跑此前固定子集（建议串行）

当前已迁移并逐路径校验：

- TextVQA：512/512 题与图片可用；
- GQA：512/512 题与图片可用；
- POPE：600/600 题与图片可用。

为了共享服务器公平和避免多卡并发，建议串行提交：

```bash
j1=$(sbatch --parsable /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/textvqa_subset_full.sbatch)
j2=$(sbatch --parsable --dependency=afterok:$j1 /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/gqa_subset_full.sbatch)
j3=$(sbatch --parsable --dependency=afterok:$j2 /labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/slurm/pope_subset_full.sbatch)
echo "$j1 $j2 $j3"
```

这些模板不带 `--force`：若输出目录中已有成功案例，批处理脚本会跳过，适合断点续跑。若要做不同配置实验，应修改为新的输出目录，不要覆盖旧实验。

结果路径：

- `results/HYG_LLaVA_SLoFo/official_fixed_subset/textvqa_512`
- `results/HYG_LLaVA_SLoFo/official_fixed_subset/gqa_512`
- `results/HYG_LLaVA_SLoFo/official_fixed_subset/pope_600`

## 7. 查看作业和日志

```bash
squeue -u 202400390068
sacct -j 作业号 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
tail -f /labmount/users/202400390068/logs/HYG_LLaVA_SLoFo/对应日志文件
```

正常完成应看到 `COMPLETED` 和 `ExitCode 0:0`。仅看到 COMPLETED 还不够，还要检查对应结果目录中的 `batch_summary.json`、`benchmark_answers.jsonl` 和案例 `result.json`。

## 8. 数据范围与完整官方下载

当前数据是此前实验实际使用的 1624 题固定子集，不是三套官方数据集的全部原始图片。它能无缝续跑已有消融和比较实验，并将迁移流量从约 28 GiB 降到约 236 MiB。

仓库保留了以下按需入口：

- `scripts/download_vqa_benchmarks.sh`
- `scripts/extract_vqa_benchmarks.sh`

完整官方图像压缩包合计约十几 GiB，且解压阶段会暂时同时占用压缩包与图片。服务器网络流量有限，因此不要直接并发执行旧下载脚本。需要全量官方评测时，按 TextVQA、GQA、COCO/POPE 分别下载；每个数据集完成完整性检查并解压后删除自己的压缩包，再进行下一个。GQA 下载与旧解压脚本的路径命名需要先统一，不能盲目一键执行。

## 9. 固定版本

版本记录：`/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/PINNED_VERSIONS.txt`

关键版本：

- 项目提交：`c452a0763d7215075ccaaaeff1c17311155b4922`
- LLaVA：`c121f0432da27facab705978f83c4ada465e46fd`
- LLaVA-1.5-7B 权重 revision：`4481d270cc22fd5c4d1bb5df129622006ccd9234`
- CLIP ViT-L/14-336 revision：`ce19dc912ca5cd21c8a653c79e251e808ccabcd1`
- PyTorch 2.1.2+cu121；Transformers 4.37.2；NumPy 1.26.4。

当前使用 FP16，不依赖 bitsandbytes。不要为 4-bit 随意升级 PyTorch 或 CUDA；如需 4-bit，应另建独立环境。

## 10. 禁止事项

- 不在登录节点直接加载模型或运行 GPU 推理；
- 不使用、修改或停止其他人的作业；
- 不把模型权重、完整 benchmark 或 Conda 环境提交进 Git；
- 不在现有环境中无版本约束地执行 `pip install -U`；
- 不同时提交三套全量任务抢占多张卡；
- 不删除旧服务器的实验归档，直到 GitHub Release 上传并校验完成。
