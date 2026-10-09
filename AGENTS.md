# Repository instructions

<!-- fastctx:begin -->
## Local file inspection

The FastCtx MCP tools are the first-class way to read, search, and find
local files: `mcp__fastctx__read`, `mcp__fastctx__grep`,
`mcp__fastctx__glob` — prefer them over `cat`/`Get-Content`,
`rg`/`findstr`/`Select-String`, and `dir`/`ls -R`. Pass absolute paths. The
last line of every result says `Complete` or `Partial` — continue only with
the exact parameters a `Partial` note provides.

### Batch replacement

Use `mcp__fastctx__replace` for mechanical find-and-replace across files.
It preserves each file's encoding and line endings, supports dry-run previews,
and rejects concurrent changes before writing. Use apply_patch for generated
content, semantic rewrites, or small local edits.
<!-- fastctx:end -->

## GitHub checkpoints

用户要求：每次修改前后都要提交到 GitHub。

- 每组逻辑修改前，检查分支、远程和工作区，将本任务范围内已有的有效改动提交并推送到对应 GitHub 分支，形成修改前检查点。若没有待提交的有效改动且当前 HEAD 已在远程，直接使用该提交，不创建无意义的空提交。
- 每组逻辑修改后，进行与修改风险相称的验证，提交改动并推送，再核对远程分支哈希。不能把仅完成本地 commit 描述为已经提交到 GitHub。
- 提交前检查差异、文件大小和敏感信息。保留源代码、配置、测试、论文源文件和需要交付的图稿；不提交密钥、数据集、模型权重、实验缓存、临时传输、依赖目录、编译日志或字节码。
- 精确暂存本任务文件，保留其他任务的未提交改动。发现同一文件正在被其他进程修改时，先确认稳定快照；不能覆盖并发改动。
- 推送失败或远程分叉时保留本地提交并报告，不强推、不重置用户改动；修改前检查点未同步成功时不继续新的修改。
- 只读检查和健康实验监控不需要空提交。本规则不授权更改运行中的实验协议、参数、种子、代码快照或封存产物。
