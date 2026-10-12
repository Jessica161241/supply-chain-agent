# 事故记录

> 本文件记录开发过程中影响交付的两次事故及其处理方式，供后续维护参考。
> 环境相关的代理问题已在 [setup_guide.md](setup_guide.md) §2 展开，此处只记事件本身。

---

## 1. 2026-10-01 · git 拉取事故导致部分文件丢失

**经过**：一次 `git pull` 后，仓库内若干未跟踪/新提交的文件从工作区消失：

| 丢失内容 | 影响 |
|---|---|
| `docs/setup_guide.md` | 环境搭建指南，README 中多处引用 |
| `docs/acceptance_checklist.md` | 验收清单 |
| `docs/risk_log.md` | 框架排查日志 |
| `scripts/` 整个目录 | 环境自检、源码拉取、骨架重建脚本 |

**恢复**：无法从 git 历史找回（这些文件当时尚未提交）。处理方式是按当前代码的真实行为**重建**：

* `scripts/quickstart_verify.py` — 按需求重写为 8 项环境自检；
* `scripts/fetch_source.sh` — 从 PyPI 拉 wheel 解压，规避框架 GitHub 404；
* `scripts/init_project_skeleton.sh` — 幂等重建目录骨架；
* `docs/setup_guide.md`、`docs/acceptance_checklist.md` — 按当前实现重新撰写。

**教训与对策**：

1. **产出即提交**。上述文件丢失的共同点是"写好了但还没 commit"。
2. 新增 `.gitattributes` 强制 `.sh` 为 LF，避免 Windows checkout 后脚本无法执行。
3. 修正 `.gitignore`：数据目录由"排除整个目录"改为"排除目录内容（`data/raw/*`）"，
   否则 git 不进入该目录，`!data/raw/.gitkeep` 之类的否定规则不生效。

---

## 2. 2026-10-10 · 本机代理失效导致 pip / git 全部失败

**症状**：

* `pip install <任何包>` → `SSLError: UNEXPECTED_EOF_WHILE_READING`，
  有时进一步误导为 `No matching distribution found`；
* `git fetch/pull/push` → `TLS connect error: unexpected eof while reading`；
* 同一时刻 `curl` 访问同一地址**正常**。

**根因**：本机配置了一个本地 HTTP 代理 `http://127.0.0.1:7899`（Clash/V2Ray 类工具留下），
代理进程已退出但配置仍在。curl 只读环境变量，pip 的 urllib 还会读 Windows 注册表，
git 读自己的 config —— 三者取代理的路径不同，所以表现为"curl 好、pip/git 坏"。

**隐蔽之处**：`git fetch` 失败时，`git status` 仍显示
"Your branch is up to date with origin/main" —— 失败的 fetch **不会**更新
`origin/main`，因此远程的新提交看起来像"对方还没推"，极易误判。

**处理**：

```bash
# pip：清环境变量
export HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" NO_PROXY="*"

# git：不读环境变量，走自己的 config，用一次性覆盖
git -c http.proxy= -c https.proxy= fetch origin
git -c http.proxy= -c https.proxy= pull origin main
```

> 该代理后来恢复正常运行，此时**直连反而更慢或被重置**，应改回走代理。
> 排查时先确认端口上到底有没有服务监听：
> `netstat -an | grep 7899`，再分别用 `curl --max-time 12` 与
> `curl -x http://127.0.0.1:7899 --max-time 12` 各测一次，用实测结果决定走哪条路。

**教训**：网络类故障先确认"代理是否活着"，不要默认某一种走法永远正确。
