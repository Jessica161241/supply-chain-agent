#!/usr/bin/env bash
# ============================================================================
# init_project_skeleton.sh —— 重建项目骨架（幂等，绝不覆盖已有文件）
#
# 用途：新队友拉下仓库后补齐目录结构；或意外删了空目录/空包时一键恢复。
#
# 行为：
#   * 只创建**缺失的目录**；
#   * 只创建**缺失的 __init__.py**（已存在的文件一个字节都不动）；
#   * 在 data/raw、data/processed 放 .gitkeep，让空目录能被 git 跟踪；
#   * 末尾打印实际创建了什么，没创建任何东西也明确说明。
#
# 用法：
#   bash scripts/init_project_skeleton.sh
#   bash scripts/init_project_skeleton.sh --dry-run    # 只报告将要创建的内容
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

created_dirs=()
created_files=()
skipped=()

mkdir_if_missing() {
  local d="${PROJECT_ROOT}/$1"
  if [[ -d "${d}" ]]; then
    skipped+=("dir  $1")
  else
    created_dirs+=("$1")
    [[ "${DRY_RUN}" -eq 1 ]] || mkdir -p "${d}"
  fi
}

touch_if_missing() {
  local f="${PROJECT_ROOT}/$1"
  if [[ -e "${f}" ]]; then
    skipped+=("file $1")
  else
    created_files+=("$1")
    [[ "${DRY_RUN}" -eq 1 ]] || { mkdir -p "$(dirname "${f}")"; : > "${f}"; }
  fi
}

# ---- 目录 ----------------------------------------------------------------
for d in \
  config data data/raw data/processed \
  src src/forecasting src/network src/coordination src/simulation src/visualization \
  agent notebooks tests scripts docs \
  outputs outputs/figures outputs/results ; do
  mkdir_if_missing "${d}"
done

# ---- Python 包标记 --------------------------------------------------------
touch_if_missing "src/__init__.py"
touch_if_missing "src/forecasting/__init__.py"
touch_if_missing "src/network/__init__.py"
touch_if_missing "src/coordination/__init__.py"
touch_if_missing "src/simulation/__init__.py"
touch_if_missing "src/visualization/__init__.py"
touch_if_missing "agent/__init__.py"
touch_if_missing "tests/__init__.py"

# ---- 空目录占位（data/ 下的数据文件本身被 .gitignore 忽略）----------------
# 注：outputs/figures、outputs/results 已有产物，不需要 .gitkeep
touch_if_missing "data/raw/.gitkeep"
touch_if_missing "data/processed/.gitkeep"

# ---- 报告 ----------------------------------------------------------------
echo "======================================================================"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "  项目骨架 —— DRY RUN（未做任何改动）  root=${PROJECT_ROOT}"
else
  echo "  项目骨架 —— 已就绪  root=${PROJECT_ROOT}"
fi
echo "======================================================================"

if [[ ${#created_dirs[@]} -gt 0 ]]; then
  echo "新建目录 (${#created_dirs[@]})："
  printf '  + %s\n' "${created_dirs[@]}"
fi
if [[ ${#created_files[@]} -gt 0 ]]; then
  echo "新建文件 (${#created_files[@]})："
  printf '  + %s\n' "${created_files[@]}"
fi
if [[ ${#created_dirs[@]} -eq 0 && ${#created_files[@]} -eq 0 ]]; then
  echo "无缺失项：所有目录与占位文件均已存在，未做任何改动。"
fi

echo
echo "已存在、跳过 (${#skipped[@]})："
printf '  = %s\n' "${skipped[@]}"
