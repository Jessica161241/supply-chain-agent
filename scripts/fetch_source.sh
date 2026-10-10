#!/usr/bin/env bash
# ============================================================================
# fetch_source.sh —— 拉取 DeepBullwhip 源码到 vendor/ 备查
#
# 背景：`https://github.com/ai-vnv/deepbullwhip` 返回 404，无法 git clone。
#       但 PyPI 上的 wheel 是**纯 Python**，解压后即含全部源码，可直接阅读、
#       在 docs/AB_interface_checklist.md 的"实测结论"里逐条比对。
#
# 用法：
#   bash scripts/fetch_source.sh              # 等价于 --wheel
#   bash scripts/fetch_source.sh --wheel      # pip download wheel 并解压
#   bash scripts/fetch_source.sh --version 0.4.1
#   bash scripts/fetch_source.sh --clean      # 先清空 vendor/ 再拉取
#
# 产物：vendor/deepbullwhip-<ver>-py3-none-any.whl
#       vendor/deepbullwhip-wheel/deepbullwhip/...
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${PROJECT_ROOT}/vendor"
VERSION="0.4.1"
DO_CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheel)   shift ;;                       # 默认行为，保留兼容
    --version) VERSION="$2"; shift 2 ;;
    --clean)   DO_CLEAN=1; shift ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

echo "==> 项目根: ${PROJECT_ROOT}"
echo "==> 目标版本: deepbullwhip==${VERSION}"

# 失效代理是本机常见故障源（pip 全量失败但 curl 正常）——先清干净
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy 2>/dev/null || true
export NO_PROXY="*"

if [[ "${DO_CLEAN}" -eq 1 ]]; then
  echo "==> --clean: 清空 ${VENDOR_DIR}"
  rm -rf "${VENDOR_DIR}"
fi
mkdir -p "${VENDOR_DIR}"

# 用项目自带虚拟环境的 python 优先，否则退回 PATH 上的 python
PYTHON="python"
for cand in "${PROJECT_ROOT}/.venv/Scripts/python.exe" "${PROJECT_ROOT}/.venv/bin/python"; do
  if [[ -x "${cand}" ]]; then PYTHON="${cand}"; break; fi
done
echo "==> 使用解释器: ${PYTHON}"

echo "==> pip download ..."
"${PYTHON}" -m pip download "deepbullwhip==${VERSION}" \
  --no-deps --no-binary :none: \
  -d "${VENDOR_DIR}"

WHEEL="$(ls -1 "${VENDOR_DIR}"/deepbullwhip-"${VERSION}"-*.whl 2>/dev/null | head -n1 || true)"
if [[ -z "${WHEEL}" ]]; then
  echo "!! 未找到 wheel，检查 ${VENDOR_DIR}" >&2
  exit 1
fi
echo "==> wheel: ${WHEEL}"

# wheel 就是 zip，直接解压即得源码树
EXTRACT_DIR="${VENDOR_DIR}/deepbullwhip-wheel"
rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"
"${PYTHON}" -c "
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(sys.argv[2])
    n = len(z.namelist())
print(f'    解压 {n} 个条目')
" "${WHEEL}" "${EXTRACT_DIR}"

echo
echo "==> 完成。源码位置："
echo "    ${EXTRACT_DIR}/deepbullwhip/"
echo
echo "    常用查阅入口（对应 docs/AB_interface_checklist.md 的实测结论）："
echo "      forecast/base.py       §1.1  forecast() 返回标量"
echo "      datasets/loader.py     §1.2  M5 只有 store + dept"
echo "      chain/network_sim.py   §1.4  forecasts 只对叶子节点生效"
echo "                             §1.5  反向拓扑序 -> 跨节点信号可行"
echo "      metrics/inventory.py   §1.6  fill_rate 口径"
echo "      chain/echelon.py       §1.7  EdgeConfig.lead_time 无效"
