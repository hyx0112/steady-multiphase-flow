#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OLGA 批量调用全流程（单文件 · 只用标准库）

从 ramp4000 战役里实际跑过 3,544 例的批处理代码抽取而成，去掉对仓库目录结构、
pandas/numpy/yaml 的依赖，拷到一个文件即可在任意机器上用。

    python olga_batch_flow.py check  --genkey-dir <目录> [--pvt-dir <PVT目录>]
    python olga_batch_flow.py run    --genkey-dir <目录> --work <目录> --workers 12
    python olga_batch_flow.py status --work <目录>
    python olga_batch_flow.py curves --work <目录> --out curves.csv [--every 10]

输入约定
--------
* 一个算例 = 一个 ``<case_id>.genkey``（+ 它引用的 PVT 表）；
* genkey 内必须有 ``FILES PVTFILE=./<名字>.tab`` 这一行，工具据此把 PVT 表一起拷进
  算例工作目录（OLGA 只在当前目录找它）；
* ``--genkey-dir`` 递归扫描 ``*.genkey``，算例号取文件名主干；也可以用
  ``--cases cases.csv`` 显式给表（列：case_id, genkey[, pvt][, fluid][, scenario]）。

跑法
----
每个算例复制到 ``<work>/<case_id>/`` 后，在该目录内执行::

    OLGA-2022.1.0.exe -consoleLog <case_id>.genkey

环境变量 ``LM_LICENSE_FILE`` / ``SLBSLS_LICENSE_FILE`` 指向 license。
判定「跑完」= ``.out`` 里有 ``TOTAL EXECUTION TIME:`` **且** ``.tpl`` **且** ``.ppl``
三个都在（OLGA 退出码为 0 但不写全这几个文件时按失败处理）。

断点续跑 ``--resume``：上述判定已成立的算例直接跳过，可反复中断/重启。

产物
----
* ``<work>/<case_id>/``        原始产物（.out/.tpl/.ppl/.log/.rsw …）
* ``<work>/_logs/<case_id>.log``  stdout+stderr
* ``<work>/_status/<case_id>.json`` 单例状态（含每次尝试的 rc/耗时）
* ``<work>/run_summary.csv``     一行一例的汇总
* ``curves`` 输出的长表：``case_id,pipe,branch,scope,nr,x_m,t_s,PT_MPa,TM_C,HOL,
  UG_m_s,UL_m_s`` —— 每个 (pipe, 截面) 一条时间序列，``x_m`` 取单元中心，
  ``scope`` 区分 SECTION/BOUNDARY；``--every N`` 抽稀但保留最后一个时刻。

注意
----
* genkey 的算例设计（模板/参数改写）不在本工具内，输入是现成的 .genkey。
* 工作目录建议保持纯 ASCII 路径（opi/OLGA 对非 ASCII 工作目录历史上有问题）。
* ``curves`` 默认先查 ``.out`` 里的 ``TOTAL EXECUTION TIME:`` 再解析：OLGA 运行中
  会持续往 ``.tpl`` 追加，半截文件会读出假的时间序列（``--allow-partial`` 可跳过该闸）。
* 同一 BRANCH 下串多个 PIPE 时，用 ``--pipe-segments PIPE-1=100,PIPE-2=50`` 指定
  各管单元数；不给则按 PIPE 个数均分。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------- 常量
PVT_RE = re.compile(r"^FILES PVTFILE=\./([^,\s]+)\s*$", re.M)
DONE_MARKER = b"TOTAL EXECUTION TIME:"
SUMMARY_COLS = ["case_id", "fluid", "scenario", "status", "rc", "attempts",
                "elapsed_sec", "out_bytes", "tpl", "ppl", "error", "finished_utc"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------- 环境
def find_solver(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if os.environ.get("OLGA_EXE"):
        return Path(os.environ["OLGA_EXE"])
    roots = [Path("D:/Program Files/Schlumberger"), Path("C:/Program Files/Schlumberger"),
             Path("C:/Program Files (x86)/Schlumberger")]
    pats = ["OLGA*/OlgaExecutables/OLGA-*.exe", "OLGA*/OlgaExecutables/opi.exe"]
    for root in roots:
        if not root.is_dir():
            continue
        for pat in pats:
            hits = sorted(root.glob(pat))
            if hits:
                return hits[-1]
    return Path("OLGA-2022.1.0.exe")


def find_license(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    for key in ("SLBSLS_LICENSE_FILE", "LM_LICENSE_FILE"):
        if os.environ.get(key):
            return Path(os.environ[key])
    for p in ("D:/Program Files/Schlumberger/license.lic",
              "C:/Program Files/Schlumberger/license.lic"):
        if Path(p).exists():
            return Path(p)
    return None


# --------------------------------------------------------------------- 算例表
@dataclass
class Case:
    case_id: str
    genkey: Path
    pvt: Path | None = None
    fluid: str = ""
    scenario: str = ""

    def resolve_pvt(self) -> Path | None:
        """PVT 表：显式指定优先；否则取 genkey 里 FILES PVTFILE=./<名> 的同目录文件。"""
        if self.pvt is not None and self.pvt.exists():
            return self.pvt
        try:
            m = PVT_RE.search(self.genkey.read_text(encoding="ascii", errors="replace"))
        except OSError:
            m = None
        if m:
            name = m.group(1)
            for cand in (self.genkey.parent / name, Path(name)):
                if cand.exists():
                    return cand
        return self.pvt          # 找不到就原样返回（不存在时由调用方报明细）


def _case_from_genkey(path: Path, pvt_dir: Path | None) -> Case:
    gk = path.resolve()
    c = Case(case_id=gk.stem, genkey=gk, fluid=gk.parent.name)
    if pvt_dir:
        try:
            m = PVT_RE.search(gk.read_text(encoding="ascii", errors="replace"))
            if m:
                c.pvt = pvt_dir / m.group(1)
        except OSError:
            pass
    return c


def _resolve_path(raw: str, base: Path) -> Path:
    """相对路径先按表所在目录解析，找不到再按当前目录（两种习惯都常见）。"""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    cand = base / p
    return cand if cand.exists() or not (Path.cwd() / p).exists() else Path.cwd() / p


def load_cases(args) -> list[Case]:
    cases: list[Case] = []
    if args.cases:
        pvt_dir = Path(args.pvt_dir) if args.pvt_dir else None
        base = Path(args.cases).resolve().parent
        with open(args.cases, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                gk = _resolve_path(row["genkey"], base)
                c = Case(case_id=row.get("case_id") or gk.stem, genkey=gk,
                         fluid=row.get("fluid", "") or gk.parent.name,
                         scenario=row.get("scenario", ""))
                if row.get("pvt"):
                    c.pvt = _resolve_path(row["pvt"], base)
                elif pvt_dir:
                    c.pvt = _case_from_genkey(gk, pvt_dir).pvt
                cases.append(c)
    elif args.genkey_dir:
        root = Path(args.genkey_dir)
        pvt_dir = Path(args.pvt_dir) if args.pvt_dir else None
        for gk in sorted(root.rglob("*.genkey")):
            cases.append(_case_from_genkey(gk, pvt_dir))
    else:
        raise SystemExit("需要 --genkey-dir 或 --cases")

    if args.only:
        cases = [c for c in cases if args.only in c.case_id]
    if args.scenario:
        cases = [c for c in cases if c.scenario == args.scenario]
    if args.limit:
        cases = cases[: args.limit]
    return cases


# --------------------------------------------------------------------- 判定
def is_completed(work_dir: Path, stem: str) -> bool:
    out = work_dir / f"{stem}.out"
    if not (out.exists() and (work_dir / f"{stem}.tpl").exists()
            and (work_dir / f"{stem}.ppl").exists()):
        return False
    try:
        return DONE_MARKER in out.read_bytes()
    except OSError:
        return False


def run_once(case: Case, solver: Path, license_path: Path | None,
             work_root: Path, timeout: int) -> dict:
    pvt = case.resolve_pvt()
    missing = []
    if not case.genkey.exists():
        missing.append(f"genkey {case.genkey}")
    elif pvt is None:
        missing.append("genkey 里没有 FILES PVTFILE=./<名> 行")
    elif not pvt.exists():
        missing.append(f"PVT 表 {pvt}")
    if missing:                       # 先校验再清目录：不缺文件才动已有结果
        return {"status": "failed", "error": "缺: " + "; ".join(missing),
                "elapsed_sec": 0.0, "rc": None}

    work_dir = work_root / case.case_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    shutil.copy2(case.genkey, work_dir / case.genkey.name)
    shutil.copy2(pvt, work_dir / pvt.name)

    env = os.environ.copy()
    if license_path is not None:
        env["LM_LICENSE_FILE"] = str(license_path)
        env["SLBSLS_LICENSE_FILE"] = str(license_path)

    log_path = work_root / "_logs" / f"{case.case_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rc, error = None, ""
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.run([str(solver), "-consoleLog", case.genkey.name],
                                  cwd=str(work_dir), env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            error = f"timeout {timeout}s"
        except OSError as exc:
            error = f"OSError: {exc}"
    elapsed = time.monotonic() - started

    stem = case.genkey.stem
    done = is_completed(work_dir, stem)
    out = work_dir / f"{stem}.out"
    if not done and not error:
        error = f"rc={rc}, 未见完成标志（TOTAL EXECUTION TIME + .tpl + .ppl）"
    return {"status": "ok" if done else "failed", "elapsed_sec": round(elapsed, 3),
            "rc": rc, "error": "" if done else error, "work_dir": str(work_dir),
            "out_bytes": out.stat().st_size if out.exists() else 0,
            "tpl": (work_dir / f"{stem}.tpl").exists(),
            "ppl": (work_dir / f"{stem}.ppl").exists()}


def run_case(case: Case, solver: Path, license_path: Path | None, work_root: Path,
             timeout: int, retries: int, retry_delay: float) -> dict:
    attempts, result = [], {}
    for attempt in range(retries + 1):
        result = run_once(case, solver, license_path, work_root, timeout)
        attempts.append({k: result.get(k) for k in ("status", "elapsed_sec", "rc", "error")})
        if result["status"] == "ok":
            break
        if attempt < retries:
            time.sleep(retry_delay)
    result.update(case_id=case.case_id, fluid=case.fluid, scenario=case.scenario,
                  attempts=len(attempts), attempt_log=attempts, finished_utc=_now())
    status_dir = work_root / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{case.case_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def write_summary(work_root: Path) -> int:
    status_dir = work_root / "_status"
    rows, seen = [], set()
    for path in sorted(status_dir.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        seen.add(d.get("case_id") or path.stem)
        rows.append({k: d.get(k, "") for k in SUMMARY_COLS})
    # --resume 跳过的算例不写状态 JSON；按完成判定补一行，免得汇总漏例
    if work_root.is_dir():
        for d in sorted(p for p in work_root.iterdir()
                        if p.is_dir() and not p.name.startswith("_")):
            if d.name in seen or not is_completed(d, d.name):
                continue
            out = d / f"{d.name}.out"
            rows.append({"case_id": d.name, "fluid": "", "scenario": "",
                         "status": "ok", "rc": 0, "attempts": 0, "elapsed_sec": "",
                         "out_bytes": out.stat().st_size,
                         "tpl": True, "ppl": True, "error": "", "finished_utc": ""})
    with (work_root / "run_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# --------------------------------------------------------------------- 子命令
def cmd_check(args) -> int:
    solver = find_solver(args.solver)
    lic = find_license(args.license)
    cases = load_cases(args)
    print(f"solver   : {solver}  {'OK' if solver.exists() else 'MISSING'}")
    print(f"license  : {lic}  {'OK' if (lic and lic.exists()) else 'MISSING'}")
    print(f"cases    : {len(cases)}")
    bad_gk = [c.case_id for c in cases if not c.genkey.exists()]
    bad_pv = [c.case_id for c in cases if c.resolve_pvt() is None
              or not c.resolve_pvt().exists()]
    bad_id = [c.case_id for c in cases if not c.case_id.isascii()]
    print(f"缺 genkey: {len(bad_gk)}  缺 PVT: {len(bad_pv)}  非 ASCII 算例号: {len(bad_id)}")
    for tag, lst in (("genkey", bad_gk), ("pvt", bad_pv), ("non-ascii", bad_id)):
        for cid in lst[:10]:
            print(f"  MISS {tag}: {cid}")
    if args.work and not str(Path(args.work).resolve()).isascii():
        print(f"  注意: 工作目录含非 ASCII 字符，OLGA 可能不认: {args.work}")
    ok = (solver.exists() and lic is not None and lic.exists()
          and not bad_gk and not bad_pv and not bad_id)
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_run(args) -> int:
    solver = find_solver(args.solver)
    lic = find_license(args.license)
    if not solver.exists():
        raise SystemExit(f"缺 OLGA 可执行文件: {solver}（用 --solver / OLGA_EXE 指定）")
    work_root = Path(args.work or "olga_runs").resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    if not str(work_root).isascii():
        print(f"[warn] 工作目录含非 ASCII 字符，OLGA 可能不认: {work_root}", flush=True)
    cases = load_cases(args)
    if not cases:
        raise SystemExit("没有算例")

    if args.dry_run:
        print(f"[dry-run] {len(cases)} 例, work={work_root}")
        for c in cases[:5]:
            print(f"  {c.case_id}: cd {work_root / c.case_id} && "
                  f"{solver} -consoleLog {c.genkey.name}")
        return 0

    todo = []
    skipped = 0
    for c in cases:
        if args.resume and is_completed(work_root / c.case_id, c.genkey.stem):
            skipped += 1
            continue
        todo.append(c)

    print(f"OLGA: {len(cases)} 例（跳过已完成 {skipped}，本次跑 {len(todo)}），"
          f"workers={args.workers}", flush=True)
    counter = {"done": 0, "ok": 0, "failed": 0}
    lock = threading.Lock()
    started = time.monotonic()

    def task(c: Case) -> dict:
        return run_case(c, solver, lic, work_root, args.timeout, args.retries,
                        args.retry_delay)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(task, c): c for c in todo}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {"case_id": c.case_id, "status": "failed",
                       "error": f"runner exception: {exc}", "elapsed_sec": 0.0}
            with lock:
                counter["done"] += 1
                counter["ok" if res.get("status") == "ok" else "failed"] += 1
                el = time.monotonic() - started
                eta = el / counter["done"] * (len(todo) - counter["done"])
                print(f"[{counter['done']}/{len(todo)}] "
                      f"{'OK' if res.get('status') == 'ok' else 'FAIL'} {c.case_id} "
                      f"{res.get('elapsed_sec', 0):.1f}s "
                      f"ok={counter['ok']} fail={counter['failed']} "
                      f"elapsed={el / 60:.1f}m eta={eta / 60:.1f}m", flush=True)

    rows = write_summary(work_root)
    print(f"完成: ok={counter['ok']} fail={counter['failed']} skipped={skipped} "
          f"summary={rows} 行 -> {work_root / 'run_summary.csv'}", flush=True)
    return 0


def cmd_status(args) -> int:
    work_root = Path(args.work or "olga_runs").resolve()
    status_dir = Path(args.status_dir) if args.status_dir else work_root / "_status"
    if not status_dir.is_dir():
        raise SystemExit(f"没有状态目录: {status_dir}（用 --status-dir 指定）")
    agg, rc_hist, fails = {}, {}, []
    for path in sorted(status_dir.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = (d.get("status", "?"), d.get("rc"))
        agg[key] = agg.get(key, 0) + 1
        rc_hist[str(d.get("rc"))] = rc_hist.get(str(d.get("rc")), 0) + 1
        if d.get("status") != "ok":
            fails.append(d)
    total = sum(agg.values())
    print(f"总计 {total} 例")
    for (st, rc), n in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {st:<7} rc={rc!s:<5} {n:>6}")
    print("rc 分布: " + ", ".join(f"{k}:{v}" for k, v in
                                 sorted(rc_hist.items(), key=lambda kv: -kv[1])))
    if fails:
        print(f"\n失败 {len(fails)} 例（前 {min(10, len(fails))}）:")
        for d in fails[:10]:
            print(f"  {d.get('case_id')}: rc={d.get('rc')} {d.get('error', '')}")
    return 0


# --------------------------------------------------------------------- .tpl
# OLGA 的 .tpl（TREND PLOT）是纯 ASCII，结构固定：头部 → NETWORK → GEOMETRY
# （BRANCH 名、单元数、单元边界坐标、高程）→ CATALOG（每个趋势通道一行）
# → TIME SERIES（首列时间 s，其后每个 CATALOG 条目一列）。按行读即可，无需 numpy。
_FLOAT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
VAR_NAMES = {"PT": "PT_MPa", "TM": "TM_C", "HOL": "HOL", "UG": "UG_m_s",
             "UL": "UL_m_s", "ID": "FLOWREGIME"}


def _floats(line: str) -> list[str]:
    return _FLOAT.findall(line)


def _one_int(line: str) -> int | None:
    """整行只有一个整数（允许 400 / 400.0）时返回它，用于单元数、CATALOG 条数。"""
    toks = _floats(line)
    if len(toks) != 1 or not toks[0].replace(".", "", 1).isdigit():
        return None
    return int(float(toks[0]))


def _parse_catalog(lines: list[str], start: int, count: int) -> list[dict]:
    entries = []
    for raw in lines[start:]:
        if raw.strip().upper().startswith("TIME SERIES"):
            break
        try:
            toks = shlex.split(raw)
        except ValueError:
            toks = raw.split()
        if not toks:
            continue
        code = toks[0].upper()
        upper = [t.upper() for t in toks]
        scope = "BOUNDARY" if "BOUNDARY" in upper else "SECTION"
        pipe = branch = ""
        nr = None
        unit = ""
        desc = []
        i = 1
        while i < len(toks):
            tok = toks[i]
            key = tok.upper()
            if key in ("SECTION:", "BOUNDARY:"):
                i += 1
            elif key == "BRANCH:" and i + 1 < len(toks):
                branch = toks[i + 1]; i += 2
            elif key == "PIPE:" and i + 1 < len(toks):
                pipe = toks[i + 1]; i += 2
            elif key == "NR:" and i + 1 < len(toks):
                nr = int(float(toks[i + 1])); i += 2
            elif tok.startswith("(") and tok.endswith(")"):
                unit = tok[1:-1]; i += 1
            else:
                desc.append(tok); i += 1
        if not pipe or nr is None:
            continue
        entries.append({"code": code, "scope": scope, "pipe": pipe, "branch": branch,
                        "nr": nr, "unit": unit, "desc": " ".join(desc)})
        if len(entries) >= count:
            break
    return entries


def parse_tpl(path: Path, pipe_segments: dict[str, int] | None = None) -> dict:
    """解析 .tpl，返回 {branch, cells, centers, entries, t, data, section_x}。

    pipe_segments（如 {"PIPE-1": 100, "PIPE-2": 100}）用于同一个 BRANCH 下串了
    多个 PIPE 的情形；不给时按 PIPE 个数均分单元（余数给最后一个 PIPE）。
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    up = [ln.strip().upper() for ln in lines]
    if "TIME PLOT" not in up[:10]:
        raise ValueError(f"{path.name}: 不是 OLGA TREND PLOT（缺 TIME PLOT 头）")
    if "NETWORK" not in up:
        raise ValueError(f"{path.name}: 缺 NETWORK 块")
    net_i = up.index("NETWORK")
    geo_i = next((i for i in range(net_i, len(up)) if up[i].startswith("GEOMETRY")), None)
    if geo_i is None:
        raise ValueError(f"{path.name}: 缺 GEOMETRY 块")
    br_i = next((i for i in range(geo_i, len(up)) if up[i] == "BRANCH"), None)
    if br_i is None:
        raise ValueError(f"{path.name}: 缺 BRANCH 块")
    name_i = next((i for i in range(br_i + 1, len(up)) if up[i]), None)
    if name_i is None:
        raise ValueError(f"{path.name}: BRANCH 后没有名字")
    branch = lines[name_i].strip().strip("'\"")

    cells = body_i = None
    for i in range(name_i + 1, len(up)):
        if up[i].startswith("CATALOG"):
            break
        v = _one_int(lines[i])
        if v is not None:
            cells, body_i = v, i
            break
    if not cells or body_i is None:
        raise ValueError(f"{path.name}: BRANCH 后找不到单元数")

    cat_i = next((i for i in range(body_i, len(up)) if up[i].startswith("CATALOG")), None)
    if cat_i is None:
        raise ValueError(f"{path.name}: 缺 CATALOG 块")
    nums = []
    for i in range(body_i + 1, cat_i):
        nums.extend(float(x) for x in _floats(lines[i]))
    if len(nums) < cells + 1:
        raise ValueError(f"{path.name}: 几何坐标不足（{len(nums)} < {cells + 1}）")
    pos = nums[: cells + 1]
    elev = nums[cells + 1: cells + 1 + cells]
    if len(pos) == cells + 1:                      # 剖面给的是单元边界 → 取单元中心
        centers = [0.5 * (pos[i] + pos[i + 1]) for i in range(cells)]
    else:
        centers = pos[:cells]

    count = None
    for i in range(cat_i + 1, min(cat_i + 4, len(up))):
        v = _one_int(lines[i])
        if v is not None:
            count, entries_i = v, i + 1
            break
    if count is None:
        raise ValueError(f"{path.name}: 缺 CATALOG 条数")
    entries = _parse_catalog(lines, entries_i, count)
    if len(entries) != count:
        raise ValueError(f"{path.name}: CATALOG 只解析出 {len(entries)}/{count} 条")

    ts_i = next((i for i in range(entries_i, len(up))
                 if up[i].startswith("TIME SERIES")), None)
    if ts_i is None:
        raise ValueError(f"{path.name}: 缺 TIME SERIES 块")
    ncol = 1 + count
    t: list[float] = []
    data: list[list[float]] = []
    for raw in lines[ts_i + 1:]:
        toks = _floats(raw)
        if len(toks) != ncol:                      # 空行/残行直接跳过
            continue
        vals = [float(x) for x in toks]
        t.append(vals[0])
        data.append(vals[1:])
    if not t:
        raise ValueError(f"{path.name}: TIME SERIES 没有数据行")

    pipe_order: list[str] = []
    for e in entries:
        if e["pipe"] not in pipe_order:
            pipe_order.append(e["pipe"])
    if pipe_segments:
        seg = {k: int(v) for k, v in pipe_segments.items()}
        if sum(seg.values()) != cells:
            raise ValueError(f"{path.name}: --pipe-segments 合计 {sum(seg.values())} "
                             f"!= 单元数 {cells}")
    else:
        share, rem = divmod(cells, len(pipe_order))
        seg = {p: share for p in pipe_order}
        seg[pipe_order[-1]] += rem
    offset, acc = {}, 0
    for p in pipe_order:
        offset[p] = acc
        acc += seg.get(p, 0)

    section_x = {}
    for e in entries:
        idx = offset.get(e["pipe"], 0) + (e["nr"] - 1)
        if 0 <= idx < len(centers):
            section_x[(e["pipe"], e["nr"])] = centers[idx]
    return {"branch": branch, "cells": cells, "centers": centers, "elev": elev,
            "entries": entries, "t": t, "data": data, "section_x": section_x}


def _scaled(entry: dict, raw: float) -> float:
    """按 CATALOG 单位归一到常用单位：PT→MPa，TM→°C，其余原样。"""
    unit = entry["unit"].strip().upper()
    if entry["code"] == "PT":
        return (raw * 1.0e5 if unit == "BAR" else raw) / 1.0e6
    if entry["code"] == "TM":
        return raw - 273.15 if unit == "K" else raw
    return raw


def _fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.6g}"


def _fmt_t(v: float) -> str:
    """时间轴留 4 位小数：%.6g 会把 21600.776 舍成 21601，对不上原始时刻。"""
    return f"{v:.4f}"


def _out_done(work_dir: Path, stem: str) -> bool:
    out = work_dir / f"{stem}.out"
    if not out.exists():
        return False
    size = out.stat().st_size
    with out.open("rb") as f:                      # 完成标志在 .out 尾部
        f.seek(max(0, size - 200_000))
        return DONE_MARKER in f.read()


def cmd_curves(args) -> int:
    if args.tpl:
        files = [Path(args.tpl)]
    else:
        root = Path(args.work or "olga_runs")
        files = sorted(root.glob("*/*.tpl"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("没有 .tpl（用 --work <工作根> 或 --tpl <文件>）")

    segments = None
    if args.pipe_segments:
        segments = {}
        for part in args.pipe_segments.split(","):
            k, _, v = part.partition("=")
            if k.strip() and v.strip():
                segments[k.strip()] = int(v)

    out_path = Path(args.out or "curves.csv")
    opener = gzip.open if out_path.suffix == ".gz" else open
    written = parsed = skipped = 0
    with opener(out_path, "wt", encoding="utf-8", newline="") as fh:
        writer = None
        for path in files:
            work_dir, case_id = path.parent, path.parent.name
            if not args.allow_partial and not _out_done(work_dir, path.stem):
                skipped += 1
                continue
            try:
                tr = parse_tpl(path, segments)
            except (ValueError, OSError) as exc:
                print(f"[curves] SKIP {case_id}: {exc}", flush=True)
                skipped += 1
                continue
            entries = tr["entries"]
            cols = []
            for e in entries:
                name = VAR_NAMES.get(e["code"], e["code"].lower())
                if name not in cols:
                    cols.append(name)
            col_index = {name: i for i, name in enumerate(cols)}
            # (pipe, nr) -> {列名: CATALOG 下标}
            slots: dict[tuple[str, int], dict[str, int]] = {}
            for j, e in enumerate(entries):
                key = (e["pipe"], e["nr"])
                name = VAR_NAMES.get(e["code"], e["code"].lower())
                slots.setdefault(key, {}).setdefault(name, j)
            if writer is None:
                writer = csv.writer(fh)
                writer.writerow(["case_id", "pipe", "branch", "scope", "nr", "x_m",
                                 "t_s", *cols])

            n = len(tr["t"])
            idx = list(range(0, n, max(1, args.every)))
            if idx[-1] != n - 1:
                idx.append(n - 1)
            for (pipe, nr), mapping in slots.items():
                scope = next(e["scope"] for e in entries
                             if e["pipe"] == pipe and e["nr"] == nr)
                x = tr["section_x"].get((pipe, nr))
                for i in idx:
                    row = tr["data"][i]
                    writer.writerow(
                        [case_id, pipe, tr["branch"], scope, nr,
                         _fmt(x), _fmt_t(tr["t"][i])]
                        + [_fmt(_scaled(entries[mapping[c]], row[mapping[c]])
                                if c in mapping else None) for c in cols])
                    written += 1
            parsed += 1

    size = out_path.stat().st_size
    print(f"[curves] {parsed} 例 -> {written} 行, 跳过 {skipped} 例, "
          f"{size / 1e6:.2f} MB -> {out_path}", flush=True)
    return 0


# --------------------------------------------------------------------- 主入口
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OLGA 批量调用全流程（check / run / status / curves）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, need_cases=True):
        if need_cases:
            sp.add_argument("--genkey-dir", help="递归扫描 *.genkey 的目录")
            sp.add_argument("--cases", help="case_id,genkey[,pvt][,fluid][,scenario] 的 CSV")
            sp.add_argument("--pvt-dir", help="PVT 表所在目录（genkey 同目录找不到时）")
            sp.add_argument("--only", default="", help="case_id 子串过滤")
            sp.add_argument("--scenario", default="", help="按 scenario 过滤（用 --cases 时有效）")
            sp.add_argument("--limit", type=int, default=0, help="只取前 N 例（冒烟）")
        sp.add_argument("--work", help="工作根目录（默认 olga_runs）")
        sp.add_argument("--solver", help="OLGA 可执行文件路径（默认自动探测 / OLGA_EXE）")
        sp.add_argument("--license", help="license 文件（默认 SLBSLS/LM_LICENSE_FILE 或自动探测）")

    sp = sub.add_parser("check", help="预检：求解器/license/genkey/PVT 是否齐")
    common(sp)
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("run", help="批量运行")
    common(sp)
    sp.add_argument("--workers", type=int, default=12)
    sp.add_argument("--timeout", type=int, default=7200, help="单例超时秒")
    sp.add_argument("--retries", type=int, default=1)
    sp.add_argument("--retry-delay", type=float, default=15.0)
    sp.add_argument("--resume", action="store_true", help="跳过已完成算例")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status", help="汇总状态目录")
    common(sp, need_cases=False)
    sp.add_argument("--status-dir", help="状态 JSON 目录（默认 <work>/_status）")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("curves", help="把 .tpl 趋势解析成长表 CSV")
    sp.add_argument("--work", help="工作根目录，扫描 <work>/*/*.tpl（默认 olga_runs）")
    sp.add_argument("--tpl", help="只解析单个 .tpl")
    sp.add_argument("--out", default="curves.csv", help="输出 CSV（.gz 结尾自动压缩）")
    sp.add_argument("--every", type=int, default=1, help="每隔 N 个时间点取一次")
    sp.add_argument("--limit", type=int, default=0, help="只取前 N 例（冒烟）")
    sp.add_argument("--pipe-segments", help="同一 BRANCH 多管时各管单元数，如 PIPE-1=100,PIPE-2=50")
    sp.add_argument("--allow-partial", action="store_true",
                    help="不看 .out 完成标志，直接解析（会读到半截 .tpl）")
    sp.set_defaults(func=cmd_curves)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd in ("check", "run"):
        if not args.cases and not args.genkey_dir:
            raise SystemExit("需要 --genkey-dir 或 --cases")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
