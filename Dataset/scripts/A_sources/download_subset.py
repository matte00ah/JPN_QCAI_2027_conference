#!/usr/bin/env python3
"""Download every URL listed in downloads.txt into the folder containing that file.

With no --list, every Dataset/<body>/<instrument>/downloads.txt is processed
(body = Moon or Mars), e.g. Dataset/Moon/LOLA/downloads.txt -> Dataset/Moon/LOLA/.
Lists without URLs (e.g. LROC, whose products are requested separately) are
skipped with a warning. --list accepts one or more list files or directories
containing downloads.txt; relative paths are resolved from the current
directory first, then from the Dataset folder.

Usage:
    python download_subset.py                          # all instruments, missing files only
    python download_subset.py --dry-run                # only list what would be done
    python download_subset.py --list Moon/LOLA Moon/LAMP   # only some instruments
    python download_subset.py --force                  # re-download everything
    python download_subset.py --list Moon/LOLA --out /scratch/LOLA   # other destination

The directory structure of each URL is preserved inside the destination
folder, after removing the "https://pds-geosciences.wustl.edu/lro/" prefix and
the mission-specific directory prefix shared by all URLs in the list (this also
works for other servers, e.g. LAMP on planetarydata.jpl.nasa.gov).
Files already present in the destination directory are skipped without
contacting the server (use --force to download them again). Files found in an
earlier directory along their target path are moved to their correct final
subdirectory first. Duplicate entries that resolve to the same destination
path are ignored. Downloads are resumable: partial data is kept in a
"<name>.part" file and a re-run continues from where it stopped. Only the
Python standard library is required (no external packages).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK_SIZE = 1 << 20  # 1 MiB
USER_AGENT = "subset-downloader/1.0 (+python urllib)"
CONTENT_RANGE_RE = re.compile(r"bytes\s+\d+-\d+/(\d+|\*)")
PDS_PREFIX = "https://pds-geosciences.wustl.edu/lro/"
PDS_PREFIX_PATH = urllib.parse.urlparse(PDS_PREFIX).path


def human(num: float) -> str:
    """Format a byte count in a readable way."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TiB"


def load_urls(list_path: Path) -> list[str]:
    """Read one URL per line, skipping blank lines and comments."""
    lines = list_path.read_text(encoding="utf-8").splitlines()
    return [url for line in lines if (url := line.strip()) and not url.startswith("#")]


def relative_path_for(url: str, mission: Path | None = None) -> Path:
    """Derive the local path from the URL, ignoring the PDS and mission prefixes."""
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
    if url.startswith(PDS_PREFIX):
        path = path[len(PDS_PREFIX_PATH):]
    path = path.lstrip("/")
    if not path or path.endswith("/"):
        raise ValueError(f"cannot determine a file path for {url!r}")

    parts = Path(path).parts
    if any(part in (".", "..") or Path(part).is_absolute() for part in parts):
        raise ValueError(f"cannot determine a safe file path for {url!r}")
    if mission is not None:
        prefix = mission.parts
        if parts[: len(prefix)] == prefix:
            parts = parts[len(prefix):]
    return Path(*parts)


def mission_prefix(urls: list[str]) -> Path | None:
    """Return the directory prefix shared by every URL, if there is one."""
    directories = []
    for url in urls:
        parts = relative_path_for(url).parts
        if len(parts) < 2:
            return None
        directories.append(parts[:-1])

    common = []
    for column in zip(*directories):
        if len(set(column)) != 1:
            break
        common.append(column[0])
    return Path(*common) if common else None


def possible_locations(out_dir: Path, rel_path: Path, full_rel_path: Path) -> list[Path]:
    """Return candidate file locations from the current directory to the final path."""
    parts = rel_path.parts
    candidates = [out_dir.joinpath(*parts[:k], parts[-1]) for k in range(len(parts))]

    # Also include layouts from older versions that kept part of the URL prefix.
    full_parts = full_rel_path.parts
    for k in range(len(full_parts)):
        candidate = out_dir.joinpath(*full_parts[k:-1], full_parts[-1])
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def make_ssl_context(insecure: bool) -> ssl.SSLContext | None:
    """Use certifi's CA bundle when available (Anaconda's Python often ships an
    incomplete default store), otherwise fall back to Python's defaults."""
    if insecure:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def remote_size(url: str, timeout: float, context: ssl.SSLContext | None) -> int | None:
    """Best-effort Content-Length lookup; None when unavailable."""
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        # HEAD may be unsupported; the GET request will report real problems.
        return None


def format_progress(done: int, total: int | None, elapsed: float) -> str:
    speed = human(done / elapsed if elapsed > 0 else 0.0)
    if total:
        return f"{100 * done / total:5.1f}%  {human(done)} / {human(total)}  ({speed}/s)"
    return f"{human(done)}  ({speed}/s)"


def print_status(prefix: str, name: str, detail: str = "", end: str = "") -> None:
    width = max(20, shutil.get_terminal_size((100, 20)).columns - 1)
    line = f"{prefix} {name}" + (f"  {detail}" if detail else "")
    print("\r" + line[:width].ljust(width), end=end, flush=True)


def download(url: str, dest: Path, *, prefix: str, timeout: float, retries: int,
             force: bool, context: ssl.SSLContext | None, legacy: list[Path] | None = None) -> str:
    """Download one URL to dest. Returns 'skipped', 'downloaded' or 'resumed'."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Move a file from an older layout into its current destination.
    moved = False
    if legacy and not dest.exists() and not force:
        for old in legacy:
            if old != dest and old.is_file():
                os.replace(old, dest)
                moved = True
                break
        else:
            for old in legacy:
                old_part = old.with_name(old.name + ".part")
                if old != dest and old_part.is_file() and not part.exists():
                    os.replace(old_part, part)
                    break

    # Skip files already in the destination directory without any network access.
    if dest.is_file() and not force:
        print_status(prefix, dest.name, "(moved, skipping)" if moved else "(skipping)")
        return "skipped"

    if force:
        part.unlink(missing_ok=True)

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                status = response.getcode()

                # Server ignored our Range header -> start over.
                if start and status != 206:
                    start = 0
                    part.unlink(missing_ok=True)

                total = None
                if status == 206:
                    match = CONTENT_RANGE_RE.match(response.headers.get("Content-Range", ""))
                    if match and match.group(1) != "*":
                        total = int(match.group(1))
                elif (length := response.headers.get("Content-Length")):
                    total = int(length)

                done = start
                began = last_tick = time.monotonic()

                with open(part, "ab" if start else "wb") as fh:
                    while chunk := response.read(CHUNK_SIZE):
                        fh.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last_tick >= 0.5:
                            print_status(prefix, dest.name, format_progress(done, total, now - began))
                            last_tick = now

                print_status(prefix, dest.name, format_progress(done, total, time.monotonic() - began))

            if total is not None and done != total:
                raise OSError(f"incomplete transfer ({done} of {total} bytes)")

            os.replace(part, dest)
            return "resumed" if start else "downloaded"

        except urllib.error.HTTPError as exc:
            if exc.code == 416 and start:
                # Range not satisfiable: our .part file may already be complete.
                actual = remote_size(url, timeout, context)
                if actual is not None and part.exists() and part.stat().st_size == actual:
                    os.replace(part, dest)
                    print_status(prefix, dest.name, "completed from partial file")
                    return "resumed"
                part.unlink(missing_ok=True)
                last_error = exc
            elif 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise RuntimeError(f"HTTP {exc.code} {exc.reason} for {url}") from exc
            else:
                last_error = exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc

        if attempt < retries:
            delay = min(2**attempt, 30)
            print_status(prefix, dest.name, f"retrying in {delay}s (attempt {attempt}/{retries})")
            time.sleep(delay)

    raise RuntimeError(f"failed to download {url}: {last_error}")


DATASET_ROOT = Path(__file__).resolve().parents[2]   # Dataset/
BODIES = ("Moon", "Mars")                              # one sub-folder per instrument inside each


def resolve_path(arg: str) -> Path:
    """Resolve a user path: absolute, else relative to the current directory if it
    exists there, else relative to the Dataset folder."""
    path = Path(arg)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else DATASET_ROOT / path


def find_lists(list_args: list[str] | None) -> list[Path]:
    """Return the URL lists to process: the ones given with --list, or by default
    every <Dataset>/<body>/<instrument>/downloads.txt."""
    if not list_args:
        return sorted(p.resolve() for body in BODIES
                      for p in (DATASET_ROOT / body).glob("*/downloads.txt"))
    lists = []
    for arg in list_args:
        path = resolve_path(arg)
        path = path / "downloads.txt" if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError(f"URL list not found: {path}")
        lists.append(path.resolve())
    return lists


def build_jobs(urls: list[str], out_dir: Path) -> list[tuple[str, Path, list[Path]]]:
    """Map every URL to its destination file (and older candidate locations)."""
    mission = mission_prefix(urls)
    jobs: list[tuple[str, Path, list[Path]]] = []
    seen: set[Path] = set()
    for url in urls:
        rel_path = relative_path_for(url, mission)
        if rel_path in seen:
            continue
        seen.add(rel_path)
        legacy_paths = possible_locations(out_dir, rel_path, relative_path_for(url))
        jobs.append((url, out_dir / rel_path, legacy_paths))
    return jobs


def run_jobs(jobs: list[tuple[str, Path, list[Path]]], args: argparse.Namespace,
             context: ssl.SSLContext | None) -> list[str]:
    """Download every job; return the failure messages."""
    failures: list[str] = []
    total = len(jobs)
    for index, (url, dest, legacy_paths) in enumerate(jobs, start=1):
        prefix = f"[{index}/{total}]"
        try:
            download(url, dest, prefix=prefix, timeout=args.timeout, retries=args.retries,
                     force=args.force, context=context, legacy=legacy_paths)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print_status(prefix, dest.name, f"FAILED - {exc}")
            failures.append(f"  {dest}: {exc}")
    print()
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download all URLs of one or more downloads.txt files into the folder that contains each list.")
    parser.add_argument("--list", metavar="FILE_OR_DIR", nargs="+",
                        help="one or more downloads.txt files, or directories containing them "
                             "(default: every Dataset/{Moon,Mars}/<instrument>/downloads.txt). "
                             "Relative paths are resolved from the current directory, then from Dataset/")
    parser.add_argument("--out", metavar="DIR",
                        help="destination folder, only with a single --list (default: the folder "
                             "containing the URL list); the path after the mission prefix is preserved inside DIR")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--timeout", type=float, default=60.0, help="network timeout in seconds (default: 60)")
    parser.add_argument("--retries", type=int, default=4, help="attempts per URL (default: 4)")
    parser.add_argument("--insecure", action="store_true", help="skip TLS certificate verification (only if downloads fail with SSL errors)")
    parser.add_argument("--dry-run", action="store_true", help="show the planned downloads and exit")
    args = parser.parse_args(argv)

    try:
        lists = find_lists(args.list)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not lists:
        print(f"error: no downloads.txt found in {DATASET_ROOT}/{{{','.join(BODIES)}}}/*/", file=sys.stderr)
        return 2
    if args.out and len(lists) > 1:
        print("error: --out can be used only with a single URL list", file=sys.stderr)
        return 2

    context = None if args.dry_run else make_ssl_context(args.insecure)
    failures: list[str] = []
    processed = 0
    for list_path in lists:
        urls = load_urls(list_path)
        if not urls:
            print(f"warning: no URLs in {list_path}, skipped", file=sys.stderr)
            continue
        processed += 1
        out_dir = resolve_path(args.out) if args.out else list_path.parent
        jobs = build_jobs(urls, out_dir)

        print(f"URL list   : {list_path}")
        print(f"Destination: {out_dir}")
        print(f"Files      : {len(jobs)}")
        if args.dry_run:
            for url, dest, _ in jobs:
                print(f"  {dest.relative_to(out_dir).as_posix()}  <-  {url}")
            print()
            continue
        try:
            failures += run_jobs(jobs, args, context)
        except KeyboardInterrupt:
            print()
            print("interrupted; partial data kept in .part files for the next run", file=sys.stderr)
            return 130

    if not processed:
        print("error: none of the URL lists contains URLs", file=sys.stderr)
        return 2
    if failures:
        print("Failed files:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
