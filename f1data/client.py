"""Polite Jolpica-F1 client: rate limiting, retries, disk cache, pagination."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from pathlib import Path

import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_LIMIT = 100  # the API's maximum page size
USER_AGENT = "f1data-elo/0.1 (personal F1 ELO project)"  # Jolpica asks for a named UA

log = logging.getLogger("f1data")


class RateLimiter:
    """Keeps under 4 requests/second and a rolling hourly budget."""

    def __init__(self, per_second: float = 4, per_hour: int = 480,
                 clock=time.monotonic, sleep=time.sleep):
        self.min_gap = 1.0 / per_second
        self.per_hour = per_hour
        self.calls: deque[float] = deque()
        self.clock, self.sleep = clock, sleep

    def _expire(self, now: float) -> None:
        while self.calls and now - self.calls[0] >= 3600:
            self.calls.popleft()

    def wait(self) -> None:
        now = self.clock()
        self._expire(now)
        if len(self.calls) >= self.per_hour:
            pause = 3600 - (now - self.calls[0]) + 1
            log.warning("Hourly budget of %d requests used; pausing %.0f min.",
                        self.per_hour, pause / 60)
            self.sleep(pause)
            now = self.clock()
            self._expire(now)
        if self.calls and now - self.calls[-1] < self.min_gap:
            self.sleep(self.min_gap - (now - self.calls[-1]))
            now = self.clock()
        self.calls.append(now)


class JolpicaClient:
    def __init__(self, cache_dir: Path | str = "cache", per_hour: int = 480,
                 session=None, limiter: RateLimiter | None = None,
                 sleep=time.sleep, max_retries: int = 6):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.limiter = limiter or RateLimiter(per_second=1, per_hour=per_hour)
        self.sleep = sleep
        self.max_retries = max_retries
        self.network_calls = 0

    # -- caching -----------------------------------------------------------
    def _cache_file(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def _read_cache(self, url: str, ttl: float | None):
        f = self._cache_file(url)
        if ttl == 0 or not f.exists():
            return None
        entry = json.loads(f.read_text(encoding="utf-8"))
        if ttl is not None and time.time() - entry["fetched_at"] > ttl:
            return None
        return entry["data"]

    def _write_cache(self, url: str, data) -> None:
        payload = {"url": url, "fetched_at": time.time(), "data": data}
        self._cache_file(url).write_text(json.dumps(payload), encoding="utf-8")

    # -- single request ----------------------------------------------------
    def get(self, path: str, limit: int = PAGE_LIMIT, offset: int = 0,
            ttl: float | None = None) -> dict:
        """GET one page. ttl=None caches forever, ttl=0 always refetches."""
        path = path.strip("/")
        url = f"{BASE_URL}/{path}.json?limit={limit}&offset={offset}"
        cached = self._read_cache(url, ttl)
        if cached is not None:
            return cached

        backoff = 30.0
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            self.network_calls += 1
            try:
                resp = self.session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            except requests.RequestException as exc:
                log.warning("Network error on %s (%s); retry %d in %.0fs",
                            url, exc, attempt, backoff)
                self.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 200:
                data = resp.json()
                self._write_cache(url, data)
                return data
            if resp.status_code == 429 or resp.status_code >= 500:
                log.warning("HTTP %d on %s; retry %d in %.0fs",
                            resp.status_code, url, attempt, backoff)
                self.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")
        raise RuntimeError(f"Gave up on {url} after {self.max_retries} attempts")

    # -- pagination --------------------------------------------------------
    def get_all_pages(self, path: str, ttl: float | None = None) -> list[dict]:
        pages, offset = [], 0
        while True:
            mr = self.get(path, offset=offset, ttl=ttl)["MRData"]
            pages.append(mr)
            step = int(mr.get("limit") or PAGE_LIMIT)
            offset += step
            if offset >= int(mr.get("total") or 0):
                return pages

    def get_table(self, path: str, table: str, items: str,
                  ttl: float | None = None) -> list[dict]:
        """For list endpoints like drivers/: concatenate every page's rows."""
        out: list[dict] = []
        for mr in self.get_all_pages(path, ttl):
            out.extend(mr[table][items])
        return out

    def get_races(self, path: str, child: str | None = None,
                  ttl: float | None = None) -> list[dict]:
        """For RaceTable endpoints. Pages count child rows, so one race can be
        split across two pages; this stitches each race back together."""
        merged: dict[tuple, dict] = {}
        for mr in self.get_all_pages(path, ttl):
            for race in mr["RaceTable"]["Races"]:
                key = (race["season"], race["round"])
                if key not in merged:
                    merged[key] = dict(race)
                    if child:
                        merged[key][child] = list(race.get(child, []))
                elif child:
                    merged[key][child].extend(race.get(child, []))
        return list(merged.values())