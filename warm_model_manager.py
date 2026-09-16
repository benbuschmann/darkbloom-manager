#!/usr/bin/env python3
"""Portable revenue-aware Darkbloom single-model warm loader.

The score combines average network pressure, a fixed mix of 85% input and
15% output token prices, and a model weight. It compares models at the
same assumed token mix without measuring provider throughput or actual payouts.

Every selection requests exactly one warm model. The catalog, network capacity,
and local scan supply the model inventory. Ignored and auto-ignored models stay
visible by default. Use --hide-ignored to hide them from the display and its ranking;
their calculations remain in saved state.

The file is intentionally standalone: copy only this script to a Mac running
Darkbloom.  It uses Python's standard library, the installed ``darkbloom``
command, ``~/.darkbloom/daemon-state.json``, and the provider TOML config.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import ipaddress
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from http.client import HTTPException
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen


MANAGER_VERSION = "0.1.11"
# State formats change only when their stored data changes, independently of
# the release number. A major release alone must not erase switching state.
STATE_SCHEMA = 4
INPUT_TOKEN_SHARE = 0.85
OUTPUT_TOKEN_SHARE = 0.15
PRICING_CACHE_SCHEMA = 2
DEFAULT_WEIGHTS = {
    "qwen3.5-35b-a3b": 1.25,
    "qwen3.6-35b-a3b-vl-mtp-mxfp8": 1.20,
    "gemma-4-26b-qat-4bit": 1.05,
    "gpt-oss-20b": 1.00,
}
DEFAULT_BASE_URL = os.environ.get(
    "DARKBLOOM_BASE_URL", "https://console.darkbloom.dev"
).rstrip("/")
DEFAULT_PRICING_URL = os.environ.get(
    "DARKBLOOM_PRICING_URL", "https://api.darkbloom.dev/v1/pricing"
)
DEFAULT_STATE_PATH = Path.home() / ".darkbloom" / "warm-model-manager-state.json"
DEFAULT_PROD_TOKEN_PATH = Path.home() / ".darkbloom" / "warm-model-manager-prod-token"
DEFAULT_LOCAL_ENDPOINT_PATH = Path(os.environ.get(
    "DARKBLOOM_LOCAL_DIR", str(Path.home() / ".darkbloom")
)) / "local.json"
PROD_PROBE_URL = "https://api.darkbloom.dev/v1/chat/completions"
PROBE_SPACING = 1800
PROBE_TIMEOUT = 30
SWITCH_PROBE_DELAY = 180
RECOVERY_GRACE = 900
RECOVERY_CHECK_SPACING = 300
RECOVERY_OFFLINE_TIME = 900
RECOVERY_VERIFY_TIME = 900
RECOVERY_ACTIVE_PHASES = {"checking", "stopping", "offline", "starting", "verifying"}
DEFAULT_PROVIDER_CONFIG_PATH = (
    Path.home() / ".config" / "darkbloom" / "provider.toml"
)
DEFAULT_DAEMON_STATE_PATH = Path(
    os.environ.get(
        "DARKBLOOM_STATE_FILE",
        str(Path.home() / ".darkbloom" / "daemon-state.json"),
    )
)
# Only retain state used by this release. Removed features cannot leave stale
# instructions behind or keep accumulating private runtime observations.
MANAGER_STATE_KEYS = {
    "active_target", "current_model", "current_residency", "discovery", "catalog",
    "last_decision_at", "last_decision_reason", "last_decision_target",
    "last_score_snapshot", "last_switch_at", "manager_version",
    "pending_switch", "preload_sync_schema", "pressure_cadence", "pressure_history",
    "pricing_cache", "pricing_status", "scoring_policy", "selection_policy", "switch_policy", "state_schema",
    "live_challenger_model", "live_challenger_streak",
    "dry_challenger_model", "dry_challenger_streak",
    "probes",
    "switch_probe", "last_switch_probe_result",
    "routing_recovery",
}


@dataclass(frozen=True)
class CapacitySample:
    model_id: str
    warm_providers: int
    active_requests: int
    pressure: float


@dataclass(frozen=True)
class ModelPrice:
    input_usd: float | None
    output_usd: float | None

    @property
    def blended_usd(self) -> float | None:
        if self.input_usd is None or self.output_usd is None:
            return None
        blended = INPUT_TOKEN_SHARE * self.input_usd + OUTPUT_TOKEN_SHARE * self.output_usd
        return blended if math.isfinite(blended) and blended > 0 else None

    def to_dict(self) -> dict[str, float | None]:
        return {"input_usd": self.input_usd, "output_usd": self.output_usd}


@dataclass(frozen=True)
class ScoreCheck:
    samples: dict[str, CapacitySample]
    averages: dict[str, float]
    prices: dict[str, ModelPrice]
    scores: dict[str, float]
    local_eligible: list[str]
    eligible: list[str]
    discovery_available: bool


@dataclass(frozen=True)
class LocalDaemonState:
    current_model: str | None
    warm_models: tuple[str, ...]
    inference_active: bool
    pid: int
    started_at: float
    fresh: bool
    alive: bool = True
    load_error_model: str | None = None
    load_error_message: str | None = None
    load_error_at: float = 0.0
    requests_served: int | None = None
    reconnect_count: int | None = None


@dataclass(frozen=True)
class Decision:
    target: str | None
    reason: str
    challenger: str | None = None
    challenger_streak: int = 0
    warming: bool = False


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{timestamp} {message}", flush=True)


def get_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"darkbloom-warm-model-manager/{MANAGER_VERSION}",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {url}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected non-object response from {url}")
    return payload


def fetch_capacity(base_url: str) -> dict[str, CapacitySample]:
    payload = get_json(f"{base_url.rstrip('/')}/api/models/capacity")
    return pressure_samples(payload.get("models") or [])


def price_value(value: Any, divisor: float = 1.0) -> float | None:
    """Keep missing or invalid prices distinct from an explicit zero price."""
    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(value) / divisor
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price >= 0 else None


def fetch_model_prices(pricing_url: str) -> tuple[dict[str, ModelPrice], ModelPrice]:
    """Return platform input/output USD per million tokens and the fallback."""
    payload = get_json(pricing_url)
    fallback = ModelPrice(
        price_value(payload.get("fallback_input_price"), 1_000_000),
        price_value(payload.get("fallback_output_price"), 1_000_000),
    )
    prices: dict[str, ModelPrice] = {}
    for row in payload.get("prices") or []:
        if not isinstance(row, dict) or not row.get("model"):
            continue
        prices[str(row["model"])] = ModelPrice(
            price_value(row.get("input_price"), 1_000_000),
            price_value(row.get("output_price"), 1_000_000),
        )
    if not prices and fallback.blended_usd is None:
        raise RuntimeError("Darkbloom returned no usable input/output prices")
    return prices, fallback


def cached_model_prices(
    manager_state: dict[str, Any],
    models: list[str],
    pricing_url: str,
    now: float,
    refresh_seconds: float,
    force_refresh: bool = False,
) -> dict[str, ModelPrice]:
    cache = manager_state.get("pricing_cache")
    cache = cache if isinstance(cache, dict) else {}
    # Older caches contain only output prices. Never invent an input price
    # from them; fetch both components before allowing a blended score.
    if cache.get("source") != pricing_url or cache.get("schema") != PRICING_CACHE_SCHEMA:
        cache = {}
        manager_state.pop("pricing_cache", None)
    cached_at = price_value(cache.get("fetched_at")) or 0.0

    def decode(value: Any) -> ModelPrice:
        row = value if isinstance(value, dict) else {}
        return ModelPrice(price_value(row.get("input_usd")), price_value(row.get("output_usd")))

    entries = cache.get("prices")
    cached_prices = {
        model: decode(value) for model, value in entries.items()
    } if isinstance(entries, dict) else {}
    fallback = decode(cache.get("fallback"))
    status = "cached"
    if force_refresh or not cached_at or now - cached_at >= refresh_seconds:
        try:
            live_prices, live_fallback = fetch_model_prices(pricing_url)
        except Exception as error:
            status = "stale cache" if cached_at else "unavailable"
            log(f"pricing refresh failed ({error}); prices: {status}")
        else:
            status = "live"
            cached_prices = live_prices
            fallback = live_fallback
            cached_at = now
            manager_state["pricing_cache"] = {
                "schema": PRICING_CACHE_SCHEMA,
                "fetched_at": now,
                "prices": {model: price.to_dict() for model, price in live_prices.items()},
                "fallback": live_fallback.to_dict(),
                "source": pricing_url,
            }

    manager_state["pricing_status"] = {
        "status": status,
        "fetched_at": cached_at or None,
        "source": pricing_url,
    }

    # A fallback is for an absent model, not a missing component of a listed
    # model's price. Keep partial prices visible but exclude them from scoring.
    return {model: cached_prices.get(model, fallback) for model in models}


def pressure_samples(rows: Iterable[dict[str, Any]]) -> dict[str, CapacitySample]:
    samples: dict[str, CapacitySample] = {}
    for row in rows:
        model_id = str(row.get("id") or row.get("model_id") or "")
        if not model_id:
            continue
        warm = max(0, int(row.get("warm_providers", row.get("loaded", 0)) or 0))
        active = max(
            0,
            int(row.get("active_requests", row.get("in_progress", 0)) or 0),
        )
        samples[model_id] = CapacitySample(
            model_id=model_id,
            warm_providers=warm,
            active_requests=active,
            pressure=active / max(1, warm),
        )
    if not samples:
        raise RuntimeError("Darkbloom returned no model capacity records")
    return samples


def update_pressure_history(
    models: list[str],
    samples: dict[str, CapacitySample],
    previous: dict[str, Any],
    history_size: int,
    now: float,
    window_seconds: float,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, float]]:
    history: dict[str, list[dict[str, float]]] = {}
    averages: dict[str, float] = {}
    cutoff = now - window_seconds
    for model in models:
        sample = samples.get(model)
        old_entries = previous.get(model, [])
        if not isinstance(old_entries, list):
            old_entries = []
        clean: list[dict[str, float]] = []
        for item in old_entries:
            if not isinstance(item, dict):
                # Pre-v3.2 float-only samples cannot be aged safely.
                continue
            try:
                observed_at = float(item["at"])
                pressure = max(0.0, float(item["pressure"]))
            except (TypeError, ValueError):
                continue
            except KeyError:
                continue
            if math.isfinite(pressure) and cutoff < observed_at <= now:
                clean.append({"at": observed_at, "pressure": pressure})
        if sample is not None:
            clean.append({"at": now, "pressure": sample.pressure})
        clean.sort(key=lambda item: item["at"])
        entries = clean[-history_size:]
        history[model] = entries
        if entries:
            averages[model] = sum(item["pressure"] for item in entries) / len(entries)
    return history, averages


def ensure_pressure_cadence(
    manager_state: dict[str, Any],
    interval_seconds: float,
    history_size: int,
) -> bool:
    desired = {
        "interval_seconds": float(interval_seconds),
        "history_size": int(history_size),
        "window_seconds": float(interval_seconds * history_size),
    }
    previous = manager_state.get("pressure_cadence")
    changed = isinstance(previous, dict) and previous != desired
    if previous != desired:
        manager_state["pressure_history"] = {}
        manager_state["pressure_cadence"] = desired
        for key in (
            "live_challenger_model",
            "live_challenger_streak",
            "dry_challenger_model",
            "dry_challenger_streak",
        ):
            manager_state.pop(key, None)
    return changed


def ensure_preload_sync_policy(manager_state: dict[str, Any]) -> bool:
    """Discard a legacy pending switch that did not synchronize preloading."""
    if manager_state.get("preload_sync_schema") == 1:
        return False
    discarded_pending = isinstance(manager_state.get("pending_switch"), dict)
    manager_state.pop("pending_switch", None)
    manager_state["preload_sync_schema"] = 1
    return discarded_pending


def ensure_scoring_policy(manager_state: dict[str, Any]) -> bool:
    """Start new passing-check counts when the formula changes, keeping warmth."""
    desired = {"input_token_share": INPUT_TOKEN_SHARE, "output_token_share": OUTPUT_TOKEN_SHARE}
    if manager_state.get("scoring_policy") == desired:
        return False
    for key in (
        "live_challenger_model", "live_challenger_streak",
        "dry_challenger_model", "dry_challenger_streak",
    ):
        manager_state.pop(key, None)
    manager_state["scoring_policy"] = desired
    return True


def ensure_switch_policy(
    manager_state: dict[str, Any],
    improvement_percent: float,
    switch_cost_seconds: float,
    decision_horizon_seconds: float,
) -> bool:
    """Count consecutive passes only under the same percentage and cost rule."""
    desired = {
        "improvement_percent": improvement_percent,
        "switch_cost_seconds": switch_cost_seconds,
        "decision_horizon_seconds": decision_horizon_seconds,
    }
    if manager_state.get("switch_policy") == desired:
        return False
    for apply in (False, True):
        for key in challenger_state_keys(apply):
            manager_state.pop(key, None)
    manager_state["switch_policy"] = desired
    return True


def revenue_scores(
    pressures: dict[str, float],
    weights: dict[str, float],
    prices: dict[str, ModelPrice],
) -> dict[str, float]:
    return {
        model: pressure * prices[model].blended_usd * weights.get(model, 1.0)
        for model, pressure in pressures.items()
        if model in prices and prices[model].blended_usd is not None
    }


def build_score_snapshot(
    models: list[str],
    samples: dict[str, CapacitySample],
    averages: dict[str, float],
    history: dict[str, list[dict[str, float]]],
    prices: dict[str, ModelPrice],
    weights: dict[str, float],
    scores: dict[str, float],
    observed_at: float,
    window_seconds: float,
    ignored_models: Iterable[str] = (),
    eligible_models: Iterable[str] = (),
    local_models: set[str] | None = None,
    selected_models: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the persisted audit record for the values driving a decision."""
    model_values: dict[str, dict[str, Any]] = {}
    for model in models:
        sample = samples.get(model)
        locally_available = model in local_models if local_models is not None else None
        excluded_by_model_flag = selected_models is not None and model not in selected_models
        ignored = model in ignored_models
        auto_ignored = not ignored and (locally_available is not True or excluded_by_model_flag)
        status = []
        if ignored:
            status.append("IGNORED")
        elif auto_ignored:
            status.append("AUTO-IGNORED")
        elif model not in eligible_models:
            status.append("INELIGIBLE")
        if locally_available is None:
            status.append("local scan unavailable")
        elif not locally_available:
            status.append("not downloaded or filtered out")
        if excluded_by_model_flag:
            status.append("outside --model selection")
        if sample is None:
            status.append("capacity unavailable; AVG retained" if history.get(model) else "capacity unavailable")
        price = prices.get(model, ModelPrice(None, None))
        if price.input_usd is None:
            status.append("input price unavailable")
        if price.output_usd is None:
            status.append("output price unavailable")
        if price.input_usd is not None and price.output_usd is not None and price.blended_usd is None:
            status.append("no positive blended price")
        model_values[model] = {
            "now_pressure": sample.pressure if sample is not None else None,
            "average_pressure": averages.get(model),
            "capacity_available": sample is not None,
            "retained_samples": len(history.get(model, [])),
            "input_usd_per_million": price.input_usd,
            "output_usd_per_million": price.output_usd,
            "blended_usd_per_million": price.blended_usd,
            "weight": weights.get(model, 1.0),
            "score": scores.get(model),
            "ignored": ignored,
            "auto_ignored": auto_ignored,
            "local_available": locally_available,
            "excluded_by_model_flag": excluded_by_model_flag,
            "status": status,
            "eligible": model in eligible_models,
        }
    return {
        "observed_at": observed_at,
        "window_seconds": window_seconds,
        "input_token_share": INPUT_TOKEN_SHARE,
        "output_token_share": OUTPUT_TOKEN_SHARE,
        "models": model_values,
    }


def challenger_state_keys(apply: bool) -> tuple[str, str]:
    prefix = "live" if apply else "dry"
    return f"{prefix}_challenger_model", f"{prefix}_challenger_streak"


def choose_scored_target(
    models: list[str],
    scores: dict[str, float],
    current_model: str | None,
    previous_challenger: str | None,
    previous_streak: int,
    improvement_percent: float,
    confirmations: int,
    switch_cost_seconds: float,
    decision_horizon_seconds: float,
) -> Decision:
    available = [model for model in models if model in scores]
    if not available:
        raise RuntimeError("none of the configured models appeared in the capacity feed")
    priority_index = {model: index for index, model in enumerate(models)}

    if current_model not in available:
        target = max(available, key=lambda model: (scores[model], -priority_index[model]))
        return Decision(target, f"no eligible current model; highest score is {scores[target]:.3f}")

    cost_discount = max(
        0.0,
        (decision_horizon_seconds - switch_cost_seconds)
        / decision_horizon_seconds,
    )
    adjusted = {
        model: scores[model] if model == current_model else scores[model] * cost_discount
        for model in available
    }
    challenger = max(
        available,
        key=lambda model: (adjusted[model], -priority_index[model]),
    )
    current_score = scores[current_model]
    challenger_score = adjusted[challenger]
    # A percentage of zero is zero. Require a strict gain to avoid switching
    # between zero-score models or equal scores when the requirement is 0%.
    if challenger == current_model or challenger_score <= current_score:
        return Decision(
            current_model,
            f"no allowed model scores higher after switch cost (current {current_score:.6g})",
        )

    required_score = current_score * (1.0 + improvement_percent / 100.0)
    comparison = f"{challenger_score:.6g} after switch cost vs current {current_score:.6g}"
    gain = (
        f"{(challenger_score / current_score - 1.0) * 100.0:.2f}% improvement"
        if current_score > 0 else "positive score above a zero current score"
    )
    requirement = f"{improvement_percent:g}% improvement requirement (--switch-improvement-percent)"
    if challenger_score < required_score and not math.isclose(challenger_score, required_score, rel_tol=1e-12, abs_tol=0.0):
        return Decision(
            current_model,
            f"{challenger} does not meet the {requirement} "
            f"({gain}; {comparison}; need >= {required_score:.6g})",
            challenger=challenger,
        )

    streak = previous_streak + 1 if previous_challenger == challenger else 1
    if streak < confirmations:
        return Decision(
            current_model,
            f"{challenger} meets the {requirement} ({gain}); "
            f"{streak}/{confirmations} consecutive checks passed (--switch-after-checks)",
            challenger=challenger,
            challenger_streak=streak,
        )
    return Decision(
        challenger,
        f"{challenger} met the {requirement} for {streak} consecutive checks "
        f"({gain}; {comparison}; --switch-after-checks)",
        challenger=challenger,
        challenger_streak=streak,
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def valid_probe_token(token: Any) -> bool:
    return (isinstance(token, str) and 0 < len(token) <= 8192
            and all(33 <= ord(c) <= 126 for c in token))


def save_prod_token(path: Path) -> None:
    """Read interactively so credentials never enter command arguments or state."""
    token = getpass.getpass("Darkbloom production API token: ").strip()
    if not valid_probe_token(token):
        raise ValueError("token must be nonempty and contain no whitespace")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".prod-token-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(token + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    log("production token saved with owner-only permissions")


class ProbeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        # A redirect must not forward either token or escape exclusive self-route.
        return None


def probe_credentials(kind: str, daemon: LocalDaemonState,
                      local_path: Path, prod_token_path: Path) -> tuple[str, str]:
    if kind == "production":
        try:
            with prod_token_path.open(encoding="utf-8") as stream:
                mode = os.fstat(stream.fileno()).st_mode
                if not stat.S_ISREG(mode) or mode & 0o077:
                    raise ValueError("production token file needs owner-only permissions (chmod 600)")
                token = stream.read(8193).strip()
        except OSError:
            raise ValueError("production token unavailable; run set-prod-token") from None
        if len(token) > 8192 or not valid_probe_token(token):
            raise ValueError("production token invalid; run set-prod-token")
        return PROD_PROBE_URL, token

    try:
        info = json.loads(local_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("local endpoint record is missing; enable --local-endpoint on Darkbloom or check --local-endpoint-file") from None
    except OSError:
        raise ValueError("local endpoint record cannot be read; check file permissions") from None
    except (ValueError, UnicodeDecodeError):
        raise ValueError("local endpoint record contains invalid JSON") from None
    if not isinstance(info, dict) or type(info.get("pid")) is not int or info["pid"] <= 0:
        raise ValueError("local endpoint record has no valid process ID")
    if info["pid"] != daemon.pid:
        raise ValueError(f"local endpoint process mismatch (endpoint pid {info['pid']}, provider pid {daemon.pid}); check --local-endpoint-file and --daemon-state")
    try:
        url = urlsplit(info.get("base_url", ""))
        loopback = ipaddress.ip_address(url.hostname or "").is_loopback
        valid = (loopback and url.scheme in ("http", "https") and url.port
                 and not url.username and not url.password and not url.query
                 and not url.fragment and url.path.rstrip("/") == "/v1")
    except (ValueError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise ValueError("local endpoint must be a loopback /v1 URL with a port")
    token = info.get("api_key", "")
    if token != "" and not valid_probe_token(token):
        raise ValueError("local endpoint token is invalid")
    return info["base_url"].rstrip("/") + "/chat/completions", token


def probe_debug_text(value: Any, token: str, limit: int = 400) -> str:
    """Keep diagnostics short, single-line and free of bearer credentials."""
    if not isinstance(value, (str, int, float)):
        return ""
    text = str(value)
    if token:
        for secret in (token, quote(token, safe="")):
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)\b(?:sk-db-|dk-local-|github_pat_|gh[pousr]_)[A-Za-z0-9_+./=-]+",
                  "[redacted]", text)
    text = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [redacted]", text)
    text = re.sub(r'''(?i)\b(?:authorization|api[_ -]?key|token)["']?\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;}]+)''',
                  "credential=[redacted]", text)
    text = " ".join("".join(c if c.isprintable() else " " for c in text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def probe_response_details(body: bytes, token: str, http_status: int) -> dict[str, Any]:
    failed = {"outcome": "FAILED", "status": "invalid response"}
    if len(body) > 65536:
        return {**failed, "status": "response exceeded size limit", "error": "response exceeds 64 KiB"}
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        if not 200 <= http_status < 300:
            excerpt = probe_debug_text(body.decode("utf-8", errors="replace"), token)
            return {**failed, "error": "non-JSON error response: " + (excerpt or "empty body")}
        # A malformed success response can contain generated text. Do not dump it.
        return {**failed, "error": "expected a JSON chat completion; received non-JSON or empty response"}
    if not isinstance(payload, dict):
        return {**failed, "error": "expected a JSON object"}
    if "error" in payload:
        error = payload["error"]
        if isinstance(error, dict):
            fields = (error.get("code") or error.get("type"), error.get("message"))
        else:
            fields = (error,)
        detail = ": ".join(filter(None, (probe_debug_text(field, token) for field in fields)))
        result = {"outcome": "FAILED", "status": "API error", "error": detail or "server returned an error"}
        code = error.get("code") if isinstance(error, dict) else None
        if (isinstance(code, str) and re.fullmatch(r"[a-z_]{1,80}", code)
                and probe_debug_text(code, token) == code):
            result["error_code"] = code
        return result
    if not 200 <= http_status < 300:
        return {"outcome": "FAILED", "status": "HTTP error", "error": "server returned no structured error details"}
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    if not isinstance(message, dict) or not any(
        isinstance(message.get(key), str) and message[key].strip()
        for key in ("content", "reasoning", "reasoning_content", "refusal")
    ):
        return {**failed, "error": "HTTP succeeded, but no assistant output was returned"}
    result = {"outcome": "SUCCESS", "status": "completion received"}
    finish = probe_debug_text(choice.get("finish_reason"), token, 60)
    if finish:
        result["finish_reason"] = finish
    if finish == "length":
        result["status"] = "completion received; output token limit reached"
    usage = payload.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens"):
            if type(usage.get(key)) is int and usage[key] >= 0:
                result[key] = usage[key]
    return result


def send_probe(kind: str, model: str, url: str, token: str) -> dict[str, Any]:
    started = time.monotonic()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if kind == "production":
        headers["X-Darkbloom-Route"] = "self"
    request = Request(url, data=json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 64, "stream": False,
    }).encode("utf-8"), headers=headers, method="POST")
    # In particular, never send a local token through an environment proxy.
    opener = build_opener(ProxyHandler({}), ProbeRedirectHandler())
    result = {"http_status": None, "outcome": "FAILED"}
    try:
        with opener.open(request, timeout=PROBE_TIMEOUT) as response:
            result["http_status"] = response.status
            provider = response.headers.get("X-Provider-Id", "")
            if (kind == "production" and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", provider)
                    and probe_debug_text(provider, token) == provider):
                result["provider_id"] = provider
            # Finish the small non-streaming response before closing the socket.
            # Retain only bounded diagnostics, never generated text or raw bodies.
            result.update(probe_response_details(response.read(65537), token, response.status))
    except HTTPError as error:
        result.update(http_status=error.code, status="HTTP error")
        try:
            details = probe_response_details(error.read(65537), token, error.code)
            reason = probe_debug_text(error.reason, token)
            result["error"] = "; ".join(filter(None, (reason, details.get("error"))))
            if details.get("error_code"):
                result["error_code"] = details["error_code"]
        except (OSError, ValueError, HTTPException) as read_error:
            result["error"] = "could not read error response: " + probe_debug_text(str(read_error), token)
        finally:
            error.close()
    except (URLError, OSError, ValueError, HTTPException) as error:
        cause = error.reason if isinstance(error, URLError) else error
        error_type = type(cause).__name__ if isinstance(cause, BaseException) else type(error).__name__
        result.update(status="connection failed or timed out",
                      error=probe_debug_text(f"{error_type}: {cause}", token))
    result["elapsed_seconds"] = round(max(0, time.monotonic() - started), 2)
    return result


def migrate_default_state(state_path: Path) -> bool:
    """Copy the previous default once, with old and new manager locks held."""
    if state_path != DEFAULT_STATE_PATH or state_path.exists():
        return False
    previous = DEFAULT_STATE_PATH.with_name("warm-model-manager-v4.json")
    if not previous.exists():
        return False
    try:
        payload = json.loads(previous.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"could not migrate previous manager state at {previous}: {error}") from error
    write_json_atomic(state_path, {
        key: value for key, value in payload.items() if key in MANAGER_STATE_KEYS
    })
    log(f"copied previous manager state to {state_path}; original file retained")
    return True


def read_daemon_state(path: Path, now: float | None = None) -> LocalDaemonState | None:
    payload = read_json(path)
    if not payload:
        return None
    current_time = time.time() if now is None else now
    written_at = float(payload.get("written_at") or 0)
    load_error = (
        payload.get("last_model_load_error")
        if isinstance(payload.get("last_model_load_error"), dict)
        else {}
    )
    warm = payload.get("warm_models") or []
    pid = int(payload.get("pid") or 0)
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    connectivity = payload.get("connectivity") if isinstance(payload.get("connectivity"), dict) else {}
    return LocalDaemonState(
        current_model=(
            str(payload["current_model"]) if payload.get("current_model") else None
        ),
        warm_models=tuple(str(model) for model in warm if model),
        inference_active=bool(payload.get("inference_active")),
        pid=pid,
        started_at=float(payload.get("started_at") or 0),
        fresh=written_at > 0 and current_time - written_at <= 90,
        alive=process_alive(pid),
        load_error_model=(
            str(load_error["model"]) if load_error.get("model") else None
        ),
        load_error_message=(
            str(load_error["message"]) if load_error.get("message") else None
        ),
        load_error_at=float(load_error.get("at") or 0),
        requests_served=nonnegative_int(stats.get("requests_served")),
        reconnect_count=nonnegative_int(connectivity.get("reconnect_count")),
    )


def nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def provider_services() -> dict[str, int]:
    """Inspect only Darkbloom's two supported user services; never guess a PID."""
    services = {}
    for label in ("io.darkbloom.provider", "dev.darkbloom.provider"):
        result = subprocess.run(
            ["/bin/launchctl", "print", f"gui/{os.getuid()}/{label}"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode:
            if "could not find service" in result.stderr.lower():
                continue
            raise RuntimeError("cannot verify Darkbloom's launchd service")
        match = re.search(r"(?m)^\s*pid = (\d+)\s*$", result.stdout)
        services[label] = int(match[1]) if match else 0
    return services


def stop_provider(darkbloom: str, expected_pid: int, *, before_stop: Callable[[], bool] | None = None) -> None:
    # `darkbloom stop` is user-service scoped, not --config scoped. Check that
    # it will stop exactly the process observed through the selected state file.
    services = provider_services()
    if len(services) != 1 or set(services.values()) != {expected_pid}:
        raise RuntimeError("launchd provider does not match the observed process; stop refused")
    if before_stop is not None and not before_stop():
        raise RuntimeError("provider activity or identity changed before stop; shutdown cancelled")
    result = subprocess.run([darkbloom, "stop"], capture_output=True, text=True,
                            timeout=60, check=False)
    if result.returncode:
        raise RuntimeError("Darkbloom stop failed: " + probe_debug_text(result.stderr or result.stdout, ""))


def recovery_config_digest(config: Path | None) -> str:
    return hashlib.sha256((config or DEFAULT_PROVIDER_CONFIG_PATH).read_bytes()).hexdigest()


def routing_recovery(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("routing_recovery")
    if value is None:
        return {}
    if (not isinstance(value, dict) or value.get("schema") != 1
            or value.get("phase") not in RECOVERY_ACTIVE_PHASES | {"monitoring", "locked"}
            or type(value.get("attempted")) is not bool):
        raise ValueError("invalid routing recovery state; inspect it before enabling changes")
    for field in ("next_check_at", "warm_since", "command_at", "restart_at", "stopped_monotonic", "verify_by", "selection_check_at"):
        if field in value and (type(value[field]) not in (int, float)
                              or not math.isfinite(value[field]) or value[field] < 0):
            raise ValueError("invalid routing recovery timing; inspect saved state")
    return value


def recovery_blocks_selection(state: dict[str, Any], enabled: bool, apply: bool) -> bool:
    phase = routing_recovery(state).get("phase")
    # Once warm, ordinary scoring runs alongside routing verification. Saved
    # recovery still requires explicit live opt-in to resume any model changes.
    return phase in RECOVERY_ACTIVE_PHASES and (phase != "verifying" or not enabled or not apply)


def warm_selection_matches(
    daemon: LocalDaemonState | None,
    target: str | None,
) -> bool:
    if target is None or not daemon or not daemon.alive or not daemon.fresh:
        return False
    return daemon.warm_models == (target,)


def current_warm_model(
    daemon: LocalDaemonState | None,
    models: list[str],
) -> str | None:
    """Recognize only a single eligible model confirmed warm by the daemon."""
    if not daemon or not daemon.alive or not daemon.fresh or len(daemon.warm_models) != 1:
        return None
    current = daemon.warm_models[0]
    return current if current in models else None


def reconcile_pending_switch(
    manager_state: dict[str, Any],
    daemon: LocalDaemonState | None,
    now: float,
    warmup_timeout: float,
) -> Decision | None:
    pending = manager_state.get("pending_switch")
    if not isinstance(pending, dict) or not pending.get("target"):
        return None
    target = str(pending["target"])
    command_at = float(pending.get("command_at") or now)
    elapsed = max(0.0, now - command_at)

    if warm_selection_matches(daemon, target):
        manager_state.pop("pending_switch", None)
        manager_state["active_target"] = target
        # Dwell starts when the model is actually warm, not when launchd
        # accepted a command that may still spend minutes loading weights.
        manager_state["last_switch_at"] = now
        return Decision(
            target,
            f"warm-up confirmed after {format_duration(elapsed)}; starting minimum warm time (--min-warm-time)",
        )

    if pending.get("command_error"):
        return Decision(
            target,
            f"switch attempt failed: {pending['command_error']}; automatic restart is blocked",
            warming=True,
        )

    if (
        daemon
        and daemon.load_error_model == target
        and daemon.load_error_message
        and daemon.load_error_at >= command_at
    ):
        return Decision(
            target,
            f"model load failed: {daemon.load_error_message}; automatic restart is blocked",
            warming=True,
        )

    if elapsed < warmup_timeout:
        remaining = warmup_timeout - elapsed
        return Decision(
            target,
            f"waiting for Darkbloom to finish loading; "
            f"{format_duration(remaining)} remaining before --warmup-timeout",
            warming=True,
        )

    return Decision(
        target,
        f"model loading exceeded {format_duration(warmup_timeout)} (--warmup-timeout); automatic restart is "
        "blocked; inspect `darkbloom status` and logs",
        warming=True,
    )


def parse_json_output(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        # The CLI can print an update banner before JSON, including [brackets].
        decoder = json.JSONDecoder()
        for match in re.finditer(r"(?m)^[ \t]*(?=[{\[])", text):
            try:
                payload, _ = decoder.raw_decode(text[match.end():])
            except json.JSONDecodeError:
                continue
            return payload
        raise error


def local_model_ids(darkbloom: str, config_path: Path | None = None) -> set[str]:
    """Read Darkbloom's local scan; --all bypasses config, not memory filtering."""
    command = [darkbloom, "models", "list"]
    if config_path:
        command.extend(["--config", str(config_path)])
    command.extend(["--all", "--json"])
    return command_model_ids(command, "local")


def catalog_model_ids(darkbloom: str, config_path: Path | None = None) -> set[str]:
    """Read concrete catalog IDs from the configured coordinator, without aliases."""
    command = [darkbloom, "models", "catalog"]
    if config_path:
        command.extend(["--config", str(config_path)])
    command.append("--json")
    return command_model_ids(command, "catalog")


def command_model_ids(command: list[str], source: str) -> set[str]:
    """Accept the CLI's local object or catalog array, never partial inventories."""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"could not list {source} models: {detail}")
    payload = parse_json_output(result.stdout)
    records = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or any(
        not isinstance(model, dict)
        or not isinstance(model.get("id"), str)
        or not model["id"].strip()
        for model in records
    ):
        raise RuntimeError(f"unexpected {source} models JSON; expected a models array with string ids")
    return {model["id"] for model in records}


def ordered_model_ids(models: Iterable[str]) -> list[str]:
    priority = {model: index for index, model in enumerate(DEFAULT_WEIGHTS)}
    return sorted(set(models), key=lambda model: (priority.get(model, len(priority)), model))


def eligible_local_targets(
    models: Iterable[str],
    local: set[str],
    ignored: set[str],
) -> list[str]:
    """Apply the same local-model eligibility in live and dry-run modes."""
    return [
        model for model in models
        if model in local and model not in ignored
    ]


def reconcile_selection_policy(
    manager_state: dict[str, Any],
    eligible: list[str],
    ignored: set[str],
    discovery_available: bool = True,
) -> None:
    """Never resume an old command that the current policy forbids."""
    policy = {"eligible": eligible, "ignored": sorted(ignored)}
    if manager_state.get("selection_policy") != policy:
        for prefix in ("live", "dry"):
            manager_state.pop(f"{prefix}_challenger_model", None)
            manager_state.pop(f"{prefix}_challenger_streak", None)
    manager_state["selection_policy"] = policy
    pending = manager_state.get("pending_switch")
    if isinstance(pending, dict):
        target = pending.get("target")
        saved_warm = pending.get("warm_models")
        if (
            not isinstance(target, str)
            or not target
            or (discovery_available and target not in eligible)
            or target in ignored
            or saved_warm != [target]
        ):
            manager_state.pop("pending_switch", None)
            log("discarded pending switch: target or saved warm set is no longer eligible")
    for key in ("active_target", "last_decision_target"):
        if manager_state.get(key) not in eligible:
            manager_state.pop(key, None)


def _toml_array_end(text: str, start: int) -> int:
    """Return the position after a TOML array without reformatting the file."""
    if start >= len(text) or text[start] != "[":
        raise RuntimeError("preload_models must be a TOML array")
    depth = 0
    quote: str | None = None
    escaped = False
    in_comment = False
    for position in range(start, len(text)):
        character = text[position]
        if in_comment:
            if character in "\r\n":
                in_comment = False
            continue
        if quote:
            if quote == '"' and character == "\\" and not escaped:
                escaped = True
                continue
            if character == quote and not escaped:
                quote = None
            escaped = False
            continue
        if character in "'\"":
            quote = character
        elif character == "#":
            in_comment = True
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return position + 1
    raise RuntimeError("preload_models has an unterminated TOML array")


def render_preload_model_config(text: str, model_id: str) -> str:
    """Replace only backend.preload_models while preserving the surrounding TOML."""
    if not isinstance(model_id, str) or not model_id.strip() or "\n" in model_id or "\r" in model_id:
        raise RuntimeError("the preload model id must be one non-empty single-line string")

    rendered = json.dumps([model_id], ensure_ascii=False)
    newline = "\r\n" if "\r\n" in text else "\n"
    backend = re.search(
        r"(?m)^[ \t]*\[backend\][ \t]*(?:#.*)?(?:\r?\n|$)",
        text,
    )
    if not backend:
        separator = "" if not text or text.endswith(("\n", "\r")) else newline
        return (
            text
            + separator
            + (newline if text else "")
            + f"[backend]{newline}preload_models = {rendered}{newline}"
        )

    section_start = backend.end()
    next_section = re.search(
        r"(?m)^[ \t]*\[\[?[^\r\n]+\]\]?[ \t]*(?:#.*)?$",
        text[section_start:],
    )
    section_end = (
        section_start + next_section.start() if next_section else len(text)
    )
    assignment = re.search(
        r"(?m)^[ \t]*preload_models[ \t]*=[ \t]*",
        text[section_start:section_end],
    )
    if not assignment:
        prefix = (
            ""
            if section_start == 0 or text[section_start - 1] in "\r\n"
            else newline
        )
        return (
            text[:section_start]
            + prefix
            + f"preload_models = {rendered}{newline}"
            + text[section_start:]
        )

    value_start = section_start + assignment.end()
    value_end = _toml_array_end(text, value_start)
    return text[:value_start] + rendered + text[value_end:]


def synchronize_preload_model(
    config_path: Path,
    model_id: str,
) -> None:
    """Atomically set Darkbloom's startup preload to exactly one model."""
    path = config_path.expanduser()
    try:
        original = path.read_text(encoding="utf-8")
        file_stat = path.stat()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Darkbloom provider config not found at {path}; pass --config if needed"
        ) from error
    updated = render_preload_model_config(original, model_id)
    if updated == original:
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.warm-model-manager-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, file_stat.st_mode & 0o777)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def local_endpoint_start_flags(path: Path, daemon: LocalDaemonState | None) -> list[str]:
    """Keep a live endpoint's settings, or request Darkbloom's authenticated default."""
    flags = ["--local-endpoint"]
    try:
        info = read_json(path)
    except UnicodeDecodeError:
        return flags
    if not daemon or not daemon.alive or not daemon.fresh or info.get("pid") != daemon.pid:
        return flags
    try:
        url = urlsplit(info.get("base_url", ""))
        host = info.get("host") or url.hostname
        ipaddress.ip_address(host)
        port = info.get("port", url.port)
        if type(port) is not int or not 1 <= port <= 65535:
            return flags
    except (ValueError, TypeError, AttributeError):
        return flags
    flags.extend(["--port", str(port), "--bind", host])
    if info.get("api_key") == "":
        flags.append("--no-auth")
    return flags


def switch_model(
    darkbloom: str,
    model_id: str,
    config_path: Path | None,
    ignored_models: Iterable[str] = (),
    local_flags: list[str] | None = None,
) -> None:
    if model_id in ignored_models:
        raise RuntimeError("refusing to load an ignored model")
    synchronize_preload_model(
        config_path or DEFAULT_PROVIDER_CONFIG_PATH,
        model_id,
    )
    command = [darkbloom, "start"]
    if config_path:
        command.extend(["--config", str(config_path)])
    command.extend(["--model", model_id, "--idle-timeout", "0"])
    command.extend(local_flags or [])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"Darkbloom refused the switch: {detail}")


def display_name(model: str | None) -> str:
    """Keep the exact CLI model ID in every report line."""
    return model or "none"


def daemon_status_line(daemon: LocalDaemonState | None) -> str:
    if daemon is None:
        return "STOPPED or unavailable (no daemon state file)"
    if not daemon.alive:
        return f"STOPPED (stale state belongs to pid {daemon.pid})"
    if not daemon.fresh:
        return f"STALE (pid {daemon.pid} is alive but state is over 90 seconds old)"
    activity = "SERVING A REQUEST" if daemon.inference_active else "idle"
    warm = ", ".join(display_name(model) for model in daemon.warm_models) or "none"
    return f"RUNNING (pid {daemon.pid}) | warm: {warm} | {activity}"


def format_duration(seconds: float) -> str:
    rounded = max(1, math.ceil(seconds))
    if rounded < 120:
        return f"{rounded}s"
    if rounded < 7200:
        return f"{math.ceil(rounded / 60)}m"
    return f"{math.ceil(rounded / 3600)}h"


def format_human_duration(seconds: float) -> str:
    """Format an elapsed or estimated duration without decimal clock math."""
    total = max(0, int(seconds))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)

    def amount(value: int, unit: str) -> str:
        suffix = "" if value == 1 else "s"
        return f"{value} {unit}{suffix}"

    if days:
        parts = [amount(days, "day")]
        if hours:
            parts.append(amount(hours, "hour"))
        return " ".join(parts)
    if hours:
        parts = [amount(hours, "hour")]
        if minutes:
            parts.append(amount(minutes, "minute"))
        return " ".join(parts)
    if minutes:
        return amount(minutes, "minute")
    return amount(secs, "second")


def format_local_time(timestamp: float, now: float) -> str:
    moment = datetime.fromtimestamp(timestamp).astimezone()
    current = datetime.fromtimestamp(now).astimezone()
    if moment.date() == current.date():
        rendered = moment.strftime("%I:%M %p %Z")
    else:
        rendered = moment.strftime("%b %d, %I:%M %p %Z")
    return rendered.lstrip("0").replace(" 0", " ")


def next_probe_slot(manager_state: dict[str, Any], now: float) -> tuple[str, float]:
    schedule = manager_state.get("probes")
    if (not isinstance(schedule, dict)
            or schedule.get("next_kind") not in ("local", "production")
            or not isinstance(schedule.get("next_at"), (float, int))
            or not math.isfinite(schedule["next_at"])):
        return "local", now
    return schedule["next_kind"], max(0, min(schedule["next_at"], now + PROBE_SPACING))


def pending_switch_probe(manager_state: dict[str, Any]) -> dict[str, Any] | None:
    event = manager_state.get("switch_probe")
    if (not isinstance(event, dict) or not isinstance(event.get("target"), str)
            or not event["target"] or type(event.get("pid")) is not int or event["pid"] <= 0
            or type(event.get("due_at")) not in (int, float)
            or not math.isfinite(event["due_at"]) or event["due_at"] < 0):
        return None
    return event


def next_probe_event(manager_state: dict[str, Any], now: float) -> tuple[str, float, bool]:
    kind, at = next_probe_slot(manager_state, now)
    event = pending_switch_probe(manager_state)
    if event and event["due_at"] <= at:
        return "production", event["due_at"], True
    return kind, at, False


def track_current_residency(
    manager_state: dict[str, Any],
    daemon: LocalDaemonState | None,
    current: str | None,
    now: float,
) -> tuple[float, float] | None:
    """Track continuous residency of the current warm model."""
    if (
        current is None
        or daemon is None
        or not daemon.alive
        or not daemon.fresh
        or not warm_selection_matches(daemon, current)
    ):
        return None

    warm_models = sorted(daemon.warm_models)
    previous = manager_state.get("current_residency")
    same_selection = (
        isinstance(previous, dict)
        and previous.get("target") == current
        and previous.get("warm_models") == warm_models
        and int(previous.get("pid") or 0) == daemon.pid
        and float(previous.get("daemon_started_at") or 0) == daemon.started_at
    )
    if same_selection:
        warm_since = float(previous.get("warm_since") or now)
    else:
        warm_since = daemon.started_at if daemon.started_at > 0 else now
        last_switch = float(manager_state.get("last_switch_at") or 0)
        if manager_state.get("active_target") == current and last_switch > warm_since:
            # Manager-triggered switches start residency only once warm-up has
            # been confirmed, not when the replacement daemon first launched.
            warm_since = last_switch
    manager_state["current_residency"] = {
        "target": current,
        "warm_models": warm_models,
        "pid": daemon.pid,
        "daemon_started_at": daemon.started_at,
        "warm_since": warm_since,
    }
    return max(0.0, now - warm_since), warm_since


def dwell_anchor(
    manager_state: dict[str, Any], daemon: LocalDaemonState | None
) -> float:
    # A running daemon with no warm model has no useful residency to protect.
    # Let the manager repair it immediately; pending-switch reconciliation
    # separately prevents repeated restarts during a legitimate warm-up.
    if daemon is None or not daemon.warm_models:
        return 0.0
    return max(
        float(manager_state.get("last_switch_at") or 0),
        float(daemon.started_at or 0),
    )


def switch_block_reason(
    decision: Decision,
    eligible_models: list[str],
    manager_state: dict[str, Any],
    daemon: LocalDaemonState | None,
    now: float,
    min_dwell: float,
) -> str | None:
    if decision.target not in eligible_models:
        return "no eligible scored selection"
    if not daemon or not daemon.alive or not daemon.fresh:
        return "Darkbloom is stopped or stale; run `darkbloom status`"
    if daemon.inference_active:
        return "this provider is actively serving a request"
    anchor = dwell_anchor(manager_state, daemon)
    if anchor and now - anchor < min_dwell:
        return f"minimum warm time: {math.ceil(min_dwell - (now - anchor))} seconds remaining (--min-warm-time)"
    return None


@dataclass(frozen=True)
class TimelineEvent:
    at: float
    text: str
    attention: bool = False


def switch_ready_at(current: str | None, decision: Decision, state: dict[str, Any],
                    daemon: LocalDaemonState | None, now: float, next_check: float,
                    interval: float, confirmations: int, minimum: float) -> float | None:
    """Earliest conditional action on the score-check cadence, without changing it."""
    if not decision.target or decision.warming:
        return None
    if decision.target == current and (not decision.challenger or decision.challenger_streak <= 0):
        return None
    remaining = max(0, confirmations - decision.challenger_streak) if decision.target == current else 0
    passing_at = next_check + (remaining - 1) * interval if remaining else now
    anchor = dwell_anchor(state, daemon)
    ready = max(passing_at, anchor + minimum if anchor else now)
    if ready <= now:
        return now
    return next_check + max(0, math.ceil((ready - next_check) / interval)) * interval


def report_timeline(state: dict[str, Any], daemon: LocalDaemonState | None,
                    current: str | None, decision: Decision, now: float,
                    next_check: float | None, interval: float, confirmations: int,
                    minimum: float, hourly_probes: bool, recover_routing: bool,
                    apply: bool) -> tuple[list[TimelineEvent], list[str]]:
    """Read all timers together. Conditional/paused actions have no invented date."""
    events: list[TimelineEvent] = []
    notes: list[str] = []
    recovery = routing_recovery(state)
    phase = recovery.get("phase")
    recovering = phase in RECOVERY_ACTIVE_PHASES
    blocking = recovery_blocks_selection(state, recover_routing, apply)
    continuous = next_check is not None
    if continuous:
        check_at = max(now, next_check)
        text = "refresh recovery status; score checks paused" if blocking else "check scores"
        ready_at = None if blocking or state.get("pending_switch") else switch_ready_at(
            current, decision, state, daemon, now, check_at, interval, confirmations, minimum)
        contender = decision.challenger or decision.target
        if not blocking and (decision.warming or state.get("pending_switch")):
            text = "check pending model warm-up"
        elif not blocking and decision.challenger and decision.target == current and decision.challenger_streak > 0:
            count = min(confirmations, decision.challenger_streak + 1)
            text = f"check {count} of {confirmations} for {contender}"
        switch_verb = "would switch" if not apply else "switch"
        condition = "if checks still pass and the provider is online and idle"
        if ready_at is not None and ready_at == check_at:
            text += f"; earliest {switch_verb} {condition}"
        elif ready_at is not None:
            events.append(TimelineEvent(ready_at, f"earliest {switch_verb} to {contender}, {condition}", True))
        events.append(TimelineEvent(check_at, text, bool(decision.challenger) and not blocking))
    else:
        notes.append("one check only; no recurring events scheduled")

    if blocking:
        notes.append("model switching and regular/after-switch probes paused during routing recovery")
    elif recovering:
        notes.append("regular/after-switch probes paused during routing verification; score checks continue")
    if recover_routing and apply and continuous:
        if phase == "offline":
            at = max(recovery["restart_at"], recovery.get("selection_check_at", 0))
            remaining = recovery["restart_at"] - now
            wait = f"; {format_duration(remaining)} remaining after confirmed stop" if remaining > 0 else ""
            events.append(TimelineEvent(at, "check fresh scores, then start the highest eligible model" + wait, True))
        elif phase in {"monitoring", "checking", "verifying"} and recovery.get("next_check_at") is not None:
            deadline = recovery.get("verify_by") if phase == "verifying" else None
            if deadline is None or max(now, recovery["next_check_at"]) < deadline:
                events.append(TimelineEvent(recovery["next_check_at"], "recovery check, local then production self-route"))
            if phase == "verifying" and recovery.get("verify_by") is not None:
                events.append(TimelineEvent(recovery["verify_by"], "recovery verification deadline; no repeat recovery shutdown"))
        elif phase in {"starting", "stopping"}:
            notes.append("waiting for confirmed " + ("warm-up" if phase == "starting" else "shutdown; 15m offline wait starts then"))
        elif not recovery:
            notes.append("recovery observation starts once one model is confirmed warm")
    elif recovering:
        notes.append("saved recovery is paused; resume with run --apply --recover-routing")

    if hourly_probes and apply and not recovering:
        kind, at = next_probe_slot(state, now)
        other = "production" if kind == "local" else "local"
        candidates = [(kind, at, False), (other, max(now, at) + PROBE_SPACING, at <= now)]
        extra = pending_switch_probe(state)
        if extra:
            candidates.insert(0, ("after switch", extra["due_at"], False))
        if not continuous:
            # `once` can attempt just one due probe before exiting.
            candidates = sorted(candidates, key=lambda item: item[1])[:1]
            candidates = [item for item in candidates if item[1] <= now]
        for kind, at, estimated in candidates:
            endpoint = "local endpoint" if kind == "local" else "production self-route"
            text = f"probe, {endpoint}"
            if kind == "after switch":
                text += f" after switch to {extra['target']}"
            if estimated:
                text += "; estimated, 30m after the preceding regular attempt"
            events.append(TimelineEvent(at, text))
        if continuous and not extra:
            notes.append("+3m after a switch is confirmed warm: one production self-route request")
    elif hourly_probes and not apply:
        notes.append("probes disabled in DRY RUN; --apply required")
    return sorted(events, key=lambda event: event.at), notes


def terminal_style(text: str, style: str) -> str:
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return text
    code = {"muted": "90", "warm": "1", "lead": "33"}[style]
    return f"\033[{code}m{text}\033[0m"


def section_line(section: str, text: str, style: str | None = None) -> None:
    body = terminal_style(text, style) if style else text
    print(terminal_style(f"{section:<9}", "muted") + body, flush=True)


def timeline_clock(at: float, now: float) -> str:
    moment = datetime.fromtimestamp(at).astimezone()
    today = datetime.fromtimestamp(now).astimezone()
    return moment.strftime("%H:%M:%S" if moment.date() == today.date() else "%b %d %H:%M:%S")


def print_timeline(events: list[TimelineEvent], notes: list[str], now: float) -> None:
    label = "next"
    for event in events:
        timing = timeline_clock(event.at, now)
        if event.at <= now:
            timing += " (due now)" if event.at == now else f" (overdue {format_duration(now - event.at)})"
        section_line(label, f"{timing}  {event.text}", "lead" if event.attention else None)
        label = ""
    for note in notes:
        section_line(label, note, "muted")
        label = ""


def score_distance(score: float | None, warm_score: float | None,
                   switch_cost: float, horizon: float) -> str:
    if score is None or warm_score is None:
        return "N/A"
    adjusted = score * max(0, (horizon - switch_cost) / horizon)
    if warm_score == 0:
        return "> zero" if adjusted > 0 else "equal zero"
    return f"{(adjusted / warm_score - 1) * 100:+.0f}%"


def print_report(
    models: list[str], samples: dict[str, CapacitySample], averages: dict[str, float],
    weights: dict[str, float], prices: dict[str, ModelPrice], scores: dict[str, float],
    pressure_history: dict[str, list[dict[str, float]]], manager_state: dict[str, Any],
    daemon: LocalDaemonState | None, current: str | None, decision: Decision,
    apply: bool, interval_seconds: float, history_size: int, confirmations: int,
    now: float, residency: tuple[float, float] | None,
    ignored_models: set[str], eligible_models: list[str], blocked: str | None,
    improvement_percent: float, hide_ignored: bool, hourly_probes: bool = False,
    recover_routing: bool = False, columns: str = "ladder", switch_cost: float = 300,
    horizon: float = 3600, minimum: float = 2700, next_check: float | None = None,
) -> None:
    snapshot_models = (manager_state.get("last_score_snapshot") or {}).get("models", {})
    ignored = {model for model in models if model in ignored_models or snapshot_models.get(model, {}).get("ignored")}
    auto = {model for model in models if model not in ignored and snapshot_models.get(model, {}).get("auto_ignored")}
    shown = [model for model in models if not hide_ignored or model not in ignored | auto]
    shown_scores = {model: scores[model] for model in shown if model in scores}
    warm_models = daemon.warm_models if daemon and daemon.alive and daemon.fresh else ()
    warm = warm_models[0] if len(warm_models) == 1 else None
    recovery = routing_recovery(manager_state)
    recovering = recovery_blocks_selection(manager_state, recover_routing, apply)
    selection_ready = warm_selection_matches(daemon, decision.target)
    if recovering:
        action = "RECOVERY " + recovery["phase"].upper()
    elif not decision.target:
        action = "WAIT"
    elif decision.warming:
        action = "WARMING"
    elif selection_ready:
        action = "KEEP"
    elif blocked:
        action = "DEFERRED"
    else:
        action = "SWITCH" if apply else "WOULD SWITCH"
    model_width = max([25, *(len(model) + 2 for model in shown)])
    average_label = f"AVG {format_duration(interval_seconds * history_size)}"
    table_header = (f"{'MODEL ID':<{model_width}} {'NOW':>6} {average_label:>8} {'N':>3} "
                    f"{'IN$/M':>7} {'OUT$/M':>7} {'BLEND$/M':>8} {'WEIGHT':>6} {'SCORE':>7} STATUS")
    timestamp = datetime.fromtimestamp(now).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print("\n" + "=" * (len(table_header) + 9 if columns == "full" else 112), flush=True)
    print(f"Darkbloom warm model manager {MANAGER_VERSION}    {timestamp}\n", flush=True)
    mode = "LIVE" if apply else "DRY RUN"
    if warm_models:
        elapsed = f"warm {format_human_duration(residency[0])}  " if residency else "warm  "
        activity = "serving a request" if daemon.inference_active else "idle"
        description = f"{', '.join(warm_models)}  {elapsed}{activity}"
    else:
        description = daemon_status_line(daemon)
    section_line("now", f"{description}    {mode}, {action}", "warm")
    if not recovering:
        if (columns == "ladder" and decision.challenger_streak > 0
                and decision.challenger in scores and current == warm and warm in scores
                and not decision.warming):
            adjusted = scores[decision.challenger] * max(0, (horizon - switch_cost) / horizon)
            gain = (f"{(adjusted / scores[warm] - 1) * 100:+.2f}% improvement"
                    if scores[warm] > 0 else "positive score above a zero warm score")
            section_line("", f"{decision.challenger}: {gain} after switch cost; "
                         f"{decision.challenger_streak}/{confirmations} consecutive checks passed")
        else:
            section_line("", decision.reason)
        if blocked and decision.target and not selection_ready and not decision.warming:
            section_line("", "waiting: " + blocked)
        if decision.target and decision.target != warm:
            section_line("", f"target: {decision.target}")
    events, notes = report_timeline(manager_state, daemon, current, decision, now, next_check,
                                   interval_seconds, confirmations, minimum, hourly_probes,
                                   recover_routing, apply)
    print("", flush=True)
    print_timeline(events, notes, now)
    print("", flush=True)

    def number(value: float | None, decimals: int = 3) -> str:
        return f"{value:.{decimals}f}" if value is not None else "N/A"

    if recovering:
        section_line("score", "paused during routing recovery; no fresh ranking")
    elif not shown:
        section_line("score", "No models to show.")
    elif columns == "full":
        section_line("score", table_header)
        section_line("", "-" * len(table_header))
        for model in shown:
            sample = samples.get(model)
            label = f"{'*' if model in warm_models else ' '} {model}"
            price = prices.get(model, ModelPrice(None, None))
            section_line("", f"{label:<{model_width}} {number(sample.pressure if sample else None):>6} "
                         f"{number(averages.get(model)):>8} {len(pressure_history.get(model, [])):>3} "
                         f"{number(price.input_usd, 4):>7} {number(price.output_usd, 4):>7} "
                         f"{number(price.blended_usd, 4):>8} {weights.get(model, 1):>6.2f} "
                         f"{number(scores.get(model)):>7} " + "; ".join(snapshot_models.get(model, {}).get("status", [])))
    else:
        top = max(shown_scores.values(), default=0)
        eligible_top = max((scores[model] for model in shown if model in eligible_models and model in scores), default=None)
        # Stable sorting keeps the configured tie order. Unknown scores go last.
        ranked = sorted(shown, key=lambda model: (model not in scores, -scores.get(model, 0)))
        bar_char = "█" if (sys.stdout.encoding or "").lower().replace("-", "") == "utf8" else "#"
        section_line("score", f"{'MODEL ID':<{model_width}} {'SCORE':>7}  {'':12} {'VS WARM':>10}  {'AVG':>9} {'BLEND$/M':>10} {'WEIGHT':>7}", "muted")
        for model in ranked:
            score = scores.get(model)
            bar_size = max(1, round(score / top * 12)) if score is not None and score > 0 and top > 0 else 0
            bar = bar_char * bar_size
            distance = "warm" if model == warm else score_distance(score, scores.get(warm), switch_cost, horizon)
            price = prices.get(model, ModelPrice(None, None))
            label = f"{'*' if model in warm_models else ' '} {model}"
            status = "; ".join(snapshot_models.get(model, {}).get("status", []))
            price_cell = f"${price.blended_usd:.4f}" if price.blended_usd is not None else "N/A"
            line = (f"{label:<{model_width}} {number(score):>7}  {bar:<12} {distance:>10}  "
                    f"avg {number(averages.get(model)):>5}  {price_cell:>8}  "
                    f"×{weights.get(model, 1):.2f}" + (f"  {status}" if status else ""))
            style = "warm" if model == warm else "lead" if score is not None and score == eligible_top and model in eligible_models else None
            section_line("", line, style)
    if not recovering:
        print("", flush=True)
        if shown_scores:
            highest = max(shown_scores.values())
            leaders = [model + (" [IGNORED]" if model in ignored else " [AUTO-IGNORED]" if model in auto else "")
                       for model in shown if shown_scores.get(model) == highest]
            ranking_label = "Highest raw score (shown models)" if hide_ignored else "Highest raw score"
            section_line("", ranking_label + ": " + " = ".join(leaders) + f" ({highest:.3f}).")
        section_line("", "score = average pressure × blend price × weight; blend = 85% input price + 15% output price", "muted")
        if columns == "ladder":
            section_line("", "% is after switch cost vs the warm model; raw ranking is not an approved switch", "muted")
            section_line("", "now, sample count, input and output prices: --columns full", "muted")
        else:
            section_line("", f"N is retained samples (max {history_size}); samples expire after {format_duration(interval_seconds * history_size)}.", "muted")
        section_line("", f"switch requires at least {improvement_percent:g}% score improvement after switch cost, "
                     f"{confirmations} consecutive passing checks and minimum warm time; ignored models cannot load", "muted")
        if hide_ignored:
            section_line("", f"{len(shown)} shown; {len(ignored)} ignored hidden; {len(auto)} auto-ignored hidden", "muted")
    print("", flush=True)
    discovery = manager_state.get("discovery") or {}
    catalog = manager_state.get("catalog") or {}
    pricing = manager_state.get("pricing_status") or {}
    def fetched(source: dict[str, Any]) -> str:
        at = source.get("fetched_at")
        return f", fetched {timeline_clock(at, now)}" if at else ""
    section_line("sources", f"discovery {discovery.get('status', 'unknown')}, local scan  "
                 f"catalog {catalog.get('status', 'unavailable')}, {len(catalog.get('models') or [])} models{fetched(catalog)}  "
                 f"prices {pricing.get('status', 'unknown')}{fetched(pricing)}")
    if recovering and (not recover_routing or not apply):
        recovery_status = "saved recovery is paused; run --apply --recover-routing required"
    elif not recover_routing:
        recovery_status = "off (--recover-routing)"
    elif not apply:
        recovery_status = "DRY RUN; no recovery actions"
    else:
        recovery_status = f"{recovery.get('phase', 'waiting')}, {recovery.get('reason', 'waiting for one warm model')}"
    section_line("", "recovery " + recovery_status)
    if not hourly_probes:
        section_line("", "probes off (--hourly-probes)", "muted")


class Manager:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.models: list[str] = []
        self.ignored = set(args.ignore_model)
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update(args.weight)
        self.state_path: Path = args.state
        self.stop_requested = False
        self.next_probe_at = 0.0
        self.next_recovery_at = 0.0
        self.next_check_at: float | None = None
        self.report_decision = Decision(None, "waiting for the next score check")
        self.report_current: str | None = None

    def stop(self, _signum: int, _frame: Any) -> None:
        self.stop_requested = True

    def recovery_if_due(self) -> None:
        if (not self.args.recover_routing or not self.args.apply
                or self.args.mode != "run" or self.stop_requested):
            return
        now = time.time()
        if now < self.next_recovery_at:
            return
        self.next_recovery_at = now + 15
        # Unlike a score cache, an unreadable recovery intent cannot be reset.
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if not isinstance(state, dict):
            raise ValueError("invalid manager state")
        saved = routing_recovery(state)
        scope = {"config": str((self.args.config or DEFAULT_PROVIDER_CONFIG_PATH).resolve()),
                 "daemon": str(self.args.daemon_state.resolve()), "darkbloom": self.args.darkbloom,
                 "endpoint": str(self.args.local_endpoint_file.resolve()),
                 "token": str(self.args.prod_token_file.resolve())}
        if saved and saved.get("scope") != scope:
            raise ValueError("recovery paths changed; resume with the original paths or use reset-recovery when inactive")
        recovery = saved or {"schema": 1, "phase": "monitoring", "attempted": False,
                             "scope": scope, "failures": 0}
        state["routing_recovery"] = recovery

        def save(reason: str, phase: str | None = None) -> None:
            changed = reason != recovery.get("reason") or (phase and phase != recovery["phase"])
            recovery["reason"] = reason
            if phase:
                recovery["phase"] = phase
            write_json_atomic(self.state_path, state)
            if changed:
                log("routing recovery: " + reason)

        def permitted(model: str | None) -> bool:
            return bool(model and model not in self.ignored
                        and (self.args.model is None or model in self.args.model))

        def same_process(daemon: LocalDaemonState | None) -> bool:
            return bool(daemon and daemon.pid == recovery.get("pid")
                        and daemon.started_at == recovery.get("started_at"))

        def traffic(daemon: LocalDaemonState | None) -> bool:
            return bool(same_process(daemon) and daemon.alive and daemon.fresh
                        and daemon.requests_served is not None
                        and recovery.get("last_requests") is not None
                        and daemon.requests_served > recovery["last_requests"])

        def recovered(daemon: LocalDaemonState) -> None:
            recovery.update(attempted=False, failures=0, last_requests=daemon.requests_served,
                            next_check_at=time.time() + RECOVERY_CHECK_SPACING)
            save("network requests reached this provider; recovery armed for a future failure", "monitoring")

        def probe(kind: str, daemon: LocalDaemonState) -> dict[str, Any]:
            url, token = probe_credentials(kind, daemon, self.args.local_endpoint_file, self.args.prod_token_file)
            log(f"recovery {kind} probe: SENDING; model={json.dumps(recovery['model'])}; "
                + ("route=self" if kind == "production" else "loopback"))
            result = send_probe(kind, recovery["model"], url, token)
            recovery["last_" + kind + "_result"] = {**result, "at": time.time(), "model": recovery["model"]}
            detail = "; ".join(f"{key}={result[key]}" for key in
                               ("status", "elapsed_seconds", "provider_id", "error") if key in result)
            log(f"recovery {kind} probe: {result['outcome']}; HTTP {result.get('http_status') or 'N/A'}; {detail}")
            write_json_atomic(self.state_path, state)
            return result

        daemon = read_daemon_state(self.args.daemon_state, now=now)
        phase = recovery["phase"]
        if phase in {"stopping", "offline"}:
            services = provider_services()
            running = bool(services or process_alive(recovery["pid"]) or (daemon and daemon.alive))
            if running:
                if phase == "offline":
                    save("provider started during the offline wait; recovery cancelled, no further stop", "locked")
                elif now - recovery["command_at"] >= 60:
                    save("stop was not confirmed; no retry; inspect darkbloom status", "locked")
                return
            if phase == "stopping":
                recovery.update(restart_at=time.time() + RECOVERY_OFFLINE_TIME, stopped_monotonic=time.monotonic())
                save("provider stopped; waiting 15 minutes before a fresh model selection", "offline")
                return
            # A forward wall-clock adjustment must not shorten the wait. A
            # reboot resets monotonic time; conservatively wait again then.
            elapsed = time.monotonic() - recovery["stopped_monotonic"]
            if elapsed < 0:
                recovery.update(restart_at=now + RECOVERY_OFFLINE_TIME, stopped_monotonic=time.monotonic())
                save("clock restarted; waiting a full 15 minutes")
                return
            if now < recovery["restart_at"] or elapsed < RECOVERY_OFFLINE_TIME:
                if recovery["restart_at"] < now + RECOVERY_OFFLINE_TIME - elapsed:
                    recovery["restart_at"] = now + RECOVERY_OFFLINE_TIME - elapsed
                    save("wall clock moved; offline countdown still uses elapsed time")
                return
            if now < recovery.get("selection_check_at", 0):
                return
            if recovery_config_digest(self.args.config) != recovery["config_digest"]:
                save("provider config changed during recovery; restart cancelled", "locked")
                return
            # Commit the retry cadence before any slow reads. No old averages,
            # passing checks or price cache may choose this cold-start model.
            recovery["selection_check_at"] = now + self.args.check_every
            save("offline wait complete; checking fresh scores for a new start")
            check = self.refresh_scores(state, time.time(), fresh_start=True)
            if not check.eligible:
                save("no eligible model with fresh pressure and prices; staying offline until the next score check")
                return
            decision = choose_scored_target(
                check.eligible, check.scores, None, None, 0,
                self.args.switch_improvement_percent, self.args.switch_after_checks,
                self.args.switch_cost, self.args.decision_horizon)
            target = decision.target
            # A download/removal, config edit or external start can race the
            # network reads. Check again before recording any launch intent.
            if not permitted(target) or target not in self.discover_models(state, time.time()):
                save("selected model is no longer allowed or locally available; staying offline until the next score check")
                return
            if recovery_config_digest(self.args.config) != recovery["config_digest"]:
                save("provider config changed during score check; restart cancelled", "locked")
                return
            services = provider_services()
            fresh_daemon = read_daemon_state(self.args.daemon_state, now=time.time())
            if services or process_alive(recovery["pid"]) or (fresh_daemon and fresh_daemon.alive):
                save("provider appeared before restart; recovery cancelled", "locked")
                return
            if self.stop_requested:
                return
            recovery.update(previous_model=recovery["model"], model=target,
                            phase="starting", command_at=time.time())
            recovery.pop("selection_check_at", None)
            state.update(last_decision_at=recovery["command_at"], last_decision_target=target,
                         last_decision_reason="fresh recovery start: " + decision.reason,
                         manager_version=MANAGER_VERSION)
            state["pending_switch"] = {"target": target, "warm_models": [target],
                                       "command_at": recovery["command_at"]}
            save(f"fresh recovery check selected {target} (score {check.scores[target]:.6g}); "
                 f"previous model {recovery['previous_model']}; starting once, no launch retry")
            try:
                switch_model(self.args.darkbloom, recovery["model"], self.args.config, self.ignored,
                             local_flags=recovery["local_flags"])
            except Exception as error:
                # The command may have succeeded before an interruption. Keep
                # waiting for fresh warm state, but never issue it a second time.
                state["pending_switch"]["command_error"] = probe_debug_text(str(error), "")
                save("start command failed or timed out; waiting for observed warm-up, no retry")
            return

        if phase == "starting":
            if not permitted(recovery["model"]):
                save("saved model is now excluded; no more recovery actions", "locked")
                return
            if warm_selection_matches(daemon, recovery["model"]) and not same_process(daemon):
                state.pop("pending_switch", None)
                state.update(active_target=recovery["model"], last_switch_at=now)
                recovery.update(pid=daemon.pid, started_at=daemon.started_at, last_requests=0,
                                reconnect_count=daemon.reconnect_count, warm_since=now,
                                verify_by=now + RECOVERY_VERIFY_TIME, next_check_at=now + SWITCH_PROBE_DELAY)
                save("model warm; normal score checks resumed; checking routing in 3 minutes", "verifying")
                if traffic(daemon):
                    recovered(daemon)
            elif now - recovery["command_at"] >= self.args.warmup_timeout:
                save("warm-up not confirmed; no retry; inspect darkbloom status and logs", "locked")
            return

        warm = bool(daemon and daemon.alive and daemon.fresh and len(daemon.warm_models) == 1)
        model = daemon.warm_models[0] if warm else None
        if not warm or not permitted(model) or daemon.requests_served is None or state.get("pending_switch"):
            recovery.update(failures=0, warm_since=now, next_check_at=now + RECOVERY_GRACE)
            save("waiting for one permitted warm model, fresh network counter and completed warm-up",
                 "locked" if recovery["attempted"] else "monitoring")
            return
        if phase == "verifying" and (not same_process(daemon) or model != recovery.get("model")):
            save("provider or model changed during verification; recovery unconfirmed", "locked")
            return
        if traffic(daemon):
            recovered(daemon)
            return
        identity_changed = (not same_process(daemon) or model != recovery.get("model")
                            or daemon.reconnect_count != recovery.get("reconnect_count")
                            or (recovery.get("last_requests") is not None
                                and daemon.requests_served < recovery["last_requests"]))
        if identity_changed:
            recovery.update(model=model, pid=daemon.pid, started_at=daemon.started_at,
                            reconnect_count=daemon.reconnect_count, last_requests=daemon.requests_served,
                            warm_since=now, failures=0, next_check_at=now + RECOVERY_GRACE)
            save("observing the warm model for 15 minutes before routing checks",
                 "locked" if recovery["attempted"] else "monitoring")
            return
        if recovery["attempted"] and phase != "verifying":
            save("attempt used; waiting for this provider's network traffic or reset-recovery", "locked")
            return
        if phase == "verifying" and now >= recovery["verify_by"]:
            save("this provider's routing is unconfirmed after 15 minutes; left running, no retry", "locked")
            return
        if daemon.inference_active:
            recovery.update(failures=0, next_check_at=now + RECOVERY_CHECK_SPACING)
            save("provider busy; recovery checks deferred", "verifying" if phase == "verifying" else "monitoring")
            return
        if now < recovery.get("next_check_at", now):
            return
        # A long pause breaks consecutive checks; never count days-old failures.
        if now > recovery.get("next_check_at", now) + RECOVERY_CHECK_SPACING:
            recovery["failures"] = 0
        recovery["next_check_at"] = now + RECOVERY_CHECK_SPACING
        save("checking local inference and production self-routing")
        try:
            # Validate both records before sending either request.
            for kind in ("local", "production"):
                probe_credentials(kind, daemon, self.args.local_endpoint_file, self.args.prod_token_file)
            local = probe("local", daemon)
            fresh = read_daemon_state(self.args.daemon_state, now=time.time())
            if traffic(fresh):
                recovered(fresh)
                return
            if (self.stop_requested or not same_process(fresh) or not warm_selection_matches(fresh, model)
                    or fresh.inference_active or local.get("outcome") != "SUCCESS"):
                recovery["failures"] = 0
                save("local check failed, provider changed or became busy; shutdown not permitted",
                     "verifying" if phase == "verifying" else "monitoring")
                return
            production = probe("production", fresh)
        except (OSError, ValueError) as error:
            recovery["failures"] = 0
            save(probe_debug_text(str(error), ""), "verifying" if phase == "verifying" else "monitoring")
            return
        fresh = read_daemon_state(self.args.daemon_state, now=time.time())
        if traffic(fresh):
            recovered(fresh)
            return
        if phase == "verifying":
            save("self-route succeeded for the account; this provider remains unconfirmed"
                 if production.get("outcome") == "SUCCESS" else "routing check failed; left running, no further shutdown")
            return
        if (not same_process(fresh) or not warm_selection_matches(fresh, model) or fresh.inference_active
                or fresh.reconnect_count != recovery.get("reconnect_count")
                or production.get("http_status") != 503 or production.get("error_code") != "model_not_loaded"):
            recovery["failures"] = 0
            save("no confirmed routing failure; shutdown not permitted", "monitoring")
            return
        recovery["failures"] += 1
        save(f"{recovery['failures']}/3 local successes with self-route model_not_loaded", "checking")
        if recovery["failures"] < 3 or self.stop_requested:
            return
        # Scan again before consuming the attempt. Catalog/price outages alone
        # must neither trigger nor authorize a provider operation.
        if model not in local_model_ids(self.args.darkbloom, self.args.config):
            recovery["failures"] = 0
            save("model not locally available; shutdown cancelled", "monitoring")
            return
        digest = recovery_config_digest(self.args.config)
        fresh = read_daemon_state(self.args.daemon_state, now=time.time())
        if traffic(fresh):
            recovered(fresh)
            return
        if (self.stop_requested or not same_process(fresh) or not warm_selection_matches(fresh, model)
                or fresh.inference_active or fresh.reconnect_count != recovery.get("reconnect_count")):
            recovery["failures"] = 0
            save("provider changed or became busy before shutdown; cancelled", "monitoring")
            return
        recovery.update(attempted=True, phase="stopping", command_at=time.time(), config_digest=digest,
                        local_flags=local_endpoint_start_flags(self.args.local_endpoint_file, fresh))
        state.pop("switch_probe", None)
        for key in ("live_challenger_model", "live_challenger_streak", "dry_challenger_model", "dry_challenger_streak"):
            state.pop(key, None)
        save("stopping this provider once; model switching and other probes paused")
        def still_idle() -> bool:
            current = read_daemon_state(self.args.daemon_state, now=time.time())
            return bool(not self.stop_requested and same_process(current)
                        and warm_selection_matches(current, model) and not current.inference_active
                        and current.requests_served == fresh.requests_served
                        and current.reconnect_count == fresh.reconnect_count)
        try:
            if not self.stop_requested:
                stop_provider(self.args.darkbloom, fresh.pid, before_stop=still_idle)
        except Exception as error:
            save("stop failed or timed out; checking actual process state, no retry: " + probe_debug_text(str(error), ""))

    def probe_if_due(self) -> None:
        if not self.args.hourly_probes or not self.args.apply or self.stop_requested:
            return
        now = time.time()
        if now < self.next_probe_at:
            return
        # Back off on state I/O failures too, without risking an unrecorded request.
        self.next_probe_at = now + 60
        manager_state = (json.loads(self.state_path.read_text(encoding="utf-8"))
                         if self.state_path.exists() else {})
        if not isinstance(manager_state, dict):
            raise ValueError("invalid manager state")
        if routing_recovery(manager_state).get("phase") in RECOVERY_ACTIVE_PHASES:
            return
        # Keep clock correction confined to the regular schedule.
        _, regular_at = next_probe_slot(manager_state, now)
        if isinstance(manager_state.get("probes"), dict) and regular_at != manager_state["probes"].get("next_at"):
            manager_state["probes"]["next_at"] = regular_at
            write_json_atomic(self.state_path, manager_state)
        kind, due_at, after_switch = next_probe_event(manager_state, now)
        if now < due_at:
            self.next_probe_at = due_at
            return

        event = pending_switch_probe(manager_state) if after_switch else None
        probe_label = "production probe (after switch)" if after_switch else f"{kind} probe"
        daemon = read_daemon_state(self.args.daemon_state, now=time.time())
        model = event["target"] if event else None
        reason = None
        if not daemon or not daemon.alive or not daemon.fresh:
            reason = "provider state is unavailable, stale or offline"
        elif len(daemon.warm_models) != 1:
            reason = "no single warm model"
        elif event and (daemon.warm_models[0] != event["target"] or daemon.pid != event["pid"]
                        or daemon.started_at != event.get("daemon_started_at")):
            reason = "model or provider process changed after warm-up; cancelled"
        else:
            model = daemon.warm_models[0]
            if model in self.ignored or (self.args.model is not None and model not in self.args.model):
                reason = "warm model is ignored or excluded by --model"
            elif manager_state.get("pending_switch"):
                reason = "model switch is pending"
            elif daemon.inference_active:
                reason = "provider is serving a request"
        url, token = "", ""
        if not reason:
            try:
                url, token = probe_credentials(kind, daemon, self.args.local_endpoint_file,
                                               self.args.prod_token_file)
            except ValueError as error:
                reason = str(error)  # Only fixed, credential-free messages above.
        result = {"at": now, "kind": kind, "model": model, "http_status": None,
                  "outcome": "SKIPPED" if reason else "UNKNOWN",
                  "status": "skipped: " + reason if reason else "request started; result unknown"}
        if after_switch:
            manager_state.pop("switch_probe", None)
            manager_state["last_switch_probe_result"] = result
        else:
            manager_state["probes"] = {
                "next_kind": "production" if kind == "local" else "local",
                "next_at": now + PROBE_SPACING,
                "last_result": result,
            }
        # Commit the next slot before POST: interruption or timeout cannot replay it.
        # After downtime, try one slot and space the next 30 minutes later.
        write_json_atomic(self.state_path, manager_state)
        self.next_probe_at = next_probe_event(manager_state, now)[1]
        if not reason and not self.stop_requested:
            route = "; route=self" if kind == "production" else ""
            log(f"{probe_label}: SENDING; model={json.dumps(model)}; POST {url}"
                f"{route}; max_tokens=64; timeout={PROBE_TIMEOUT}s")
            result.update(send_probe(kind, model, url, token))
        elif self.stop_requested and not reason:
            result["outcome"] = "SKIPPED"
            result["status"] = "skipped: manager is stopping"
        http_result = f"HTTP {result['http_status']}" if result["http_status"] is not None else "HTTP N/A"
        details = [result["status"]]
        for key, label in (("elapsed_seconds", "seconds"), ("finish_reason", "finish_reason"),
                           ("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"),
                           ("provider_id", "provider"), ("error", "error")):
            if key in result:
                details.append(f"{label}={result[key]}")
        log(f"{probe_label}: {result['outcome']}; model={json.dumps(model)}; {http_result}; "
            + "; ".join(details))
        report_now = time.time()
        # A provider/model change between score checks invalidates that forecast.
        report_decision = (self.report_decision if warm_selection_matches(daemon, self.report_current)
                           else Decision(None, "waiting for a fresh score check"))
        events, notes = report_timeline(
            manager_state, daemon, self.report_current, report_decision, report_now,
            (self.next_check_at or report_now + self.args.check_every) if self.args.mode == "run" else None, self.args.check_every,
            self.args.switch_after_checks, self.args.min_warm_time, True,
            self.args.recover_routing, True)
        print_timeline(events, notes, report_now)
        # Show the HTTP result even if saving its diagnostic summary fails.
        write_json_atomic(self.state_path, manager_state)

    def discover_models(self, manager_state: dict[str, Any], now: float) -> set[str]:
        previous = manager_state.get("discovery") or {}
        try:
            local = local_model_ids(self.args.darkbloom, self.args.config)
        except Exception as error:
            # Retain the last inventory in state. An unsuccessful scan
            # cannot authorize a launch, even when yesterday's inventory could.
            local = set()
            visible = previous.get("models", [])
            manager_state["discovery"] = {
                **previous, "status": "unavailable", "error": str(error),
                "checked_at": now,
            }
            log(f"local discovery unavailable; loading disabled: {error}")
        else:
            visible = ordered_model_ids(local)
            manager_state["discovery"] = {
                "models": visible, "status": "live", "checked_at": now,
                "fetched_at": now,
            }
            if visible != previous.get("models"):
                log(f"local discovery refreshed: {len(visible)} models")
        return local

    def discover_catalog(self, manager_state: dict[str, Any], now: float) -> set[str]:
        source = {"darkbloom": self.args.darkbloom, "config": str(self.args.config) if self.args.config else None}
        previous = manager_state.get("catalog") or {}
        if previous.get("source") != source:
            previous = {}
        try:
            catalog = catalog_model_ids(self.args.darkbloom, self.args.config)
        except Exception as error:
            catalog = set(previous.get("models") or [])
            status = "stale cache" if previous.get("fetched_at") else "unavailable"
            manager_state["catalog"] = {
                **previous, "models": ordered_model_ids(catalog), "source": source,
                "status": status, "checked_at": now, "error": str(error),
            }
            log(f"model catalog unavailable; {status}: {error}")
        else:
            manager_state["catalog"] = {
                "models": ordered_model_ids(catalog), "source": source,
                "status": "live", "fetched_at": now, "checked_at": now,
            }
        return catalog

    def refresh_scores(self, manager_state: dict[str, Any], now: float,
                       fresh_start: bool = False) -> ScoreCheck:
        """Use one scoring/eligibility path for ordinary checks and recovery starts."""
        if fresh_start:
            self.report_current = None
            self.report_decision = Decision(None, "waiting for the first warm score check")
            manager_state["pressure_history"] = {}
            manager_state["state_schema"] = STATE_SCHEMA
            for key in ("live_challenger_model", "live_challenger_streak",
                        "dry_challenger_model", "dry_challenger_streak",
                        "challenger_model", "challenger_streak", "current_residency",
                        "current_model", "active_target", "last_switch_at", "pending_switch",
                        "switch_probe", "last_decision_at", "last_decision_target", "last_decision_reason"):
                manager_state.pop(key, None)
            ensure_pressure_cadence(manager_state, self.args.check_every, self.args.average_samples)
            ensure_preload_sync_policy(manager_state)
            ensure_scoring_policy(manager_state)
            ensure_switch_policy(manager_state, self.args.switch_improvement_percent,
                                 self.args.switch_cost, self.args.decision_horizon)
        catalog = self.discover_catalog(manager_state, now)
        local = self.discover_models(manager_state, now)
        discovery_available = manager_state["discovery"]["status"] == "live"
        try:
            samples = fetch_capacity(self.args.base_url)
        except Exception as error:
            samples = {}
            log(f"network capacity unavailable: {error}")

        # The catalog and capacity feed expand the saved inventory only. They cannot
        # authorize loading a model absent from a successful local scan.
        visible = catalog | set(samples) | set(manager_state["discovery"].get("models") or []) | self.ignored
        self.models = list(dict.fromkeys([
            *(self.args.model or []), *ordered_model_ids(visible - self.ignored),
            *self.args.ignore_model,
        ]))
        local_eligible = eligible_local_targets(
            self.args.model if self.args.model is not None else self.models,
            local, self.ignored,
        )
        reconcile_selection_policy(
            manager_state, local_eligible, self.ignored,
            discovery_available,
        )
        history, averages = update_pressure_history(
            self.models,
            samples,
            manager_state.get("pressure_history") or {},
            self.args.average_samples,
            now,
            self.args.check_every * self.args.average_samples,
        )
        prices = cached_model_prices(
            manager_state,
            self.models,
            self.args.pricing_url,
            now,
            self.args.pricing_refresh,
            force_refresh=fresh_start,
        )
        # Retained history remains visible during a feed gap, but is not a
        # fresh score and must not drive selection.
        all_scores = revenue_scores(
            {model: average for model, average in averages.items() if model in samples},
            self.weights, prices,
        )
        eligible_models = [model for model in local_eligible if model in all_scores]
        if fresh_start and manager_state["pricing_status"]["status"] != "live":
            eligible_models = []
        manager_state["pressure_history"] = history
        manager_state["last_score_snapshot"] = build_score_snapshot(
            self.models,
            samples,
            averages,
            history,
            prices,
            self.weights,
            all_scores,
            now,
            self.args.check_every * self.args.average_samples,
            self.ignored,
            eligible_models,
            local if discovery_available else None,
            self.args.model,
        )

        return ScoreCheck(samples, averages, prices, all_scores, local_eligible,
                          eligible_models, discovery_available)

    def iteration(self) -> None:
        now = time.time()
        existing = (json.loads(self.state_path.read_text()) if self.state_path.exists() else {})
        if not isinstance(existing, dict):
            raise ValueError("invalid manager state")
        if recovery_blocks_selection(existing, self.args.recover_routing, self.args.apply):
            print_report(
                [], {}, {}, {}, {}, {}, {}, existing,
                read_daemon_state(self.args.daemon_state), None,
                Decision(None, "routing recovery in progress"), self.args.apply,
                self.args.check_every, self.args.average_samples, self.args.switch_after_checks,
                now, None, self.ignored, [], None, self.args.switch_improvement_percent,
                self.args.hide_ignored, self.args.hourly_probes, self.args.recover_routing,
                self.args.columns, self.args.switch_cost, self.args.decision_horizon,
                self.args.min_warm_time, (self.next_check_at or now + self.args.check_every) if self.args.mode == "run" else None)
            return
        manager_state = {
            key: value for key, value in existing.items()
            if key in MANAGER_STATE_KEYS
        }
        migrated_history = manager_state.get("state_schema") != STATE_SCHEMA
        if migrated_history:
            # Float-only samples from older releases have no observation time,
            # so retaining them would defeat expiration after a restart.
            manager_state["pressure_history"] = {}
            for key in (
                "challenger_model",
                "challenger_streak",
                "live_challenger_model",
                "live_challenger_streak",
                "dry_challenger_model",
                "dry_challenger_streak",
            ):
                manager_state.pop(key, None)
            manager_state["state_schema"] = STATE_SCHEMA
        cadence_changed = ensure_pressure_cadence(
            manager_state,
            self.args.check_every,
            self.args.average_samples,
        )
        discarded_legacy_pending = ensure_preload_sync_policy(manager_state)
        scoring_changed = ensure_scoring_policy(manager_state)
        switch_rule_changed = ensure_switch_policy(
            manager_state, self.args.switch_improvement_percent,
            self.args.switch_cost, self.args.decision_horizon,
        )
        if migrated_history:
            log("initialized timestamped pressure samples")
        elif cadence_changed:
            log("saved pressure samples and consecutive check counts reset because --check-every or --average-samples changed")
        if scoring_changed and not migrated_history:
            log("scoring formula changed; consecutive passing checks reset")
        if switch_rule_changed and not migrated_history:
            log("percentage or switch-cost rule changed; consecutive passing checks reset")
        if discarded_legacy_pending:
            log(
                "discarded a legacy pending switch so startup preload can be "
                "synchronized and retried"
            )
        check = self.refresh_scores(manager_state, now)
        samples, averages, prices, all_scores = check.samples, check.averages, check.prices, check.scores
        local_eligible, eligible_models = check.local_eligible, check.eligible
        discovery_available = check.discovery_available
        history = manager_state["pressure_history"]
        scores = {model: all_scores[model] for model in eligible_models}
        # Score reads can be slow. Assess the daemon after they finish.
        daemon = read_daemon_state(self.args.daemon_state, now=time.time())

        current = current_warm_model(daemon, local_eligible)
        challenger_model_key, challenger_streak_key = challenger_state_keys(
            self.args.apply
        )

        pending_decision = reconcile_pending_switch(
            manager_state,
            daemon,
            now,
            self.args.warmup_timeout,
        ) if discovery_available else None
        if pending_decision is not None:
            decision = pending_decision
            if not decision.warming and self.args.hourly_probes and self.args.apply:
                confirmed_at = time.time()
                manager_state["switch_probe"] = {
                    "target": decision.target, "confirmed_at": confirmed_at,
                    "due_at": confirmed_at + SWITCH_PROBE_DELAY,
                    "pid": daemon.pid, "daemon_started_at": daemon.started_at,
                }
                # The extra request may precede the cached hourly deadline.
                self.next_probe_at = 0.0
            manager_state[challenger_model_key] = None
            manager_state[challenger_streak_key] = 0
        elif not eligible_models or (current in local_eligible and current not in scores):
            decision = Decision(None, "no eligible scored target or current score unavailable; waiting for data")
            manager_state[challenger_model_key] = None
            manager_state[challenger_streak_key] = 0
        else:
            decision = choose_scored_target(
                eligible_models,
                scores,
                current,
                (
                    str(manager_state[challenger_model_key])
                    if manager_state.get(challenger_model_key)
                    else None
                ),
                int(manager_state.get(challenger_streak_key) or 0),
                self.args.switch_improvement_percent,
                self.args.switch_after_checks,
                self.args.switch_cost,
                self.args.decision_horizon,
            )
            manager_state[challenger_model_key] = decision.challenger
            manager_state[challenger_streak_key] = decision.challenger_streak

        residency = track_current_residency(
            manager_state,
            daemon,
            current,
            now,
        )
        blocked = switch_block_reason(
            decision, eligible_models, manager_state, daemon, now, self.args.min_warm_time,
        )

        self.report_current, self.report_decision = current, decision
        print_report(
            self.models,
            samples,
            averages,
            self.weights,
            prices,
            all_scores,
            history,
            manager_state,
            daemon,
            current,
            decision,
            self.args.apply,
            self.args.check_every,
            self.args.average_samples,
            self.args.switch_after_checks,
            now,
            residency,
            self.ignored,
            eligible_models,
            blocked,
            self.args.switch_improvement_percent,
            self.args.hide_ignored,
            self.args.hourly_probes,
            self.args.recover_routing,
            self.args.columns, self.args.switch_cost, self.args.decision_horizon,
            self.args.min_warm_time,
            (self.next_check_at or now + self.args.check_every) if self.args.mode == "run" else None,
        )
        selection_matches = warm_selection_matches(
            daemon,
            decision.target,
        )
        manager_state["current_model"] = current
        manager_state["last_decision_at"] = now
        manager_state["last_decision_target"] = decision.target
        manager_state["last_decision_reason"] = decision.reason
        manager_state["manager_version"] = MANAGER_VERSION

        if decision.target is None:
            log("no switch: waiting for an eligible scored model")
        elif decision.warming:
            log(f"warm-up pending: {decision.target}; no restart will be issued")
        elif selection_matches:
            log(f"no switch: desired model is already warm ({decision.target})")
        elif blocked:
            log(f"switch deferred: {blocked}")
        elif not self.args.apply:
            log(f"dry run: would switch to {decision.target}")
        else:
            # Persist intent before any provider change. A timed-out command
            # can still have restarted Darkbloom, and must never be retried
            # automatically just because it did not return successfully.
            manager_state.pop("switch_probe", None)
            self.next_probe_at = 0.0
            manager_state["pending_switch"] = {
                "target": decision.target,
                "warm_models": [decision.target],
                "command_at": time.time(),
            }
            write_json_atomic(self.state_path, manager_state)
            log("switching the launchd provider to " + decision.target)
            try:
                endpoint_options = ({"local_flags": local_endpoint_start_flags(self.args.local_endpoint_file, daemon)}
                                    if self.args.hourly_probes or self.args.recover_routing else {})
                switch_model(self.args.darkbloom, decision.target, self.args.config, self.ignored,
                             **endpoint_options)
            except Exception as error:
                manager_state["pending_switch"]["command_error"] = str(error)
                write_json_atomic(self.state_path, manager_state)
                raise
            log(f"switch command accepted: {decision.target}; waiting for Darkbloom to report the model warm")

        if selection_matches:
            manager_state["active_target"] = decision.target
        write_json_atomic(self.state_path, manager_state)

    def run(self) -> None:
        if self.args.mode == "reset-recovery":
            state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
            if routing_recovery(state).get("phase") in {"stopping", "offline", "starting"}:
                raise ValueError("cannot reset an active recovery; resume run --apply --recover-routing first")
            state.pop("routing_recovery", None)
            write_json_atomic(self.state_path, state)
            log("routing recovery reset; no provider commands issued")
            return
        if self.args.mode == "once":
            self.iteration()
            self.probe_if_due()
            return
        if self.args.apply:
            log(
                f"manager ({MANAGER_VERSION}) started in LIVE mode; "
                "model changes are enabled"
            )
        else:
            log(
                f"manager ({MANAGER_VERSION}) started in DRY-RUN mode; "
                "add --apply to enable model changes"
            )
        if self.args.hourly_probes:
            log("hourly probes enabled: local and production self-route, 30 minutes apart; "
                "extra self-route request 3 minutes after confirmed switch warm-up"
                if self.args.apply else "hourly probes disabled in dry run; --apply is required to send prompts")
        if self.args.recover_routing:
            log("routing recovery enabled: 15m warm observation, 3 checks 5m apart, one 15m shutdown"
                if self.args.apply else "routing recovery disabled in dry run; no prompts or provider commands")
        while not self.stop_requested:
            started = time.monotonic()
            self.next_check_at = time.time() + self.args.check_every
            try:
                self.recovery_if_due()
                self.iteration()
            except Exception as error:
                log(f"manager error: {error}")
            try:
                self.probe_if_due()
            except Exception:
                log("probe state error; check that the manager state file is writable")
            remaining = max(0.0, self.args.check_every - (time.monotonic() - started))
            if not self.stop_requested:
                log(
                    f"next check in {format_duration(remaining)}. Ctrl-C stops."
                )
            deadline = time.monotonic() + remaining
            while not self.stop_requested and time.monotonic() < deadline:
                try:
                    self.recovery_if_due()
                    self.probe_if_due()
                except Exception as error:
                    log("probe/recovery check failed: " + probe_debug_text(str(error), ""))
                time.sleep(max(0.0, min(1.0, deadline - time.monotonic())))
        recovery = routing_recovery(read_json(self.state_path))
        if recovery.get("phase") in {"stopping", "offline"}:
            log("manager stopped during recovery; Darkbloom may remain stopped. "
                "Resume the same run --apply --recover-routing command to finish the saved wait.")


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_weight(value: str) -> tuple[str, float]:
    try:
        model, raw_weight = value.rsplit("=", 1)
        weight = positive_float(raw_weight)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("use MODEL=WEIGHT with a positive weight") from error
    if not model:
        raise argparse.ArgumentTypeError("model id must not be empty")
    return model, weight


def legacy_margin_percent(value: str) -> float:
    percent = nonnegative_float(value) * 100.0
    if not math.isfinite(percent):
        raise argparse.ArgumentTypeError("percentage must be finite")
    return percent


def removed_absolute_margin(value: str) -> None:
    raise argparse.ArgumentTypeError(
        "fixed score margins were removed; use --switch-improvement-percent 25 for 25% (or 1 for 1%)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {MANAGER_VERSION}",
    )
    parser.add_argument("mode", choices=("once", "run", "set-prod-token", "reset-recovery"), nargs="?", default="once")
    parser.add_argument("--apply", action="store_true", help="enable model changes, opted-in probes and routing recovery; default is dry-run")
    parser.add_argument("--hourly-probes", action="store_true", help="send local and production self-route prompts once each per hour, 30 minutes apart, plus one self-route prompt 3 minutes after switch warm-up; enables the local endpoint on model switches; requires --apply")
    parser.add_argument("--recover-routing", action="store_true", help="optional routing recovery: after 15m warm and 3 local successes/self-route model_not_loaded failures 5m apart, stop for 15m, then use fresh scores to start the highest eligible model once; requires run --apply, local endpoint and production token; reset-recovery clears an inactive attempt")
    parser.add_argument("--prod-token-file", type=Path, default=DEFAULT_PROD_TOKEN_PATH, help="private production token file; save with set-prod-token (default ~/.darkbloom/warm-model-manager-prod-token)")
    parser.add_argument("--local-endpoint-file", type=Path, default=DEFAULT_LOCAL_ENDPOINT_PATH, help="Darkbloom local endpoint metadata (default ~/.darkbloom/local.json; respects DARKBLOOM_LOCAL_DIR)")
    parser.add_argument("--columns", choices=("ladder", "full"), default="ladder", help="score display: ladder (default) or full table with current pressure, sample count and separate prices")
    parser.add_argument("--model", action="append", metavar="MODEL", help="restrict loading candidates; repeat in tie-break order; other catalog rows stay visible unless --hide-ignored is set (default: all locally discovered models)")
    parser.add_argument(
        "--ignore-model", "--ignore", action="extend", nargs="+", default=[], metavar="MODEL_ID",
        help="never select or load these models; accepts one or more space-separated IDs; repeat as needed; visible unless --hide-ignored is set",
    )
    parser.add_argument(
        "--hide-ignored", action="store_true",
        help="hide ignored and auto-ignored models from the display and its ranking; keep their calculations in saved state",
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=parse_weight,
        default=[],
        metavar="MODEL=WEIGHT",
        help="model score multiplier; repeat as needed",
    )
    parser.add_argument("--check-every", "--interval", type=int, default=60, metavar="SECONDS", help="check scores this often; minimum 60 seconds (default 60)")
    parser.add_argument("--average-samples", "--history", type=int, default=15, metavar="COUNT", help="average up to this many recent pressure samples (default 15)")
    improvement = parser.add_mutually_exclusive_group()
    improvement.add_argument("--switch-improvement-percent", type=nonnegative_float, default=25.0, metavar="PERCENT", help="required score improvement after switch cost; 25 means 25%% (default 25)")
    improvement.add_argument("--relative-margin", dest="switch_improvement_percent", type=legacy_margin_percent, default=argparse.SUPPRESS, metavar="FRACTION", help="legacy form of --switch-improvement-percent: 0.25 means 25%%; use only one form")
    parser.add_argument("--absolute-margin", type=removed_absolute_margin, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--switch-after-checks", "--confirmations", type=int, default=3, metavar="COUNT", help="require this many consecutive checks meeting the percentage improvement requirement (default 3)")
    parser.add_argument("--min-warm-time", "--min-dwell", type=int, default=2700, metavar="SECONDS", help="keep the current model warm at least this long before switching (default 2700)")
    parser.add_argument("--warmup-timeout", type=positive_float, default=180, help="seconds to wait for the requested model to become warm before reporting loading as overdue (default 180)")
    parser.add_argument("--switch-cost", type=nonnegative_float, default=300, help="estimated unavailable seconds per switch (default 300)")
    parser.add_argument("--decision-horizon", type=positive_float, default=3600, help="seconds over which a switch must repay its cost (default 3600)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Darkbloom public console base URL")
    parser.add_argument("--pricing-url", default=DEFAULT_PRICING_URL, help="Darkbloom public pricing endpoint")
    parser.add_argument("--pricing-refresh", type=positive_float, default=900, help="seconds to cache input/output token prices (default 900)")
    parser.add_argument("--darkbloom", default="darkbloom", help="path to the darkbloom executable")
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "non-default provider.toml path; live switches synchronize its "
            "backend preload_models value"
        ),
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH, help="manager state JSON path (default ~/.darkbloom/warm-model-manager-state.json)")
    parser.add_argument("--daemon-state", type=Path, default=DEFAULT_DAEMON_STATE_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.prod_token_file = args.prod_token_file.expanduser()
    args.local_endpoint_file = args.local_endpoint_file.expanduser()
    args.daemon_state = args.daemon_state.expanduser()
    if args.mode == "set-prod-token":
        try:
            save_prod_token(args.prod_token_file)
        except (OSError, ValueError, EOFError):
            log("token was not saved; check the token and destination permissions")
            return 1
        except KeyboardInterrupt:
            return 130
        return 0
    if args.recover_routing and args.mode != "run":
        raise SystemExit("--recover-routing requires run mode so the manager can complete the offline wait")
    args.weight = dict(args.weight)
    models = args.model or []
    if len(set(models)) != len(models):
        raise SystemExit("managed models must be unique")
    # Weight overrides may name models downloaded after this process starts.
    if args.config:
        args.config = args.config.expanduser()
    args.state = args.state.expanduser()
    if args.check_every < 60:
        raise SystemExit("--check-every must be at least 60 seconds")
    if args.pricing_refresh < 60:
        raise SystemExit("--pricing-refresh must be at least 60 seconds")
    if args.average_samples < 1:
        raise SystemExit("--average-samples must be at least 1")
    if args.switch_after_checks < 1:
        raise SystemExit("--switch-after-checks must be at least 1")
    if args.min_warm_time < 0:
        raise SystemExit("--min-warm-time must be zero or greater")
    if args.warmup_timeout < 60:
        raise SystemExit("--warmup-timeout must be at least 60 seconds")
    if args.switch_cost >= args.decision_horizon:
        raise SystemExit("--switch-cost must be less than --decision-horizon")

    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock_paths = [args.state.with_suffix(args.state.suffix + ".lock")]
    # Older managers use these locks. Hold both before copying state or running
    # a check so an old process cannot race a renamed manager into a restart.
    for name in ("warm-model-manager.json.lock", "warm-model-manager-v4.json.lock"):
        legacy_lock = DEFAULT_STATE_PATH.with_name(name)
        if legacy_lock.parent.exists() and legacy_lock not in lock_paths:
            lock_paths.append(legacy_lock)
    with ExitStack() as stack:
        for lock_path in lock_paths:
            lock = stack.enter_context(lock_path.open("w", encoding="utf-8"))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log("another warm-model manager is already running")
                return 2
        manager = Manager(args)
        signal.signal(signal.SIGINT, manager.stop)
        signal.signal(signal.SIGTERM, manager.stop)
        try:
            migrate_default_state(args.state)
            manager.run()
        except Exception as error:
            log(f"manager error: {error}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
