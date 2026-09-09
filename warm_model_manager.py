#!/usr/bin/env python3
"""Portable revenue-aware Darkbloom single-model warm loader.

The score combines average network pressure, a fixed mix of 85% input and
15% output token prices, and a model weight. It compares models at the
same assumed token mix without measuring provider throughput or actual payouts.

Every selection requests exactly one warm model. The catalog, network capacity,
and local scan supply the model table. Models absent from the local scan are
automatically ignored; explicit ignores also remain visible for comparison.

The file is intentionally standalone: copy only this script to a Mac running
Darkbloom.  It uses Python's standard library, the installed ``darkbloom``
command, ``~/.darkbloom/daemon-state.json``, and the provider TOML config.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MANAGER_VERSION = "0.1.6"
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
    if not cached_at or now - cached_at >= refresh_seconds:
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
    )


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


def switch_model(
    darkbloom: str,
    model_id: str,
    config_path: Path | None,
    ignored_models: Iterable[str] = (),
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


def switch_forecast(
    current: str | None,
    decision: Decision,
    manager_state: dict[str, Any],
    daemon: LocalDaemonState | None,
    now: float,
    interval_seconds: float,
    confirmations: int,
    min_dwell_seconds: float,
) -> str | None:
    """Describe the earliest conditional switch time at the current cadence."""
    if decision.target is None:
        return None
    if decision.warming:
        # A pending command may have failed or timed out. Its reason explains
        # the state; it does not provide a reliable loading ETA.
        return None

    if current is None:
        if decision.target:
            return (
                f"{display_name(decision.target)} can switch at the next safe action "
                "once Darkbloom is online and idle and any minimum warm time has elapsed (--min-warm-time)"
            )
        return None

    if decision.target == current and decision.challenger is None:
        return None

    contender = decision.challenger or decision.target
    if decision.target == current and decision.challenger_streak <= 0:
        return (
            f"no estimate yet; {display_name(contender)} "
            "does not meet the required percentage improvement after switch cost"
        )

    remaining_checks = 0
    if decision.target == current:
        remaining_checks = max(0, confirmations - decision.challenger_streak)
    confirmation_wait = remaining_checks * interval_seconds

    anchor = dwell_anchor(manager_state, daemon)
    dwell_wait = (
        max(0.0, min_dwell_seconds - (now - anchor)) if anchor > 0 else 0.0
    )
    wait = max(confirmation_wait, dwell_wait)
    if wait > 0 and interval_seconds > 0:
        # Decisions only happen on ticks, so round to the first actual check at
        # or after all confirmation and dwell requirements have cleared.
        wait = math.ceil(wait / interval_seconds) * interval_seconds

    condition = (
        f"if {display_name(contender)} keeps meeting the percentage requirement and the provider is idle"
    )
    if wait <= 0:
        if daemon and daemon.inference_active:
            return f"as soon as the provider becomes idle, {condition}"
        return f"now, {condition}"
    switch_at = now + wait
    return (
        f"about {format_human_duration(wait)} "
        f"(around {format_local_time(switch_at, now)}), {condition}"
    )


def print_report(
    models: list[str],
    samples: dict[str, CapacitySample],
    averages: dict[str, float],
    weights: dict[str, float],
    prices: dict[str, ModelPrice],
    scores: dict[str, float],
    pressure_history: dict[str, list[dict[str, float]]],
    manager_state: dict[str, Any],
    daemon: LocalDaemonState | None,
    current: str | None,
    decision: Decision,
    apply: bool,
    interval_seconds: float,
    history_size: int,
    confirmations: int,
    now: float,
    residency: tuple[float, float] | None,
    forecast: str | None,
    ignored_models: set[str],
    eligible_models: list[str],
    blocked: str | None,
    improvement_percent: float,
) -> None:
    timestamp = datetime.fromtimestamp(now).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )
    mode = "LIVE — changes enabled" if apply else "DRY RUN — no changes enabled"
    model_width = max([25, *(len(display_name(model)) + 2 for model in models)])
    average_label = f"AVG {format_duration(interval_seconds * history_size)}"
    table_header = (
        f"{'MODEL ID':<{model_width}} {'NOW':>6} {average_label:>8} {'N':>3} "
        f"{'IN$/M':>7} {'OUT$/M':>7} {'BLEND$/M':>8} {'WEIGHT':>6} {'SCORE':>7} STATUS"
    )

    print("", flush=True)
    print("=" * len(table_header), flush=True)
    print(
        f"Darkbloom Warm Model Manager ({MANAGER_VERSION})  |  {timestamp}",
        flush=True,
    )
    print(f"Mode:      {mode}", flush=True)
    print(f"Darkbloom: {daemon_status_line(daemon)}", flush=True)
    discovery = manager_state.get("discovery") or {}
    print(f"Discovery: {discovery.get('status', 'unknown')} (local scan; refreshed every check)", flush=True)
    catalog = manager_state.get("catalog") or {}
    catalog_at = catalog.get("fetched_at")
    catalog_age = f"; fetched {format_local_time(catalog_at, now)}" if catalog_at else ""
    catalog_count = len(catalog.get("models") or [])
    catalog_detail = (
        f"{catalog_count} model{'s' if catalog_count != 1 else ''}"
        if catalog.get("status") in {"live", "stale cache"} else "no cached catalog"
    )
    print(f"Catalog:   {catalog.get('status', 'unavailable')}; {catalog_detail}{catalog_age}", flush=True)
    pricing = manager_state.get("pricing_status") or {}
    fetched_at = pricing.get("fetched_at")
    price_age = f"; fetched {format_local_time(fetched_at, now)}" if fetched_at else ""
    print(f"Prices:    {pricing.get('status', 'unknown')}{price_age}", flush=True)
    if current and residency:
        elapsed, warm_since = residency
        print(
            f"Current:   {display_name(current)} warm for "
            f"{format_human_duration(elapsed)} "
            f"(since {format_local_time(warm_since, now)})",
            flush=True,
        )
    print("", flush=True)
    print(table_header, flush=True)
    print("-" * len(table_header), flush=True)
    def number(value: float | None, decimals: int = 3) -> str:
        return f"{value:.{decimals}f}" if value is not None else "N/A"

    snapshot_models = (manager_state.get("last_score_snapshot") or {}).get("models", {})
    for model in models:
        sample = samples.get(model)
        marker = (
            "*"
            if daemon
            and daemon.alive
            and daemon.fresh
            and model in daemon.warm_models
            else " "
        )
        label = f"{marker} {display_name(model)}"
        price = prices.get(model, ModelPrice(None, None))
        sample_count = len(pressure_history.get(model, []))
        status = snapshot_models.get(model, {}).get("status", [])
        print(
            f"{label:<{model_width}} {number(sample.pressure if sample else None):>6} "
            f"{number(averages.get(model)):>8} {sample_count:>3} "
            f"{number(price.input_usd, 4):>7} {number(price.output_usd, 4):>7} "
            f"{number(price.blended_usd, 4):>8} "
            f"{weights.get(model, 1.0):>6.2f} {number(scores.get(model)):>7} "
            + "; ".join(status),
            flush=True,
        )
    if scores:
        print("", flush=True)
        highest = max(scores.values())
        leaders = [model for model in models if scores.get(model) == highest]
        names = [display_name(model) + (
            " [IGNORED]" if model in ignored_models else
            " [AUTO-IGNORED]" if snapshot_models.get(model, {}).get("auto_ignored") else ""
        ) for model in leaders]
        print("Highest raw score: " + " = ".join(names) + f" ({highest:.3f}).", flush=True)
        print("Ranking is before switch cost, required score improvement, consecutive passing checks and minimum warm time.", flush=True)
        print("Ignored and auto-ignored models cannot be loaded.", flush=True)
    print("", flush=True)
    target_changed = current != decision.target
    selection_ready = warm_selection_matches(
        daemon,
        decision.target,
    )
    if decision.target is None:
        action = "WAIT"
    elif decision.warming:
        action = "WARMING"
    elif selection_ready:
        action = "KEEP"
    elif blocked:
        action = "DEFERRED"
    elif target_changed and apply:
        action = "SWITCH"
    elif target_changed:
        action = "WOULD SWITCH"
    else:
        action = "KEEP"
    print(f"Decision:  {action} → {display_name(decision.target)}", flush=True)
    print(f"Switch rule: at least {improvement_percent:g}% score improvement after switch cost (--switch-improvement-percent).", flush=True)
    if decision.challenger and decision.target == current:
        progress = (
            f" — {decision.challenger_streak}/{confirmations} consecutive checks passed (--switch-after-checks)"
            if decision.challenger_streak > 0
            else " — needs a larger score advantage"
        )
        print(
            f"Candidate: {display_name(decision.challenger)}{progress}",
            flush=True,
        )
    if forecast:
        print(f"Earliest switch: {forecast}", flush=True)
    print(f"Reason:    {decision.reason}", flush=True)
    if blocked and decision.target is not None and not selection_ready and not decision.warming:
        print(f"Deferred:  {blocked}", flush=True)
    print(
        f"BLEND$/M = {INPUT_TOKEN_SHARE:.0%} input price + {OUTPUT_TOKEN_SHARE:.0%} output price per million total tokens.",
        flush=True,
    )
    print("Score = average pressure × BLEND$/M × weight (ranking estimate).", flush=True)
    print(
        f"N is the number of retained samples (max {history_size}, --average-samples); samples expire "
        f"after {format_duration(interval_seconds * history_size)}.",
        flush=True,
    )
    print("* marks currently warm models; NOW is the current network snapshot.", flush=True)


class Manager:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.models: list[str] = []
        self.ignored = set(args.ignore_model)
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update(args.weight)
        self.state_path: Path = args.state
        self.stop_requested = False

    def stop(self, _signum: int, _frame: Any) -> None:
        self.stop_requested = True

    def discover_models(self, manager_state: dict[str, Any], now: float) -> set[str]:
        previous = manager_state.get("discovery") or {}
        try:
            local = local_model_ids(self.args.darkbloom, self.args.config)
        except Exception as error:
            # Retain the last inventory for display only. An unsuccessful scan
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

    def iteration(self) -> None:
        now = time.time()
        manager_state = {
            key: value for key, value in read_json(self.state_path).items()
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
        catalog = self.discover_catalog(manager_state, now)
        local = self.discover_models(manager_state, now)
        discovery_available = manager_state["discovery"]["status"] == "live"
        try:
            samples = fetch_capacity(self.args.base_url)
        except Exception as error:
            samples = {}
            log(f"network capacity unavailable: {error}")

        # The catalog and capacity feed expand visibility only. They cannot
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
        )
        # Catalog and price requests can be slow. Check freshness against the
        # time after those reads, rather than the start of the iteration.
        daemon = read_daemon_state(self.args.daemon_state, now=time.time())
        # Retained history remains visible during a feed gap, but is not a
        # fresh score and must not drive selection.
        all_scores = revenue_scores(
            {model: average for model, average in averages.items() if model in samples},
            self.weights, prices,
        )
        eligible_models = [model for model in local_eligible if model in all_scores]
        scores = {
            model: score
            for model, score in all_scores.items()
            if model in eligible_models
        }
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
        forecast = switch_forecast(
            current,
            decision,
            manager_state,
            daemon,
            now,
            self.args.check_every,
            self.args.switch_after_checks,
            self.args.min_warm_time,
        )
        blocked = switch_block_reason(
            decision, eligible_models, manager_state, daemon, now, self.args.min_warm_time,
        )

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
            forecast,
            self.ignored,
            eligible_models,
            blocked,
            self.args.switch_improvement_percent,
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
            manager_state["pending_switch"] = {
                "target": decision.target,
                "warm_models": [decision.target],
                "command_at": time.time(),
            }
            write_json_atomic(self.state_path, manager_state)
            log("switching the launchd provider to " + decision.target)
            try:
                switch_model(self.args.darkbloom, decision.target, self.args.config, self.ignored)
            except Exception as error:
                manager_state["pending_switch"]["command_error"] = str(error)
                write_json_atomic(self.state_path, manager_state)
                raise
            log(f"switch command accepted: {decision.target}; waiting for Darkbloom to report the model warm")

        if selection_matches:
            manager_state["active_target"] = decision.target
        write_json_atomic(self.state_path, manager_state)

    def run(self) -> None:
        if self.args.mode == "once":
            self.iteration()
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
        while not self.stop_requested:
            started = time.monotonic()
            try:
                self.iteration()
            except Exception as error:
                log(f"manager error: {error}")
            remaining = max(0.0, self.args.check_every - (time.monotonic() - started))
            if not self.stop_requested:
                log(
                    f"manager is running; next check in {format_duration(remaining)} (--check-every) "
                    "(Ctrl-C to stop)"
                )
            deadline = time.monotonic() + remaining
            while not self.stop_requested and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))


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
    parser.add_argument("mode", choices=("once", "run"), nargs="?", default="once")
    parser.add_argument("--apply", action="store_true", help="actually switch models; default is dry-run")
    parser.add_argument("--model", action="append", metavar="MODEL", help="restrict loading candidates; repeat in tie-break order; other catalog rows stay visible (default: all locally discovered models)")
    parser.add_argument(
        "--ignore-model", "--ignore", action="append", default=[], metavar="MODEL_ID",
        help="score and display this model but never select or load it; repeat as needed",
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
