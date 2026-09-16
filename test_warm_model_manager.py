import fcntl
import io
import json
import subprocess
import sys
import tempfile
import unittest

import warm_model_manager as manager_module
from pathlib import Path
from unittest.mock import ANY, patch
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from urllib.error import HTTPError, URLError

from warm_model_manager import (
    CapacitySample,
    DEFAULT_PROVIDER_CONFIG_PATH,
    DEFAULT_STATE_PATH,
    DEFAULT_WEIGHTS,
    Decision,
    LocalDaemonState,
    MANAGER_VERSION,
    ModelPrice,
    PRICING_CACHE_SCHEMA,
    STATE_SCHEMA,
    Manager,
    PROD_PROBE_URL,
    ProbeRedirectHandler,
    build_score_snapshot,
    build_parser,
    cached_model_prices,
    catalog_model_ids,
    challenger_state_keys,
    choose_scored_target,
    current_warm_model,
    daemon_status_line,
    dwell_anchor,
    ensure_pressure_cadence,
    ensure_preload_sync_policy,
    ensure_scoring_policy,
    ensure_switch_policy,
    eligible_local_targets,
    fetch_model_prices,
    format_duration,
    format_human_duration,
    local_model_ids,
    local_endpoint_start_flags,
    main,
    migrate_default_state,
    pressure_samples,
    probe_credentials,
    probe_debug_text,
    probe_response_details,
    report_timeline,
    print_timeline,
    score_distance,
    switch_ready_at,
    read_daemon_state,
    reconcile_pending_switch,
    render_preload_model_config,
    revenue_scores,
    send_probe,
    switch_model,
    switch_block_reason,
    track_current_residency,
    update_pressure_history,
    warm_selection_matches,
)


MODELS = ["q35", "q36", "gemma", "gpt"]
ORIGINAL_STOP_PROVIDER = manager_module.stop_provider
ORIGINAL_PROVIDER_SERVICES = manager_module.provider_services
ORIGINAL_SEND_PROBE = manager_module.send_probe


def setUpModule() -> None:
    # Every test must explicitly fake external I/O. An accidental real provider
    # command or network connection is a test failure, including on developer Macs.
    for target in ("subprocess.run", "socket.create_connection"):
        guard = patch(target, side_effect=AssertionError("external I/O forbidden in offline tests"))
        guard.start()
        unittest.addModuleCleanup(guard.stop)


def timeline_lines(state, now, enabled=True, apply=True):
    events, notes = report_timeline(state, None, None, Decision(None, "test"), now,
                                   now + 60, 60, 3, 2700, enabled, False, apply)
    with redirect_stdout(io.StringIO()) as output:
        print_timeline(events, notes, now)
    return output.getvalue().splitlines()


class CapacityTests(unittest.TestCase):
    def test_manager_release_version(self) -> None:
        self.assertEqual(MANAGER_VERSION, "0.1.11")

    def test_default_model_weights(self) -> None:
        self.assertEqual(DEFAULT_WEIGHTS["qwen3.5-35b-a3b"], 1.25)
        self.assertEqual(
            DEFAULT_WEIGHTS["qwen3.6-35b-a3b-vl-mtp-mxfp8"],
            1.20,
        )
        self.assertEqual(DEFAULT_WEIGHTS["gemma-4-26b-qat-4bit"], 1.05)
        self.assertEqual(DEFAULT_WEIGHTS["gpt-oss-20b"], 1.00)

    def test_public_capacity_shape_and_legacy_tracker_shape(self) -> None:
        samples = pressure_samples(
            [
                {"id": "q35", "warm_providers": 50, "active_requests": 5},
                {"model_id": "gpt", "loaded": 200, "in_progress": 100},
            ]
        )
        self.assertAlmostEqual(samples["q35"].pressure, 0.1)
        self.assertAlmostEqual(samples["gpt"].pressure, 0.5)

    def test_three_observation_rolling_average(self) -> None:
        samples = {
            "q35": CapacitySample("q35", 10, 4, 0.4),
        }
        history, averages = update_pressure_history(
            MODELS,
            samples,
            {
                "q35": [
                    {"at": 100, "pressure": 0.1},
                    {"at": 160, "pressure": 0.2},
                    {"at": 220, "pressure": 0.3},
                ]
            },
            history_size=3,
            now=280,
            window_seconds=180,
        )
        self.assertEqual(
            history["q35"],
            [
                {"at": 160.0, "pressure": 0.2},
                {"at": 220.0, "pressure": 0.3},
                {"at": 280, "pressure": 0.4},
            ],
        )
        self.assertAlmostEqual(averages["q35"], 0.3)

    def test_stale_and_untimestamped_samples_expire(self) -> None:
        samples = {"q35": CapacitySample("q35", 10, 8, 0.8)}
        history, averages = update_pressure_history(
            MODELS,
            samples,
            {
                "q35": [
                    0.99,
                    {"at": 100, "pressure": 0.7},
                ]
            },
            history_size=3,
            now=1000,
            window_seconds=180,
        )
        self.assertEqual(history["q35"], [{"at": 1000, "pressure": 0.8}])
        self.assertEqual(averages["q35"], 0.8)

    def test_cadence_change_resets_history_and_confirmation_state(self) -> None:
        state = {
            "pressure_cadence": {
                "interval_seconds": 60.0,
                "history_size": 3,
                "window_seconds": 180.0,
            },
            "pressure_history": {"q35": [{"at": 100, "pressure": 0.5}]},
            "live_challenger_model": "q35",
            "live_challenger_streak": 1,
        }
        changed = ensure_pressure_cadence(state, interval_seconds=900, history_size=3)
        self.assertTrue(changed)
        self.assertEqual(state["pressure_history"], {})
        self.assertNotIn("live_challenger_model", state)
        self.assertEqual(state["pressure_cadence"]["window_seconds"], 2700.0)

        changed_again = ensure_pressure_cadence(
            state,
            interval_seconds=900,
            history_size=3,
        )
        self.assertFalse(changed_again)

    def test_discards_only_legacy_pending_preload_switch_once(self) -> None:
        state = {
            "pending_switch": {"target": "gpt", "command_at": 1000},
            "active_target": "q36",
        }
        self.assertTrue(ensure_preload_sync_policy(state))
        self.assertNotIn("pending_switch", state)
        self.assertEqual(state["active_target"], "q36")
        self.assertEqual(state["preload_sync_schema"], 1)
        self.assertFalse(ensure_preload_sync_policy(state))

    def test_revenue_score_combines_pressure_price_and_weight(self) -> None:
        scores = revenue_scores(
            {"q35": 0.5, "gpt": 0.5, "q9": 2.0},
            {"q35": 1.25, "gpt": 1.0},
            {"q35": ModelPrice(0.08, 0.75), "gpt": ModelPrice(0.02, 0.10), "q9": ModelPrice(0.08, 0.13)},
        )
        self.assertAlmostEqual(scores["q35"], 0.1128125)
        self.assertAlmostEqual(scores["gpt"], 0.016)
        self.assertAlmostEqual(scores["q9"], 0.175)

    def test_score_snapshot_records_every_input_to_the_decision_score(self) -> None:
        samples = {"q35": CapacitySample("q35", 10, 4, 0.4)}
        history = {
            "q35": [
                {"at": 100.0, "pressure": 0.2},
                {"at": 160.0, "pressure": 0.4},
            ]
        }
        snapshot = build_score_snapshot(
            ["q35"],
            samples,
            {"q35": 0.3},
            history,
            {"q35": ModelPrice(0.08, 0.75)},
            {"q35": 1.15},
            {"q35": 0.0622725},
            observed_at=160.0,
            window_seconds=180.0,
        )
        self.assertEqual(snapshot["observed_at"], 160.0)
        self.assertEqual(snapshot["window_seconds"], 180.0)
        self.assertEqual(snapshot["input_token_share"], 0.85)
        self.assertEqual(snapshot["output_token_share"], 0.15)
        values = snapshot["models"]["q35"]
        self.assertEqual(values["now_pressure"], 0.4)
        self.assertEqual(values["average_pressure"], 0.3)
        self.assertEqual(values["retained_samples"], 2)
        self.assertEqual(values["input_usd_per_million"], 0.08)
        self.assertEqual(values["output_usd_per_million"], 0.75)
        self.assertAlmostEqual(values["blended_usd_per_million"], 0.1805)
        self.assertEqual(values["weight"], 1.15)
        self.assertEqual(values["score"], 0.0622725)

    def test_public_micro_usd_prices_convert_to_usd_per_million(self) -> None:
        payload = {
            "fallback_input_price": 50_000,
            "fallback_output_price": 200_000,
            "prices": [
                {"model": "q35", "input_price": 80_000, "output_price": 750_000},
                {"model": "gpt", "input_price": 20_000, "output_price": 100_000},
            ],
        }
        with patch("warm_model_manager.get_json", return_value=payload):
            prices, fallback = fetch_model_prices("https://example.test/pricing")
        self.assertEqual(prices, {"q35": ModelPrice(0.08, 0.75), "gpt": ModelPrice(0.02, 0.10)})
        self.assertEqual(fallback, ModelPrice(0.05, 0.20))

    def test_cached_prices_cover_missing_model_with_public_fallback(self) -> None:
        state = {}
        with patch(
            "warm_model_manager.fetch_model_prices",
            return_value=({"q35": ModelPrice(0.08, 0.75)}, ModelPrice(0.05, 0.20)),
        ):
            prices = cached_model_prices(
                state,
                ["q35", "new-model"],
                "https://example.test/pricing",
                now=1000,
                refresh_seconds=900,
            )
        self.assertEqual(prices, {"q35": ModelPrice(0.08, 0.75), "new-model": ModelPrice(0.05, 0.20)})

    def test_supported_model_blends_use_the_same_fixed_token_mix(self) -> None:
        cases = [
            ("qwen3.5-35b-a3b", 0.08, 0.75, 0.1805, 0.225625),
            ("qwen3.6-35b-a3b-vl-mtp-mxfp8", 0.05, 0.70, 0.1475, 0.177),
            ("gemma-4-26b-qat-4bit", 0.042, 0.22, 0.0687, 0.072135),
            ("gemma-4-26b", 0.042, 0.22, 0.0687, 0.0687),
            ("gemma-4-26b-8bit", 0.042, 0.22, 0.0687, 0.0687),
            ("gpt-oss-20b", 0.02, 0.10, 0.032, 0.032),
            ("Qwen3.5-9B", 0.08, 0.13, 0.0875, 0.0875),
            ("qwen3-vl-30b-a3b-instruct", 0.09, 0.40, 0.1365, 0.1365),
            ("EigenLabs/Qwen3.8-27B-4bit-mtp", 0.15, 2.0, 0.4275, 0.4275),
        ]
        for model, input_price, output_price, blend, score in cases:
            with self.subTest(model=model):
                price = ModelPrice(input_price, output_price)
                self.assertAlmostEqual(price.blended_usd, blend)
                self.assertAlmostEqual(revenue_scores({model: 1}, DEFAULT_WEIGHTS, {model: price})[model], score)

    def test_missing_or_invalid_price_components_cannot_be_scored(self) -> None:
        for field in ("input_price", "output_price"):
            for invalid in (None, -1, "bad", float("nan"), float("inf"), True):
                with self.subTest(field=field, value=invalid):
                    payload = {
                        "prices": [{"model": "partial", "input_price": 80_000, "output_price": 130_000, field: invalid}],
                        "fallback_input_price": 50_000, "fallback_output_price": 200_000,
                    }
                    with patch("warm_model_manager.get_json", return_value=payload):
                        prices = cached_model_prices({}, ["partial", "unlisted"], "https://example.test/pricing", 1000, 900)
                    self.assertIsNone(prices["partial"].blended_usd)
                    self.assertEqual(set(revenue_scores({"partial": 100, "unlisted": 1}, {}, prices)), {"unlisted"})

    def test_explicit_zero_component_is_valid_but_all_zero_is_not_scored(self) -> None:
        payload = {"prices": [
            {"model": "free-input", "input_price": 0, "output_price": 200_000},
            {"model": "free-output", "input_price": 100_000, "output_price": 0},
            {"model": "free", "input_price": 0, "output_price": 0},
            {"model": "missing-input", "output_price": 200_000},
        ]}
        with patch("warm_model_manager.get_json", return_value=payload):
            prices, fallback = fetch_model_prices("https://example.test/pricing")
        scores = revenue_scores(dict.fromkeys(prices, 1), {}, prices)
        self.assertEqual(set(scores), {"free-input", "free-output"})
        self.assertAlmostEqual(scores["free-input"], 0.03)
        self.assertAlmostEqual(scores["free-output"], 0.085)
        self.assertIsNone(fallback.blended_usd)

    def test_both_prices_and_fallback_survive_cache_roundtrip_and_outage(self) -> None:
        state = {}
        url = "https://example.test/pricing"
        expected = {"q9": ModelPrice(0.08, 0.13), "new": ModelPrice(0.05, 0.20)}
        with patch("warm_model_manager.fetch_model_prices", return_value=({"q9": expected["q9"]}, expected["new"])):
            self.assertEqual(cached_model_prices(state, list(expected), url, 1000, 900), expected)
        state = json.loads(json.dumps(state))
        self.assertEqual(state["pricing_cache"]["schema"], PRICING_CACHE_SCHEMA)
        with patch("warm_model_manager.fetch_model_prices", side_effect=RuntimeError("offline")) as fetch:
            self.assertEqual(cached_model_prices(state, list(expected), url, 1060, 900), expected)
            fetch.assert_not_called()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cached_model_prices(state, list(expected), url, 2000, 900), expected)
            fetch.assert_called_once()
        self.assertEqual(state["pricing_status"]["status"], "stale cache")
        self.assertEqual(state["pricing_status"]["fetched_at"], 1000)

    def test_output_only_cache_requires_fresh_input_prices_even_during_outage(self) -> None:
        url = "https://example.test/pricing"
        for offline in (False, True):
            with self.subTest(offline=offline):
                state = {"pricing_cache": {"source": url, "fetched_at": 1000,
                         "prices": {"q9": 0.13}, "fallback_output_usd": 0.2}}
                with patch("warm_model_manager.fetch_model_prices", return_value=({"q9": ModelPrice(0.08, 0.13)}, ModelPrice(0.05, 0.2)),
                           side_effect=RuntimeError("offline") if offline else None) as fetch, redirect_stdout(io.StringIO()):
                    prices = cached_model_prices(state, ["q9"], url, 1060, 900)
                fetch.assert_called_once_with(url)
                if offline:
                    self.assertEqual(revenue_scores({"q9": 10}, {}, prices), {})
                    self.assertEqual(state["pricing_status"]["status"], "unavailable")
                    self.assertIsNone(state["pricing_status"]["fetched_at"])
                    self.assertNotIn("pricing_cache", state)
                else:
                    self.assertAlmostEqual(prices["q9"].blended_usd, 0.0875)
                    self.assertEqual(state["pricing_cache"]["schema"], PRICING_CACHE_SCHEMA)

    def test_price_cache_does_not_cross_pricing_sources(self) -> None:
        state = {}
        with patch("warm_model_manager.fetch_model_prices", return_value=({}, ModelPrice(0.05, 0.2))):
            cached_model_prices(state, ["new"], "https://first.test/pricing", 1000, 900)
        with patch("warm_model_manager.fetch_model_prices", side_effect=RuntimeError("offline")), redirect_stdout(io.StringIO()):
            prices = cached_model_prices(state, ["new"], "https://second.test/pricing", 1060, 900)
        self.assertIsNone(prices["new"].blended_usd)
        self.assertEqual(state["pricing_status"]["status"], "unavailable")

    def test_formula_change_resets_counts_once_without_erasing_warmup_or_history(self) -> None:
        preserved = {
            "pressure_history": {"good": [{"at": 1000, "pressure": 1}]},
            "pending_switch": {"target": "good", "warm_models": ["good"], "command_at": 1000},
            "last_switch_at": 500, "active_target": "good",
        }
        state = {**preserved, "live_challenger_model": "next", "live_challenger_streak": 2,
                 "dry_challenger_model": "next", "dry_challenger_streak": 2}
        self.assertTrue(ensure_scoring_policy(state))
        self.assertTrue(all(state[key] == value for key, value in preserved.items()))
        self.assertFalse(any("challenger" in key for key in state))
        state["live_challenger_streak"] = 1
        self.assertFalse(ensure_scoring_policy(state))
        self.assertEqual(state["live_challenger_streak"], 1)

    def test_switch_rule_resets_counts_only_when_percentage_or_cost_changes(self) -> None:
        preserved = {
            "pressure_history": {"good": [{"at": 1000, "pressure": 1}]},
            "pricing_cache": {"fetched_at": 1000},
            "pending_switch": {"target": "good", "warm_models": ["good"], "command_at": 1000},
            "last_switch_at": 500,
        }
        state = dict(preserved)
        for percent, cost, horizon in ((25, 300, 3600), (1, 300, 3600), (1, 0, 3600), (1, 0, 7200)):
            state.update(live_challenger_model="next", live_challenger_streak=2,
                         dry_challenger_model="next", dry_challenger_streak=2)
            self.assertTrue(ensure_switch_policy(state, percent, cost, horizon))
            self.assertFalse(any("challenger" in key for key in state))
            self.assertTrue(all(state[key] == value for key, value in preserved.items()))
            state["live_challenger_streak"] = 1
            self.assertFalse(ensure_switch_policy(state, percent, cost, horizon))
            self.assertEqual(state["live_challenger_streak"], 1)

    def test_duration_labels_match_test_and_production_intervals(self) -> None:
        self.assertEqual(format_duration(59.2), "60s")
        self.assertEqual(format_duration(3 * 60), "3m")
        self.assertEqual(format_duration(3 * 900), "45m")

    def test_human_duration_is_readable(self) -> None:
        self.assertEqual(format_human_duration(45), "45 seconds")
        self.assertEqual(format_human_duration(60), "1 minute")
        self.assertEqual(format_human_duration(3_660), "1 hour 1 minute")
        self.assertEqual(format_human_duration(90_000), "1 day 1 hour")

    def test_dry_run_and_live_confirmations_use_different_state(self) -> None:
        self.assertEqual(
            challenger_state_keys(False),
            ("dry_challenger_model", "dry_challenger_streak"),
        )
        self.assertEqual(
            challenger_state_keys(True),
            ("live_challenger_model", "live_challenger_streak"),
        )


class ScoreDecisionTests(unittest.TestCase):
    def test_percentage_threshold_is_independent_of_score_scale(self) -> None:
        for scale in (0.000001, 0.005, 0.01, 0.08, 1.0, 1000.0):
            for percent in (1.0, 25.0, 200.0):
                for gain, passes in ((percent - 0.001, False), (percent, True), (percent + 1, True)):
                    with self.subTest(scale=scale, percent=percent, gain=gain):
                        decision = choose_scored_target(
                            ["current", "candidate"],
                            {"current": scale, "candidate": scale * (1 + gain / 100)},
                            "current", None, 0, improvement_percent=percent,
                            confirmations=1, switch_cost_seconds=0, decision_horizon_seconds=3600,
                        )
                        self.assertEqual(decision.target, "candidate" if passes else "current")
                        self.assertIn(f"{percent:g}% improvement requirement", decision.reason)

    def test_percentage_requirement_is_applied_after_switch_cost(self) -> None:
        for scale in (0.000001, 0.01, 10.0):
            for raw, passes in ((scale * 1.25, False), (scale * 1.25 / (11 / 12), True)):
                with self.subTest(scale=scale, raw=raw):
                    decision = choose_scored_target(
                        ["current", "candidate"], {"current": scale, "candidate": raw},
                        "current", None, 0, improvement_percent=25,
                        confirmations=1, switch_cost_seconds=300, decision_horizon_seconds=3600,
                    )
                    self.assertEqual(decision.target, "candidate" if passes else "current")
                    self.assertIn("after switch cost", decision.reason)

    def test_equal_scores_keep_current_even_when_an_earlier_model_wins_tie_order(self) -> None:
        for score in (0.0, 0.01, 1.0):
            for percent in (0, 25):
                with self.subTest(score=score, percent=percent):
                    decision = choose_scored_target(
                        ["candidate", "current"], {"candidate": score, "current": score},
                        "current", "candidate", 2, improvement_percent=percent,
                        confirmations=3, switch_cost_seconds=0, decision_horizon_seconds=3600,
                    )
                    self.assertEqual(decision.target, "current")
                    self.assertIsNone(decision.challenger)
                    self.assertEqual(decision.challenger_streak, 0)

    def test_positive_score_can_replace_zero_only_after_consecutive_checks(self) -> None:
        previous, streak = None, 0
        for check in range(1, 4):
            decision = choose_scored_target(
                ["current", "candidate"], {"current": 0, "candidate": 0.000001},
                "current", previous, streak, improvement_percent=25,
                confirmations=3, switch_cost_seconds=300, decision_horizon_seconds=3600,
            )
            self.assertEqual(decision.target, "candidate" if check == 3 else "current")
            self.assertIn("positive score above a zero current score", decision.reason)
            self.assertNotIn("inf%", decision.reason)
            previous, streak = decision.challenger, decision.challenger_streak

    def test_percentage_failure_breaks_consecutive_streak(self) -> None:
        decision = choose_scored_target(
            ["current", "candidate"], {"current": 0.01, "candidate": 0.0124},
            "current", "candidate", 2, improvement_percent=25,
            confirmations=3, switch_cost_seconds=0, decision_horizon_seconds=3600,
        )
        self.assertEqual(decision.challenger_streak, 0)
        next_check = choose_scored_target(
            ["current", "candidate"], {"current": 0.01, "candidate": 0.013},
            "current", decision.challenger, decision.challenger_streak, improvement_percent=25,
            confirmations=3, switch_cost_seconds=0, decision_horizon_seconds=3600,
        )
        self.assertEqual(next_check.target, "current")
        self.assertEqual(next_check.challenger_streak, 1)

    def test_initial_choice_uses_highest_blended_score_not_pressure(self) -> None:
        pressures = {"q35": 0.101, "q36": 0.054, "gemma": 0.384, "gpt": 0.629}
        scores = revenue_scores(
            pressures,
            {"q35": 1.15, "q36": 1.10, "gemma": 1.05, "gpt": 1.0},
            {"q35": ModelPrice(0.08, 0.75), "q36": ModelPrice(0.05, 0.70),
             "gemma": ModelPrice(0.042, 0.22), "gpt": ModelPrice(0.02, 0.10)},
        )
        decision = choose_scored_target(
            MODELS,
            scores,
            None,
            None,
            0,
            improvement_percent=25,
            confirmations=2,
            switch_cost_seconds=300,
            decision_horizon_seconds=3600,
        )
        self.assertEqual(decision.target, "gemma")

    def test_challenger_requires_two_consecutive_checks(self) -> None:
        first = choose_scored_target(
            MODELS,
            {"q35": 0.1, "q36": 0.2, "gemma": 0.5, "gpt": 0.8},
            "gemma",
            None,
            0,
            improvement_percent=25,
            confirmations=2,
            switch_cost_seconds=300,
            decision_horizon_seconds=3600,
        )
        self.assertEqual(first.target, "gemma")
        self.assertEqual(first.challenger, "gpt")
        self.assertEqual(first.challenger_streak, 1)

        second = choose_scored_target(
            MODELS,
            {"q35": 0.1, "q36": 0.2, "gemma": 0.5, "gpt": 0.8},
            "gemma",
            first.challenger,
            first.challenger_streak,
            improvement_percent=25,
            confirmations=2,
            switch_cost_seconds=300,
            decision_horizon_seconds=3600,
        )
        self.assertEqual(second.target, "gpt")
        self.assertEqual(second.challenger_streak, 2)

    def test_switch_cost_can_block_a_marginal_challenger(self) -> None:
        decision = choose_scored_target(
            MODELS,
            {"q35": 0.1, "q36": 0.2, "gemma": 0.5, "gpt": 0.65},
            "gemma",
            None,
            0,
            improvement_percent=25,
            confirmations=1,
            switch_cost_seconds=300,
            decision_horizon_seconds=3600,
        )
        self.assertEqual(decision.target, "gemma")
        self.assertIn("does not meet the 25% improvement requirement", decision.reason)
        self.assertEqual(decision.challenger, "gpt")
        self.assertEqual(decision.challenger_streak, 0)


class RuntimeForecastTests(unittest.TestCase):
    @staticmethod
    def daemon(
        model: str = "gemma",
        *,
        pid: int = 7,
        started_at: float = 1_000,
        active: bool = False,
    ) -> LocalDaemonState:
        return LocalDaemonState(
            current_model=model,
            warm_models=(model,),
            inference_active=active,
            pid=pid,
            started_at=started_at,
            fresh=True,
            alive=True,
        )

    def test_current_residency_uses_confirmed_warm_time_and_persists(self) -> None:
        state = {
            "active_target": "gemma",
            "last_switch_at": 1_200,
        }
        daemon = self.daemon(started_at=1_000)
        first = track_current_residency(
            state, daemon, "gemma", now=1_500
        )
        self.assertEqual(first, (300, 1_200))

        second = track_current_residency(
            state, daemon, "gemma", now=1_560
        )
        self.assertEqual(second, (360, 1_200))

    def test_current_residency_resets_when_daemon_changes(self) -> None:
        state = {}
        first = track_current_residency(
            state,
            self.daemon(pid=7, started_at=1_000),
            "gemma",
            now=1_500,
        )
        self.assertEqual(first, (500, 1_000))

        restarted = track_current_residency(
            state,
            self.daemon(pid=8, started_at=1_490),
            "gemma",
            now=1_550,
        )
        self.assertEqual(restarted, (60, 1_490))

    def test_no_warm_model_bypasses_minimum_dwell_for_recovery(self) -> None:
        daemon = LocalDaemonState(
            current_model=None,
            warm_models=(),
            inference_active=False,
            pid=11,
            started_at=1_990,
            fresh=True,
            alive=True,
        )
        self.assertEqual(
            dwell_anchor({"last_switch_at": 1_950}, daemon),
            0.0,
        )

    def test_new_daemon_start_preserves_dwell_after_an_older_saved_switch(self) -> None:
        daemon = self.daemon(started_at=9_900)
        state = {"last_switch_at": 1_000}
        self.assertEqual(dwell_anchor(state, daemon), 9_900)
        self.assertIn("1700 seconds remaining", switch_block_reason(
            Decision("q36", "confirmed"), ["q36"], state, daemon, 10_000, 1_800,
        ))
        self.assertIsNone(switch_block_reason(
            Decision("q36", "confirmed"), ["q36"], state, daemon, 11_700, 1_800,
        ))
        self.assertEqual(dwell_anchor({"last_switch_at": 10_100}, daemon), 10_100)

    def test_switch_eta_waits_for_confirmations_and_dwell(self) -> None:
        at = switch_ready_at("gemma", Decision("gemma", "passing", "q36", 1),
                             {"last_switch_at": 1500}, self.daemon(started_at=1000),
                             1600, 1660, 60, 3, 1800)
        self.assertEqual(at, 3340)  # First check at/after both requirements.


    def test_margin_failing_contender_has_no_false_countdown(self) -> None:
        at = switch_ready_at("gemma", Decision("gemma", "below margin", "q36", 0),
                             {"last_switch_at": 1000}, self.daemon(), 2000, 2060, 60, 3, 1800)
        self.assertIsNone(at)



class DaemonStateTests(unittest.TestCase):
    def test_reads_daemon_warmth_and_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daemon.json"
            path.write_text(
                json.dumps(
                    {
                        "written_at": 995,
                        "started_at": 900,
                        "pid": 7,
                        "current_model": "gpt",
                        "warm_models": ["gpt"],
                        "inference_active": True,
                    }
                ),
                encoding="utf-8",
            )
            state = read_daemon_state(path, now=1000)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertTrue(state.fresh)
        self.assertTrue(state.inference_active)

    def test_stale_daemon_is_not_treated_as_currently_warm(self) -> None:
        daemon = LocalDaemonState(
            current_model="gpt",
            warm_models=("gpt",),
            inference_active=False,
            pid=7,
            started_at=900,
            fresh=False,
            alive=True,
        )
        self.assertFalse(warm_selection_matches(daemon, "gpt"))
        self.assertIn("STALE", daemon_status_line(daemon))


class PendingSwitchTests(unittest.TestCase):
    @staticmethod
    def daemon(
        warm_models: tuple[str, ...] = (),
        load_error_model: str | None = None,
        load_error_message: str | None = None,
        load_error_at: float = 0,
    ) -> LocalDaemonState:
        return LocalDaemonState(
            current_model=warm_models[0] if warm_models else None,
            warm_models=warm_models,
            inference_active=False,
            pid=10,
            started_at=100,
            fresh=True,
            alive=True,
            load_error_model=load_error_model,
            load_error_message=load_error_message,
            load_error_at=load_error_at,
        )

    def test_pending_switch_waits_past_short_test_dwell_without_restart(self) -> None:
        state = {"pending_switch": {"target": "q35", "command_at": 1000}}
        decision = reconcile_pending_switch(
            state,
            self.daemon(),
            now=1120,
            warmup_timeout=180,
        )
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertTrue(decision.warming)
        self.assertEqual(decision.target, "q35")
        self.assertIn("60s remaining", decision.reason)
        self.assertIn("pending_switch", state)

    def test_pending_switch_clears_only_after_target_is_warm(self) -> None:
        state = {"pending_switch": {"target": "q35", "command_at": 1000}}
        decision = reconcile_pending_switch(
            state,
            self.daemon(("q35",)),
            now=1125,
            warmup_timeout=180,
        )
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertFalse(decision.warming)
        self.assertNotIn("pending_switch", state)
        self.assertEqual(state["last_switch_at"], 1125)

    def test_timed_out_or_failed_load_is_blocked_instead_of_restarted(self) -> None:
        timed_out_state = {
            "pending_switch": {"target": "q35", "command_at": 1000}
        }
        timed_out = reconcile_pending_switch(
            timed_out_state,
            self.daemon(),
            now=1181,
            warmup_timeout=180,
        )
        assert timed_out is not None
        self.assertTrue(timed_out.warming)
        self.assertIn("automatic restart is blocked", timed_out.reason)

        failed = reconcile_pending_switch(
            {"pending_switch": {"target": "q35", "command_at": 1000}},
            self.daemon((), "q35", "weights are corrupt", 1005),
            now=1010,
            warmup_timeout=180,
        )
        assert failed is not None
        self.assertTrue(failed.warming)
        self.assertIn("weights are corrupt", failed.reason)

        old_error = reconcile_pending_switch(
            {"pending_switch": {"target": "q35", "command_at": 1000}},
            self.daemon((), "q35", "old failure", 900),
            now=1010,
            warmup_timeout=180,
        )
        assert old_error is not None
        self.assertNotIn("old failure", old_error.reason)
        self.assertIn("finish loading", old_error.reason)


IGNORED = "EigenLabs/Qwen3.8-27B-4bit-mtp"


class SingleModelLoadingTests(unittest.TestCase):
    def test_current_model_requires_exactly_one_eligible_warm_model(self) -> None:
        for warm, expected in (((), None), (("good",), "good"), (("other",), None),
                               (("good", "other"), None), (("good", "good"), None)):
            with self.subTest(warm=warm):
                daemon = PendingSwitchTests.daemon(warm)
                self.assertEqual(current_warm_model(daemon, ["good"]), expected)
                self.assertEqual(warm_selection_matches(daemon, "good"), expected == "good")

    def test_warmup_waits_until_only_the_requested_model_is_warm(self) -> None:
        state = {"pending_switch": {"target": "gemma", "warm_models": ["gemma"], "command_at": 1000}}
        waiting = reconcile_pending_switch(
            state, PendingSwitchTests.daemon(("gemma", "gpt")), 1060, 180,
        )
        self.assertTrue(waiting.warming)
        self.assertIn("pending_switch", state)
        confirmed = reconcile_pending_switch(
            state, PendingSwitchTests.daemon(("gemma",)), 1120, 180,
        )
        self.assertFalse(confirmed.warming)
        self.assertNotIn("pending_switch", state)
        self.assertEqual(state["last_switch_at"], 1120)

    def test_launch_sets_one_model_and_replaces_stale_preloads_before_start(self) -> None:
        for model in ("gemma-4-26b-qat-4bit", "gpt-oss-20b", "new-model"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "provider.toml"
                config.write_text(
                    '[provider]\nname = "test"\n\n[backend]\n'
                    'preload_models = [\n "old", "other", # stale\n]\n'
                    'startup_preload = true\n\n[network]\ncoordinator = "example"\n'
                )
                def fake_start(command, **kwargs):
                    text = config.read_text()
                    self.assertIn("preload_models = " + json.dumps([model]), text)
                    self.assertNotIn('"old"', text)
                    self.assertIn('coordinator = "example"', text)
                    self.assertIn('startup_preload = true', text)
                    self.assertEqual(command, [
                        "darkbloom", "start", "--config", str(config),
                        "--model", model, "--idle-timeout", "0",
                    ])
                    return subprocess.CompletedProcess(command, 0, "", "")
                with patch("warm_model_manager.subprocess.run", side_effect=fake_start) as run:
                    switch_model("darkbloom", model, config)
                run.assert_called_once()

    def test_default_config_receives_exactly_one_preload(self) -> None:
        with patch("warm_model_manager.synchronize_preload_model") as sync, \
                patch("warm_model_manager.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            switch_model("darkbloom", "gpt-oss-20b", None)
        sync.assert_called_once_with(DEFAULT_PROVIDER_CONFIG_PATH, "gpt-oss-20b")
        self.assertEqual(run.call_args.args[0], [
            "darkbloom", "start", "--model", "gpt-oss-20b", "--idle-timeout", "0",
        ])

    def test_switch_passes_local_endpoint_options_without_credentials(self):
        with patch("warm_model_manager.synchronize_preload_model") as sync, \
                patch("warm_model_manager.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            switch_model("darkbloom", "good", None,
                         local_flags=["--local-endpoint", "--port", "9000", "--bind", "::1"])
        self.assertEqual(run.call_args.args[0], [
            "darkbloom", "start", "--model", "good", "--idle-timeout", "0",
            "--local-endpoint", "--port", "9000", "--bind", "::1",
        ])
        sync.assert_called_once_with(DEFAULT_PROVIDER_CONFIG_PATH, "good")

    def test_preload_renderer_preserves_other_tables(self) -> None:
        original = '[other]\npreload_models = ["unrelated"]\n[backend]\n# operator comment\n'
        rendered = render_preload_model_config(original, "good")
        self.assertIn('[other]\npreload_models = ["unrelated"]', rendered)
        self.assertIn('[backend]\npreload_models = ["good"]\n# operator comment', rendered)

    def test_preload_renderer_rejects_model_collections_and_empty_ids(self) -> None:
        for invalid in (("gemma", "gpt"), ["gemma"], "", " ", "good\nother"):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                render_preload_model_config("[backend]\n", invalid)


class DiscoveryTests(unittest.TestCase):
    def test_catalog_uses_official_array_and_selected_config(self) -> None:
        rows = [{"id": "Qwen3.5-9B", "min_ram_gb": 16}, {"id": IGNORED}, {"id": "Qwen3.5-9B"}]
        with patch("warm_model_manager.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, json.dumps(rows), "")
            self.assertEqual(catalog_model_ids("custom-darkbloom", Path("custom.toml")), {"Qwen3.5-9B", IGNORED})
            self.assertEqual(run.call_args.args[0], [
                "custom-darkbloom", "models", "catalog", "--config", "custom.toml", "--json",
            ])

    def test_empty_catalog_is_valid_but_partial_or_malformed_catalog_is_not(self) -> None:
        with patch("warm_model_manager.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "[]", "")
            self.assertEqual(catalog_model_ids("darkbloom"), set())
            for payload in ({}, [{"id": "good"}, {}], [{"id": 1}], [{"id": " "}]):
                with self.subTest(payload=payload):
                    run.return_value = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
                    with self.assertRaisesRegex(RuntimeError, "catalog models JSON"):
                        catalog_model_ids("darkbloom")

    def test_catalog_failure_and_update_banner(self) -> None:
        with patch("warm_model_manager.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, "", "coordinator unavailable")
            with self.assertRaisesRegex(RuntimeError, "catalog models: coordinator unavailable"):
                catalog_model_ids("darkbloom")
            run.return_value = subprocess.CompletedProcess([], 0, '[update] Available\n[{"id":"good"}]\n', "")
            self.assertEqual(catalog_model_ids("darkbloom"), {"good"})

    def test_official_json_shape_and_config_are_used(self) -> None:
        payload = {"cacheDirectory": "/cache", "filteredByConfig": False,
                   "models": [{"id": "new"}, {"id": IGNORED}, {"id": "new"}]}
        with patch("warm_model_manager.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
            self.assertEqual(local_model_ids("/bin/darkbloom", Path("/custom.toml")), {"new", IGNORED})
        self.assertEqual(run.call_args.args[0], [
            "/bin/darkbloom", "models", "list", "--config", "/custom.toml", "--all", "--json",
        ])

    def test_empty_list_is_valid_but_malformed_output_is_not(self) -> None:
        with patch("warm_model_manager.subprocess.run") as run:
            for payload in ([], {"models": []}):
                run.return_value = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
                self.assertEqual(local_model_ids("darkbloom"), set())
            for payload in ({}, {"models": None}, {"models": {}}, {"models": ["oops"]}, {"models": [{"id": 7}]}):
                run.return_value = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
                with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                    local_model_ids("darkbloom")

    def test_list_command_failure_is_reported(self) -> None:
        with patch("warm_model_manager.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, "", "bad config")
            with self.assertRaisesRegex(RuntimeError, "bad config"):
                local_model_ids("darkbloom")

    def test_update_banner_with_brackets_does_not_hide_json(self) -> None:
        with patch("warm_model_manager.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 0, '[update] New release\n{"models":[{"id":"new"}]}\n', "",
            )
            self.assertEqual(local_model_ids("darkbloom"), {"new"})

    def test_ignore_is_repeatable_with_alias(self) -> None:
        args = build_parser().parse_args(["--ignore-model", IGNORED, "--ignore", "other"])
        self.assertEqual(args.ignore_model, [IGNORED, "other"])
        self.assertIsNone(args.model)
        self.assertFalse(args.hide_ignored)

    def test_ignore_accepts_multiple_ids_per_flag_and_keeps_following_options(self) -> None:
        args = build_parser().parse_args([
            "run", "--ignore-model", IGNORED, "Qwen3.5-9B",
            "--ignore", "gpt-oss-20b", "gemma-4-26b-8bit",
            "--ignore-model", IGNORED, "--hide-ignored", "--check-every", "120", "--apply",
        ])
        self.assertEqual(args.ignore_model, [IGNORED, "Qwen3.5-9B", "gpt-oss-20b", "gemma-4-26b-8bit", IGNORED])
        self.assertEqual(args.mode, "run")
        self.assertEqual(args.check_every, 120)
        self.assertTrue(args.apply)
        self.assertTrue(args.hide_ignored)

    def test_ignore_requires_at_least_one_id_per_occurrence(self) -> None:
        for flag in ("--ignore-model", "--ignore"):
            for following in ([], ["--apply"]):
                with self.subTest(flag=flag, following=following), redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(["once", flag, *following])
                self.assertEqual(error.exception.code, 2)

    def test_each_model_is_eligible_independently(self) -> None:
        gemma, gpt = "gemma-4-26b-qat-4bit", "gpt-oss-20b"
        self.assertEqual(eligible_local_targets([gemma, gpt], {gemma, gpt}, {gpt}), [gemma])
        self.assertEqual(eligible_local_targets([gemma], {gemma}, set()), [gemma])
        self.assertEqual(eligible_local_targets([gpt], {gpt}, set()), [gpt])

    def test_low_level_load_guard_precedes_preload_and_command(self) -> None:
        with patch("warm_model_manager.synchronize_preload_model") as sync, \
                patch("warm_model_manager.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "ignored"):
                switch_model("darkbloom", IGNORED, None, {IGNORED})
            sync.assert_not_called()
            run.assert_not_called()


class CliAndStateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.state_path = self.directory / DEFAULT_STATE_PATH.name
        self.previous_path = self.directory / "warm-model-manager-v4.json"
        default = patch("warm_model_manager.DEFAULT_STATE_PATH", self.state_path)
        default.start()
        self.addCleanup(default.stop)
        self.previous = {
            "state_schema": STATE_SCHEMA,
            "preload_sync_schema": 1,
            "manager_version": "legacy",
            "pressure_history": {"good": [{"at": 10000, "pressure": 0.5}]},
            "pending_switch": {"target": "good", "warm_models": ["good"], "command_at": 10000},
            "last_switch_at": 8000,
            "live_challenger_model": "good",
            "live_challenger_streak": 2,
        }
        self.previous_path.write_text(json.dumps(self.previous))

    def test_old_and_new_flag_names_have_identical_settings(self) -> None:
        old = build_parser().parse_args([
            "run", "--interval", "60", "--history", "5",
            "--confirmations", "3", "--min-dwell", "1800",
        ])
        new = build_parser().parse_args([
            "run", "--check-every", "60", "--average-samples", "5",
            "--switch-after-checks", "3", "--min-warm-time", "1800",
        ])
        self.assertEqual(vars(old), vars(new))
        defaults = build_parser().parse_args([])
        self.assertEqual(
            (defaults.check_every, defaults.average_samples,
             defaults.switch_after_checks, defaults.min_warm_time),
            (60, 15, 3, 2700),
        )

    def test_validation_uses_new_flag_names_before_external_io(self) -> None:
        for flag, value in (("--check-every", "59"), ("--average-samples", "0"),
                            ("--switch-after-checks", "0"), ("--min-warm-time", "-1")):
            with self.subTest(flag=flag), patch.object(sys, "argv", ["warm_model_manager.py", flag, value]), \
                    self.assertRaisesRegex(SystemExit, flag):
                main()
        self.assertFalse(self.state_path.exists())

    def test_percentage_flag_uses_percent_and_legacy_flag_preserves_fraction_units(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).switch_improvement_percent, 25)
        for percent in (0, 0.25, 1, 25, 200):
            with self.subTest(percent=percent):
                new = parser.parse_args(["--switch-improvement-percent", str(percent)])
                old = parser.parse_args(["--relative-margin", str(percent / 100)])
                self.assertEqual(new.switch_improvement_percent, percent)
                self.assertAlmostEqual(old.switch_improvement_percent, percent)
                self.assertFalse(hasattr(new, "absolute_margin"))

    def test_invalid_or_ambiguous_margin_flags_stop_before_external_io(self) -> None:
        commands = [
            ["--switch-improvement-percent", value] for value in ("-1", "nan", "inf", "bad")
        ] + [
            ["--relative-margin", "1e308"],
            ["--relative-margin", "0.25", "--switch-improvement-percent", "25"],
            ["--absolute-margin", "0.01"],
        ]
        for flags in commands:
            with self.subTest(flags=flags), patch.object(sys, "argv", ["warm_model_manager.py", *flags]), \
                    patch("sys.stderr", new=io.StringIO()) as error_text, self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(self.state_path.exists())
            if flags[0] == "--absolute-margin":
                self.assertIn("use --switch-improvement-percent 25 for 25% (or 1 for 1%)", error_text.getvalue())

    def test_default_filename_does_not_depend_on_major_release(self) -> None:
        for version in ("5.0.0", "6.0.0"):
            with self.subTest(version=version), patch("warm_model_manager.MANAGER_VERSION", version):
                self.assertEqual(build_parser().parse_args([]).state.name, "warm-model-manager-state.json")

    def test_migration_keeps_switch_state_and_original_with_owner_only_permissions(self) -> None:
        original = self.previous_path.read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertTrue(migrate_default_state(self.state_path))
        self.assertEqual(json.loads(self.state_path.read_text()), self.previous)
        self.assertEqual(self.previous_path.read_bytes(), original)
        self.assertEqual(self.state_path.stat().st_mode & 0o777, 0o600)

    def test_existing_destination_and_custom_state_are_never_overwritten(self) -> None:
        self.state_path.write_text('{"active_target": "existing"}')
        self.assertFalse(migrate_default_state(self.state_path))
        self.assertEqual(self.state_path.read_text(), '{"active_target": "existing"}')
        self.state_path.unlink()
        custom = self.directory / "custom.json"
        self.assertFalse(migrate_default_state(custom))
        self.assertFalse(custom.exists())
        self.assertFalse(self.state_path.exists())
        self.previous_path.unlink()
        self.assertFalse(migrate_default_state(self.state_path))

    def test_damaged_previous_state_blocks_run_instead_of_starting_empty(self) -> None:
        for content in ("{broken", "[]"):
            with self.subTest(content=content):
                self.previous_path.write_text(content)
                with patch.object(sys, "argv", ["warm_model_manager.py", "once"]), \
                        patch("warm_model_manager.signal.signal"), \
                        patch("warm_model_manager.Manager.run") as run, \
                        redirect_stdout(io.StringIO()) as report:
                    self.assertEqual(main(), 1)
                run.assert_not_called()
                self.assertIn("could not migrate previous manager state", report.getvalue())
                self.assertFalse(self.state_path.exists())

    def test_old_and_new_process_locks_block_migration_and_checks(self) -> None:
        for name in (self.state_path.name + ".lock", "warm-model-manager.json.lock",
                     "warm-model-manager-v4.json.lock"):
            with self.subTest(lock=name), (self.directory / name).open("w") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(sys, "argv", ["warm_model_manager.py", "once"]), \
                        patch("warm_model_manager.Manager.run") as run, \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 2)
                run.assert_not_called()
                self.assertFalse(self.state_path.exists())
                self.assertEqual(json.loads(self.previous_path.read_text()), self.previous)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.state = self.directory / "state.json"
        self.local = self.directory / "local.json"
        self.token = self.directory / "prod-token"
        self.token.write_text("fixture-production-token\n")
        self.token.chmod(0o600)
        self.local.write_text(json.dumps({
            "pid": 123, "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "fixture-local-token",
        }))
        self.args = build_parser().parse_args([
            "run", "--apply", "--hourly-probes", "--state", str(self.state),
            "--prod-token-file", str(self.token), "--local-endpoint-file", str(self.local),
        ])
        self.manager = Manager(self.args)
        self.warm = LocalDaemonState("good", ("good",), False, 123, 100, True)
        self.clock = self.mock("time.time", return_value=10000)
        self.daemon = self.mock("read_daemon_state", return_value=self.warm)
        self.send = self.mock("send_probe", return_value={"http_status": 200, "outcome": "SUCCESS",
                                                        "status": "completion received", "elapsed_seconds": 0.25})
        self.saved = {"pressure_history": {"good": [{"at": 10000, "pressure": 1}]},
                      "live_challenger_streak": 2, "last_switch_at": 9000}
        self.state.write_text(json.dumps(self.saved))

    def mock(self, name, **kwargs):
        patcher = patch("warm_model_manager." + name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def tick(self, at):
        self.clock.return_value = at
        with redirect_stdout(io.StringIO()) as output:
            self.manager.probe_if_due()
        return json.loads(self.state.read_text()), output.getvalue()

    def test_alternates_hourly_and_reads_changed_warm_model(self):
        self.tick(10000)
        self.tick(11799)
        self.daemon.return_value = replace(self.warm, current_model="new", warm_models=("new",))
        self.tick(11800)
        state, report = self.tick(13600)
        self.assertEqual([(c.args[0], c.args[1]) for c in self.send.call_args_list],
                         [("local", "good"), ("production", "new"), ("local", "new")])
        self.assertEqual(state["probes"]["next_at"], 15400)
        self.assertIn('model="new"; HTTP 200', report)
        self.assertIn("local probe: SENDING", report)
        self.assertIn("local probe: SUCCESS", report)
        self.assertIn("seconds=0.25", report)
        self.assertIn("probe, production self-route", report)
        self.assertIn("probe, local endpoint", report)
        for key, value in self.saved.items():
            self.assertEqual(state[key], value)
        self.assertNotIn("fixture-", self.state.read_text() + report)
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o600)

    def test_restart_and_downtime_do_not_burst_or_replay(self):
        self.tick(10000)
        self.manager = Manager(self.args)
        self.tick(10060)
        self.send.assert_called_once()
        self.manager = Manager(self.args)
        state, _ = self.tick(20000)
        self.assertEqual(self.send.call_count, 2)
        self.assertEqual(self.send.call_args.args[0], "production")
        self.assertEqual(state["probes"]["next_at"], 21800)
        self.tick(20001)
        self.assertEqual(self.send.call_count, 2)

    def test_off_dry_run_and_stopping_do_not_read_credentials_or_advance_schedule(self):
        for enabled, apply, stopped in ((False, True, False), (True, False, False), (True, True, True)):
            with self.subTest(enabled=enabled, apply=apply, stopped=stopped):
                self.args.hourly_probes, self.args.apply = enabled, apply
                self.manager = Manager(self.args)
                self.manager.stop_requested = stopped
                with patch("warm_model_manager.probe_credentials") as credentials:
                    state, _ = self.tick(10000)
                credentials.assert_not_called()
                self.assertEqual(state, self.saved)
        self.send.assert_not_called()
        self.assertFalse(build_parser().parse_args([]).hourly_probes)

    def test_skip_unavailable_busy_ignored_and_pending_models(self):
        cases = [
            (None, {}, [], "unavailable"),
            (replace(self.warm, fresh=False), {}, [], "stale"),
            (replace(self.warm, alive=False), {}, [], "offline"),
            (replace(self.warm, warm_models=()), {}, [], "no single warm"),
            (replace(self.warm, warm_models=("good", "other")), {}, [], "no single warm"),
            (replace(self.warm, inference_active=True), {}, [], "serving"),
            (self.warm, {}, ["good"], "ignored"),
            (self.warm, {"pending_switch": {"target": "new"}}, [], "pending"),
        ]
        for daemon, saved, ignored, reason in cases:
            with self.subTest(reason=reason):
                self.daemon.return_value = daemon
                self.state.write_text(json.dumps(saved))
                self.manager = Manager(self.args)
                self.manager.ignored = set(ignored)
                state, output = self.tick(10000)
                self.assertIn(reason, output)
                self.assertIn("probe: SKIPPED", output)
                self.assertNotIn("SENDING", output)
                self.assertEqual(state["probes"]["next_at"], 11800)
                self.assertIsNone(state["probes"]["last_result"]["http_status"])
        self.send.assert_not_called()

    def test_model_restriction_is_also_honored(self):
        self.args.model = ["other"]
        _, report = self.tick(10000)
        self.assertIn("excluded by --model", report)
        self.send.assert_not_called()

    def test_missing_tokens_do_not_stop_alternating_schedule(self):
        self.tick(10000)
        self.token.unlink()
        _, output = self.tick(11800)
        self.assertIn("production token unavailable", output)
        self.tick(13600)
        self.assertEqual([c.args[0] for c in self.send.call_args_list], ["local", "local"])

    def test_save_before_request_and_interrupt_blocks_replay(self):
        def interrupt(*_):
            state = json.loads(self.state.read_text())["probes"]
            self.assertEqual(state["next_at"], 11800)
            self.assertEqual(state["last_result"]["status"], "request started; result unknown")
            raise KeyboardInterrupt
        self.send.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.tick(10000)
        self.manager = Manager(self.args)
        self.tick(10001)
        self.send.assert_called_once()

    def test_failed_state_write_or_corruption_prevents_request(self):
        with patch("warm_model_manager.write_json_atomic", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.tick(10000)
        self.state.write_text("{damaged")
        self.manager = Manager(self.args)
        with self.assertRaises(ValueError):
            self.tick(10000)
        self.send.assert_not_called()

    def test_schedule_runs_between_slow_score_checks(self):
        self.args.check_every = 7200
        elapsed = [10000.0]
        self.clock.side_effect = lambda: elapsed[0]
        def sleep(_):
            elapsed[0] += 900
            if elapsed[0] > 13600:
                self.manager.stop_requested = True
        with patch.object(self.manager, "iteration") as iteration, \
                patch("warm_model_manager.time.monotonic", side_effect=lambda: elapsed[0]), \
                patch("warm_model_manager.time.sleep", side_effect=sleep), redirect_stdout(io.StringIO()):
            self.manager.run()
        iteration.assert_called_once()
        self.assertEqual([c.args[0] for c in self.send.call_args_list], ["local", "production", "local"])

    def test_local_endpoint_rejects_wrong_process_and_nonlocal_urls(self):
        for base in ("http://example.com:8000/v1", "http://127.0.0.1:8000/v1?x=1",
                     "http://user:secret@127.0.0.1:8000/v1", "file:///v1", "http://127.0.0.1:bad/v1"):
            with self.subTest(base=base):
                self.local.write_text(json.dumps({"pid": 123, "base_url": base, "api_key": "fixture"}))
                with self.assertRaisesRegex(ValueError, "loopback"):
                    probe_credentials("local", self.warm, self.local, self.token)
        self.local.write_text(json.dumps({"pid": 999, "base_url": "http://127.0.0.1:8000/v1"}))
        with self.assertRaisesRegex(ValueError, "endpoint pid 999, provider pid 123"):
            probe_credentials("local", self.warm, self.local, self.token)

    def test_local_saved_token_ipv6_and_no_auth_mode(self):
        url, token = probe_credentials("local", self.warm, self.local, self.token)
        self.assertEqual(url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(token, "fixture-local-token")
        self.local.write_text(json.dumps({"pid": 123, "base_url": "http://[::1]:9000/v1/", "api_key": ""}))
        self.assertEqual(probe_credentials("local", self.warm, self.local, self.token),
                         ("http://[::1]:9000/v1/chat/completions", ""))

    def test_local_record_errors_identify_the_failure_without_printing_contents(self):
        self.local.unlink()
        with self.assertRaisesRegex(ValueError, "record is missing; enable --local-endpoint"):
            probe_credentials("local", self.warm, self.local, self.token)
        for contents, message in ((b"{PRIVATE", "invalid JSON"), (b"\xff", "invalid JSON"),
                                  (b"[]", "no valid process ID"), (b'{"pid": true}', "no valid process ID")):
            with self.subTest(message=message):
                self.local.write_bytes(contents)
                with self.assertRaisesRegex(ValueError, message) as caught:
                    probe_credentials("local", self.warm, self.local, self.token)
                self.assertNotIn("PRIVATE", str(caught.exception))
        with patch.object(Path, "read_text", side_effect=PermissionError("PRIVATE")):
            with self.assertRaisesRegex(ValueError, "cannot be read; check file permissions"):
                probe_credentials("local", self.warm, self.local, self.token)

    def test_switch_endpoint_flags_preserve_live_settings_and_default_if_unavailable(self):
        for host, base in (("127.0.0.1", "http://127.0.0.1:9000/v1"),
                           ("::1", "http://[::1]:9000/v1"),
                           ("0.0.0.0", "http://127.0.0.1:9000/v1")):
            for token in ("PRIVATE", ""):
                with self.subTest(host=host, authenticated=bool(token)):
                    self.local.write_text(json.dumps({"pid": 123, "host": host, "port": 9000,
                                                      "base_url": base, "api_key": token}))
                    self.assertEqual(local_endpoint_start_flags(self.local, self.warm), [
                        "--local-endpoint", "--port", "9000", "--bind", host,
                        *([] if token else ["--no-auth"]),
                    ])
        for daemon in (None, replace(self.warm, pid=999), replace(self.warm, alive=False),
                       replace(self.warm, fresh=False)):
            self.assertEqual(local_endpoint_start_flags(self.local, daemon), ["--local-endpoint"])
        for contents in ("{broken", "[]", '{"pid": 123, "port": "bad"}'):
            self.local.write_text(contents)
            self.assertEqual(local_endpoint_start_flags(self.local, self.warm), ["--local-endpoint"])
        self.local.unlink()
        self.assertEqual(local_endpoint_start_flags(self.local, self.warm), ["--local-endpoint"])

    def queue_switch_probe(self, regular_at=11800):
        state = {**self.saved, "probes": {"next_kind": "production", "next_at": regular_at},
                 "switch_probe": {"target": "good", "confirmed_at": 10000, "due_at": 10180,
                                  "pid": 123, "daemon_started_at": 100}}
        self.state.write_text(json.dumps(state))
        return state

    def test_switch_probe_is_once_after_three_minutes_and_leaves_hourly_cadence_alone(self):
        original = self.queue_switch_probe()
        self.tick(10000)
        self.manager = Manager(self.args)
        self.tick(10179)
        self.send.assert_not_called()
        state, output = self.tick(10180)
        self.send.assert_called_once_with("production", "good", PROD_PROBE_URL, "fixture-production-token")
        self.assertNotIn("switch_probe", state)
        self.assertEqual(state["probes"], original["probes"])
        self.assertEqual(state["last_switch_probe_result"]["outcome"], "SUCCESS")
        self.assertIn("production probe (after switch): SENDING", output)
        self.assertIn("route=self", output)
        self.assertIn("production probe (after switch): SUCCESS", output)
        self.assertIn("probe, production self-route", output)
        self.assertIn("probe, local endpoint", output)
        self.manager = Manager(self.args)
        self.tick(10181)
        self.send.assert_called_once()
        self.tick(11800)
        self.assertEqual(self.send.call_count, 2)

    def test_switch_probe_consumes_intent_before_post_and_does_not_replay_after_interruption(self):
        original = self.queue_switch_probe()
        def interrupt(*_):
            state = json.loads(self.state.read_text())
            self.assertNotIn("switch_probe", state)
            self.assertEqual(state["probes"], original["probes"])
            self.assertEqual(state["last_switch_probe_result"]["outcome"], "UNKNOWN")
            raise KeyboardInterrupt
        self.send.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.tick(10180)
        self.manager = Manager(self.args)
        self.tick(10181)
        self.send.assert_called_once()

    def test_switch_probe_requires_same_warm_process_and_normal_probe_guards(self):
        for daemon, ignored, models, pending, reason in (
            (replace(self.warm, warm_models=("new",)), [], None, False, "changed after warm-up"),
            (replace(self.warm, pid=999), [], None, False, "changed after warm-up"),
            (replace(self.warm, started_at=999), [], None, False, "changed after warm-up"),
            (replace(self.warm, inference_active=True), [], None, False, "serving"),
            (replace(self.warm, fresh=False), [], None, False, "stale"),
            (replace(self.warm, warm_models=()), [], None, False, "no single warm"),
            (self.warm, ["good"], None, False, "ignored"),
            (self.warm, [], ["other"], False, "excluded"),
            (self.warm, [], None, True, "pending"),
        ):
            with self.subTest(reason=reason):
                original = self.queue_switch_probe()
                if pending:
                    original["pending_switch"] = {"target": "next"}
                    self.state.write_text(json.dumps(original))
                self.daemon.return_value = daemon
                self.args.model = models
                self.manager = Manager(self.args)
                self.manager.ignored = set(ignored)
                state, output = self.tick(10180)
                self.assertIn(reason, output)
                self.assertIn("production probe (after switch): SKIPPED", output)
                self.assertNotIn("switch_probe", state)
                self.assertEqual(state["probes"], original["probes"])
        self.send.assert_not_called()

    def test_switch_probe_missing_token_and_http_failure_are_not_retried(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                original = self.queue_switch_probe()
                self.manager = Manager(self.args)
                self.send.reset_mock()
                if missing:
                    self.token.unlink()
                self.send.return_value = {"http_status": 503, "outcome": "FAILED", "status": "HTTP error",
                                          "error": "model_not_loaded: No owned machine serves this model"}
                state, output = self.tick(10180)
                self.assertIn("production token unavailable" if missing else "error=model_not_loaded", output)
                self.assertNotIn("switch_probe", state)
                self.assertEqual(state["probes"], original["probes"])
                self.manager = Manager(self.args)
                self.tick(10181)
                self.assertEqual(self.send.call_count, 0 if missing else 1)

    def test_extra_probe_appears_in_next_two_and_regular_probe_can_run_first(self):
        state = self.queue_switch_probe(10100)
        lines = [line for line in timeline_lines(state, 10000, True, True) if "probe," in line]
        self.assertIn("probe, production self-route", lines[0])
        self.assertIn("probe, production self-route after switch", lines[1])
        self.assertEqual(len(lines), 3)  # Neither regular endpoint is hidden by the extra probe.
        self.tick(10100)
        self.tick(10179)
        self.send.assert_called_once()
        state, _ = self.tick(10180)
        self.assertEqual(self.send.call_count, 2)
        self.assertEqual(state["probes"]["next_at"], 11900)
        self.assertEqual(state["probes"]["next_kind"], "local")
        self.assertEqual(state["probes"]["last_result"]["at"], 10100)

    def test_switch_probe_write_failure_prevents_post(self):
        self.queue_switch_probe()
        with patch("warm_model_manager.write_json_atomic", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.tick(10180)
        self.send.assert_not_called()

    def test_token_setup_is_private_and_does_not_start_manager(self):
        self.token.chmod(0o644)
        with patch("warm_model_manager.getpass.getpass", return_value="replacement-fixture"), \
                patch.object(sys, "argv", ["manager", "set-prod-token", "--prod-token-file", str(self.token)]), \
                patch.object(Manager, "run") as run, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 0)
        run.assert_not_called()
        self.assertEqual(self.token.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.token.read_text(), "replacement-fixture\n")
        self.assertNotIn("replacement-fixture", output.getvalue())
        self.assertEqual(json.loads(self.state.read_text()), self.saved)

    def test_bad_token_and_broad_permissions_fail_without_echoing_secret(self):
        for value in ("", "fixture\ninjected", "fixture\rheader"):
            self.token.write_text(value)
            with self.assertRaisesRegex(ValueError, "invalid"):
                probe_credentials("production", self.warm, self.local, self.token)
        self.token.write_text("fixture")
        self.token.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "owner-only"):
            probe_credentials("production", self.warm, self.local, self.token)

    def test_requests_have_correct_model_auth_route_and_limits(self):
        for kind, url, token in (("production", PROD_PROBE_URL, "prod-fixture"),
                                 ("local", "http://127.0.0.1:8000/v1/chat/completions", "local-fixture")):
            with self.subTest(kind=kind), patch("warm_model_manager.build_opener") as build:
                response = build.return_value.open.return_value.__enter__.return_value
                response.status = 200
                response.headers = {"X-Provider-Id": "provider-another-mac"}
                response.read.return_value = b'{"choices": [{"message": {"content": "PRIVATE CONTENT"}}]}'
                result = send_probe(kind, 'model/with"quote', url, token)
                request = build.return_value.open.call_args.args[0]
                headers = {k.lower(): v for k, v in request.header_items()}
                self.assertEqual(request.full_url, url)
                self.assertEqual(request.method, "POST")
                body = json.loads(request.data)
                self.assertEqual(body["model"], 'model/with"quote')
                self.assertEqual(body["max_tokens"], 64)
                self.assertFalse(body["stream"])
                self.assertEqual(headers["authorization"], "Bearer " + token)
                self.assertEqual(headers.get("x-darkbloom-route"), "self" if kind == "production" else None)
                self.assertEqual(result.get("provider_id"), "provider-another-mac" if kind == "production" else None)
                self.assertEqual(result["outcome"], "SUCCESS")
                self.assertEqual(build.call_args.args[0].proxies, {})
                self.assertIsInstance(build.call_args.args[1], ProbeRedirectHandler)
                self.assertEqual(build.return_value.open.call_args.kwargs["timeout"], 30)
                response.read.assert_called_once_with(65537)
                self.assertNotIn("PRIVATE CONTENT", json.dumps(result))

    def test_response_read_failure_preserves_http_status_and_redacts_details(self):
        with patch("warm_model_manager.build_opener") as build:
            response = build.return_value.open.return_value.__enter__.return_value
            response.status = 200
            response.headers = {}
            response.read.side_effect = TimeoutError("SECRET")
            result = send_probe("production", "model", PROD_PROBE_URL, "SECRET")
        self.assertEqual(result["http_status"], 200)
        self.assertIn("timed out", result["status"])
        self.assertEqual(result["outcome"], "FAILED")
        self.assertIn("TimeoutError", result["error"])
        self.assertNotIn("SECRET", json.dumps(result))

    def test_malicious_response_headers_and_oversized_body_are_not_logged(self):
        for provider in ("SECRET", "provider\nforged log"):
            with self.subTest(provider=provider), patch("warm_model_manager.build_opener") as build:
                response = build.return_value.open.return_value.__enter__.return_value
                response.status = 200
                response.headers = {"X-Provider-Id": provider}
                response.read.return_value = b"x" * 65537
                result = send_probe("production", "model", PROD_PROBE_URL, "SECRET")
                self.assertNotIn("provider_id", result)
                self.assertEqual(result["status"], "response exceeded size limit")

    def test_http_errors_and_timeout_are_bounded_and_redacted(self):
        errors = [HTTPError(PROD_PROBE_URL, n, "SECRET", {}, io.BytesIO(b"SECRET"))
                  for n in (302, 401, 429, 503)] + [URLError("SECRET"), TimeoutError("SECRET")]
        for error in errors:
            with self.subTest(error=type(error).__name__), patch("warm_model_manager.build_opener") as build:
                build.return_value.open.side_effect = error
                result = send_probe("production", "model", PROD_PROBE_URL, "SECRET")
                self.assertEqual(result["http_status"], getattr(error, "code", None))
                self.assertEqual(result["outcome"], "FAILED")
                self.assertNotIn("SECRET", json.dumps(result))
                build.return_value.open.assert_called_once()
        self.assertIsNone(ProbeRedirectHandler().redirect_request(None, None, 302, "", {}, "https://other.invalid"))

    def test_report_schedule_shows_both_endpoints_and_overdue_estimate(self):
        state = {"probes": {"next_kind": "production", "next_at": 11800}}
        events, _ = report_timeline(state, None, None, Decision(None, "test"), 10000,
                                    10060, 60, 3, 2700, True, False, True)
        self.assertEqual([event.at for event in events], [10060, 11800, 13600])
        self.assertIn("production", events[1].text)
        self.assertIn("local", events[2].text)
        overdue = "\n".join(timeline_lines(state, 12000))
        self.assertIn("overdue", overdue)
        self.assertIn("estimated, 30m after the preceding regular attempt", overdue)
        self.assertEqual(state["probes"]["next_at"], 11800)

    def test_initial_and_disabled_schedules_are_honest(self):
        initial = "\n".join(timeline_lines({}, 10000))
        self.assertIn("local", initial)
        self.assertIn("due now", initial)
        self.assertIn("production", initial)
        for enabled, apply in ((False, True), (True, False)):
            text = "\n".join(timeline_lines(self.saved, 10000, enabled, apply))
            self.assertNotIn("probe,", text)
            if enabled:
                self.assertIn("DRY RUN", text)

    def test_api_error_code_and_message_are_available_without_credentials(self):
        body = json.dumps({"error": {"code": "model_not_loaded",
                                   "message": "No owned machine serves model good; token=SECRET"}}).encode()
        for status in (200, 503):
            with self.subTest(status=status), patch("warm_model_manager.build_opener") as build:
                if status == 503:
                    build.return_value.open.side_effect = HTTPError(PROD_PROBE_URL, 503, "Service Unavailable", {}, io.BytesIO(body))
                else:
                    response = build.return_value.open.return_value.__enter__.return_value
                    response.status, response.headers, response.read.return_value = 200, {}, body
                result = send_probe("production", "good", PROD_PROBE_URL, "SECRET")
                self.assertEqual(result["outcome"], "FAILED")
                self.assertEqual(result["http_status"], status)
                self.assertIn("model_not_loaded", result["error"])
                self.assertIn("No owned machine serves model good", result["error"])
                self.assertNotIn("SECRET", json.dumps(result))

    def test_http_success_with_bad_or_empty_completion_is_failed(self):
        for body in (b"<html>SECRET</html>", b"", b"[]", b"{}", b'{"choices": []}',
                     b'{"choices": [{"message": {"content": ""}}]}'):
            with self.subTest(body=body):
                result = probe_response_details(body, "SECRET", 200)
                self.assertEqual(result["outcome"], "FAILED")
                self.assertNotIn("SECRET", json.dumps(result))

    def test_non_json_http_error_gives_a_bounded_redacted_excerpt(self):
        result = probe_response_details(b"upstream connection refused; Authorization: Bearer SECRET\nretry later", "SECRET", 502)
        self.assertIn("upstream connection refused", result["error"])
        self.assertIn("retry later", result["error"])
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertNotIn("\n", result["error"])

    def test_completed_inference_reports_finish_reason_and_usage_without_reply_text(self):
        body = json.dumps({"choices": [{"finish_reason": "length", "message": {"content": "PRIVATE REASONING"}}],
                           "usage": {"prompt_tokens": 15, "completion_tokens": 64}}).encode()
        result = probe_response_details(body, "SECRET", 200)
        self.assertEqual(result["outcome"], "SUCCESS")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["completion_tokens"], 64)
        self.assertEqual(result["prompt_tokens"], 15)
        self.assertIn("token limit", result["status"])
        self.assertNotIn("PRIVATE REASONING", json.dumps(result))

    def test_debug_text_redacts_keys_and_prevents_terminal_control_or_multiline_output(self):
        other_key = "sk-db-" + "f" * 64
        text = "SECRET\nAuthorization: Bearer OTHER\r\x1b[31m api_key=HIDDEN " + other_key + " x" * 500
        result = probe_debug_text(text, "SECRET")
        for hidden in ("SECRET", "OTHER", "HIDDEN", other_key, "\n", "\r", "\x1b"):
            self.assertNotIn(hidden, result)
        self.assertLessEqual(len(result), 403)

    def test_result_and_both_times_are_logged_even_if_final_state_save_fails(self):
        from warm_model_manager import write_json_atomic
        saves = [0]
        def save(path, state):
            saves[0] += 1
            if saves[0] == 2:
                raise OSError("disk full")
            write_json_atomic(path, state)
        with patch("warm_model_manager.write_json_atomic", side_effect=save), \
                redirect_stdout(io.StringIO()) as report:
            with self.assertRaises(OSError):
                self.manager.probe_if_due()
        self.assertIn("probe: SUCCESS", report.getvalue())
        self.assertIn("HTTP 200", report.getvalue())
        self.assertIn("probe, production self-route", report.getvalue())
        self.assertIn("probe, local endpoint", report.getvalue())
        self.assertEqual(json.loads(self.state.read_text())["probes"]["next_at"], 11800)

    def test_failed_probe_logs_specific_error_and_keeps_the_schedule(self):
        self.send.return_value = {"outcome": "FAILED", "http_status": 401, "status": "HTTP error",
                                  "error": "invalid_api_key: key is invalid", "elapsed_seconds": 0.1}
        state, report = self.tick(10000)
        self.assertIn("probe: FAILED", report)
        self.assertIn("HTTP 401", report)
        self.assertIn("error=invalid_api_key: key is invalid", report)
        self.assertIn("probe, production self-route", report)
        self.assertEqual(state["probes"]["next_at"], 11800)


class RoutingRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.path = self.directory / "state.json"
        self.config = self.directory / "provider.toml"
        self.config.write_text('[backend]\npreload_models = ["good"]\n')
        self.local = self.directory / "local.json"
        self.local.write_text(json.dumps({"pid": 123, "base_url": "http://127.0.0.1:8000/v1",
                                         "api_key": "fixture-local-token"}))
        self.token = self.directory / "token"
        self.token.write_text("fixture-production-token")
        self.token.chmod(0o600)
        self.args = build_parser().parse_args([
            "run", "--apply", "--recover-routing", "--hourly-probes",
            "--state", str(self.path), "--config", str(self.config),
            "--daemon-state", str(self.directory / "daemon.json"),
            "--local-endpoint-file", str(self.local), "--prod-token-file", str(self.token),
        ])
        self.manager = Manager(self.args)
        self.now = self.mock("time.time", return_value=10000)
        self.monotonic = self.mock("time.monotonic", side_effect=lambda: self.now.return_value)
        self.warm = LocalDaemonState("good", ("good",), False, 123, 100, True,
                                     requests_served=0, reconnect_count=1)
        self.daemon = self.mock("read_daemon_state", return_value=self.warm)
        self.services = self.mock("provider_services", return_value={"io.darkbloom.provider": 123})
        self.alive = self.mock("process_alive", return_value=True)
        self.stop = self.mock("stop_provider")
        self.launch = self.mock("switch_model")
        self.discovery = self.mock("local_model_ids", return_value={"good"})
        self.catalog = self.mock("catalog_model_ids", return_value={"good"})
        self.capacity = self.mock("fetch_capacity", return_value={"good": CapacitySample("good", 1, 1, 1)})
        self.prices = self.mock("fetch_model_prices", return_value=({"good": ModelPrice(1, 1)}, ModelPrice(None, None)))
        self.send = self.mock("send_probe", side_effect=lambda kind, *args: self.ok() if kind == "local" else self.failure())

    def mock(self, name, **kwargs):
        p = patch("warm_model_manager." + name, **kwargs)
        result = p.start()
        self.addCleanup(p.stop)
        return result

    @staticmethod
    def ok():
        return {"outcome": "SUCCESS", "http_status": 200, "status": "completion received",
                "provider_id": "another-provider"}

    @staticmethod
    def failure():
        return {"outcome": "FAILED", "http_status": 503, "error_code": "model_not_loaded",
                "status": "HTTP error", "error": "model_not_loaded: not available"}

    def tick(self, at):
        self.now.return_value = at
        with redirect_stdout(io.StringIO()) as output:
            self.manager.recovery_if_due()
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        return state.get("routing_recovery", {}), output.getvalue()

    def trigger(self):
        self.tick(10000)
        self.tick(10900)
        self.tick(11200)
        return self.tick(11500)

    def stopped(self):
        self.trigger()
        self.daemon.return_value = None
        self.services.return_value = {}
        self.alive.return_value = False
        return self.tick(11515)

    def restart(self):
        self.stopped()
        self.tick(12415)
        self.warm = replace(self.warm, pid=124, started_at=12415)
        self.daemon.return_value = self.warm
        self.local.write_text(json.dumps({"pid": 124, "base_url": "http://127.0.0.1:8000/v1", "api_key": "fixture-local-token"}))
        return self.tick(12430)

    def test_requires_opt_in_live_continuous_run(self):
        self.assertFalse(build_parser().parse_args([]).recover_routing)
        for setting, value in (("apply", False), ("recover_routing", False), ("mode", "once")):
            original = getattr(self.args, setting)
            setattr(self.args, setting, value)
            self.manager = Manager(self.args)
            self.tick(10000)
            setattr(self.args, setting, original)
        self.assertFalse(self.path.exists())
        self.send.assert_not_called()
        self.stop.assert_not_called()
        with patch.object(sys, "argv", ["manager", "once", "--apply", "--recover-routing"]):
            with self.assertRaisesRegex(SystemExit, "requires run"):
                main()

    def test_observes_full_warm_grace_then_three_spaced_failures(self):
        self.tick(10000)
        self.tick(10899)
        self.send.assert_not_called()
        first, output = self.tick(10914)
        self.assertEqual(first["failures"], 1)
        self.assertIn("local probe: SUCCESS", output)
        self.assertIn("production probe: FAILED", output)
        self.assertIn("HTTP 503", output)
        self.tick(11213)
        self.assertEqual(self.send.call_count, 2)
        self.manager = Manager(self.args)
        self.tick(11214)
        self.stop.assert_not_called()
        recovery, _ = self.tick(11514)
        self.assertEqual(recovery["phase"], "stopping")
        self.stop.assert_called_once_with("darkbloom", 123, before_stop=ANY)
        self.assertEqual(self.send.call_count, 6)
        self.assertNotIn("fixture-", self.path.read_text())
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_confirmed_stop_starts_full_countdown_and_resume_preserves_it(self):
        recovery, _ = self.stopped()
        self.assertEqual(recovery["restart_at"], 12415)
        self.manager = Manager(self.args)
        self.tick(12414)
        self.launch.assert_not_called()
        self.tick(12429)
        self.launch.assert_called_once()
        self.assertEqual(self.launch.call_args.args, ("darkbloom", "good", self.config, set()))
        self.assertIn("--local-endpoint", self.launch.call_args.kwargs["local_flags"])
        self.assertEqual(self.config.read_text(), '[backend]\npreload_models = ["good"]\n')

    def new_winner(self):
        self.discovery.return_value = {"good", "better"}
        self.catalog.return_value = {"good", "better"}
        self.capacity.return_value = {
            "good": CapacitySample("good", 1, 1, 1),
            "better": CapacitySample("better", 1, 3, 3),
        }
        self.prices.return_value = ({"good": ModelPrice(1, 1), "better": ModelPrice(0.5, 0.5)}, ModelPrice(None, None))

    def test_recovery_first_check_replaces_old_winner_and_resets_selection_state(self):
        self.stopped()
        self.new_winner()
        self.args.average_samples = 1000  # Old samples would still fit this long window.
        self.args.min_warm_time = 99999
        self.args.switch_after_checks = 100
        self.args.switch_improvement_percent = 10000
        self.args.switch_cost = 3599
        state = json.loads(self.path.read_text())
        state.update(pressure_history={"good": [{"at": 11500, "pressure": 10000}]},
                     current_residency={"target": "good"}, current_model="good", active_target="good",
                     last_switch_at=11500, live_challenger_model="good", live_challenger_streak=99,
                     dry_challenger_model="good", dry_challenger_streak=99,
                     probes={"next_kind": "production", "next_at": 13000})
        ensure_pressure_cadence(state, self.args.check_every, self.args.average_samples)
        state["pressure_history"] = {"good": [{"at": 11500, "pressure": 10000}]}
        ensure_scoring_policy(state)
        # Force refresh even for a valid, recent/future-dated price cache.
        state["pricing_cache"] = {"schema": PRICING_CACHE_SCHEMA, "source": self.args.pricing_url,
                                  "fetched_at": 20000, "prices": {"good": ModelPrice(100, 100).to_dict()},
                                  "fallback": ModelPrice(0, 0).to_dict()}
        self.path.write_text(json.dumps(state))
        self.manager = Manager(self.args)
        self.manager.weights["better"] = 2
        recovery, output = self.tick(12415)
        self.assertEqual(recovery["phase"], "starting")
        self.assertEqual(recovery["model"], "better")
        self.assertEqual(recovery["previous_model"], "good")
        self.assertTrue(recovery["attempted"])
        self.assertIn("fresh recovery check selected better (score 3)", output)
        self.launch.assert_called_once_with("darkbloom", "better", self.config, set(), local_flags=ANY)
        self.prices.assert_called_once_with(self.args.pricing_url)
        state = json.loads(self.path.read_text())
        self.assertEqual(state["pending_switch"]["warm_models"], ["better"])
        self.assertEqual(state["probes"], {"next_kind": "production", "next_at": 13000})
        self.assertEqual(state["pressure_history"]["good"], [{"at": 12415, "pressure": 1}])
        self.assertEqual(state["pressure_history"]["better"], [{"at": 12415, "pressure": 3}])
        row = state["last_score_snapshot"]["models"]["better"]
        self.assertEqual(row["score"], 3)
        self.assertEqual(row["weight"], 2)
        for key in ("live_challenger_model", "live_challenger_streak", "dry_challenger_model",
                    "dry_challenger_streak", "last_switch_at", "current_residency", "active_target"):
            self.assertNotIn(key, state)

    def test_recovery_keeps_excluded_scores_but_only_starts_a_downloaded_allowed_model(self):
        self.stopped()
        self.new_winner()
        self.discovery.return_value |= {"ignored", "outside"}
        self.catalog.return_value |= {"ignored", "outside", "remote"}
        for model in ("ignored", "outside", "remote"):
            self.capacity.return_value[model] = CapacitySample(model, 1, 1000, 1000)
            self.prices.return_value[0][model] = ModelPrice(10, 10)
        self.args.ignore_model = ["ignored"]
        self.args.model = ["good", "better", "ignored", "remote"]
        self.manager = Manager(self.args)
        self.tick(12415)
        self.assertEqual(self.launch.call_args.args[1], "better")
        rows = json.loads(self.path.read_text())["last_score_snapshot"]["models"]
        for model in ("ignored", "outside", "remote"):
            self.assertEqual(rows[model]["score"], 10000)
            self.assertFalse(rows[model]["eligible"])

    def test_recovery_can_choose_replacement_when_old_model_is_removed_or_ignored(self):
        self.stopped()
        original = self.path.read_text()
        self.new_winner()
        for ignored in (False, True):
            self.path.write_text(original)
            self.manager = Manager(self.args)
            if ignored:
                self.manager.ignored.add("good")
            else:
                self.discovery.return_value = {"better"}
            self.launch.reset_mock()
            recovery, _ = self.tick(12415)
            self.assertEqual(recovery["model"], "better")
            self.launch.assert_called_once()

    def test_recovery_waits_for_fresh_data_then_retries_without_another_stop(self):
        self.stopped()
        original = json.loads(self.path.read_text())
        ensure_scoring_policy(original)
        original["pricing_cache"] = {"schema": PRICING_CACHE_SCHEMA, "source": self.args.pricing_url,
                                     "fetched_at": 12400, "prices": {"good": ModelPrice(100, 100).to_dict()},
                                     "fallback": ModelPrice(None, None).to_dict()}
        for failing in (self.discovery, self.capacity, self.prices):
            with self.subTest(failing=failing):
                self.path.write_text(json.dumps(original))
                self.manager = Manager(self.args)
                self.launch.reset_mock()
                failing.side_effect = OSError("fixture unavailable")
                recovery, output = self.tick(12415)
                self.assertEqual(recovery["phase"], "offline")
                self.assertEqual(recovery["selection_check_at"], 12475)
                self.assertIn("staying offline", output)
                self.launch.assert_not_called()
                self.assertNotIn("pending_switch", json.loads(self.path.read_text()))
                attempts = failing.call_count
                self.manager = Manager(self.args)  # Saved cadence survives process restart.
                self.tick(12430)
                self.assertEqual(failing.call_count, attempts)
                failing.side_effect = None
                self.tick(12475)
                self.launch.assert_called_once()
        self.stop.assert_called_once()

    def test_empty_eligible_set_can_recover_after_a_download(self):
        self.stopped()
        self.discovery.return_value = set()
        self.tick(12415)
        self.launch.assert_not_called()
        self.new_winner()
        recovery, _ = self.tick(12475)
        self.assertEqual(recovery["model"], "better")
        self.launch.assert_called_once()
        self.stop.assert_called_once()

    def test_unavailable_catalog_does_not_block_locally_verified_fresh_scores(self):
        self.stopped()
        self.new_winner()
        self.catalog.side_effect = OSError("catalog unavailable")
        self.tick(12415)
        self.assertEqual(self.launch.call_args.args[1], "better")
        self.discovery.assert_called_with("darkbloom", self.config)
        self.catalog.assert_called_with("darkbloom", self.config)

    def test_recovery_checks_winner_again_after_slow_score_fetch(self):
        self.stopped()
        self.new_winner()
        self.discovery.side_effect = [{"good", "better"}, {"good"}]
        recovery, _ = self.tick(12415)
        self.assertEqual(recovery["phase"], "offline")
        self.assertIn("no longer allowed or locally available", recovery["reason"])
        self.launch.assert_not_called()
        self.discovery.side_effect = None
        recovery, _ = self.tick(12475)
        self.assertEqual(recovery["model"], "better")
        self.launch.assert_called_once()

    def test_external_start_config_edit_and_cancel_during_scoring_prevent_launch(self):
        self.stopped()
        original = self.path.read_text()
        changes = [lambda: setattr(self.daemon, "return_value", replace(self.warm, pid=999)),
                   lambda: self.config.write_text('[backend]\npreload_models = ["external"]\n'),
                   lambda: self.manager.stop(None, None)]
        for change in changes:
            self.path.write_text(original)
            self.config.write_text('[backend]\npreload_models = ["good"]\n')
            self.daemon.return_value = None
            self.manager = Manager(self.args)
            def fetch():
                change()
                return {"good": CapacitySample("good", 1, 1, 1)}
            self.capacity.side_effect = lambda *args: fetch()
            self.tick(12415)
            self.launch.assert_not_called()
            self.assertNotIn("pending_switch", json.loads(self.path.read_text()))
        self.stop.assert_called_once()

    def test_new_winner_is_saved_before_interrupted_launch_and_warmup_confirms_it(self):
        self.stopped()
        self.new_winner()
        self.launch.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.tick(12415)
        state = json.loads(self.path.read_text())
        self.assertEqual(state["pending_switch"]["target"], "better")
        self.assertEqual(state["routing_recovery"]["model"], "better")
        self.manager = Manager(self.args)
        self.tick(12430)
        self.launch.assert_called_once()
        self.daemon.return_value = replace(self.warm, current_model="better", warm_models=("better",),
                                          pid=124, started_at=12415)
        recovery, _ = self.tick(12445)
        self.assertEqual(recovery["phase"], "verifying")
        state = json.loads(self.path.read_text())
        self.assertEqual(state["last_switch_at"], 12445)
        self.assertEqual(state["active_target"], "better")
        self.assertNotIn("pending_switch", state)
        self.assertEqual(recovery["next_check_at"], 12625)
        self.launch.assert_called_once()

    def test_new_winner_uses_real_single_model_preload_sync_with_mocked_cli(self):
        self.stopped()
        self.new_winner()
        self.config.write_text('# keep this comment\n[backend]\npreload_models = ["good"]\n')
        state = json.loads(self.path.read_text())
        state["routing_recovery"]["config_digest"] = manager_module.recovery_config_digest(self.config)
        self.path.write_text(json.dumps(state))
        self.launch.side_effect = switch_model
        with patch("warm_model_manager.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as command:
            self.tick(12415)
        self.assertIn('preload_models = ["better"]', self.config.read_text())
        self.assertIn('# keep this comment', self.config.read_text())
        args = command.call_args.args[0]
        self.assertEqual(args[args.index('--model') + 1], 'better')
        self.assertIn('--local-endpoint', args)
        self.assertIn(str(self.config), args)

    def test_fresh_selection_wait_and_score_reset_are_recovery_only(self):
        self.stopped()
        self.args.apply = False
        before = self.path.read_text()
        self.tick(12415)
        self.assertEqual(self.path.read_text(), before)
        self.capacity.assert_not_called()
        self.prices.assert_not_called()
        self.launch.assert_not_called()


    def test_warm_recovery_resumes_scoring_and_preserves_new_warm_time(self):
        self.stopped()
        self.new_winner()
        self.tick(12415)
        self.daemon.return_value = replace(self.warm, current_model="better", warm_models=("better",),
                                          pid=124, started_at=12415)
        self.tick(12430)
        self.capacity.return_value["good"] = CapacitySample("good", 1, 100, 100)
        for now in (12430, 12490, 12550):
            self.now.return_value = now
            with redirect_stdout(io.StringIO()) as output:
                self.manager.iteration()
        state = json.loads(self.path.read_text())
        self.assertEqual(state["routing_recovery"]["phase"], "verifying")
        self.assertTrue(state["routing_recovery"]["attempted"])
        self.assertEqual(state["last_decision_target"], "good")
        self.assertEqual(state["live_challenger_streak"], 3)
        self.assertEqual(state["last_switch_at"], 12430)
        self.assertEqual(state["current_residency"]["warm_since"], 12430)
        self.assertIn("minimum warm time", output.getvalue())
        self.assertIn("Highest raw score", output.getvalue())
        self.assertIn("score checks continue", output.getvalue())
        self.assertNotIn("no fresh ranking", output.getvalue())
        self.assertEqual(len(state["pressure_history"]["better"]), 4)
        self.launch.assert_called_once()  # Cold start only; the new warm time still blocks switching.
        self.stop.assert_called_once()
        before = self.send.call_count
        self.manager.probe_if_due()
        self.assertEqual(self.send.call_count, before)  # Recovery owns the probes until verification ends.

    def test_disabled_recovery_cannot_bypass_saved_verification(self):
        self.restart()
        original = self.path.read_text()
        self.capacity.reset_mock()
        for apply, enabled in ((False, True), (True, False)):
            self.args.apply, self.args.recover_routing = apply, enabled
            self.manager = Manager(self.args)
            with redirect_stdout(io.StringIO()) as output:
                self.manager.iteration()
            self.assertIn("saved recovery is paused", output.getvalue())
            self.assertEqual(self.path.read_text(), original)
            self.capacity.assert_not_called()
        self.launch.assert_called_once()

    def test_normal_switch_during_verification_does_not_rearm_shutdown(self):
        self.restart()
        self.new_winner()
        self.args.min_warm_time = 0
        self.args.switch_after_checks = 1
        self.args.pricing_refresh = 60
        self.now.return_value = 12500
        with redirect_stdout(io.StringIO()):
            self.manager.iteration()
        state = json.loads(self.path.read_text())
        self.assertEqual(state["pending_switch"]["target"], "better")
        recovery, _ = self.tick(12515)
        self.assertEqual(recovery["phase"], "locked")
        self.assertTrue(recovery["attempted"])
        self.stop.assert_called_once()
        self.assertEqual(self.launch.call_count, 2)  # One recovery start, one allowed normal switch.

    def test_zero_scores_use_normal_initial_tie_order(self):
        self.stopped()
        self.new_winner()
        self.args.model = ["better", "good"]
        self.capacity.return_value = {name: CapacitySample(name, 1, 0, 0) for name in ("good", "better")}
        self.manager = Manager(self.args)
        self.tick(12415)
        self.assertEqual(self.launch.call_args.args[1], "better")

    def test_recovery_does_not_fetch_selection_data_until_full_offline_wait(self):
        self.stopped()
        self.capacity.assert_not_called()
        self.prices.assert_not_called()
        self.tick(12414)
        self.capacity.assert_not_called()
        self.prices.assert_not_called()
        self.tick(12429)
        self.capacity.assert_called_once()
        self.prices.assert_called_once()
        self.launch.assert_called_once()


    def test_clock_jump_cannot_shorten_offline_wait(self):
        self.stopped()
        self.monotonic.side_effect = None
        self.monotonic.return_value = 11600
        self.tick(20000)
        self.launch.assert_not_called()
        self.manager = Manager(self.args)
        self.monotonic.return_value = 500
        recovery, _ = self.tick(20015)
        self.assertEqual(recovery["restart_at"], 20915)
        self.launch.assert_not_called()

    def test_offline_timer_begins_after_slow_stop_verification(self):
        self.trigger()
        self.daemon.return_value = None
        self.alive.return_value = False
        def service_check():
            self.now.return_value += 20
            return {}
        self.services.side_effect = service_check
        recovery, _ = self.tick(11515)
        self.assertEqual(recovery["restart_at"], 12435)
        self.assertEqual(recovery["stopped_monotonic"], 11535)

    def test_warmup_restarts_minimum_warm_time_and_other_provider_success_is_unconfirmed(self):
        recovery, _ = self.restart()
        self.assertEqual(recovery["phase"], "verifying")
        state = json.loads(self.path.read_text())
        self.assertEqual(state["last_switch_at"], 12430)
        self.assertNotIn("pending_switch", state)
        self.send.side_effect = lambda *args: self.ok()
        recovery, output = self.tick(12610)
        self.assertEqual(recovery["phase"], "verifying")
        self.assertIn("this provider remains unconfirmed", output)
        recovery, _ = self.tick(13330)
        self.assertEqual(recovery["phase"], "locked")
        self.assertTrue(recovery["attempted"])
        self.manager = Manager(self.args)
        self.tick(20000)
        self.stop.assert_called_once()
        self.launch.assert_called_once()

    def test_only_local_network_counter_confirms_and_rearms(self):
        self.restart()
        self.daemon.return_value = replace(self.warm, requests_served=1)
        recovery, output = self.tick(12445)
        self.assertEqual(recovery["phase"], "monitoring")
        self.assertFalse(recovery["attempted"])
        self.assertIn("network requests reached this provider", output)

    def test_requests_arriving_during_failure_checks_cancel_recovery(self):
        self.tick(10000)
        self.tick(10900)
        self.daemon.return_value = replace(self.warm, requests_served=1)
        recovery, _ = self.tick(10915)
        self.assertEqual(recovery["phase"], "monitoring")
        self.assertEqual(recovery["failures"], 0)
        self.stop.assert_not_called()

    def test_requests_or_busy_state_after_probe_or_discovery_prevent_stop(self):
        for when in ("local", "production", "discovery"):
            for traffic in (False, True):
                with self.subTest(when=when, traffic=traffic):
                    self.path.unlink(missing_ok=True)
                    self.manager = Manager(self.args)
                    self.daemon.return_value = self.warm
                    self.send.side_effect = lambda kind, *a: self.ok() if kind == "local" else self.failure()
                    self.discovery.side_effect = None
                    self.tick(10000)
                    self.tick(10900)
                    self.tick(11200)
                    def change(*args):
                        self.daemon.return_value = replace(self.warm, requests_served=1) if traffic else replace(self.warm, inference_active=True)
                        return {"good"}
                    if when == "discovery":
                        self.discovery.side_effect = change
                    else:
                        def reply(kind, *args):
                            if kind == when:
                                change()
                            return self.ok() if kind == "local" else self.failure()
                        self.send.side_effect = reply
                    self.tick(11500)
                    self.stop.assert_not_called()

    def test_other_api_failures_never_count_as_routing_failure(self):
        responses = [self.ok(), {"outcome": "FAILED", "http_status": 503, "error": "model_not_loaded"},
                     {"outcome": "FAILED", "http_status": None, "error": "timeout"}]
        responses += [{**self.failure(), "http_status": code} for code in (200, 401, 403, 429, 500)]
        responses += [{**self.failure(), "error_code": code} for code in ("model_unavailable", "rate_limit_exceeded")]
        for response in responses:
            with self.subTest(response=response):
                self.path.unlink(missing_ok=True)
                self.manager = Manager(self.args)
                self.send.side_effect = lambda kind, *a: self.ok() if kind == "local" else response
                recovery, _ = self.trigger()
                self.assertEqual(recovery["failures"], 0)
                self.stop.assert_not_called()

    def test_local_failure_and_missing_token_cannot_trigger_stop(self):
        self.send.side_effect = lambda *a: self.failure()
        recovery, _ = self.trigger()
        self.assertEqual(recovery["failures"], 0)
        self.assertTrue(all(call.args[0] == "local" for call in self.send.call_args_list))
        self.send.reset_mock()
        self.token.unlink()
        self.tick(11800)
        self.send.assert_not_called()
        self.stop.assert_not_called()

    def test_process_model_and_reconnect_changes_reset_observation(self):
        for changed in (replace(self.warm, pid=999), replace(self.warm, started_at=10905),
                        replace(self.warm, warm_models=("different",)), replace(self.warm, reconnect_count=2),
                        replace(self.warm, fresh=False), replace(self.warm, requests_served=None)):
            with self.subTest(changed=changed):
                self.path.unlink(missing_ok=True)
                self.manager = Manager(self.args)
                self.daemon.return_value = self.warm
                self.tick(10000)
                self.tick(10900)
                self.daemon.return_value = changed
                recovery, _ = self.tick(10915)
                self.assertEqual(recovery["failures"], 0)
                self.assertEqual(recovery["next_check_at"], 11815)
                self.stop.assert_not_called()

    def test_long_manager_pause_breaks_consecutive_checks(self):
        self.tick(10000)
        self.tick(10900)
        self.manager = Manager(self.args)
        recovery, _ = self.tick(15000)
        self.assertEqual(recovery["failures"], 1)
        self.stop.assert_not_called()

    def test_ignore_or_exclusion_prevents_checks_and_restart(self):
        for change in (lambda: self.manager.ignored.add("good"), lambda: setattr(self.args, "model", ["other"])):
            self.path.unlink(missing_ok=True)
            self.args.model = None
            self.manager = Manager(self.args)
            self.daemon.return_value = self.warm
            self.services.return_value = {"io.darkbloom.provider": 123}
            self.alive.return_value = True
            self.stopped()
            change()
            recovery, _ = self.tick(12415)
            self.assertEqual(recovery["phase"], "offline")
            self.launch.assert_not_called()
        self.path.unlink()
        self.manager = Manager(self.args)
        self.daemon.return_value = self.warm
        self.send.reset_mock()
        self.trigger()
        self.send.assert_not_called()

    def test_removed_model_or_changed_config_cancels_restart(self):
        for change in (lambda: self.config.write_text('[backend]\npreload_models = ["other"]\n'),
                       lambda: setattr(self.discovery, "return_value", set())):
            self.path.unlink(missing_ok=True)
            self.config.write_text('[backend]\npreload_models = ["good"]\n')
            self.discovery.return_value = {"good"}
            self.manager = Manager(self.args)
            self.daemon.return_value = self.warm
            self.services.return_value = {"io.darkbloom.provider": 123}
            self.alive.return_value = True
            self.stopped()
            change()
            recovery, _ = self.tick(12415)
            self.assertEqual(recovery["phase"], "locked" if self.discovery.return_value else "offline")
            self.launch.assert_not_called()

    def test_stop_and_start_interruptions_are_not_replayed(self):
        self.stop.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.trigger()
        self.manager = Manager(self.args)
        recovery, _ = self.tick(11560)
        self.assertEqual(recovery["phase"], "locked")
        self.stop.assert_called_once()
        self.path.unlink()
        self.stop.side_effect = None
        self.manager = Manager(self.args)
        self.stopped()
        self.launch.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.tick(12415)
        self.manager = Manager(self.args)
        self.tick(12430)
        recovery, _ = self.tick(12600)
        self.assertEqual(recovery["phase"], "locked")
        self.launch.assert_called_once()
        self.assertIn("pending_switch", json.loads(self.path.read_text()))

    def test_timed_out_stop_can_still_be_confirmed_without_retry(self):
        self.stop.side_effect = subprocess.TimeoutExpired("darkbloom", 60)
        recovery, _ = self.stopped()
        self.assertEqual(recovery["phase"], "offline")
        self.stop.assert_called_once()

    def test_write_failure_prevents_stop_or_start(self):
        original = manager_module.write_json_atomic
        def write(path, state):
            if state.get("routing_recovery", {}).get("phase") in {"stopping", "starting"}:
                raise OSError("disk full")
            original(path, state)
        with patch("warm_model_manager.write_json_atomic", side_effect=write):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.trigger()
        self.stop.assert_not_called()
        self.path.unlink()
        self.manager = Manager(self.args)
        self.stopped()
        with patch("warm_model_manager.write_json_atomic", side_effect=write):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.tick(12415)
        self.launch.assert_not_called()

    def test_pause_survives_disabled_flag_dry_run_and_manager_restart(self):
        self.stopped()
        state_before = self.path.read_text()
        self.args.recover_routing = False
        self.manager = Manager(self.args)
        with redirect_stdout(io.StringIO()) as output:
            self.manager.iteration()
            self.manager.probe_if_due()
        self.assertIn("saved recovery is paused", output.getvalue())
        self.assertEqual(self.path.read_text(), state_before)
        self.args.recover_routing = True
        self.args.apply = False
        self.tick(13000)
        self.launch.assert_not_called()
        self.assertEqual(self.path.read_text(), state_before)

    def test_external_start_aborts_offline_recovery(self):
        self.stopped()
        self.daemon.return_value = replace(self.warm, pid=999)
        recovery, _ = self.tick(11800)
        self.assertEqual(recovery["phase"], "locked")
        self.stop.assert_called_once()
        self.launch.assert_not_called()

    def test_changed_paths_and_corrupt_state_fail_closed(self):
        self.stopped()
        self.args.config = self.directory / "other.toml"
        self.manager = Manager(self.args)
        with self.assertRaisesRegex(ValueError, "paths changed"):
            self.tick(12415)
        for value in ("not-json", '{"routing_recovery": "bad"}',
                      '{"routing_recovery":{"schema":1,"phase":"offline","attempted":true,"restart_at":"bad"}}'):
            self.path.write_text(value)
            self.manager = Manager(self.args)
            with self.assertRaises(ValueError):
                self.tick(13000)
        self.launch.assert_not_called()

    def test_explicit_reset_preserves_other_state_and_refuses_active_wait(self):
        self.stopped()
        self.args.mode = "reset-recovery"
        with self.assertRaisesRegex(ValueError, "active recovery"):
            self.manager.run()
        state = json.loads(self.path.read_text())
        state["routing_recovery"]["phase"] = "locked"
        state["pressure_history"] = {"good": [{"at": 10000, "pressure": 1}]}
        self.path.write_text(json.dumps(state))
        with redirect_stdout(io.StringIO()):
            self.manager.run()
        remaining = json.loads(self.path.read_text())
        self.assertNotIn("routing_recovery", remaining)
        self.assertEqual(remaining["pressure_history"], state["pressure_history"])
        self.launch.assert_not_called()

    def test_daemon_reader_accepts_only_real_network_counters(self):
        with patch("warm_model_manager.read_json", return_value={
                "pid": 123, "written_at": 10000, "stats": {"requests_served": 12},
                "connectivity": {"reconnect_count": 3}}):
            daemon = read_daemon_state(self.args.daemon_state, now=10000)
        self.assertEqual(daemon.requests_served, 12)
        self.assertEqual(daemon.reconnect_count, 3)
        for value in (None, -1, True, "12", 1.5):
            self.assertIsNone(manager_module.nonnegative_int(value))

    def test_stop_helper_only_targets_matching_launchd_pid(self):
        # Call the actual implementation while keeping all OS commands mocked.
        with patch("warm_model_manager.provider_services", return_value={"io.darkbloom.provider": 123}), \
                patch("warm_model_manager.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            ORIGINAL_STOP_PROVIDER("/custom/darkbloom", 123)
            self.assertEqual(run.call_args.args[0], ["/custom/darkbloom", "stop"])
        for services in ({}, {"io.darkbloom.provider": 999},
                         {"io.darkbloom.provider": 123, "dev.darkbloom.provider": 456}):
            with patch("warm_model_manager.provider_services", return_value=services), \
                    patch("warm_model_manager.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "match"):
                    ORIGINAL_STOP_PROVIDER("darkbloom", 123)
                run.assert_not_called()

    def test_service_lookup_distinguishes_absent_unknown_and_multiple_services(self):
        absent = subprocess.CompletedProcess([], 113, "", "Could not find service in domain")
        loaded = subprocess.CompletedProcess([], 0, "service = {\n    pid = 123\n}\n", "")
        with patch("warm_model_manager.subprocess.run", side_effect=[loaded, absent]) as run:
            self.assertEqual(ORIGINAL_PROVIDER_SERVICES(), {"io.darkbloom.provider": 123})
            self.assertEqual(run.call_args_list[0].args[0][:2], ["/bin/launchctl", "print"])
        with patch("warm_model_manager.subprocess.run", return_value=absent):
            self.assertEqual(ORIGINAL_PROVIDER_SERVICES(), {})
        with patch("warm_model_manager.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "permission denied")):
            with self.assertRaisesRegex(RuntimeError, "cannot verify"):
                ORIGINAL_PROVIDER_SERVICES()

    def test_traffic_or_cancellation_during_service_lookup_prevents_stop(self):
        for change in (lambda: setattr(self.daemon, "return_value", replace(self.warm, inference_active=True)),
                       lambda: setattr(self.daemon, "return_value", replace(self.warm, requests_served=1)),
                       lambda: self.manager.stop(None, None)):
            self.path.unlink(missing_ok=True)
            self.manager = Manager(self.args)
            self.daemon.return_value = self.warm
            self.stop.side_effect = ORIGINAL_STOP_PROVIDER
            def check_services():
                change()
                return {"io.darkbloom.provider": 123}
            self.services.side_effect = check_services
            with patch("warm_model_manager.subprocess.run") as run:
                self.trigger()
            run.assert_not_called()

    def test_real_http_error_parser_preserves_only_structured_failure_code(self):
        with patch("warm_model_manager.build_opener") as build:
            build.return_value.open.side_effect = HTTPError(
                PROD_PROBE_URL, 503, "Unavailable", {},
                io.BytesIO(b'{"error":{"code":"model_not_loaded","message":"not available"}}'))
            result = ORIGINAL_SEND_PROBE("production", "good", PROD_PROBE_URL, "fixture-secret")
        self.assertEqual(result["error_code"], "model_not_loaded")
        self.assertEqual(result["http_status"], 503)
        self.assertEqual(result["outcome"], "FAILED")
        result = probe_response_details(b'{"error":{"code":"secret","message":"secret"}}', "secret", 503)
        self.assertNotIn("secret", json.dumps(result))

    def test_recovery_checks_run_between_long_score_intervals(self):
        self.args.check_every = 3600
        times = []
        def recover():
            times.append(self.now.return_value)
            if len(times) == 3:
                self.manager.stop(None, None)
        def advance(seconds):
            self.now.return_value += seconds
        with patch.object(self.manager, "iteration") as iteration, \
                patch.object(self.manager, "recovery_if_due", side_effect=recover), \
                patch.object(self.manager, "probe_if_due"), \
                patch("warm_model_manager.time.sleep", side_effect=advance), redirect_stdout(io.StringIO()):
            self.manager.run()
        iteration.assert_called_once()
        self.assertLess(times[-1] - times[0], self.args.check_every)

    def test_active_recovery_cannot_be_bypassed_by_normal_iteration(self):
        self.stopped()
        state = json.loads(self.path.read_text())
        with patch("warm_model_manager.fetch_capacity") as capacity, redirect_stdout(io.StringIO()) as output:
            self.manager.iteration()
            self.manager.probe_if_due()
        capacity.assert_not_called()
        self.launch.assert_not_called()
        self.assertIn("15m remaining", output.getvalue())
        self.assertEqual(json.loads(self.path.read_text()), state)

    def test_post_start_timeout_is_not_retried_but_warm_state_can_confirm(self):
        self.stopped()
        self.launch.side_effect = subprocess.TimeoutExpired("darkbloom", 300)
        self.tick(12415)
        self.daemon.return_value = replace(self.warm, pid=124, started_at=12415, requests_served=1)
        self.manager = Manager(self.args)
        recovery, _ = self.tick(12430)
        self.assertFalse(recovery["attempted"])
        self.assertEqual(recovery["phase"], "monitoring")
        self.assertNotIn("pending_switch", json.loads(self.path.read_text()))
        self.launch.assert_called_once()


class TimelineReportTests(unittest.TestCase):
    def setUp(self):
        self.now = 10000
        self.daemon = LocalDaemonState("warm", ("warm",), False, 123, 9000, True)
        self.decision = Decision("warm", "candidate passes; 2/3 consecutive checks passed", "candidate", 2)
        self.state = {
            "probes": {"next_kind": "local", "next_at": 10800},
            "switch_probe": {"target": "warm", "pid": 123, "due_at": 10180},
            "routing_recovery": {"schema": 1, "phase": "monitoring", "attempted": False,
                                 "next_check_at": 10090, "reason": "network requests reached this provider"},
        }

    def timeline(self, **overrides):
        values = dict(state=self.state, daemon=self.daemon, current="warm", decision=self.decision,
                      now=self.now, next_check=10060, interval=60, confirmations=3, minimum=1120,
                      hourly_probes=True, recover_routing=True, apply=True)
        values.update(overrides)
        return report_timeline(**values)

    def render(self, **overrides):
        values = dict(
            models=["warm", "candidate", "ignored", "missing"],
            samples={model: CapacitySample(model, 1, 1, 1) for model in ("warm", "candidate", "ignored")},
            averages={"warm": 1, "candidate": 3, "ignored": 20},
            weights={"candidate": 1.25},
            prices={model: ModelPrice(1, 1) for model in ("warm", "candidate", "ignored")},
            scores={"warm": 1, "candidate": 3.75, "ignored": 20},
            pressure_history={"warm": [{"at": 10000, "pressure": 1}]},
            manager_state={**self.state, "last_score_snapshot": {"models": {
                "ignored": {"ignored": True, "status": ["IGNORED"]},
                "missing": {"auto_ignored": True, "status": ["AUTO-IGNORED", "capacity unavailable"]},
            }}},
            daemon=self.daemon, current="warm", decision=self.decision, apply=True,
            interval_seconds=60, history_size=15, confirmations=3, now=self.now,
            residency=(1000, 9000), ignored_models={"ignored"}, eligible_models=["warm", "candidate"],
            blocked=None, improvement_percent=25, hide_ignored=False, hourly_probes=True,
            recover_routing=True, next_check=10060, minimum=1120,
        )
        values.update(overrides)
        with redirect_stdout(io.StringIO()) as output:
            manager_module.print_report(**values)
        return output.getvalue()

    def test_every_timer_sorted_without_losing_either_regular_probe(self):
        before = json.dumps(self.state, sort_keys=True)
        events, notes = self.timeline()
        self.assertEqual([event.at for event in events], [10060, 10090, 10120, 10180, 10800, 12600])
        self.assertIn("check 3 of 3", events[0].text)
        self.assertIn("recovery check", events[1].text)
        self.assertIn("earliest switch", events[2].text)
        self.assertIn("if checks still pass", events[2].text)
        self.assertIn("after switch to warm", events[3].text)
        self.assertIn("local endpoint", events[4].text)
        self.assertIn("production self-route", events[5].text)
        self.assertEqual(notes, [])
        self.assertEqual(json.dumps(self.state, sort_keys=True), before)

    def test_final_passing_check_and_earliest_switch_share_one_event(self):
        events, _ = self.timeline(minimum=0)
        score_events = [event for event in events if event.attention]
        self.assertEqual(len(score_events), 1)
        self.assertIn("check 3 of 3", score_events[0].text)
        self.assertIn("earliest switch if checks still pass", score_events[0].text)

    def test_no_false_switch_time_for_below_margin_or_pending_warmup(self):
        for decision, pending in ((Decision("warm", "below margin", "candidate", 0), False),
                                  (Decision("candidate", "loading", warming=True), False),
                                  (self.decision, True)):
            state = dict(self.state)
            if pending:
                state["pending_switch"] = {"target": "candidate"}
            events, _ = self.timeline(state=state, decision=decision)
            self.assertFalse(any("earliest" in event.text for event in events))
            if pending or decision.warming:
                self.assertTrue(any("check pending model warm-up" in event.text for event in events))

    def test_post_switch_without_confirmed_time_is_a_relative_note(self):
        state = dict(self.state)
        state.pop("switch_probe")
        events, notes = self.timeline(state=state)
        self.assertFalse(any("after switch" in event.text for event in events))
        self.assertIn("+3m after a switch is confirmed warm", " ".join(notes))

    def test_once_has_no_recurring_or_future_requests(self):
        events, notes = self.timeline(next_check=None)
        self.assertEqual(events, [])
        self.assertIn("one check only", notes[0])
        self.state["probes"]["next_at"] = 9900
        self.state["switch_probe"]["due_at"] = 9800
        events, _ = self.timeline(next_check=None)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].at, 9800)

    def test_once_same_time_probe_tie_matches_execution_priority(self):
        self.state["probes"]["next_at"] = 10000
        self.state["switch_probe"]["due_at"] = 10000
        events, _ = self.timeline(next_check=None)
        self.assertEqual(len(events), 1)
        self.assertIn("after switch", events[0].text)
        self.assertEqual(manager_module.next_probe_event(self.state, 10000), ("production", 10000, True))

    def test_recovery_does_not_promise_checks_after_its_verification_deadline(self):
        self.state["routing_recovery"].update(phase="verifying", verify_by=10080)
        events, _ = self.timeline()
        self.assertEqual([event.at for event in events], [10060, 10080, 10120])
        self.assertFalse(any("recovery check," in event.text for event in events))

    def test_dry_run_shows_conditional_selection_but_no_sending_or_recovery(self):
        events, notes = self.timeline(apply=False)
        self.assertEqual([event.at for event in events], [10060, 10120])
        self.assertIn("earliest would switch", events[1].text)
        self.assertIn("DRY RUN", " ".join(notes))

    def test_recovery_offline_has_restart_but_no_switch_or_probe(self):
        self.state["routing_recovery"].update(phase="offline", restart_at=10900, model="warm")
        events, notes = self.timeline()
        self.assertEqual([event.at for event in events], [10060, 10900])
        self.assertIn("score checks paused", events[0].text)
        self.assertIn("15m remaining", events[1].text)
        self.assertIn("probes paused", " ".join(notes))
        text = self.render()
        self.assertIn("RECOVERY OFFLINE", text)
        self.assertIn("no fresh ranking", text)
        self.assertNotIn("Highest raw score", text)

    def test_offline_recovery_timeline_shows_fresh_selection_and_retry_time(self):
        self.state["routing_recovery"].update(phase="offline", restart_at=9900,
                                             selection_check_at=10080, model="old-model")
        events, _ = self.timeline()
        self.assertEqual([event.at for event in events], [10060, 10080])
        self.assertIn("check fresh scores, then start the highest eligible model", events[1].text)
        self.assertNotIn("old-model", events[1].text)

    def test_saved_recovery_without_enabled_live_run_promises_no_restart(self):
        self.state["routing_recovery"].update(phase="offline", restart_at=10900, model="warm")
        for apply, enabled in ((False, True), (True, False)):
            events, notes = self.timeline(apply=apply, recover_routing=enabled)
            self.assertEqual(len(events), 1)
            self.assertNotIn("restart warm", events[0].text)
            self.assertIn("saved recovery is paused", " ".join(notes))

    def test_recovery_verification_deadline_is_in_time_order(self):
        self.state["routing_recovery"].update(phase="verifying", verify_by=10200)
        events, _ = self.timeline()
        self.assertEqual([event.at for event in events], [10060, 10090, 10120, 10200])
        self.assertIn("no repeat recovery shutdown", events[-1].text)

    def test_long_minimum_rounds_to_real_check_cadence(self):
        at = switch_ready_at("warm", self.decision, {}, self.daemon,
                             10000, 10042, 60, 3, 1180)
        self.assertEqual(at, 10222)
        # An immediate approved switch stays conditional on the provider being idle.
        events, _ = self.timeline(daemon=replace(self.daemon, inference_active=True),
                                 decision=Decision("candidate", "passed", "candidate", 3), minimum=0)
        self.assertEqual(events[0].at, 10000)
        self.assertIn("online and idle", events[0].text)

    def test_distance_uses_switch_cost_without_discounting_the_warm_score(self):
        self.assertEqual(score_distance(2, 1, 300, 3600), "+83%")
        self.assertEqual(score_distance(0.5, 1, 300, 3600), "-54%")
        self.assertEqual(score_distance(1, 1, 0, 3600), "+0%")
        self.assertEqual(score_distance(2, 0, 300, 3600), "> zero")
        self.assertEqual(score_distance(0, 0, 300, 3600), "equal zero")
        self.assertEqual(score_distance(None, 1, 300, 3600), "N/A")
        self.assertEqual(score_distance(1, None, 300, 3600), "N/A")

    def test_ladder_sorted_with_ignored_leader_and_missing_data_last(self):
        text = self.render()
        rows = [line for line in text.splitlines() if "avg " in line]
        self.assertIn("ignored", rows[0])
        self.assertIn("IGNORED", rows[0])
        self.assertIn("candidate", rows[1])
        self.assertIn("+244%", rows[1])
        self.assertIn("×1.25", rows[1])
        self.assertIn("* warm", rows[2])
        self.assertIn("missing", rows[3])
        self.assertIn("N/A", rows[3])
        self.assertIn("LIVE, KEEP", text)
        self.assertIn("Highest raw score: ignored [IGNORED]", text)
        self.assertNotIn("earliest switch to ignored", text)
        self.assertNotIn("IN$/M", text)
        self.assertNotIn("OUT$/M", text)
        self.assertNotIn("\033", text)
        self.assertEqual([line.split()[0] for line in text.splitlines()
                          if line.startswith(("now ", "next ", "score ", "sources "))],
                         ["now", "next", "score", "sources"])

    def test_full_restores_all_columns_inside_same_timeline(self):
        text = self.render(columns="full")
        header = next(line for line in text.splitlines() if line.startswith("score "))
        for column in ("NOW", "AVG 15m", "N", "IN$/M", "OUT$/M", "BLEND$/M", "WEIGHT", "SCORE", "STATUS"):
            self.assertIn(column, header)
        self.assertIn("probe, local endpoint", text)
        self.assertIn("recovery check", text)
        self.assertNotIn("VS WARM", text)

    def test_hidden_rows_do_not_affect_ladder_bar_or_leader(self):
        text = self.render(hide_ignored=True)
        self.assertNotIn("  ignored ", text)
        self.assertNotIn("  missing ", text)
        self.assertIn("1 ignored hidden; 1 auto-ignored hidden", text)
        self.assertIn("Highest raw score (shown models): candidate", text)
        candidate = next(line for line in text.splitlines() if "avg " in line and "candidate" in line)
        self.assertIn("############", candidate)

    def test_no_single_fresh_warm_model_means_no_percentage_baseline(self):
        for daemon in (None, replace(self.daemon, fresh=False), replace(self.daemon, warm_models=("warm", "other"))):
            text = self.render(daemon=daemon)
            rows = [line for line in text.splitlines() if "avg " in line]
            self.assertTrue(all("N/A" in line for line in rows))
            self.assertNotIn("+244%", text)

    def test_dates_are_shown_for_events_across_midnight(self):
        from datetime import datetime
        now = datetime(2026, 9, 16, 23, 59, 40).timestamp()
        self.assertEqual(manager_module.timeline_clock(now, now), "23:59:40")
        self.assertEqual(manager_module.timeline_clock(now + 60, now), "Sep 17 00:00:40")

    def test_color_only_on_terminal_and_no_color_respected(self):
        with patch("warm_model_manager.sys.stdout.isatty", return_value=True), \
                patch.dict("os.environ", {"TERM": "xterm"}, clear=True):
            self.assertEqual(manager_module.terminal_style("warm", "warm"), "\033[1mwarm\033[0m")
            with patch.dict("os.environ", {"NO_COLOR": ""}):
                self.assertEqual(manager_module.terminal_style("warm", "warm"), "warm")
            with patch.dict("os.environ", {"TERM": "dumb"}):
                self.assertEqual(manager_module.terminal_style("warm", "warm"), "warm")

    def test_columns_defaults_and_validation(self):
        self.assertEqual(build_parser().parse_args([]).columns, "ladder")
        self.assertEqual(build_parser().parse_args(["--columns", "full"]).columns, "full")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--columns", "unknown"])


class ManagerIntegrationTests(unittest.TestCase):
    """Exercise full ticks using fake network, discovery, daemon and launch calls."""
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "state.json"
        self.args = build_parser().parse_args([
            "once", "--apply", "--state", str(self.path),
            "--ignore-model", IGNORED, "--check-every", "60", "--average-samples", "5",
            "--switch-after-checks", "3", "--min-warm-time", "1800",
        ])
        self.manager = Manager(self.args)
        self.local = self.mock("local_model_ids", return_value={"good", IGNORED})
        self.catalog = self.mock("catalog_model_ids", return_value=set())
        self.capacity = self.mock("fetch_capacity", return_value=self.samples(good=1, **{IGNORED: 100}))
        self.prices = self.mock("fetch_model_prices", return_value=(
            {"good": ModelPrice(0.5, 0.5), IGNORED: ModelPrice(0.75, 0.75)}, ModelPrice(0.2, 0.2),
        ))
        self.daemon = self.mock("read_daemon_state", return_value=LocalDaemonState(
            current_model="good", warm_models=("good",), inference_active=False,
            pid=123, started_at=100, fresh=True,
        ))
        self.launch = self.mock("switch_model")
        self.mock("subprocess.run", side_effect=AssertionError("unexpected real command"))
        self.clock = self.mock("time.time", return_value=10000)

    def mock(self, name, **kwargs):
        patcher = patch("warm_model_manager." + name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    @staticmethod
    def samples(**pressures):
        return {model: CapacitySample(model, 1, int(pressure), pressure)
                for model, pressure in pressures.items()}

    def tick(self, now=10000):
        self.clock.return_value = now
        output = io.StringIO()
        with redirect_stdout(output):
            self.manager.iteration()
        return json.loads(self.path.read_text()), output.getvalue()

    def save(self, state):
        self.path.write_text(json.dumps(state))

    def test_columns_only_changes_presentation_not_selection_or_saved_state(self):
        self.local.return_value = {"good", "candidate"}
        self.capacity.return_value = self.samples(good=1, candidate=10)
        self.args.mode = "run"
        baseline, _ = self.tick()
        results = []
        for columns in ("ladder", "full"):
            self.save(baseline)
            self.args.columns = columns
            self.manager = Manager(self.args)
            self.launch.reset_mock()
            state, output = self.tick(10060)
            results.append((state, self.launch.call_args_list))
            self.assertIn("IN$/M" if columns == "full" else "VS WARM", output)
        self.assertEqual(results[0], results[1])

    def test_normal_checks_preserve_probe_schedule_and_scoring_state(self):
        state, _ = self.tick()
        schedule = {"next_kind": "production", "next_at": 11800,
                    "last_result": {"kind": "local", "at": 10000, "http_status": 200}}
        state["probes"] = schedule
        self.save(state)
        updated, _ = self.tick(10060)
        self.assertEqual(updated["probes"], schedule)
        self.assertEqual(updated["last_score_snapshot"]["models"]["good"]["score"],
                         state["last_score_snapshot"]["models"]["good"]["score"])
        self.launch.assert_not_called()

    def test_switch_probe_starts_at_confirmed_warmup_and_survives_checks_and_restart(self):
        self.args.mode = "run"
        self.args.hourly_probes = True
        state, _ = self.tick()
        self.assertNotIn("switch_probe", state)  # Already warm at startup.
        state["probes"] = {"next_kind": "production", "next_at": 11800}
        state["pending_switch"] = {"target": "good", "warm_models": ["good"], "command_at": 10000}
        self.save(state)
        warm = self.daemon.return_value
        self.daemon.return_value = replace(warm, warm_models=())
        waiting, _ = self.tick(10060)
        self.assertNotIn("switch_probe", waiting)
        self.daemon.return_value = warm
        self.manager.next_probe_at = 11800
        confirmed, report = self.tick(10120)
        self.assertEqual(confirmed["switch_probe"]["due_at"], 10300)
        self.assertEqual(confirmed["switch_probe"]["target"], "good")
        self.assertEqual(self.manager.next_probe_at, 0)
        self.assertIn("probe, production self-route after switch", report)
        self.assertIn("probe, production self-route", report)
        self.manager = Manager(self.args)
        after, _ = self.tick(10180)
        self.assertEqual(after["switch_probe"], confirmed["switch_probe"])
        self.assertEqual(after["probes"], state["probes"])
        with patch("warm_model_manager.probe_credentials", return_value=(PROD_PROBE_URL, "fixture")), \
                patch("warm_model_manager.send_probe", return_value={"http_status": 200, "outcome": "SUCCESS"}), \
                redirect_stdout(io.StringIO()):
            self.clock.return_value = 10300
            self.manager.probe_if_due()
        after, _ = self.tick(10360)
        self.assertNotIn("switch_probe", after)
        self.assertEqual(after["last_switch_probe_result"]["outcome"], "SUCCESS")
        self.launch.assert_not_called()

    def test_switch_probe_is_not_queued_when_disabled_or_dry(self):
        state, _ = self.tick()
        state["pending_switch"] = {"target": "good", "warm_models": ["good"], "command_at": 10000}
        for enabled, apply in ((False, True), (True, False)):
            self.args.hourly_probes, self.args.apply = enabled, apply
            self.save(state)
            after, _ = self.tick(10060)
            self.assertNotIn("switch_probe", after)
        self.launch.assert_not_called()

    def test_switch_enables_local_endpoint_only_with_probes_and_cancels_previous_extra(self):
        state, _ = self.tick()
        self.daemon.return_value = replace(self.daemon.return_value, warm_models=())
        state["switch_probe"] = {"target": "previous", "pid": 123, "due_at": 10180}
        for enabled in (False, True):
            self.args.hourly_probes = enabled
            self.save(state)
            self.launch.reset_mock()
            with patch("warm_model_manager.local_endpoint_start_flags", return_value=["--local-endpoint"]) as flags:
                after, _ = self.tick(10060)
            self.assertNotIn("switch_probe", after)
            self.assertEqual(after["pending_switch"]["target"], "good")
            self.assertEqual(self.launch.call_args.kwargs, {"local_flags": ["--local-endpoint"]} if enabled else {})
            self.assertEqual(flags.call_count, int(enabled))

    def test_once_sends_one_due_probe_after_confirming_current_model(self):
        self.args.hourly_probes = True
        with patch("warm_model_manager.probe_credentials", return_value=("http://127.0.0.1:8000/v1/chat/completions", "fixture")), \
                patch("warm_model_manager.send_probe", return_value={"http_status": 200, "outcome": "SUCCESS", "status": "completion received"}) as send, \
                redirect_stdout(io.StringIO()):
            self.manager.run()
        send.assert_called_once()
        self.assertEqual(send.call_args.args[:2], ("local", "good"))
        self.assertEqual(json.loads(self.path.read_text())["probes"]["next_at"], 11800)
        self.launch.assert_not_called()

    def test_every_report_shows_next_two_probes_after_restart_without_sending(self):
        self.args.mode = "run"
        self.args.hourly_probes = True
        state, initial = self.tick()
        self.assertIn("probe, local endpoint", initial)
        self.assertIn("due now", initial)
        self.assertIn("probe, production self-route", initial)
        state["probes"] = {"next_kind": "production", "next_at": 11800}
        self.save(state)
        self.manager = Manager(self.args)
        for now in (10060, 10120):
            _, report = self.tick(now)
            self.assertIn("probe, production self-route", report)
            self.assertIn("probe, local endpoint", report)
            self.assertNotIn("SENDING", report)
        self.launch.assert_not_called()

    def test_ignored_leader_has_all_math_persisted_but_is_never_selected(self) -> None:
        self.args.hide_ignored = True
        self.prices.return_value[0][IGNORED] = ModelPrice(0.15, 2.0)
        state, report = self.tick()
        row = state["last_score_snapshot"]["models"][IGNORED]
        self.assertTrue(row["ignored"])
        self.assertFalse(row["eligible"])
        self.assertEqual(row["now_pressure"], 100)
        self.assertEqual(row["average_pressure"], 100)
        self.assertEqual(row["input_usd_per_million"], 0.15)
        self.assertEqual(row["output_usd_per_million"], 2.0)
        self.assertAlmostEqual(row["blended_usd_per_million"], 0.4275)
        self.assertEqual(row["weight"], 1)
        self.assertAlmostEqual(row["score"], 42.75)
        self.assertEqual(state["pressure_history"][IGNORED], [{"at": 10000, "pressure": 100}])
        self.assertEqual(state["last_decision_target"], "good")
        self.assertNotIn(IGNORED, report)
        self.assertIn("Highest raw score (shown models): good (0.500).", report)
        self.assertIn("1 shown; 1 ignored hidden; 0 auto-ignored hidden", report)
        self.assertNotIn("WOULD SWITCH", report)
        self.launch.assert_not_called()

    def test_hidden_models_do_not_widen_table_and_missing_data_stays_visible(self) -> None:
        self.args.columns = "full"
        self.args.hide_ignored = True
        hidden = "not-downloaded-" + "x" * 150
        self.catalog.return_value = {"good", "missing-data", hidden}
        self.local.return_value = {"good", "missing-data"}
        state, report = self.tick()
        self.assertNotIn(hidden, report)
        self.assertNotIn(IGNORED, report)
        self.assertTrue(state["last_score_snapshot"]["models"][hidden]["auto_ignored"])
        self.assertFalse(state["last_score_snapshot"]["models"]["missing-data"]["auto_ignored"])
        row = next(line for line in report.splitlines() if line.startswith("           missing-data"))
        self.assertIn("N/A", row)
        self.assertIn("capacity unavailable", row)
        header = next(line for line in report.splitlines() if line.startswith("score    MODEL ID"))
        self.assertEqual(header.index("NOW"), 38)
        self.assertIn("2 shown; 1 ignored hidden; 1 auto-ignored hidden", report)
        self.assertIn("\n\n         Highest raw score (shown models): good (0.500).", report)
        self.launch.assert_not_called()

    def test_multiple_ignored_ids_are_hidden_and_block_saved_launches_through_cli(self) -> None:
        ignored = {IGNORED, "second-hidden", "third-hidden"}
        self.local.return_value = {"good", *ignored}
        self.capacity.return_value = self.samples(good=1, **{model: 100 for model in ignored})
        original, _ = self.tick()
        self.daemon.return_value = LocalDaemonState(None, (), False, 123, 100, True)
        for apply, hide in ((False, False), (False, True), (True, False), (True, True)):
            for pending_target in ignored:
                with self.subTest(apply=apply, hide=hide, pending_target=pending_target):
                    self.launch.reset_mock()
                    state = dict(original)
                    state["pending_switch"] = {
                        "target": pending_target, "warm_models": [pending_target], "command_at": 10000,
                    }
                    state.update(live_challenger_model=pending_target, live_challenger_streak=100)
                    self.save(state)
                    command = [
                        "warm_model_manager.py", "once", "--state", str(self.path),
                        "--ignore-model", IGNORED, "second-hidden",
                        "--ignore", "third-hidden", IGNORED,
                        *(["--hide-ignored"] if hide else []),
                        *(["--apply"] if apply else []),
                    ]
                    output = io.StringIO()
                    with patch.object(sys, "argv", command), \
                            patch("warm_model_manager.DEFAULT_STATE_PATH", self.path), \
                            patch("warm_model_manager.signal.signal"), redirect_stdout(output):
                        self.assertEqual(main(), 0)
                    state = json.loads(self.path.read_text())
                    report = output.getvalue()
                    self.assertEqual(state["last_decision_target"], "good")
                    if hide:
                        self.assertIn("1 shown; 3 ignored hidden; 0 auto-ignored hidden", report)
                    else:
                        self.assertNotIn("ignored hidden", report)
                    for model in ignored:
                        if hide:
                            self.assertNotIn(model, report)
                        else:
                            self.assertIn("  " + model, report)
                        self.assertTrue(state["last_score_snapshot"]["models"][model]["ignored"])
                        self.assertFalse(state["last_score_snapshot"]["models"][model]["eligible"])
                        self.assertIsNotNone(state["last_score_snapshot"]["models"][model]["score"])
                    if apply:
                        self.launch.assert_called_once_with("darkbloom", "good", None, ignored)
                        self.assertEqual(state["pending_switch"]["warm_models"], ["good"])
                    else:
                        self.launch.assert_not_called()
                        self.assertNotIn("pending_switch", state)
                        self.assertIn("WOULD SWITCH", report)

    def test_default_shows_ignored_models_and_display_toggle_keeps_passing_checks(self) -> None:
        self.catalog.return_value = {"good", "candidate", "remote"}
        self.local.return_value = {"good", "candidate"}
        self.capacity.return_value = self.samples(good=1, candidate=10, remote=1000, **{IGNORED: 100})
        self.assertFalse(self.args.hide_ignored)
        for check, hide in enumerate((False, True, False), start=1):
            self.args.hide_ignored = hide
            state, report = self.tick(10000 + (check - 1) * 60)
            if hide:
                self.assertNotIn(IGNORED, report)
                self.assertNotIn("remote", report)
                self.assertIn("Highest raw score (shown models): candidate (2.000).", report)
            else:
                self.assertIn("  " + IGNORED, report)
                self.assertIn("  remote", report)
                self.assertIn("Highest raw score: remote [AUTO-IGNORED] (200.000).", report)
                self.assertIn("IGNORED; not downloaded or filtered out", report)
                self.assertNotIn("ignored hidden", report)
            for model in ("good", "candidate", "remote", IGNORED):
                self.assertEqual(len(state["pressure_history"][model]), check)
            if check < 3:
                self.assertEqual(state["live_challenger_streak"], check)
                self.launch.assert_not_called()
        self.assertEqual(state["last_decision_target"], "candidate")
        self.launch.assert_called_once_with("darkbloom", "candidate", None, {IGNORED})

    def test_blended_winner_obeys_cost_margins_and_three_checks_in_live_and_dry_run(self) -> None:
        self.args.columns = "full"
        q35, q9 = "qwen3.5-35b-a3b", "Qwen3.5-9B"
        self.local.return_value = {q35, q9}
        self.capacity.return_value = self.samples(**{q35: 1, q9: 4})
        self.prices.return_value = ({q35: ModelPrice(0.08, 0.75), q9: ModelPrice(0.08, 0.13)}, ModelPrice(0.05, 0.20))
        self.daemon.return_value = LocalDaemonState(q35, (q35,), False, 123, 100, True)
        # Output-only pricing ranks q35 first: 0.9375 versus q9's 0.52.
        # The blend gives q9 0.35 (0.32083 after switch cost) versus 0.225625.
        for apply in (False, True):
            with self.subTest(apply=apply):
                self.path.unlink(missing_ok=True)
                self.launch.reset_mock()
                self.args.apply = apply
                self.manager = Manager(self.args)
                for index in range(3):
                    state, report = self.tick(10000 + index * 60)
                    self.assertAlmostEqual(state["last_score_snapshot"]["models"][q9]["score"], 0.35)
                    self.assertAlmostEqual(state["last_score_snapshot"]["models"][q35]["score"], 0.225625)
                    if index < 2:
                        self.assertEqual(state["last_decision_target"], q35)
                        self.assertIn(f"{index + 1}/3 consecutive checks passed", report)
                        self.launch.assert_not_called()
                self.assertEqual(state["last_decision_target"], q9)
                if apply:
                    self.launch.assert_called_once_with("darkbloom", q9, None, {IGNORED})
                    self.assertEqual(state["pending_switch"]["warm_models"], [q9])
                else:
                    self.launch.assert_not_called()
                    self.assertIn("WOULD SWITCH", report)
                header = next(line for line in report.splitlines() if line.startswith("score    MODEL ID"))
                self.assertEqual(header.split()[1:], ["MODEL", "ID", "NOW", "AVG", "5m", "N", "IN$/M", "OUT$/M", "BLEND$/M", "WEIGHT", "SCORE", "STATUS"])
                q9_row = next(line for line in report.splitlines() if line.startswith("           " + q9))
                self.assertEqual(q9_row.split()[4:9], ["0.0800", "0.1300", "0.0875", "1.00", "0.350"])
                self.assertIn("\n" + "=" * len(header) + "\n", report)
                self.assertIn("\n\n         Highest raw score:", report)
                self.assertIn("85% input price + 15% output price", report)
                self.assertNotIn("PROJ$/M", report)

    def test_small_scores_can_switch_on_percentage_gain_in_live_and_dry_run(self) -> None:
        self.local.return_value = {"good", "candidate"}
        self.capacity.return_value = self.samples(good=1, candidate=1.4)
        self.prices.return_value = ({"good": ModelPrice(0.005, 0.005), "candidate": ModelPrice(0.005, 0.005)}, ModelPrice(0.05, 0.2))
        for apply in (False, True):
            with self.subTest(apply=apply):
                self.path.unlink(missing_ok=True)
                self.launch.reset_mock()
                self.args.apply = apply
                self.manager = Manager(self.args)
                for index in range(3):
                    state, report = self.tick(10000 + index * 60)
                    if index < 2:
                        self.assertEqual(state["last_decision_target"], "good")
                        self.launch.assert_not_called()
                self.assertEqual(state["last_decision_target"], "candidate")
                self.assertIn("28.33% improvement", report)
                self.assertIn("at least 25% score improvement after switch cost", report)
                self.assertNotIn("both score requirements", report)
                self.assertEqual(state["switch_policy"]["improvement_percent"], 25)
                if apply:
                    self.launch.assert_called_once_with("darkbloom", "candidate", None, {IGNORED})
                else:
                    self.launch.assert_not_called()
                    self.assertIn("WOULD SWITCH", report)

    def test_percentage_rule_upgrade_rechecks_counts_and_preserves_history_and_warmup(self) -> None:
        self.local.return_value = {"good", "candidate"}
        self.capacity.return_value = self.samples(good=1, candidate=100)
        state, _ = self.tick()
        state.pop("switch_policy")
        state.update(manager_version="0.1.5", live_challenger_model="candidate", live_challenger_streak=2,
                     dry_challenger_model="candidate", dry_challenger_streak=2, last_switch_at=5000)
        self.save(state)
        upgraded, report = self.tick(10060)
        self.assertEqual(upgraded["live_challenger_streak"], 1)
        self.assertEqual(upgraded.get("dry_challenger_streak", 0), 0)
        self.assertEqual(len(upgraded["pressure_history"]["good"]), 2)
        self.assertEqual(upgraded["last_switch_at"], 5000)
        self.launch.assert_not_called()
        pending = {"target": "candidate", "warm_models": ["candidate"], "command_at": 10060}
        state["pending_switch"] = pending
        self.save(state)
        upgraded, report = self.tick(10120)
        self.assertEqual(upgraded["pending_switch"], pending)
        self.assertEqual(upgraded["last_switch_at"], 5000)
        self.assertIn("WARMING", report)
        self.launch.assert_not_called()

    def test_missing_component_blocks_candidate_or_current_in_live_and_dry_run(self) -> None:
        self.local.return_value = {"good", "candidate"}
        self.capacity.return_value = self.samples(good=1, candidate=100)
        for apply in (False, True):
            for missing in ("good", "candidate"):
                with self.subTest(apply=apply, missing=missing):
                    self.path.unlink(missing_ok=True)
                    self.args.apply = apply
                    self.manager = Manager(self.args)
                    prices = {"good": ModelPrice(0.08, 0.13), "candidate": ModelPrice(0.15, 2.0)}
                    prices[missing] = ModelPrice(None, prices[missing].output_usd)
                    self.prices.return_value = (prices, ModelPrice(0.05, 0.20))
                    for index in range(4):
                        state, report = self.tick(10000 + 60 * index)
                        row = state["last_score_snapshot"]["models"][missing]
                        self.assertIsNone(row["input_usd_per_million"])
                        self.assertIsNotNone(row["output_usd_per_million"])
                        self.assertIsNone(row["blended_usd_per_million"])
                        self.assertIsNone(row["score"])
                        self.assertFalse(row["eligible"])
                        self.assertIn("input price unavailable", row["status"])
                        self.assertEqual(state["last_decision_target"], None if missing == "good" else "good")
                    self.launch.assert_not_called()

    def test_upgrade_keeps_history_but_requires_new_passing_checks(self) -> None:
        self.local.return_value = {"good", "candidate"}
        state, _ = self.tick()
        state.pop("scoring_policy")
        state.update(live_challenger_model="candidate", live_challenger_streak=2,
                     dry_challenger_model="candidate", dry_challenger_streak=2, last_switch_at=5000)
        state["pricing_cache"] = {"source": self.args.pricing_url, "fetched_at": 10000,
                                  "prices": {"good": 0.5, "candidate": 0.2}, "fallback_output_usd": 0.2}
        self.save(state)
        self.capacity.return_value = self.samples(good=1, candidate=100)
        self.manager = Manager(self.args)
        for index in (1, 2):
            state, report = self.tick(10000 + 60 * index)
            self.assertEqual(state["live_challenger_streak"], index)
            self.assertEqual(state.get("dry_challenger_streak", 0), 0)
            self.assertEqual(state["last_decision_target"], "good")
            self.assertEqual(len(state["pressure_history"]["good"]), index + 1)
            self.assertEqual(state["last_switch_at"], 5000)
            self.launch.assert_not_called()
        self.assertEqual(state["state_schema"], STATE_SCHEMA)
        self.assertEqual(self.prices.call_count, 2)
        self.assertEqual(state["pricing_cache"]["schema"], PRICING_CACHE_SCHEMA)

    def test_formula_upgrade_preserves_an_already_issued_pending_launch(self) -> None:
        state, _ = self.tick()
        state.pop("scoring_policy")
        pending = {"target": "good", "warm_models": ["good"], "command_at": 10000}
        state.update(pending_switch=pending, last_switch_at=5000)
        self.save(state)
        self.daemon.return_value = LocalDaemonState(None, (), False, 124, 10000, True)
        state, report = self.tick(10060)
        self.assertEqual(state["pending_switch"], pending)
        self.assertEqual(state["last_switch_at"], 5000)
        self.assertEqual(len(state["pressure_history"]["good"]), 2)
        self.assertIn("WARMING", report)
        self.launch.assert_not_called()

    def test_catalog_models_without_downloads_are_scored_but_never_selected(self) -> None:
        self.args.hide_ignored = True
        remote = "Qwen3.5-9B"
        self.catalog.return_value = {"good", remote, "catalog-only"}
        self.local.return_value = {"good"}
        self.capacity.return_value = self.samples(good=1, **{remote: 1000})
        for apply in (False, True):
            with self.subTest(apply=apply):
                self.path.unlink(missing_ok=True)
                self.args.apply = apply
                self.manager = Manager(self.args)
                state, report = self.tick()
                row = state["last_score_snapshot"]["models"][remote]
                self.assertEqual(row["now_pressure"], 1000)
                self.assertEqual(row["average_pressure"], 1000)
                self.assertEqual(row["score"], 200)
                self.assertEqual(row["weight"], 1)
                self.assertFalse(row["ignored"])
                self.assertTrue(row["auto_ignored"])
                self.assertFalse(row["local_available"])
                self.assertFalse(row["eligible"])
                self.assertNotIn(remote, report)
                self.assertNotIn("catalog-only", report)
                self.assertIn("1 shown; 1 ignored hidden; 2 auto-ignored hidden", report)
                self.assertIn("Highest raw score (shown models): good (0.500).", report)
                missing = state["last_score_snapshot"]["models"]["catalog-only"]
                self.assertIsNone(missing["score"])
                self.assertIsNone(missing["now_pressure"])
                self.assertIn("capacity unavailable", missing["status"])
                self.assertEqual(state["last_decision_target"], "good")
                self.assertNotIn("WOULD SWITCH", report)
                self.assertNotIn("pending_switch", state)
                self.launch.assert_not_called()

    def test_download_becomes_eligible_then_removal_revokes_pending_selection(self) -> None:
        self.args.hide_ignored = True
        remote = "Qwen3.5-9B"
        self.local.return_value = {"good"}
        self.catalog.return_value = {"good", remote, IGNORED}
        self.capacity.return_value = self.samples(good=1, **{remote: 100, IGNORED: 1000})
        state, report = self.tick()
        self.assertTrue(state["last_score_snapshot"]["models"][remote]["auto_ignored"])
        self.assertNotIn(remote, report)
        self.local.return_value = {"good", remote, IGNORED}
        for index in range(1, 4):
            state, report = self.tick(10000 + index * 60)
            self.assertFalse(state["last_score_snapshot"]["models"][remote]["auto_ignored"])
            self.assertTrue(state["last_score_snapshot"]["models"][remote]["eligible"])
            self.assertTrue(state["last_score_snapshot"]["models"][IGNORED]["ignored"])
            self.assertIn("  " + remote, report)
            self.assertNotIn(IGNORED, report)
            if index < 3:
                self.assertEqual(state["live_challenger_streak"], index)
                self.launch.assert_not_called()
        self.launch.assert_called_once_with("darkbloom", remote, None, {IGNORED})
        self.assertEqual(state["pending_switch"]["target"], remote)
        self.local.return_value = {"good", IGNORED}
        state, report = self.tick(10240)
        self.assertNotIn("pending_switch", state)
        self.assertEqual(state["last_decision_target"], "good")
        self.assertTrue(state["last_score_snapshot"]["models"][remote]["auto_ignored"])
        self.assertEqual(len(state["pressure_history"][remote]), 5)
        self.assertIn("discarded pending switch", report)
        self.assertNotIn("  " + remote, report)
        self.launch.assert_called_once()

    def test_model_flag_restricts_loading_and_hides_excluded_catalog_rows(self) -> None:
        self.args.hide_ignored = True
        self.catalog.return_value = {"first", "second", "excluded"}
        self.local.return_value = set(self.catalog.return_value)
        self.capacity.return_value = self.samples(first=1, second=1, excluded=1000)
        self.daemon.return_value = LocalDaemonState(None, (), False, 123, 100, True)
        self.args.model = ["second", "first"]
        for apply in (False, True):
            with self.subTest(apply=apply):
                self.path.unlink(missing_ok=True)
                self.args.apply = apply
                self.manager = Manager(self.args)
                state, report = self.tick()
                self.assertEqual(state["last_decision_target"], "second")
                row = state["last_score_snapshot"]["models"]["excluded"]
                self.assertTrue(row["local_available"])
                self.assertTrue(row["excluded_by_model_flag"])
                self.assertTrue(row["auto_ignored"])
                self.assertNotIn("excluded", report)
                self.assertIn("Highest raw score (shown models): second = first (0.200).", report)
                if not apply:
                    self.launch.assert_not_called()
        self.launch.assert_called_once_with("darkbloom", "second", None, {IGNORED})

    def test_catalog_refresh_does_not_reset_passing_checks_for_local_candidates(self) -> None:
        self.catalog.return_value = {"good", "candidate"}
        self.local.return_value = {"good", "candidate"}
        self.capacity.return_value = self.samples(good=1, candidate=100)
        state, _ = self.tick()
        self.assertEqual(state["live_challenger_streak"], 1)
        self.catalog.return_value.add("new-remote")
        state, _ = self.tick(10060)
        self.assertEqual(state["live_challenger_streak"], 2)
        self.assertIn("new-remote", state["last_score_snapshot"]["models"])
        self.launch.assert_not_called()

    def test_catalog_outage_retains_rows_across_restart_without_fresh_scores(self) -> None:
        self.catalog.return_value = {"good", "remote"}
        self.capacity.return_value = self.samples(good=1, remote=10)
        self.tick()
        self.catalog.side_effect = RuntimeError("catalog offline")
        self.capacity.side_effect = RuntimeError("capacity offline")
        self.manager = Manager(self.args)
        state, report = self.tick(10060)
        self.assertEqual(state["catalog"]["status"], "stale cache")
        self.assertEqual(state["catalog"]["fetched_at"], 10000)
        row = state["last_score_snapshot"]["models"]["remote"]
        self.assertEqual(row["average_pressure"], 10)
        self.assertIsNone(row["score"])
        self.assertIn("catalog stale cache", report)
        self.assertIn("capacity unavailable; AVG retained", row["status"])
        self.catalog.side_effect = None
        self.catalog.return_value = {"good", "replacement"}
        state, _ = self.tick(10120)
        self.assertNotIn("remote", self.manager.models)
        self.assertIn("replacement", self.manager.models)
        self.assertEqual(state["catalog"]["status"], "live")
        self.launch.assert_not_called()

    def test_catalog_failure_uses_live_capacity_ids_for_visibility(self) -> None:
        self.catalog.side_effect = RuntimeError("catalog unavailable")
        self.local.return_value = {"good"}
        self.capacity.return_value = self.samples(good=1, **{"network-only": 1000})
        state, report = self.tick()
        self.assertEqual(state["catalog"]["status"], "unavailable")
        self.assertTrue(state["last_score_snapshot"]["models"]["network-only"]["auto_ignored"])
        self.assertIn("catalog unavailable", report)
        self.assertEqual(state["last_decision_target"], "good")
        self.launch.assert_not_called()

    def test_local_scan_failure_shows_unknown_presence_and_blocks_loading(self) -> None:
        self.catalog.return_value = {"good", "remote"}
        self.capacity.return_value = self.samples(good=1, remote=100)
        self.tick()
        self.local.side_effect = RuntimeError("local scan failed")
        state, report = self.tick(10060)
        for model in ("good", "remote"):
            row = state["last_score_snapshot"]["models"][model]
            self.assertIsNone(row["local_available"])
            self.assertTrue(row["auto_ignored"])
            self.assertIn("local scan unavailable", row["status"])
        self.assertNotIn("not downloaded or filtered out", report)
        self.assertIsNone(state["last_decision_target"])
        self.launch.assert_not_called()

    def test_catalog_cache_does_not_cross_config_paths(self) -> None:
        self.catalog.return_value = {"catalog-only"}
        self.tick()
        self.args.config = Path("other-provider.toml")
        self.catalog.side_effect = RuntimeError("offline")
        state, _ = self.tick(10060)
        self.assertEqual(state["catalog"]["status"], "unavailable")
        self.assertEqual(state["catalog"]["models"], [])
        self.assertNotIn("catalog-only", self.manager.models)
        self.catalog.assert_called_with("darkbloom", Path("other-provider.toml"))

    def test_slow_catalog_cannot_make_an_old_daemon_snapshot_look_fresh(self) -> None:
        self.args.daemon_state = self.path.with_name("daemon.json")
        self.args.daemon_state.write_text(json.dumps({
            "pid": 123, "started_at": 100, "written_at": 10000,
            "warm_models": [], "inference_active": False,
        }))
        self.daemon.side_effect = read_daemon_state

        def slow_catalog(*args):
            self.clock.return_value = 10120
            return {"good"}

        self.catalog.side_effect = slow_catalog
        with patch("warm_model_manager.process_alive", return_value=True):
            _, report = self.tick()
        self.assertIn("STALE", report)
        self.assertIn("DEFERRED", report)
        self.launch.assert_not_called()

    def test_empty_catalog_local_inventory_and_network_feed_keep_running(self) -> None:
        self.local.return_value = set()
        self.capacity.side_effect = RuntimeError("offline")
        self.args.ignore_model = []
        self.manager = Manager(self.args)
        state, report = self.tick()
        self.assertEqual(state["last_score_snapshot"]["models"], {})
        self.assertIn("WAIT", report)
        self.launch.assert_not_called()

    def test_filtered_ignored_id_is_still_scored_but_hidden(self) -> None:
        self.args.hide_ignored = True
        self.local.return_value = {"good"}
        state, report = self.tick()
        self.assertIn(IGNORED, self.manager.models)
        self.assertEqual(state["last_score_snapshot"]["models"][IGNORED]["score"], 75)
        self.assertNotIn(IGNORED, report)
        self.assertIn("1 ignored hidden", report)

    def test_missing_capacity_preserves_aging_history_without_inventing_score(self) -> None:
        self.args.hide_ignored = True
        self.tick()
        self.capacity.return_value = self.samples(good=1)
        state, report = self.tick(10060)
        row = state["last_score_snapshot"]["models"][IGNORED]
        self.assertIsNone(row["now_pressure"])
        self.assertEqual(row["average_pressure"], 100)
        self.assertIsNone(row["score"])
        self.assertIn("capacity unavailable; AVG retained", row["status"])
        self.assertNotIn(IGNORED, report)
        state, _ = self.tick(10300)
        self.assertEqual(state["pressure_history"][IGNORED], [])
        self.assertIsNone(state["last_score_snapshot"]["models"][IGNORED]["average_pressure"])

    def test_network_outage_still_persists_ignored_row_without_showing_it(self) -> None:
        self.args.hide_ignored = True
        self.capacity.side_effect = RuntimeError("offline")
        self.prices.side_effect = RuntimeError("offline")
        state, report = self.tick()
        row = state["last_score_snapshot"]["models"][IGNORED]
        self.assertIsNone(row["score"])
        self.assertIsNone(row["output_usd_per_million"])
        self.assertIn("N/A", report)
        self.assertEqual(row["status"], ["IGNORED", "capacity unavailable", "input price unavailable", "output price unavailable"])
        self.assertNotIn(IGNORED, report)
        self.assertIn("WAIT", report)
        self.launch.assert_not_called()

    def test_missing_price_for_ignored_model_does_not_block_eligible_model(self) -> None:
        self.args.hide_ignored = True
        self.prices.return_value = ({"good": ModelPrice(0.5, 0.5)}, ModelPrice(None, None))
        state, report = self.tick()
        self.assertEqual(state["last_decision_target"], "good")
        self.assertIsNone(state["last_score_snapshot"]["models"][IGNORED]["score"])
        self.assertEqual(state["last_score_snapshot"]["models"][IGNORED]["status"],
                         ["IGNORED", "input price unavailable", "output price unavailable"])
        self.assertNotIn(IGNORED, report)

    def test_stale_pricing_cache_is_identified(self) -> None:
        self.tick()
        self.prices.side_effect = RuntimeError("offline")
        state, report = self.tick(11000)
        self.assertEqual(state["pricing_status"]["status"], "stale cache")
        self.assertEqual(state["pricing_status"]["fetched_at"], 10000)
        self.assertIn("prices stale cache, fetched", report)

    def test_discovery_refresh_adds_and_removes_models_and_applies_future_weight(self) -> None:
        self.args.weight = [("future", 1.7)]
        self.manager = Manager(self.args)
        self.tick()
        self.local.return_value = {"future"}
        self.capacity.return_value = self.samples(future=3, **{IGNORED: 100})
        state, _ = self.tick(10060)
        self.assertEqual(self.manager.models, ["future", IGNORED])
        row = state["last_score_snapshot"]["models"]["future"]
        self.assertEqual(row["weight"], 1.7)
        self.assertAlmostEqual(row["score"], 3 * 0.2 * 1.7)
        self.assertEqual(state["last_decision_target"], "future")
        self.assertNotIn("good", state["pressure_history"])
        self.assertEqual(self.local.call_count, 2)
        self.launch.assert_called_once_with("darkbloom", "future", None, {IGNORED})

    def test_explicit_model_restriction_keeps_ignored_ids_and_excludes_remote_only_targets(self) -> None:
        self.args.model = ["remote", "good"]
        self.capacity.return_value = self.samples(remote=1000, good=1, **{IGNORED: 100})
        for apply in (True, False):
            self.args.apply = apply
            state, _ = self.tick()
            self.assertEqual(self.manager.models, ["remote", "good", IGNORED])
            self.assertEqual(state["last_decision_target"], "good")
        self.launch.assert_not_called()

    def test_discovery_failure_disables_selection_and_hides_inventory(self) -> None:
        self.args.hide_ignored = True
        self.tick()
        before = list(self.manager.models)
        self.local.side_effect = RuntimeError("scan failed")
        state, report = self.tick(10060)
        self.assertEqual(self.manager.models, before)
        self.assertIsNone(state["last_decision_target"])
        self.assertIn("discovery unavailable", report)
        self.assertIn("0 shown; 1 ignored hidden; 1 auto-ignored hidden", report)
        self.assertIn("No models to show.", report)
        self.assertNotIn("Highest raw score", report)
        self.launch.assert_not_called()

    def test_all_ignored_or_empty_local_inventory_reports_without_launching(self) -> None:
        self.args.hide_ignored = True
        for local in ({IGNORED}, set()):
            with self.subTest(local=local):
                self.local.return_value = local
                state, report = self.tick()
                self.assertIsNone(state["last_decision_target"])
                self.assertNotIn(IGNORED, report)
                self.assertIn("No models to show.", report)
                self.assertNotIn("Highest raw score", report)
                self.assertIn("WAIT", report)
        self.launch.assert_not_called()

    def test_saved_pending_cannot_restore_ignored_model(self) -> None:
        state, _ = self.tick()
        state.update({
            "pending_switch": {"target": IGNORED, "warm_models": [IGNORED], "command_at": 10000},
            "active_target": IGNORED,
            "live_challenger_model": IGNORED, "live_challenger_streak": 100,
        })
        self.save(state)
        state, report = self.tick(10060)
        self.assertEqual(state["last_decision_target"], "good")
        self.assertNotIn("pending_switch", state)
        self.assertIn("discarded pending switch", report)
        self.launch.assert_not_called()

    def test_saved_warm_set_with_ignored_companion_is_discarded(self) -> None:
        state, _ = self.tick()
        state["pending_switch"] = {"target": "good", "warm_models": ["good", IGNORED]}
        self.save(state)
        state, report = self.tick(10060)
        self.assertNotIn("pending_switch", state)
        self.assertIn("discarded pending switch", report)
        self.launch.assert_not_called()

    def test_confirmations_and_dwell_survive_refresh_with_ignored_raw_leader(self) -> None:
        self.local.return_value = {"good", "challenger", IGNORED}
        self.capacity.return_value = self.samples(good=1, challenger=10, **{IGNORED: 100})
        self.daemon.return_value = LocalDaemonState(
            "good", ("good",), False, 123, 9900, True,
        )
        for i in range(3):
            state, report = self.tick(10000 + i * 60)
            self.assertEqual(state["live_challenger_streak"], i + 1)
        self.assertEqual(state["last_decision_target"], "challenger")
        self.assertIn("minimum warm time", report)
        self.launch.assert_not_called()
        self.tick(11700)
        self.launch.assert_called_once_with("darkbloom", "challenger", None, {IGNORED})

    def test_dry_run_also_defers_during_dwell(self) -> None:
        self.args.apply = False
        self.local.return_value = {"other", IGNORED}
        self.capacity.return_value = self.samples(other=1, **{IGNORED: 100})
        self.daemon.return_value = LocalDaemonState("old", ("old",), False, 123, 9990, True)
        state, report = self.tick()
        self.assertEqual(state["last_decision_target"], "other")
        self.assertIn("DEFERRED", report)
        self.assertNotIn("would switch to", report)
        self.launch.assert_not_called()

    def test_older_release_retains_history_and_valid_pending_warmup(self) -> None:
        state, _ = self.tick()
        state["manager_version"] = "legacy"
        state.pop("selection_policy")
        state["pending_switch"] = {"target": "good", "warm_models": ["good"], "command_at": 10000}
        self.save(state)
        self.manager = Manager(self.args)
        ready = self.daemon.return_value
        self.daemon.return_value = LocalDaemonState(None, (), False, 123, 10000, True)
        state, report = self.tick(10060)
        self.assertEqual(len(state["pressure_history"][IGNORED]), 2)
        self.assertIn("WARMING", report)
        self.assertIn("pending_switch", state)
        self.daemon.return_value = ready
        state, report = self.tick(10120)
        self.assertNotIn("pending_switch", state)
        self.assertEqual(state["last_switch_at"], 10120)
        self.assertIn("warm-up confirmed", report)
        self.launch.assert_not_called()

    def test_discovery_outage_does_not_erase_valid_warmup_and_cause_restart(self) -> None:
        state, _ = self.tick()
        state["pending_switch"] = {"target": "good", "warm_models": ["good"], "command_at": 10000}
        self.save(state)
        self.daemon.return_value = LocalDaemonState(None, (), False, 123, 10000, True)
        self.local.side_effect = RuntimeError("scan failed")
        state, _ = self.tick(10060)
        self.assertIn("pending_switch", state)
        self.assertIsNone(state["last_decision_target"])
        self.local.side_effect = None
        state, report = self.tick(10120)
        self.assertIn("pending_switch", state)
        self.assertIn("WARMING", report)
        self.launch.assert_not_called()

    def test_ignored_pending_is_revoked_even_when_discovery_fails(self) -> None:
        state, _ = self.tick()
        state["pending_switch"] = {"target": IGNORED, "command_at": 10000}
        self.save(state)
        self.local.side_effect = RuntimeError("scan failed")
        state, _ = self.tick(10060)
        self.assertNotIn("pending_switch", state)
        self.assertIsNone(state["last_decision_target"])
        self.launch.assert_not_called()

    def test_math_is_saved_even_if_launch_fails(self) -> None:
        self.local.return_value = {"new"}
        self.capacity.return_value = self.samples(new=1, **{IGNORED: 100})
        self.launch.side_effect = RuntimeError("launch refused")
        with self.assertRaisesRegex(RuntimeError, "launch refused"):
            self.tick()
        state = json.loads(self.path.read_text())
        self.assertEqual(state["last_score_snapshot"]["models"][IGNORED]["score"], 75)
        self.assertEqual(state["pending_switch"]["target"], "new")
        self.assertEqual(state["pending_switch"]["command_error"], "launch refused")

    def test_failed_or_timed_out_launch_is_not_repeated_after_restart(self) -> None:
        errors = (RuntimeError("launch refused"), subprocess.TimeoutExpired(["darkbloom", "start"], 300))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.path.unlink(missing_ok=True)
                self.launch.reset_mock()
                self.daemon.return_value = LocalDaemonState(None, (), False, 123, 100, True)

                def fail_launch(*args):
                    saved = json.loads(self.path.read_text())
                    self.assertEqual(saved["pending_switch"]["target"], "good")
                    raise error

                self.launch.side_effect = fail_launch
                with self.assertRaises(type(error)):
                    self.tick()
                self.manager = Manager(self.args)
                for now in (10060, 10400):
                    state, report = self.tick(now)
                    self.assertEqual(state["pending_switch"]["command_error"], str(error))
                    self.assertIn("automatic restart is blocked", report)
                    self.assertNotIn("loading now", report)
                self.launch.assert_called_once()

                # A timed-out CLI may still have succeeded. Fresh warmth can
                # confirm it without issuing another command.
                self.daemon.return_value = LocalDaemonState("good", ("good",), False, 124, 10001, True)
                state, report = self.tick(10460)
                self.assertNotIn("pending_switch", state)
                self.assertEqual(state["last_switch_at"], 10460)
                self.assertIn("warm-up confirmed", report)
                self.launch.assert_called_once()

    def test_interrupted_launch_retains_pending_state_before_returning(self) -> None:
        self.daemon.return_value = LocalDaemonState(None, (), False, 123, 100, True)
        self.launch.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.tick()
        self.manager = Manager(self.args)
        state, report = self.tick(10060)
        self.assertEqual(state["pending_switch"]["target"], "good")
        self.assertIn("finish loading", report)
        self.launch.assert_called_once()

    def test_pending_state_write_failure_prevents_launch(self) -> None:
        self.daemon.return_value = LocalDaemonState(None, (), False, 123, 100, True)
        with patch("warm_model_manager.write_json_atomic", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.tick()
        self.launch.assert_not_called()

    def test_empty_local_inventory_saves_network_models_but_hides_them_without_ignore_flags(self) -> None:
        self.args.hide_ignored = True
        self.local.return_value = set()
        self.args.ignore_model = []
        self.manager = Manager(self.args)
        state, report = self.tick()
        self.assertEqual(set(state["last_score_snapshot"]["models"]), {"good", IGNORED})
        self.assertTrue(all(row["auto_ignored"] for row in state["last_score_snapshot"]["models"].values()))
        self.assertIn("0 shown; 0 ignored hidden; 2 auto-ignored hidden", report)
        self.assertIn("No models to show.", report)
        self.assertNotIn("Highest raw score", report)
        self.assertIn("WAIT", report)
        self.launch.assert_not_called()

    def test_documented_dry_run_command_through_cli_entrypoint(self) -> None:
        command = [
            "warm_model_manager.py", "once", "--check-every", "60", "--average-samples", "5",
            "--switch-after-checks", "3", "--min-warm-time", "1800",
            "--ignore-model", IGNORED, "--state", str(self.path),
        ]
        with patch.object(sys, "argv", command), \
                patch("warm_model_manager.DEFAULT_STATE_PATH", self.path), \
                patch("warm_model_manager.signal.signal"), redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        state = json.loads(self.path.read_text())
        self.assertEqual(state["last_decision_target"], "good")
        self.assertTrue(state["last_score_snapshot"]["models"][IGNORED]["ignored"])
        self.launch.assert_not_called()

    def test_default_state_move_and_future_releases_preserve_pending_warmup(self) -> None:
        original, _ = self.tick()
        original["pending_switch"] = {
            "target": "good", "warm_models": ["good"], "command_at": 10000,
        }
        previous = self.path.with_name("warm-model-manager-v4.json")
        previous.write_text(json.dumps(original))
        self.path.unlink()
        self.daemon.return_value = LocalDaemonState(None, (), False, 123, 10000, True)
        command = [
            "warm_model_manager.py", "once", "--apply", "--check-every", "60",
            "--average-samples", "5", "--switch-after-checks", "3",
            "--min-warm-time", "1800", "--ignore-model", IGNORED,
        ]
        for index, version in enumerate(("5.0.0", "6.0.0"), start=1):
            self.clock.return_value = 10000 + 60 * index
            with self.subTest(version=version), patch.object(sys, "argv", command), \
                    patch("warm_model_manager.DEFAULT_STATE_PATH", self.path), \
                    patch("warm_model_manager.MANAGER_VERSION", version), \
                    patch("warm_model_manager.signal.signal"), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            state = json.loads(self.path.read_text())
            self.assertEqual(state["manager_version"], version)
            self.assertEqual(state["state_schema"], original["state_schema"])
            self.assertEqual(state["pending_switch"], original["pending_switch"])
            self.assertEqual(len(state["pressure_history"][IGNORED]), index + 1)
            self.assertEqual(json.loads(previous.read_text()), original)
            self.launch.assert_not_called()

    def test_removed_feature_state_is_dropped_without_bypassing_confirmations(self) -> None:
        self.local.return_value = {"good", "challenger", IGNORED}
        state, _ = self.tick()
        legacy = {
            "exploration": {"model": IGNORED, "started_at": 9900},
            "return_from_exploration": "challenger",
            "next_exploration_at": 1,
            "model_stats": {"good": {"requests": 42}},
            "provider_observation": {"requests_served": 42},
            "observation_slices": [{"model": "good", "online_seconds": 60}],
        }
        state.update(legacy)
        self.save(state)
        self.capacity.return_value = self.samples(good=1, challenger=10, **{IGNORED: 100})
        state, report = self.tick(10060)
        self.assertTrue(set(legacy).isdisjoint(state))
        self.assertEqual(state["live_challenger_streak"], 1)
        self.assertEqual(state["last_decision_target"], "good")
        self.assertEqual(len(state["pressure_history"]["good"]), 2)
        self.assertNotIn("Local observations", report)
        self.assertNotIn("req/h", report)
        self.launch.assert_not_called()

    def test_removed_flags_are_not_silently_accepted(self) -> None:
        for flag in ("--no-pair", "--no-exploration", "--exploration-duration", "--exploration-every", "--exploration-spacing"):
            with self.subTest(flag=flag), patch("sys.stderr", new=io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                build_parser().parse_args([flag])
            self.assertEqual(error.exception.code, 2)

    def test_ignored_pair_member_revokes_saved_pair_switch(self) -> None:
        gemma, gpt = "gemma-4-26b-qat-4bit", "gpt-oss-20b"
        self.local.return_value = {"good", gemma, gpt}
        state, _ = self.tick()
        state["pending_switch"] = {
            "target": gemma, "warm_models": [gemma, gpt], "command_at": 10000,
        }
        self.save(state)
        self.args.ignore_model.append(gpt)
        self.manager = Manager(self.args)
        state, report = self.tick(10060)
        self.assertNotIn("pending_switch", state)
        self.assertEqual(state["last_decision_target"], "good")
        self.assertIn("discarded pending switch", report)
        self.launch.assert_not_called()

    def test_each_former_pair_member_can_win_alone_in_live_and_dry_run(self) -> None:
        gemma, gpt = "gemma-4-26b-qat-4bit", "gpt-oss-20b"
        for model in (gemma, gpt):
            for apply in (False, True):
                with self.subTest(model=model, apply=apply):
                    self.path.unlink(missing_ok=True)
                    self.launch.reset_mock()
                    self.args.apply = apply
                    self.local.return_value = {model}
                    self.capacity.return_value = self.samples(**{model: 2})
                    self.daemon.return_value = PendingSwitchTests.daemon(())
                    state, report = self.tick()
                    self.assertEqual(state["last_decision_target"], model)
                    self.assertNotIn("Warm set:", report)
                    if apply:
                        self.launch.assert_called_once_with("darkbloom", model, None, {IGNORED})
                        self.assertEqual(state["pending_switch"]["warm_models"], [model])
                    else:
                        self.launch.assert_not_called()
                        self.assertIn("WOULD SWITCH", report)

    def test_old_pair_is_replaced_with_single_winner_without_reusing_saved_pair(self) -> None:
        gemma, gpt = "gemma-4-26b-qat-4bit", "gpt-oss-20b"
        state, _ = self.tick()
        state.update({
            "pairing_enabled": True,
            "pending_switch": {"target": gemma, "warm_models": [gemma, gpt], "command_at": 10000},
        })
        self.save(state)
        self.local.return_value = {gemma, gpt}
        self.capacity.return_value = self.samples(**{gemma: 1, gpt: 5})
        self.daemon.return_value = PendingSwitchTests.daemon((gemma, gpt))
        state, report = self.tick(10060)
        self.launch.assert_called_once_with("darkbloom", gpt, None, {IGNORED})
        self.assertEqual(state["pending_switch"]["warm_models"], [gpt])
        self.assertNotIn("pairing_enabled", state)
        self.assertIn("discarded pending switch", report)

    def test_invalid_saved_warm_lists_are_discarded_even_during_discovery_failure(self) -> None:
        original, _ = self.tick()
        invalid_records = (
            {}, {"warm_models": None}, {"warm_models": []},
            {"warm_models": ["good", "other"]}, {"warm_models": ["good", "good"]},
            {"warm_models": ["other"]}, {"warm_models": "good"},
        )
        for available in (True, False):
            self.local.side_effect = None if available else RuntimeError("scan failed")
            for record in invalid_records:
                with self.subTest(available=available, record=record):
                    self.save({
                        **original,
                        "pending_switch": {"target": "good", "command_at": 10000, **record},
                    })
                    state, report = self.tick(10060)
                    self.assertNotIn("pending_switch", state)
                    self.assertIn("discarded pending switch", report)
                    self.launch.assert_not_called()

    def test_replacing_an_old_multi_model_selection_obeys_idle_and_dwell(self) -> None:
        original, _ = self.tick()
        for apply in (False, True):
            self.args.apply = apply
            for busy, last_switch, reason in (
                (True, 100, "actively serving"),
                (False, 10000, "minimum warm time"),
            ):
                with self.subTest(apply=apply, busy=busy):
                    self.save({**original, "last_switch_at": last_switch})
                    self.daemon.return_value = LocalDaemonState(
                        current_model="good", warm_models=("good", "other"),
                        inference_active=busy, pid=123, started_at=100, fresh=True,
                    )
                    state, report = self.tick(10060)
                    self.assertEqual(state["last_decision_target"], "good")
                    self.assertNotIn("pending_switch", state)
                    self.assertIn(reason, report)
                    self.assertIn("DEFERRED", report)
                    self.launch.assert_not_called()

    def test_single_model_residency_survives_upgrade_without_old_policy_fields(self) -> None:
        state, _ = self.tick()
        warm_since = state["current_residency"]["warm_since"]
        state["pairing_enabled"] = False
        state["current_residency"]["pairing_enabled"] = False
        self.save(state)
        state, _ = self.tick(10060)
        self.assertEqual(state["current_residency"]["warm_since"], warm_since)
        self.assertNotIn("pairing_enabled", state)
        self.assertNotIn("pairing_enabled", state["current_residency"])
        self.launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
