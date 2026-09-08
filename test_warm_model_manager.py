import fcntl
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

from warm_model_manager import (
    CapacitySample,
    DEFAULT_PROVIDER_CONFIG_PATH,
    DEFAULT_STATE_PATH,
    DEFAULT_WEIGHTS,
    Decision,
    LocalDaemonState,
    MANAGER_VERSION,
    STATE_SCHEMA,
    Manager,
    build_score_snapshot,
    build_parser,
    cached_output_prices,
    challenger_state_keys,
    choose_scored_target,
    current_warm_model,
    daemon_status_line,
    dwell_anchor,
    ensure_pressure_cadence,
    ensure_preload_sync_policy,
    eligible_local_targets,
    fetch_output_prices,
    format_duration,
    format_human_duration,
    local_model_ids,
    main,
    migrate_default_state,
    pressure_samples,
    read_daemon_state,
    reconcile_pending_switch,
    render_preload_model_config,
    revenue_scores,
    switch_model,
    switch_forecast,
    track_current_residency,
    update_pressure_history,
    warm_selection_matches,
)


MODELS = ["q35", "q36", "gemma", "gpt"]


def setUpModule() -> None:
    # Every test must explicitly fake external I/O. An accidental real provider
    # command or network connection is a test failure, including on developer Macs.
    for target in ("subprocess.run", "socket.create_connection"):
        guard = patch(target, side_effect=AssertionError("external I/O forbidden in offline tests"))
        guard.start()
        unittest.addModuleCleanup(guard.stop)


class CapacityTests(unittest.TestCase):
    def test_manager_is_marked_unreleased(self) -> None:
        self.assertEqual(MANAGER_VERSION, "unreleased")

    def test_default_qwen_preference_weights(self) -> None:
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

    def test_revenue_score_combines_pressure_price_and_preference(self) -> None:
        scores = revenue_scores(
            {"q35": 0.5, "gpt": 0.5},
            {"q35": 1.15, "gpt": 1.0},
            {"q35": 0.75, "gpt": 0.10},
        )
        self.assertAlmostEqual(scores["q35"], 0.43125)
        self.assertAlmostEqual(scores["gpt"], 0.05)

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
            {"q35": 0.75},
            {"q35": 1.15},
            {"q35": 0.25875},
            observed_at=160.0,
            window_seconds=180.0,
        )
        self.assertEqual(snapshot["observed_at"], 160.0)
        self.assertEqual(snapshot["window_seconds"], 180.0)
        values = snapshot["models"]["q35"]
        self.assertEqual(values["now_pressure"], 0.4)
        self.assertEqual(values["average_pressure"], 0.3)
        self.assertEqual(values["retained_samples"], 2)
        self.assertEqual(values["output_usd_per_million"], 0.75)
        self.assertAlmostEqual(values["projected_usd_per_million"], 0.225)
        self.assertEqual(values["preference"], 1.15)
        self.assertEqual(values["score"], 0.25875)

    def test_public_micro_usd_prices_convert_to_usd_per_million(self) -> None:
        payload = {
            "fallback_output_price": 200_000,
            "prices": [
                {"model": "q35", "output_price": 750_000},
                {"model": "gpt", "output_price": 100_000},
            ],
        }
        with patch("warm_model_manager.get_json", return_value=payload):
            prices, fallback = fetch_output_prices("https://example.test/pricing")
        self.assertEqual(prices, {"q35": 0.75, "gpt": 0.10})
        self.assertEqual(fallback, 0.20)

    def test_cached_prices_cover_missing_model_with_public_fallback(self) -> None:
        state = {}
        with patch(
            "warm_model_manager.fetch_output_prices",
            return_value=({"q35": 0.75}, 0.20),
        ):
            prices = cached_output_prices(
                state,
                ["q35", "new-model"],
                "https://example.test/pricing",
                now=1000,
                refresh_seconds=900,
            )
        self.assertEqual(prices, {"q35": 0.75, "new-model": 0.20})

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
    def test_initial_choice_uses_highest_projected_revenue_not_pressure(self) -> None:
        pressures = {"q35": 0.101, "q36": 0.054, "gemma": 0.384, "gpt": 0.629}
        scores = revenue_scores(
            pressures,
            {"q35": 1.15, "q36": 1.10, "gemma": 1.05, "gpt": 1.0},
            {"q35": 0.75, "q36": 0.70, "gemma": 0.22, "gpt": 0.10},
        )
        decision = choose_scored_target(
            MODELS,
            scores,
            None,
            None,
            0,
            relative_margin=0.25,
            absolute_margin=0.01,
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
            relative_margin=0.25,
            absolute_margin=0.10,
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
            relative_margin=0.25,
            absolute_margin=0.10,
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
            relative_margin=0.0,
            absolute_margin=0.10,
            confirmations=1,
            switch_cost_seconds=300,
            decision_horizon_seconds=3600,
        )
        self.assertEqual(decision.target, "gemma")
        self.assertIn("does not clear", decision.reason)
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

    def test_switch_eta_waits_for_confirmations_and_dwell(self) -> None:
        forecast = switch_forecast(
            "gemma",
            Decision(
                "gemma",
                "q36 clears both margins; confirmation 1/3",
                challenger="q36",
                challenger_streak=1,
            ),
            {"last_switch_at": 1_500},
            self.daemon(started_at=1_000),
            now=1_600,
            interval_seconds=60,
            confirmations=3,
            min_dwell_seconds=1_800,
        )
        assert forecast is not None
        self.assertIn("about 29 minutes", forecast)
        self.assertIn("keeps clearing both margins", forecast)

    def test_margin_failing_contender_has_no_false_countdown(self) -> None:
        forecast = switch_forecast(
            "gemma",
            Decision(
                "gemma",
                "q36 does not clear both switch margins",
                challenger="q36",
            ),
            {"last_switch_at": 1_000},
            self.daemon(),
            now=2_000,
            interval_seconds=60,
            confirmations=3,
            min_dwell_seconds=1_800,
        )
        assert forecast is not None
        self.assertIn("no estimate yet", forecast)
        self.assertIn("does not clear both switch margins", forecast)


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
            (900, 3, 2, 2700),
        )

    def test_validation_uses_new_flag_names_before_external_io(self) -> None:
        for flag, value in (("--check-every", "59"), ("--average-samples", "0"),
                            ("--switch-after-checks", "0"), ("--min-warm-time", "-1")):
            with self.subTest(flag=flag), patch.object(sys, "argv", ["warm_model_manager.py", flag, value]), \
                    self.assertRaisesRegex(SystemExit, flag):
                main()
        self.assertFalse(self.state_path.exists())

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
        self.capacity = self.mock("fetch_capacity", return_value=self.samples(good=1, **{IGNORED: 100}))
        self.prices = self.mock("fetch_output_prices", return_value=({"good": 0.5, IGNORED: 0.75}, 0.2))
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

    def test_ignored_leader_has_all_math_persisted_but_is_never_selected(self) -> None:
        state, report = self.tick()
        row = state["last_score_snapshot"]["models"][IGNORED]
        self.assertTrue(row["ignored"])
        self.assertFalse(row["eligible"])
        self.assertEqual(row["now_pressure"], 100)
        self.assertEqual(row["average_pressure"], 100)
        self.assertEqual(row["output_usd_per_million"], 0.75)
        self.assertEqual(row["preference"], 1)
        self.assertEqual(row["score"], 75)
        self.assertEqual(state["pressure_history"][IGNORED], [{"at": 10000, "pressure": 100}])
        self.assertEqual(state["last_decision_target"], "good")
        self.assertIn(f"Highest raw score: {IGNORED} [IGNORED]", report)
        self.assertNotIn("WOULD SWITCH", report)
        self.launch.assert_not_called()

    def test_filtered_ignored_id_is_still_scored_and_shown(self) -> None:
        self.local.return_value = {"good"}
        state, report = self.tick()
        self.assertIn(IGNORED, self.manager.models)
        self.assertEqual(state["last_score_snapshot"]["models"][IGNORED]["score"], 75)
        self.assertIn("IGNORED", report)

    def test_missing_capacity_preserves_aging_history_without_inventing_score(self) -> None:
        self.tick()
        self.capacity.return_value = self.samples(good=1)
        state, report = self.tick(10060)
        row = state["last_score_snapshot"]["models"][IGNORED]
        self.assertIsNone(row["now_pressure"])
        self.assertEqual(row["average_pressure"], 100)
        self.assertIsNone(row["score"])
        self.assertIn("capacity unavailable; AVG retained", report)
        state, _ = self.tick(10300)
        self.assertEqual(state["pressure_history"][IGNORED], [])
        self.assertIsNone(state["last_score_snapshot"]["models"][IGNORED]["average_pressure"])

    def test_network_outage_still_prints_and_persists_ignored_row(self) -> None:
        self.capacity.side_effect = RuntimeError("offline")
        self.prices.side_effect = RuntimeError("offline")
        state, report = self.tick()
        row = state["last_score_snapshot"]["models"][IGNORED]
        self.assertIsNone(row["score"])
        self.assertIsNone(row["output_usd_per_million"])
        self.assertIn("N/A", report)
        self.assertIn("IGNORED; capacity unavailable; price unavailable", report)
        self.assertIn("WAIT", report)
        self.launch.assert_not_called()

    def test_missing_price_for_ignored_model_does_not_block_eligible_model(self) -> None:
        self.prices.return_value = ({"good": 0.5}, 0)
        state, report = self.tick()
        self.assertEqual(state["last_decision_target"], "good")
        self.assertIsNone(state["last_score_snapshot"]["models"][IGNORED]["score"])
        self.assertIn("IGNORED; price unavailable", report)

    def test_stale_pricing_cache_is_identified(self) -> None:
        self.tick()
        self.prices.side_effect = RuntimeError("offline")
        state, report = self.tick(11000)
        self.assertEqual(state["pricing_status"]["status"], "stale cache")
        self.assertEqual(state["pricing_status"]["fetched_at"], 10000)
        self.assertIn("Prices:    stale cache; fetched", report)

    def test_discovery_refresh_adds_and_removes_models_and_applies_future_weight(self) -> None:
        self.args.weight = [("future", 1.7)]
        self.manager = Manager(self.args)
        self.tick()
        self.local.return_value = {"future"}
        self.capacity.return_value = self.samples(future=3, **{IGNORED: 100})
        state, _ = self.tick(10060)
        self.assertEqual(self.manager.models, ["future", IGNORED])
        row = state["last_score_snapshot"]["models"]["future"]
        self.assertEqual(row["preference"], 1.7)
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

    def test_discovery_failure_disables_selection_but_retains_display(self) -> None:
        self.tick()
        before = list(self.manager.models)
        self.local.side_effect = RuntimeError("scan failed")
        state, report = self.tick(10060)
        self.assertEqual(self.manager.models, before)
        self.assertIsNone(state["last_decision_target"])
        self.assertIn("Discovery: unavailable", report)
        self.launch.assert_not_called()

    def test_all_ignored_or_empty_local_inventory_reports_without_launching(self) -> None:
        for local in ({IGNORED}, set()):
            with self.subTest(local=local):
                self.local.return_value = local
                state, report = self.tick()
                self.assertIsNone(state["last_decision_target"])
                self.assertIn(IGNORED, report)
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
        self.assertIn("minimum dwell", report)
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
        self.assertNotIn("pending_switch", state)

    def test_empty_inventory_without_ignore_flags_keeps_running(self) -> None:
        self.local.return_value = set()
        self.args.ignore_model = []
        self.manager = Manager(self.args)
        state, report = self.tick()
        self.assertEqual(state["last_score_snapshot"]["models"], {})
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
                (False, 10000, "minimum dwell"),
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
