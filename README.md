# Darkbloom warm-model manager

Keeps one downloaded model warm on a running Darkbloom provider. The manager
compares public network pressure and input/output prices, waits for a sustained score
advantage, then switches when the provider is idle.

## Install

You need an Apple Silicon Mac with a configured Darkbloom provider, downloaded
models, and Python 3.10–3.14. The script uses Python's standard library. It needs
no API key or extra packages.

```sh
git clone https://github.com/benbuschmann/darkbloom-manager.git
cd darkbloom-manager
python3 -m unittest discover -s . -p 'test_*.py'
```

Tests use temporary files and fake network, daemon, and launch data. They do not
need Darkbloom installed. CI runs on Linux with each supported Python version
and on macOS with Python 3.14.

Copy `warm_model_manager.py` to run it on another provider Mac, or download the
[latest released script](https://github.com/benbuschmann/darkbloom-manager/releases/latest/download/warm_model_manager.py).
The filename stays the same across releases. Check your copy with
`python3 warm_model_manager.py --version`.

## Run

Preview one check:

```sh
python3 warm_model_manager.py once \
  --ignore-model 'EigenLabs/Qwen3.8-27B-4bit-mtp'
```

For continuous loading:

```sh
python3 warm_model_manager.py run --apply \
  --ignore-model 'EigenLabs/Qwen3.8-27B-4bit-mtp'
```

The defaults check every minute, average up to 15 recent samples, require
three consecutive checks that meet both score requirements, and keep a model
warm for at least 45 minutes before replacing it. Use the timing flags below
to override them.

Each report starts with a separator. A blank line separates the model table
from the highest-score line.

Without `--apply`, the manager prints decisions and saves its own state. It
does not launch a model or edit startup preloads. It still reads the catalog
and local model list through Darkbloom, which may migrate an older provider config.

Add `--config /path/to/provider.toml` if your provider uses a custom config.
The catalog, local scan, and live launches all use that path. Use
`--darkbloom /path/to/darkbloom` if the CLI is not on your `PATH`.

A live switch runs `darkbloom start` with one `--model` and
`--idle-timeout 0`. That restarts the provider. Before launching, the manager
sets `backend.preload_models` to a one-element list containing the selected
model, preserving the rest of the config text. It waits until fresh daemon
state confirms exactly that model is warm before starting the minimum warm
time. A provider restart also starts a new minimum warm period, even if the
manager's saved switch time is older.

## Timing and scores

| Flag | Default | What it controls |
| --- | --- | --- |
| `--check-every SECONDS` | 60 | Seconds between checks. Minimum: 60. |
| `--average-samples COUNT` | 15 | Maximum number of recent pressure samples to average. |
| `--switch-after-checks COUNT` | 3 | Consecutive checks the same candidate must pass before switching. |
| `--min-warm-time SECONDS` | 2700 | Minimum time to keep a warm model before replacing it. |

The old names still work as aliases: `--interval`, `--history`,
`--confirmations`, and `--min-dwell`, respectively.

Samples expire after `check-every × average-samples` seconds, including while
the manager is stopped. With the defaults of `60` and `15`, the average contains
at most 15 samples from the past 15 minutes. The table's `N` column shows how many are
available. Changing either setting clears the old samples and resets the
number of consecutive passing checks.

```text
pressure = active requests / max(1, warm providers)
blended price = (input price × 0.85) + (output price × 0.15)
score = average pressure × blended price × weight
```

Prices are USD per million tokens. Every model uses the same fixed mix:
85% input tokens and 15% output tokens. The mix is built into the script.
Prices still refresh from Darkbloom's public endpoint.

For example, an input price of `$0.08/M` and an output price of `$0.13/M`
give a blended price of `$0.0875/M` total tokens. At average pressure `2.0`
and weight `1.00`, the score is `0.175`.

The score compares models at that assumed mix. It does not measure your
provider's throughput, actual token mix, or payouts, and does not reproduce
per-request billing rounding.

| Table column | Meaning |
| --- | --- |
| `IN$/M` | Input price per million input tokens. |
| `OUT$/M` | Output price per million output tokens. |
| `BLEND$/M` | Price per million total tokens at the fixed 85/15 mix, before pressure and weight. |
| `WEIGHT` | The model score multiplier, set with `--weight`. |
| `SCORE` | Average pressure × blended price × weight. |

Price columns show four decimal places. Calculations use the full values.
`BLEND$/M` replaces the earlier `PROJ$/M` column.

| Model ID | Default weight |
| --- | ---: |
| `qwen3.5-35b-a3b` | 1.25 |
| `qwen3.6-35b-a3b-vl-mtp-mxfp8` | 1.20 |
| `gemma-4-26b-qat-4bit` | 1.05 |
| `gpt-oss-20b` | 1.00 |
| Other models | 1.00 |

Set a weight with `--weight MODEL_ID=WEIGHT`. Repeat it for more models.
For example, `--weight Qwen3.5-9B=1.25` increases that model's score by 25%.
Weights must be positive; they can also name ignored models or future downloads.

Before comparing a candidate with the current model, the manager discounts its
score by `(decision-horizon − switch-cost) / decision-horizon`. It then checks
both score requirements. With the defaults, a current score of `0.10` and a
candidate score of `0.12` produce an adjusted candidate score of `0.11`. That misses the required
`0.125`, so that check does not count toward `--switch-after-checks`.

The switch thresholds remain 25% and `0.01`. Blended scores are usually smaller
than output-only scores, so the `0.01` absolute requirement can be relatively
stricter. Both requirements still have to pass.

| Flag | Default | What it controls |
| --- | --- | --- |
| `--relative-margin` | 0.25 | Required score advantage after the discount: 25%. |
| `--absolute-margin` | 0.01 | Required score increase, also after the discount. |
| `--switch-cost` | 300 seconds | Estimated time lost while changing models. |
| `--decision-horizon` | 3600 seconds | Period used to weigh that lost time. Must exceed switch cost. |
| `--warmup-timeout` | 180 seconds | Time to wait for the requested model to become warm before reporting loading as overdue. |
| `--pricing-refresh` | 900 seconds | Time between price-cache refreshes. |

If no eligible model is currently warm, the manager selects the highest scored
eligible model without waiting for consecutive passing checks. It still checks
daemon health, active requests, and pending warm-up. An empty warm list has no minimum
warm time to preserve; an existing warm selection does.

## Discovery and ignored models

The table combines three sources on every check:

| Source | What it supplies |
| --- | --- |
| `darkbloom models catalog --json` | Supported model IDs from the coordinator in your provider config, including models you have not downloaded. |
| Public network capacity | Model IDs and current pressure, including IDs missing from the catalog response. |
| `darkbloom models list --all --json` | Locally discovered models. Only models in a successful local scan can be loaded. |

Both Darkbloom commands respect `--config`. The catalog uses concrete model
IDs, rather than the consumer-facing aliases shown in some Darkbloom pages.
Explicit `--model` and `--ignore-model` IDs also remain visible.

Models absent from the local scan show
`AUTO-IGNORED; not downloaded or filtered out`. Their pressure, price, weight,
and score are still calculated and saved when network data is available.
They cannot be selected or loaded. Once a download appears in the local scan,
that automatic exclusion clears. Removing a download excludes it again and
revokes any saved pending selection for it. The manager does not download models.

Repeated `--model MODEL_ID` flags restrict loading candidates and set their
tie order. Other rows remain visible with `AUTO-IGNORED; outside --model selection`.
Without that restriction, the four weighted models above retain their tie
order, followed by other IDs alphabetically. Explicit ignore IDs appear last
unless placed earlier by `--model`. Each launch still requests one model.

Use `--ignore-model MODEL_ID`, or `--ignore MODEL_ID`, to exclude a model from
loading. The flag is repeatable and matches exact, case-sensitive IDs. Ignored
models stay in the table, even when they are absent from the local list. Their
rows show `IGNORED` alongside pressure, average, both prices, blend, weight, and score.
Their calculations are saved too. An explicit ignore stays in effect even if
you download the model later.

The `MODEL ID` column and decision lines use the exact IDs accepted by
`--ignore-model`, including capitalization and any namespace prefix.

`Highest raw score` includes ignored and auto-ignored models, with their labels,
and includes ties. It is measured before
switch costs and thresholds. The `Decision`, `Candidate`, `Earliest switch`, and
`Deferred` lines show what the manager can actually do. A saved pending switch
cannot authorize an ignored model to load.

`Candidate` shows progress such as `2/3 consecutive checks passed`, followed
by `--switch-after-checks`. A wait for minimum warm time names `--min-warm-time`
and shows the seconds remaining. `Earliest switch` is conditional: the candidate
must keep meeting both score requirements, and the provider must be idle.

Darkbloom's `--all` bypasses the enabled-model config filter, but its scanner can
still omit downloads that exceed available memory. The catalog's JSON output
does not report download status. The manager therefore cannot distinguish a
missing download from a filtered download; the status says both. Listing also
skips the serving command's runtime-capability filter. A listed model can still
fail to run on your GPU. Use an explicit ignore for models your provider cannot run.

This behavior was checked on September 8, 2026 against the upstream
[list command](https://github.com/Layr-Labs/d-inference/blob/efcde6334ddf95a98e7c5353329abc52e195e9f2/provider-swift/Sources/darkbloom/ModelsCommand.swift),
[catalog client](https://github.com/Layr-Labs/d-inference/blob/efcde6334ddf95a98e7c5353329abc52e195e9f2/provider-swift/Sources/ProviderCore/Models/ModelCatalogClient.swift),
[runtime filtering](https://github.com/Layr-Labs/d-inference/blob/efcde6334ddf95a98e7c5353329abc52e195e9f2/provider-swift/Sources/darkbloom/Darkbloom.swift),
and [model scanner](https://github.com/Layr-Labs/d-inference/blob/efcde6334ddf95a98e7c5353329abc52e195e9f2/provider-swift/Sources/ProviderCore/Models/ModelScanner%2BDiscovery.swift).

## FAQ

### Do I need a Darkbloom API key?

No. The manager reads public capacity and pricing endpoints without an API key.
It uses your installed Darkbloom CLI and provider config for local operations.
It does not read a `.env` file. Darkbloom itself must already be configured and
running; follow its [provider setup instructions](https://github.com/Layr-Labs/d-inference/blob/master/docs/provider/installation.md).

### What is saved locally?

The manager writes `~/.darkbloom/warm-model-manager-state.json` after each check,
including dry runs. The file is replaced atomically and has owner-only read and
write permissions.

| Saved data | Purpose |
| --- | --- |
| Timestamped pressure samples | Rebuild each model's rolling average after a restart. |
| Latest score snapshot | Retain pressure, average, input/output prices, blended price, token mix, weight, score, local availability, and explicit or automatic ignore reasons. |
| Price cache and fetch time | Reuse both prices and public fallback prices between refreshes and identify stale prices. |
| Discovered model IDs, scan times, and scan errors | Show the last inventory when discovery fails. |
| Catalog IDs, source, fetch time, and errors | Keep catalog rows visible during an outage and identify stale catalog data. |
| Current model, process identity, warm-start time, and last switch time | Track how long the model has been warm. |
| Candidate and count of consecutive passing checks | Resume progress toward `--switch-after-checks` across restarts. Live and dry-run counts are separate. |
| Pending target, launch time, and any command error | Wait for warm-up without issuing repeated restarts. |
| Last decision, scoring mix, selection policy, timing settings, and format/release metadata | Explain the last check and detect incompatible saved settings. |

Each save replaces the latest snapshot and keeps a bounded sample window.
The manager does not collect prompts, responses, API keys,
local request counts, or per-model request rates. Catalog, discovery, and launch errors
can contain local paths or CLI error text.

Use `--state /path/to/state.json` for another location. On the first run with
the new default path, the manager copies the previous default state if present
and leaves the original file in place. It holds the older managers' locks
before copying. An existing new state file wins; custom state paths are not
migrated. A release-number change does not reset state.

The saved ignore list records the last run's settings. Keep `--ignore-model`
in your launch command; saved settings do not replace command-line options.

### Can I delete the state file?

Stop the manager first. Deleting state removes rolling samples, progress toward
the required consecutive checks, and pending warm-up tracking. For a deliberate
fresh start, use a new `--state` path after stopping the old process. Deleting the default file
can cause the retained previous state to be imported again.

### What happens when data is unavailable?

Missing pressure or price appears as `N/A`. Recent averages remain visible
until they expire, but missing current pressure prevents a fresh score. A price
refresh failure can use the last cache containing both prices, labeled with
its fetch time. Models absent from the pricing response use Darkbloom's public
fallback input and output prices when available.

A listed model missing either price shows `N/A` for that component, the blend,
and the score. Its known price stays visible. The manager does not replace a
missing component with zero or a fallback. A zero price is accepted only when
explicitly supplied; the blended price must be positive to allow selection.
If the current model's score is unavailable, switching waits for usable data.

If the catalog request fails, the last catalog remains visible as `stale cache`.
Without a cached catalog, the report says `Catalog: unavailable` and still shows
IDs from the capacity feed, local scan, and your flags. Catalog membership alone
never permits a launch.

If local discovery fails, the last inventory remains visible and new launches
are blocked. Local availability is unknown, and the rows say `local scan unavailable`.

### Why hasn't it switched?

Read `Reason`, `Candidate`, and `Deferred` in the report. The candidate may need
more passing checks (`--switch-after-checks`), the current model may need more
warm time (`--min-warm-time`), or the score advantage may be too small. A busy,
stopped, or stale provider also blocks a switch. Dry runs use the same rules.

A failed or timed-out launch command, load error, or overdue warm-up blocks
automatic retries. The pending target is saved before attempting the launch;
fresh daemon state can still confirm success if the command timed out. Inspect
`darkbloom status` and the provider logs, fix the loading problem, then stop the
manager before editing pending state.

### What does Ctrl-C stop?

It stops the manager after any in-flight check or command finishes. The
provider keeps its last model, startup preload, and always-warm idle setting.
The manager runs in the foreground and installs no background service.

### How do I update an older installation?

Stop the manager first. In a Git checkout, run `git pull` to get the latest
source. For a standalone copy, run this in the folder containing your script
to replace it with the latest public release:

```sh
curl -fL https://github.com/benbuschmann/darkbloom-manager/releases/latest/download/warm_model_manager.py \
  -o warm_model_manager.py.new &&
chmod +x warm_model_manager.py.new &&
mv warm_model_manager.py.new warm_model_manager.py &&
python3 warm_model_manager.py --version
```

Then use your existing launch command. Replacing the script leaves saved state
intact. [Release notes and checksums](https://github.com/benbuschmann/darkbloom-manager/releases/latest)
are available on GitHub. The commands and filenames stay the same across releases.

When upgrading from output-only scoring, the manager fetches both prices before
scoring. If that fetch fails, prices remain unavailable until it succeeds.
The formula change resets live and dry-run passing-check counts once. Retained
pressure samples, minimum warm time, and any already-issued pending launch stay
in place; normal sample expiration and timing-setting resets still apply.

Remove old pairing and exploration flags, including `--no-pair` and
`--no-exploration`. Those features are gone. Pending selections that contain
multiple models are discarded. Only one model can satisfy warm-up.

Run one manager per provider. The lock prevents overlapping old and new
processes; do not delete a lock file while a manager is running.

## Files and support

Darkbloom's daemon state defaults to `~/.darkbloom/daemon-state.json`. Override
it with `--daemon-state` or `DARKBLOOM_STATE_FILE`. The provider config defaults
to `~/.config/darkbloom/provider.toml`.

Report bugs in [GitHub issues](https://github.com/benbuschmann/darkbloom-manager/issues)
with the manager, Darkbloom, Python, and macOS versions, your chip and RAM, model
IDs, a redacted command, and the relevant report lines. Leave out tokens and
private config files. Development checks are in [CONTRIBUTING.md](CONTRIBUTING.md).

This project's code and documentation use the [MIT license](LICENSE).
Darkbloom is installed separately and has its own
[license](https://github.com/Layr-Labs/d-inference/blob/master/LICENSE).
