# Recruitix Code Evaluation Service

Implements the "Code evaluation" row of the Recruitix BRD's AI/ML model
table: a sandboxed test-case runner plus static/complexity analysis,
grading coding-round submissions on **correctness** and **efficiency**.
This is the highest-risk Recruitix service so far — it executes arbitrary,
untrusted, candidate-submitted code.

## Sandboxing approach — and why

The obvious "reuse a proven engine" choice was
[Piston](https://github.com/engineer-man/piston) (used by many
competitive-programming judges). It builds isolation by running `nsjail`
*inside* its own container — but constructing `nsjail`'s namespaces
requires the *outer* Piston container itself to run with `--privileged`
(or an equivalent broad capability set). That would have weakened this
service's own container isolation, on the same Docker daemon as every
other Recruitix service — not a tradeoff worth making for a nested-inside
sandbox when this container is going to be spinning up code execution
containers all day.

Instead, each submission runs in its own **plain, non-privileged**
`docker run`, using only standard Docker isolation controls — verified
empirically before anything was built around it, and re-verified by
`tests/test_sandbox_live.py` on every run:

| Control | Verified behavior |
|---|---|
| `--network none` | a socket `connect()` inside the sandbox raises "Network is unreachable" |
| `--memory` / `--memory-swap` | a memory bomb gets OOM-killed (exit 137), doesn't affect the host |
| `--cap-drop=ALL`, `--security-opt no-new-privileges` | no Linux capabilities beyond the bare minimum, no setuid escalation |
| `--read-only` + a small `--tmpfs /tmp` | writing outside `/tmp` raises "Read-only file system"; `/tmp` itself works |
| `--user 1000:1000` | never runs as root inside the container |
| `--pids-limit` | bounds fork-bomb potential |
| wall-clock timeout + `docker kill` by container name | an infinite loop is actually terminated, and its container is confirmed gone afterward, not just detached from |

This is a **weaker** isolation boundary than `nsjail`/`isolate` — a Docker
runtime escape vulnerability isn't defended against by a second sandboxing
layer the way it would be with Piston/Judge0. It was chosen because it
requires no privilege that weakens this service's own container security,
which was the deciding tradeoff for a service running on a shared dev
machine. See "Known limitations" for what a hardened production
deployment should reconsider.

## Correctness and efficiency — how they're actually graded

**Correctness**: run the submission against each test case in the sandbox,
compare (whitespace-normalized) stdout to the expected output. Nothing
inferred — this is what the code actually does.

**Efficiency**: true Big-O inference from static code alone is an
unsolved general problem (undecidable in the worst case). Rather than
guess, this *measures* it — the same "measure, don't guess" philosophy as
[answer-grading](../answer-grading)'s semantic scoring:

1. Test cases can be tagged with an `size_n` (the input size they
   represent).
2. The submission runs once per size; runtimes are recorded.
3. A **measured baseline overhead** (container + interpreter startup,
   run once via a no-op program) is subtracted from each timing — found
   necessary via live testing, since that fixed overhead (several hundred
   ms) otherwise completely swamps real algorithmic work at small input
   sizes.
4. The corrected (size, runtime) points are fit against candidate growth
   models (`O(1)`, `O(log n)`, `O(n)`, `O(n log n)`, `O(n²)`, `O(n³)`,
   `O(2ⁿ)`) via least squares; whichever fits best (by a normalized-RMSE
   goodness metric — see the module docstring for why not textbook R²,
   which is mathematically degenerate for the constant-time model) is
   reported, along with the raw timing data and a confidence level.

If the caller supplies an `expected_complexity` (what a correct solution
*should* achieve), the measured class is compared against it for the
efficiency score component. Without a target — or when confidence in the
estimate is too low to trust — efficiency is **excluded** from
`overall_score` rather than filled in with a guess.

**Static analysis** (Python only currently): cyclomatic complexity and
maintainability index via `radon`, style/lint issues via `ruff` — both
pure static analysis over source text, no execution, safe to run directly
without sandboxing.

## Endpoint

```
POST /api/v1/code/evaluate
{
  "language": "python",
  "source_code": "...",
  "test_cases": [{"input": "...", "expected_output": "...", "size_n": 100}, ...],
  "expected_complexity": "O(n log n)"
}
→
{
  "correctness": {"passed": 4, "total": 4, "pass_rate": 1.0},
  "test_results": [...],
  "static_analysis": {"cyclomatic_complexity": 3, "lint_issues": [...]},
  "efficiency": {"estimated_complexity": "O(n^2)", "confidence": "high", "runtime_by_size": [...]},
  "overall_score": 0.88,
  "explanation": "..."
}
```

## Run locally

```bash
pip install -r requirements-dev.txt
docker pull python:3.11-slim
docker pull node:20-slim
uvicorn app.main:app --reload
```

No `.env` needed — this service calls no external API. It does need a
reachable Docker daemon (the same one you're running commands against
locally) to spin up sandbox containers.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/code/evaluate` | source code + test cases -> correctness/efficiency/static-analysis grade |
| `GET /health/live` | liveness — process is up |
| `GET /health/ready` | readiness — Docker reachable, language images present |
| `GET /api/v1/capabilities` | supported languages, sandbox limits |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | interactive Swagger UI |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `CODE_EVAL_PYTHON_IMAGE` / `CODE_EVAL_JAVASCRIPT_IMAGE` | `python:3.11-slim` / `node:20-slim` | sandbox language images (pin an exact digest for a real deploy) |
| `CODE_EVAL_SANDBOX_TIMEOUT_SECONDS` | `10` | wall-clock limit per sandboxed run |
| `CODE_EVAL_SANDBOX_MEMORY_MB` | `128` | memory limit per sandboxed run |
| `CODE_EVAL_SANDBOX_CPUS` | `0.5` | CPU share per sandboxed run |
| `CODE_EVAL_MAX_SOURCE_CODE_CHARS` | `20000` | caps submitted source |
| `CODE_EVAL_MAX_STDIN_CHARS` | `50000` | caps a test case's input/expected_output — generous enough for genuine efficiency-probing inputs (a few thousand integers) |
| `CODE_EVAL_MAX_TEST_CASES` | `20` | caps test cases per submission |
| `CODE_EVAL_CORRECTNESS_WEIGHT` / `_EFFICIENCY_WEIGHT` / `_STATIC_QUALITY_WEIGHT` | `0.7` / `0.2` / `0.1` | score blend (auto-renormalized when a component has no signal) |
| `CODE_EVAL_REQUIRE_API_KEY` | `0` | require an `X-API-Key` header — protects host compute resources (spawning containers), not a paid quota. **Must be turned on before any internet-reachable deploy** |
| `CODE_EVAL_API_KEYS` | empty | comma-separated shared secrets accepted by `X-API-Key` |
| `CODE_EVAL_RATE_LIMIT_STORAGE_URI` | unset | e.g. `redis://host:6379` to share rate limits across replicas |

## Tests

```bash
pip install -r requirements-dev.txt
docker pull python:3.11-slim && docker pull node:20-slim
pytest tests/ -v
```

70 tests across eight files. `test_sandbox_live.py` is the most important
one in this service — it doesn't just check correctness, it **proves the
isolation guarantees**: a memory bomb actually gets OOM-killed, network
really is unreachable, filesystem writes outside `/tmp` really are
blocked, an infinite loop is really terminated (with its container
confirmed gone afterward, not just orphaned). These same checks were run
manually before any code was written, then codified so a future change
can't silently weaken them without a test failing.

- `test_efficiency.py` — the complexity-fitting math against known
  synthetic growth patterns (including the O(1)/R² edge case that was a
  real bug during development, see the module docstring).
- `test_static_python.py` — radon/ruff static analysis, pure text
  parsing, no execution.
- `test_grading.py` — score combination logic with a fake sandbox runner.
- `test_sandbox_live.py` — the real sandbox, correctness and isolation.
- `test_api.py` — HTTP-level tests, plus one full-stack test with nothing
  mocked (a real O(n²) submission through the real endpoint against the
  real sandbox, correctly graded as correct-but-inefficient).
- `test_auth.py` / `test_body_size_limit.py` / `test_rate_limit_redis.py`
  — same patterns as the other Recruitix services.

## Deployment

```bash
docker build -t recruitix-code-eval services/code-eval
docker run -p 8000:8000 -v /var/run/docker.sock:/var/run/docker.sock recruitix-code-eval
```

**This container must be run with the host's Docker socket mounted in**
(Docker-outside-of-Docker) so its `docker run` calls reach a real daemon
that can create sibling sandbox containers.

## Known limitations

- **This is a materially higher trust level than every other Recruitix
  service.** Mounting the Docker socket gives this container real control
  over the host's Docker daemon — in a real deployment it should run on
  its own dedicated host, isolated from anything else a container escape
  could reach, not co-located with other production services the way
  local dev has them.
- The isolation boundary is standard Docker container isolation, not a
  second sandboxing layer like `nsjail`/`isolate` — a Docker runtime
  escape vulnerability isn't defended against by anything additional here.
  A future hardening pass could add gVisor (`runsc`) as the container
  runtime for an extra kernel-syscall-interception layer without needing
  the privileged-container tradeoff Piston/Judge0 require.
- Static analysis (complexity/lint) only covers Python — JavaScript
  submissions execute and grade on correctness/efficiency normally, but
  get no static-quality signal, which is excluded from `overall_score`
  rather than faked.
- Efficiency estimation needs genuinely size-varying test cases (at least
  3 distinct `size_n` values) and inputs large enough that real
  algorithmic work exceeds sandbox-startup noise (a few hundred ms) — a
  problem whose reference solution runs in microseconds even at large N
  may simply report `insufficient_data`/`low confidence` rather than a
  wrong classification, which is the intended fail-safe, not a bug.
- One `docker run` per test case (plus one baseline run when efficiency
  estimation applies) means container-startup latency dominates a
  submission's total grading time — fine for a single coding-round
  evaluation, but would need a warm-container-pool redesign to handle high
  submission throughput.
- Inbound auth is a single shared secret, not per-user — same posture as
  the other Recruitix services.
