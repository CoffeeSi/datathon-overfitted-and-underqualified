# Фокус — cashier-video review

**Overfitted and Underqualified · ForteBank Datathon**

Фокус is a prototype for uploading a `.mov` cashier video and reviewing
the resulting job. The interface shows the verdict, processing status, event
timeline, descriptions, and evidence frames when findings are available.

**Live frontend:** [Open Фокус](https://frontend-hapejow6c.gobrev.dev/)

## Final result

- **Working review flow:** The Next.js interface sends uploads through its
  server to the FastAPI job API, polls for completion, and displays the result.
  The complete flow has been verified with an explicitly labeled mock analyzer.
- **Measured visual prototype:** A separate banknote tracking experiment found
  the reviewed X-38 event at 29 seconds, four seconds from its annotation; it
  produced no candidates on three reviewed negative videos. This small check
  does not establish independent quality and is not enabled by the repository's
  default local Compose stack.
  See the [banknote result](reports/banknote_study_20261018.md).
- **Repository action analyzer:** A research checkpoint completed three real MOV jobs
  through the API, but its two evaluated action types had macro F1 = 0 on sparse
  validation windows. The default real-analyzer config therefore enables no
  event types. See the [backend result](reports/action_backend_20261018.md).

The result is a human review aid. It does not make a legal conclusion, and the
current models have not passed an independent test on new recording sessions.

## Display the result

Open the [deployed frontend](https://frontend-hapejow6c.gobrev.dev/) to see the
upload and review interface. On 18 October 2026, its service endpoint reported
`health=online`, `ready=true`, and `provider=local-cash-sequence`. This status
check did not include a video upload or a quality evaluation of the deployed model.

### Reproduce the upload flow locally

With Docker Compose available, run these commands from the repository root.
Create `.env.compose` and set a nonempty `FORTEBANK_API_KEY` before starting:

```powershell
if (-not (Test-Path .env.compose)) { Copy-Item .env.compose.example .env.compose }
# Set FORTEBANK_API_KEY in .env.compose.
docker compose --env-file .env.compose up --build -d
Invoke-RestMethod http://127.0.0.1:3000/api/review/service
```

Open <http://127.0.0.1:3000> and upload a valid `.mov`. The service check should
show `health=online`, `ready=true`, and `provider=mock`. This mode demonstrates
upload, job progress, and result display; the mock does not inspect video content
or produce real findings. To stop the containers while keeping their data volume:

```powershell
docker compose --env-file .env.compose down
```

### Show the event-review screen

The interface also includes clearly labeled synthetic examples with timecoded
events and evidence images. Stop the Compose stack first so port 3000 is free,
then run the frontend with Node.js 24 or 22.13+:

```powershell
cd frontend
npm ci
$env:DEMO_ENABLED = "true"
npm run dev -- --hostname 127.0.0.1
```

Open <http://127.0.0.1:3000/review/demo-violation>. This example needs no
backend and does not represent a model prediction. Other synthetic states are
listed in the [frontend guide](frontend/README.md#явный-демонстрационный-режим).

### Run the real analyzer when evaluated weights are available

The optional Compose override requires compatible binary and action checkpoints,
paths in a local config, and reviewed types in `enabled_types`. The supplied
config has `enabled_types: []`; starting the override without these prerequisites
leaves `/readyz` unavailable. From the repository root:

```powershell
if (-not (Test-Path configs/action_service.local.yaml)) {
    Copy-Item configs/action_service.yaml configs/action_service.local.yaml
}
# Set checkpoint paths and enabled_types in configs/action_service.local.yaml.
# Set MODEL_RUNS_PATH and ACTION_SERVICE_CONFIG in .env.compose.
docker compose --env-file .env.compose -f compose.yaml -f compose.real.yaml up --build -d
```

The override disables mock results. See the [frontend Compose guide](frontend/README.md#docker-compose)
for the model-path settings and readiness checks.
