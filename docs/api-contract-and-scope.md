# API Contract and Event Scope

Status: **draft for team alignment**. Confirm the open questions at the end with the organizers before treating the proposed details as final.

## Goal

Accept a cashier video, process it locally, and return a human-reviewable result: a video-level verdict plus any visible event findings, with timestamps, evidence frames, explanations, and Kazakhstan public-source references.

## Requirements stated in the case

- Input is a `.mov` video uploaded as the `file` field.
- Every request to the jury API must include a valid `X-API-Key` header.
- Processing is asynchronous because a video can take several minutes.
- The API has two operations:
  - `POST /api/v1/analyze` starts a job and returns a `job_id`.
  - `GET /api/v1/jobs/{job_id}` returns the job status and, when complete, the result.
- Job statuses are `processing`, `done`, and `failed`.
- A completed result uses a video-level verdict and a `violations` list. The case schema names `no_violation` and `violation`; the FAQ also allows “insufficient data” but warns against abusing it. This draft represents that third verdict as `insufficient_data` unless the jury specifies a different wire value.
- Each finding includes a `type`, `timecode_sec`, `description`, and 1–3 JPEG `evidence_frames`, encoded as base64 with a timestamp for each frame.
- The service must return a result within five minutes. The jury marks a video unprocessed if the service is unavailable or does not return a result in that time.
- The service must analyze only visual evidence, run locally, and avoid cloud AI services. Face identification and audio analysis are prohibited.

## Proposed request and response shape

The case does not specify HTTP status codes or the exact JSON token for “insufficient data.” These examples are a practical proposal for the team to implement consistently.

### Start analysis

```http
POST /api/v1/analyze
X-API-Key: <key>
Content-Type: multipart/form-data

file=<video.mov>
```

Proposed response: HTTP `202 Accepted`.

```json
{
  "job_id": "3f2a9c1e",
  "status": "processing"
}
```

### Read a job while it is processing

Proposed response: HTTP `200 OK`.

```json
{
  "job_id": "3f2a9c1e",
  "status": "processing"
}
```

### Read a completed job

Proposed response: HTTP `200 OK`.

```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "<team-defined event type>",
      "timecode_sec": 44.2,
      "description": "What is visible and why it may warrant review.",
      "evidence_frames": [
        {
          "timestamp_sec": 44.2,
          "image_base64": "<base64-encoded JPEG>"
        }
      ]
    }
  ]
}
```

For `no_violation` or `insufficient_data`, return an empty `violations` array. If any finding is supported, return `violation` with the supported findings and omit unclear segments. A video may contain multiple findings. The proposed machine value for “insufficient data” is `insufficient_data`; change it if the jury publishes a required token.

### Verdict policy

- `violation`: at least one event is supported by visible evidence. Include supported findings and omit unclear segments.
- `no_violation`: the video is sufficiently observable, and no supported violation is found.
- `insufficient_data`: use sparingly, only when visual quality, occlusion, or missing coverage makes the video as a whole too difficult to assess. Do not use it to avoid a difficult classification, to cover one unclear segment when another finding is supported, or to lower the false-positive risk without a concrete visibility problem.

### Failed jobs and request errors

The case specifies a `failed` job status but does not define its response body or HTTP error codes. Proposed behavior: include a short machine-readable error code and a safe, concise message for `failed` jobs; reject missing/invalid API keys and unsupported or malformed uploads with standard client-error responses. Do not include a `verdict` for a failed job.

## Proposed event scope

The case provides only a violation flag and timestamp for each training video; it does not provide event types. Final categories therefore depend on inspecting the dataset and researching the public rules.

Include an event type only when all of these are true:

1. The behavior can be described from visible frames without relying on transaction records, internal procedures, or other hidden facts.
2. The team can identify the event time and one or more frames that let a reviewer verify the description.
3. A relevant Kazakhstan public source can be cited accurately, with the source title and section or link.
4. The category is specific enough to label consistently and test on held-out videos.
5. The wording presents the result as a possible violation for human review, not as a final legal or disciplinary conclusion.

Defer any category that lacks visible evidence, a defensible source, or consistent examples. Do not infer identity, intent, correctness of a system posting, or a required action that cannot be observed in the video. Keep the initial event catalog small; expand it only when the data and source research support it.

## Out of scope

- Cloud-based video, image, speech, or language-model processing.
- Audio analysis or face identification.
- Assertions based on hidden banking systems, transaction correctness, internal logs, or unavailable ForteBank-only procedures.
- Publishing, sharing, or training on jury test videos.
- Definitive legal, disciplinary, or employee-identity conclusions.

## Organizer questions to resolve

1. What exact JSON token should the API use for “insufficient data,” and are there jury scoring rules beyond the warning not to abuse it?
2. What exact body and HTTP status should a `failed` job return?
3. Does the jury require a specific HTTP status/body for the initial `POST`, and are there file-size or upload limits beyond the `.mov` format and approximately 29 MB example?

## Team decisions to record after alignment

- Final response schema and error behavior, including the jury's exact “insufficient data” token if it differs from `insufficient_data`.
- Initial event catalog, source links, and cautious descriptions.
- Validation split by video and the way false alarms, misses, timestamps, and evidence quality will be reviewed.
- Whether the API returns image data inline only, or also supports a separate evidence URL. The sample response shows base64 frames.
