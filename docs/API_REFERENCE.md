# Cosmos-Reason2 API Reference

Base URL: `http://localhost:9900`

---

## Health Check

```
GET /api/health
```

```json
{"vllm": "ok", "model": "nvidia/Cosmos-Reason2-8B"}
```

---

## Run Inference

```
POST /api/infer
Content-Type: application/json
```

**Request:**

```json
{
  "prompt": "Assess the casualty's condition and recommend a triage category.",
  "videos": ["/home/dtc/scenario_01.mp4"],
  "images": ["/home/dtc/casualty_closeup.png"],
  "reasoning": true,
  "temperature": 0.6,
  "max_tokens": 4096,
  "top_p": 0.95,
  "fps": 2.0,
  "max_model_len": 16384
}
```

Only `prompt` is required. All other fields are optional.

The default system prompt is: *"You are an EMT looking at a simulated casualty scenario. You will treat it like a real scene. You will not talk about simulation aspects like training setup, fake scene, etc."*

You can override it with the `system_prompt` field.

**Response:**

```json
{
  "content": "The casualty presents with a visible laceration on the left forearm...",
  "reasoning_content": "<think>I can see the subject is conscious and responsive...</think>",
  "usage": {"prompt_tokens": 2048, "completion_tokens": 512, "total_tokens": 2560},
  "duration_s": 8.3
}
```

**Sampling defaults** (auto-applied when omitted):

| Parameter | Reasoning ON | Reasoning OFF |
|-----------|-------------|---------------|
| `temperature` | 0.0 | 0.7 |
| `top_p` | 0.95 | 0.8 |
| `top_k` | 20 | 20 |
| `max_tokens` | 4096 | 4096 |
| `repetition_penalty` | 1.0 | 1.0 |
| `presence_penalty` | 0.0 | 1.5 |
| `fps` | 2.0 | 2.0 |
| `max_model_len` | 16384 | 16384 |

---

## Browse Host Filesystem

```
GET /api/browse?path=/home/dtc/data
```

```json
{
  "path": "/home/dtc/data",
  "entries": [
    {"name": "scenarios", "type": "dir"},
    {"name": "scenario_01.mp4", "type": "video", "size": 52428800},
    {"name": "casualty_closeup.png", "type": "image", "size": 1048576}
  ]
}
```

Entry types: `dir`, `image`, `video`, `file`. Paths are relative to the host filesystem.

---

## Serve Media File

```
GET /api/media?path=/home/dtc/scenario_01.mp4
```

Returns the raw file content with appropriate MIME type. Used by the UI for video/image preview. Path must be on the host filesystem.

---

## Estimate Token Usage

```
POST /api/estimate-tokens
Content-Type: application/json
```

```json
{
  "images": ["/home/dtc/casualty_closeup.png"],
  "videos": ["/home/dtc/scenario_01.mp4"],
  "prompt": "Assess the casualty.",
  "fps": 2.0,
  "max_model_len": 16384
}
```

**Response:**

```json
{
  "image_tokens": 340,
  "video_tokens": 2048,
  "text_tokens": 4,
  "total_tokens": 2392,
  "max_model_len": 16384,
  "max_tokens": 4096,
  "budget_remaining": 9896
}
```

Token counts are approximations based on image/video resolution and `PIXELS_PER_TOKEN`. Actual tokenization happens inside vLLM.

---

## Workspaces

Workspaces group folders from different locations on the host for repeated use.

**List all:**

```
GET /api/workspaces
```

```json
{
  "workspaces": [
    {
      "id": "a1b2c3d4-...",
      "name": "workspace0",
      "folders": ["/home/dtc/videos", "/home/dtc/masks"],
      "created_at": "2026-03-02T20:00:00+00:00",
      "updated_at": "2026-03-02T20:00:00+00:00"
    }
  ]
}
```

**Create:**

```
POST /api/workspaces
Content-Type: application/json
```

```json
{"name": "triage_session_1", "folders": ["/home/dtc/videos", "/home/dtc/masks"]}
```

Both fields are optional. Name auto-generates as `workspace0`, `workspace1`, etc. if omitted.

**Get one:**

```
GET /api/workspaces/<id>
```

**Update:**

```
PUT /api/workspaces/<id>
Content-Type: application/json
```

```json
{"name": "renamed", "folders": ["/home/dtc/videos"]}
```

**Delete:**

```
DELETE /api/workspaces/<id>
```

```json
{"ok": true}
```

---

## List Models

```
GET /api/models
```

```json
{"models": ["nvidia/Cosmos-Reason2-8B"]}
```

---

## Get Sampling Defaults

```
GET /api/defaults?reasoning=true
```

Returns the default sampling parameters for the specified mode (includes `fps`).

---

## Examples (curl)

**Health check:**

```bash
curl http://localhost:9900/api/health
```

**Single video — triage assessment with reasoning:**

```bash
curl -X POST http://localhost:9900/api/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Assess the injuries visible on this casualty. What is the appropriate triage category (Immediate, Delayed, Minor, Expectant)?",
    "videos": ["/home/dtc/scenario_01.mp4"],
    "reasoning": true,
    "fps": 2.0
  }'
```

**Two videos — compare casualties across scenes:**

```bash
curl -X POST http://localhost:9900/api/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Compare the two casualties. Which one requires more urgent medical attention and why?",
    "videos": [
      "/home/dtc/casualty_A.mp4",
      "/home/dtc/casualty_B.mp4"
    ],
    "reasoning": true,
    "fps": 2.0,
    "max_tokens": 4096
  }'
```

**Video + image — injury cross-reference:**

```bash
curl -X POST http://localhost:9900/api/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "The image shows a reference chart for burn severity classification. Based on the video of this casualty, classify the burn injuries you observe using the chart categories.",
    "videos": ["/home/dtc/burn_casualty.mp4"],
    "images": ["/home/dtc/burn_severity_chart.png"],
    "reasoning": true,
    "fps": 2.0
  }'
```

**Image-only — static injury assessment:**

```bash
curl -X POST http://localhost:9900/api/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Describe the visible injuries on this casualty and list the immediate interventions needed.",
    "images": ["/home/dtc/casualty_photo.jpg"]
  }'
```

**Create a workspace and list it:**

```bash
# Create
curl -X POST http://localhost:9900/api/workspaces \
  -H 'Content-Type: application/json' \
  -d '{"name": "field_exercise", "folders": ["/home/dtc/videos", "/home/dtc/snapshots"]}'

# List all
curl http://localhost:9900/api/workspaces
```

**Estimate tokens before running a large request:**

```bash
curl -X POST http://localhost:9900/api/estimate-tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "videos": ["/home/dtc/long_scenario.mp4", "/home/dtc/followup.mp4"],
    "images": ["/home/dtc/injury_detail.png"],
    "prompt": "Assess both scenarios and provide a combined triage report.",
    "fps": 2.0,
    "max_model_len": 16384
  }'
```

---

**Note:** File paths refer to locations on the **host machine** where the Docker container is running. If accessing the API remotely, set up an SSH tunnel: `ssh -fNL 9900:localhost:9900 <host>`, then use `http://localhost:9900`.
