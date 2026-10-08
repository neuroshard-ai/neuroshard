# Try the grown assistant on your own machine

**Research preview.** This runs the assistant that passed A3's sealed confirmation,
Granite 4.1 3B with three learned modules, on your own CPU. It is not a hosted
service and not yet run by independent operators (that is A5). It was measured
only on fictional workspace tasks: drafting documents from project files and
scheduling meetings on team calendars.

## What you need

- Linux x86_64 with Python 3.12, about 32 GB of RAM (peak use was 24 GB) and
  15 GB of free disk.
- The published numbers came from Intel CPUs with AMX (Sapphire Rapids or newer,
  such as AWS r7i). Other x86_64 CPUs work, but replies are slower and may differ.

## Install

On Ubuntu 24.04, first run `sudo apt install git python3.12-venv`.

```bash
git clone --branch main https://github.com/neuroshard-ai/neuroshard.git
cd neuroshard
python3.12 -m venv ~/.venvs/neuroshard-assistant
source ~/.venvs/neuroshard-assistant/bin/activate
python -m pip install -r docs/granite-reference-requirements.txt cryptography==50.0.1
python -m pip install --no-deps -e .
neuroshard assistant fetch
```

`fetch` downloads the modules (about 250 MB) from the
[assistant-preview-1 release](https://github.com/neuroshard-ai/neuroshard/releases/tag/assistant-preview-1)
and Granite 4.1 3B (about 6.8 GB) from its pinned Hugging Face revision. Every file
is checked against its pinned digest before use.

## See the results yourself

```bash
neuroshard assistant replay --set drafting --limit 3
```

This re-runs published confirmation conversations and compares each outcome, and
each generated token, with the published episode. Use `--set scheduling` or
`--set cross`; `--limit 0` runs a whole set (the 192 drafting conversations took
about 3 hours on 8 AMX cores). Published results: drafting 192/192, scheduling
192/192, cross 47/48 ([report](ASSISTANT_REPEATED_GROWTH_COHORT3_CONFIRMATION_RESULTS.md)).

## Use it

```bash
neuroshard assistant chat
```

This opens a sample workspace of project documents and team calendars, and
suggests that workspace's own requests. Start with a suggestion, then follow up
with corrections or your own wording. Each reply shows which route answered
(drafting or scheduling), the tools it called, and the saved drafts and meetings.
A reply takes about a minute.

`--sample N` picks one of 48 sample workspaces. `--world file.json` loads your
own: `documents` (each with `id`, `project`, `title`, `revision`, `status`
`approved` or `draft`, and `content`) and optional `calendars` (each with a
`team` and its `busy` intervals by ISO date). Calendar teams are the six
fictional ones of the sample workspaces.

## What to expect

- Requests like the suggestions are what it was measured on. Further from them,
  quality is unmeasured; free-form questions are out of scope.
- Tools change only an in-memory workspace; nothing is sent. Conversations stay
  on your machine and are not training data.
- A turn that runs out of its budget ends the conversation, as it ends an
  evaluated episode; `/new` starts another.

## Help it grow

- Contribute reviewed corrections: [contributor guide](CONTRIBUTOR_ALPHA.md).
- Run one of the first independent nodes: [operator kit](OPERATOR_KIT.md).
