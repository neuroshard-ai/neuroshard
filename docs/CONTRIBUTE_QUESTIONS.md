# Write test questions for the assistant

Every new version the network grows must answer real requests better than the version it replaces.
Those requests come from a set of realistic workspaces and questions. Part of each version's sealed test
is drawn from it by public randomness, after training ends. Adding to that set is the most direct say
anyone has over what the network learns, and it needs no server.

## What to send

One **workspace**: 6 to 12 short documents that a real person might keep. For each workspace, write
**4 to 10 questions** about it.

- **Documents:** emails, receipts, notes, policies, bookings, letters. Make them untidy the way real
  ones are: an old version superseded by a newer one, near-duplicates, several dates and amounts that are
  easy to confuse. Invent everything. **No real personal data.**
- **Questions:** the way you would actually type them: short, casual, sometimes vague. Mix:
  - direct lookups;
  - questions that need the *current* version of something;
  - one or two that need a simple calculation;
  - **at least one the documents cannot answer** but that sounds as if they might.
- **Answers:** for each question, the exact value a correct answer must contain, copied as the document
  writes it, and which document(s) it comes from. For an unanswerable question, say so.

## Format

```json
{
  "documents": [
    {"id": "doc-1a2b3c4d", "title": "Boiler service invoice", "content": "Invoice 4471. Boiler serviced 3 May. Total due: 145 pounds."},
    {"id": "doc-2b3c4d5e", "title": "Old boiler quote", "content": "Quote from March: boiler service 160 pounds."}
  ],
  "questions": [
    {"user": "what did the boiler service cost in the end?", "facts": ["145 pounds"], "source_ids": ["doc-1a2b3c4d"]},
    {"user": "did they say when the next service is due?", "not_found": true}
  ]
}
```

Document IDs are `doc-` followed by 8 lowercase hex characters.

## What happens to it

Each question is kept only if it passes the same checks the existing set did
(`src/neuroshard/evolution/real_documents.py`):
- **Mechanical:** every fact appears in a cited document, and the route's scorer accepts the answer.
- **Blind:** a separate model call, given only the documents, reaches the same answer.

Accepted questions are never used for training. Owners admit only training data they can rebuild
themselves, and that rule keeps test questions out of it.

Open a pull request adding your file under `config/experiments/contributed-real-documents/`, or attach it
to a [GitHub issue](https://github.com/neuroshard-ai/neuroshard/issues/new?title=Test%20questions:%20). You
can also send it to info@neuroshard.com.
