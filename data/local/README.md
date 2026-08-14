# Local application state

**Status:** generated locally and ignored, except for this guide.

The API stores its SQLite database at `data/local/bankscope_chat.db` by default. The database owns
threads, messages, ordered session tickers, pipeline outcomes, and citation metadata.

It does not duplicate canonical filing evidence. Opening a citation resolves its target against the
current processed corpus and rejects stale citations when the recorded corpus hash differs.

Deleting this database removes local conversation history but does not affect the corpus or index.
Do not commit it or use it as an evaluation fixture.

[Data lifecycle](../README.md) · [Chat subsystem](../../src/bankscope/chat/README.md)

