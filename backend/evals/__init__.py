"""Offline evaluation of the reading agent.

`scripts/verify_*` answer "does the agent work at all" on one hand-picked
conversation. This package answers "how well, and did a change make it worse":
a golden set of questions with reference facts, each run through the same
`agent_runner._execute_turn` a reader's turn takes, then scored for

* answer correctness — which of the case's key points the answer states;
* citation precision / recall — whether each cited passage supports the
  sentence citing it, and how many factual sentences are fully backed;
* abstention — whether a question the paper cannot answer is declined;
* cost — tool calls, model calls, tokens, wall time.

Running the agent and judging its answers are separate steps, so a stored run
can be re-scored (`--rejudge`) after a judge prompt changes without paying for
the turns again.
"""
