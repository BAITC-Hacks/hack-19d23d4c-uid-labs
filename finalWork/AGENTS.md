# Handoff rules

- This project solves the original HackAlem transaction-graph case.
- Never hardcode specific real gid results, component counts, roles or winning benchmark metrics.
- Preserve all nodes including isolated seed clients and exact int64 identifiers.
- All roles are hypotheses. role_score is rule strength, not a probability of guilt.
- Boundary absence of outgoing transfers must never imply terminal role.
- Do not use seed out/in ratio for transit or terminal classification.
- Keep identical transaction rows; without a transaction identifier they may be legitimate separate transfers.
- Reconcile edges against transactions; never add their monetary totals together.
- Keep hypothetical continuation edges separate from observed data.
- UI is offline and must safely embed JSON, including strings containing </script>.
- Run `python -m unittest discover -s tests -v` after algorithm changes.
- Run a real Parquet roundtrip with pyarrow installed before delivery to the jury.
- Document missing validation instead of reporting unexecuted tests as passed.
