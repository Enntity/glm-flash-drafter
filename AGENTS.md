# Notes for coding agents

- **Joining volunteer training with an invite link:** follow [JOIN.md](JOIN.md) step by step. It lists the exact
  commands, the expected output of each check and the safety rules (GB10 shared memory, never run the target
  server and the trainer on the same node, never print the site token).
- **Developing this repository:** read [CONTRIBUTING.md](CONTRIBUTING.md). Every tensor load goes through
  `enntity_drafter/guard.py`; run the CPU test file you touched, not the whole suite
  (`OMP_NUM_THREADS=1 python -m pytest tests/test_<name>.py`).
- **Operating the network:** [docs/operators.md](docs/operators.md). The coordinator (`coordinator/`) must stay a
  stdlib control plane that never loads weights; the validator (`validator/`) and site client
  (`enntity_drafter/join.py`) share the delta checks in `enntity_drafter/validation.py`.
