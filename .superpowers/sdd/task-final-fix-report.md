## Final Fix Notes

- Passed through Ark and DeepSeek provider environment variables in `docker-compose.yml` with safe defaults so the container sees copied `.env` values without relying on `env_file`.
- Hardened `.dockerignore` to exclude nested `.env` files by default, explicitly allowlisting `demo/hitl_project/.env`, and excluding `.superpowers/` scratch output from Docker context.
- Removed tracked scratch reports `task-1-report.md` and `task-4-report.md` from the release branch.
- Updated `tests/test_production_release_pack.py` to lock in the new compose and ignore rules.
