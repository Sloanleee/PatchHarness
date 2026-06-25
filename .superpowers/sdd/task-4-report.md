Task 4 report

Scope completed:
- Added the README release-doc regression test in `tests/test_production_release_pack.py`.
- Updated `README.md` with public demo instructions for Docker Quick Start, Docker Compose Quick Start, LangGraph HITL demo flow, production demo evidence generation, and mock-default / real-provider `.env` guidance.
- Removed the public docs endpoint from `README.md` and added a regression assertion that `http://127.0.0.1:8000/docs` stays absent.

TDD evidence:
- Red run: `C:\D\code\harness\.venv\Scripts\python.exe -m unittest tests.test_production_release_pack`
  - Result: failed as expected because `README.md` did not yet contain the required public release phrases.
  - Key failure: `AssertionError: 'Docker' not found in README.md`.
- Green run: `C:\D\code\harness\.venv\Scripts\python.exe -m unittest tests.test_production_release_pack`
  - Result: `Ran 6 tests in 0.003s`, then `OK`.
- Review-fix verification: `C:\D\code\harness\.venv\Scripts\python.exe -m unittest tests.test_production_release_pack`
  - Result: `Ran 6 tests in 0.003s`, then `OK`.

Commit:
- `8433857` - `Document production demo release flow`

Notes:
- No extra files were modified beyond the requested README, test, and this report.
- No remaining concerns from the implementation itself.
