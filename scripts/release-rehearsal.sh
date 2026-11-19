#!/usr/bin/env bash
# Release rehearsal for inspect-safety-stats.
#
# Builds the package from a git ref in a throwaway worktree, checks the
# distribution metadata, installs the wheel into fresh virtual environments for
# every advertised Python version, exercises the CLI and the shipped offline
# example from OUTSIDE the repository, and finally runs the test suite against
# the installed wheel (not the source tree). Nothing here touches the dev venv.
#
# Usage: scripts/release-rehearsal.sh [ref] [workdir]
#   ref      git ref to build (default: main)
#   workdir  scratch directory (default: mktemp)
# Requires: git, uv (https://docs.astral.sh/uv/), network access for dependencies.
set -euo pipefail

REF="${1:-main}"
WORK="${2:-$(mktemp -d)}"
REPO="$(git rev-parse --show-toplevel)"
PYTHONS=(3.11 3.12 3.13)
WARNINGS=()
SRC="$WORK/src"
DIST="$WORK/dist"
mkdir -p "$WORK"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
fail() { printf '\033[31mFAIL:\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() { git -C "$REPO" worktree remove --force "$SRC" 2>/dev/null || true; }
trap cleanup EXIT

step "1. Export $REF into a clean worktree"
git -C "$REPO" worktree add --detach "$SRC" "$REF" >/dev/null
git -C "$SRC" log -1 --format='  %h %s'
VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$SRC/src/inspect_safety_stats/version.py")
echo "  version $VERSION"

step "2. Build sdist and wheel"
(cd "$SRC" && uv build --out-dir "$DIST" >/dev/null 2>&1) || fail "build"
ls -1 "$DIST"
WHEEL=$(ls "$DIST"/*.whl)

step "3. Distribution metadata (twine check)"
uvx twine check "$DIST"/* || fail "twine check"

step "4. Wheel contents"
unzip -l "$WHEEL" | awk '{print $4}' | grep -E '\.(html|typed)$' | sed 's/^/  /'
unzip -l "$WHEEL" | awk '{print $4}' | grep -E 'visualization_template\.html' | wc -l | grep -qx 2 \
  || fail "both HTML templates must be in the wheel"
unzip -l "$WHEEL" | awk '{print $4}' | grep -E '^(tests|examples|research)/|^(addition|simpleqa|test)\.py$' \
  && fail "wheel carries files that are not the package" || echo "  no stray files"

step "5. Prepare an out-of-repo copy of the offline example"
EX="$WORK/example"; rm -rf "$EX"; mkdir -p "$EX"
cp "$SRC"/examples/real_simpleqa/{safety-plan.yaml,design.csv,simpleqa-8-real.eval} "$EX/"
echo "  $EX"

for PY in "${PYTHONS[@]}"; do
  step "6. Python $PY: fresh venv, install wheel, smoke test"
  VENV="$WORK/venv-$PY"; rm -rf "$VENV"
  uv venv --python "$PY" "$VENV" >/dev/null 2>&1 || fail "uv venv $PY"
  # The smoke test reads the example's Inspect log, which needs the inspect extra.
  uv pip install --python "$VENV/bin/python" --quiet "${WHEEL}[inspect]" || fail "install on $PY"
  BIN="$VENV/bin/inspect-safety-stats"
  "$VENV/bin/python" --version | sed 's/^/  /'
  "$BIN" version | sed 's/^/  /'
  "$BIN" plan zero-failure --threshold 0.01 --alpha 0.05 | grep -q "Required rollouts: 299" \
    || fail "zero-failure planner should report 299"
  "$BIN" generative --help >/dev/null || fail "generative subcommands missing"
  OUT="$WORK/report-$PY"; rm -rf "$OUT"
  (cd "$EX" && "$BIN" validate --plan safety-plan.yaml --logs simpleqa-8-real.eval | tail -1 | sed 's/^/  validate: /')
  (cd "$EX" && "$BIN" analyze --plan safety-plan.yaml --logs simpleqa-8-real.eval --output "$OUT" \
      --no-decision-exit-codes | grep -E 'Integrity|Risk measures' | sed 's/^/  /')
  # Compare with the committed example report. A difference is a WARNING, not a
  # failure: it usually means the committed example is stale (see issue on
  # regenerating example reports), which the release must fix but which does not
  # make the wheel wrong.
  "$VENV/bin/python" - "$OUT/report.json" "$SRC/examples/real_simpleqa/report/report.json" <<'PY' || WARNINGS+=("committed real_simpleqa report differs from the $PY build")
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for d in (a, b):
    d.pop("provenance", None)
    d.get("design", {}).pop("manifest", None)
if a != b:
    print("  WARNING: report.json differs from the committed example (stale example?)")
    sys.exit(1)
print("  report.json matches the committed example (ignoring provenance)")
PY
  # The invalid-integrity path must produce diagnostics and a non-zero exit code.
  sed 's/name: model_graded_qa/name: nonexistent_scorer/' "$EX/safety-plan.yaml" > "$EX/wrong.yaml"
  rc=0; (cd "$EX" && "$BIN" validate --plan wrong.yaml --logs simpleqa-8-real.eval >/dev/null 2>&1) || rc=$?
  [[ $rc -eq 3 ]] || fail "validate should exit 3 on a missing score, got $rc"
  echo "  validate on wrong scorer exits $rc (expected 3)"
  rc=0; (cd "$EX" && "$BIN" analyze --plan wrong.yaml --logs simpleqa-8-real.eval --output "$OUT-invalid" >/dev/null 2>&1) || rc=$?
  [[ $rc -eq 3 ]] || fail "analyze should exit 3 on invalid integrity, got $rc"
  [[ -f "$OUT-invalid/report.json" ]] || fail "analyze must still write the invalid report"
  echo "  analyze on wrong scorer exits $rc (expected 3) and writes its report"
done

step "6b. Determinism: identical report.json across Python versions"
"$WORK/venv-${PYTHONS[0]}/bin/python" - "$WORK" "${PYTHONS[@]}" <<'PY' || fail "reports differ between Python versions"
import json, sys
work, *pys = sys.argv[1:]
def load(py):
    d = json.load(open(f"{work}/report-{py}/report.json")); d.pop("provenance", None); return d
base = load(pys[0])
for py in pys[1:]:
    assert load(py) == base, f"report.json differs between {pys[0]} and {py}"
print("  identical on", ", ".join(pys))
PY
for PY in "${PYTHONS[@]}"; do
  a=$(jq -S 'del(.provenance)' "$WORK/report-$PY/report.json" | sha256sum | cut -c1-16); echo "  $PY  $a"
done

step "7. Test suite against the INSTALLED wheel (Python ${PYTHONS[-1]})"
VENV="$WORK/venv-${PYTHONS[-1]}"
uv pip install --python "$VENV/bin/python" --quiet "${WHEEL}[dev]" || fail "install dev extras"
# Run from the worktree so tests/ and examples/ are present, but the package under
# test is the installed wheel: the worktree has no editable install and src/ is not on sys.path.
(cd "$SRC" && "$VENV/bin/python" -c "import inspect_safety_stats as m, sys; p=m.__file__; assert 'site-packages' in p, p; print('  package under test:', p)")
(cd "$SRC" && "$VENV/bin/python" -m pytest -q -p no:cacheprovider 2>&1 | tail -3 | sed 's/^/  /')

if ((${#WARNINGS[@]})); then
  step "Warnings"
  printf '  - %s\n' "${WARNINGS[@]}"
fi
step "Done: $VERSION built from $REF installs cleanly, analyzes the shipped example deterministically on ${PYTHONS[*]}, and passes its test suite from the wheel"
echo "artifacts in $WORK/dist"
