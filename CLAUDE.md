# Development

- Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
- Build docs: `quarto render docs`
- `docs/` has its own Project.toml for doc-specific dependencies
- Quarto YAML reference: https://quarto.org/docs/reference/
- Never edit Project.toml or Manifest.toml manually — use Pkg
- For Claude's plan mode, always write a "plan_$task.md" in .claude

# Benchmarks

1. `benchmark/` has its own Project.toml:
   ```
   julia --project=benchmark -e 'using Pkg; Pkg.add(["BenchmarkTools", "JSON3"]); Pkg.develop(path=".")'
   ```
2. `benchmark/benchmark.jl` defines a `BenchmarkGroup` suite, runs it, and writes `benchmark/results.json`
3. `benchmark/push_results.sh` pushes `results.json` to the `benchmark-results` orphan branch via a git worktree
4. Run benchmarks locally:
   ```
   julia --project=benchmark benchmark/benchmark.jl
   bash benchmark/push_results.sh
   ```
5. The Docs workflow fetches `benchmark-results` before rendering, so `docs/pages/benchmarks.qmd` picks
   up the latest results automatically — no `_quarto.yml` changes needed

# Docs Layout

- `docs/assets/` holds `styles.css`, `theme.scss`, `_version-selector.html`, and `logo.svg`
- `docs/pages/` holds `api.qmd`, `coverage.qmd`, `benchmarks.qmd`, and `changelog.qmd`
- `pages/api.qmd` must always be the last item before the "Resources" section in `_quarto.yml`
- `pages/api.qmd` lives in its own `part: "API"` to visually separate it from other doc pages
- `index.qmd` must always begin with `## Overview` and `## Quickstart` sections
- The Julia engine is set project-wide via `engines: ['julia']`; individual .qmd files do not need
  an `engine:` key

# Style

- 4-space indentation
- Docstrings on all exports
- Use `### Examples` for inline docs examples
- Segment code sections with: "#" * repeat('-', 80) * "# " * "$section_title" on a single line

# Releases

- First released version should be v0.1.0
- Preflight: tests must pass and git status must be clean
- If current version has no git tag, release it as-is (don't bump)
- If current version is already tagged, bump based on commit log:
  - **Major**: major rewrites (ask user if major bump is ok)
  - **Minor**: new features, exports, or API additions
  - **Patch**: fixes, docs, refactoring, dependency updates (default)
- Commit message: `bump version for new release: {x} to {y}`
- Generate release notes from commits since last tag (group by features, fixes, etc.)
- Important: For major or minor version bumps, release notes must include the word "breaking"
- Update CHANGELOG.md with each release (prepend new entry under `# Unreleased` or version heading)
- Register via:
  ```
  gh api repos/{owner}/{repo}/commits/{sha}/comments -f body='@JuliaRegistrator register

  Release notes:

  <release notes here>'
  ```
