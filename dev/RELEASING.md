# Releasing scrich

One GitHub release updates everything: the git tag sets the package version, the
`Release` workflow publishes to PyPI, Zenodo archives the code (new DOI version) and
Read the Docs builds the `stable` docs.

## One-time setup (done once)

- PyPI: trusted publisher for `scrich` (see the header of `.github/workflows/release.yml`).
- GitHub: Settings -> Environments -> `pypi` (created automatically on the first run, or by hand).
- Zenodo: GitHub integration switched on for this repository.

## Each release

1. On `main`, all checks green (Tests, Lint, Docs).
2. Update `CHANGELOG.md`: new section `## X.Y.Z (YYYY-MM-DD)`. Commit and push.
3. Optional dry run: Actions -> Release -> Run workflow (builds and checks, does not publish).
4. GitHub -> Releases -> Draft a new release:
   - tag `vX.Y.Z` (create on publish, target `main`), title `scRICH X.Y.Z`,
   - notes: paste the changelog section,
   - Publish release.
5. Check: Actions -> Release green; https://pypi.org/project/scrich/ shows X.Y.Z;
   new Zenodo version; https://scrich.readthedocs.io/en/stable/ shows X.Y.Z.
6. Test in a fresh environment: `pip install scrich==X.Y.Z`.

Versions: `X.Y.Z` = major.minor.patch. Bug fixes -> patch (1.0.1); new features ->
minor (1.1.0); changes that break existing code -> major (2.0.0).
A version number can be uploaded to PyPI only once; to fix a broken release, make a new one.
