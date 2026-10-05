# Workspace integration source backup

These bundles preserve the owned AnimFlow, shared game-engine, Lonewolf and
SimplexGen changes while the first three projects lack publication remotes.
They are workspace-relative patches, not standalone application repositories.
Do not apply them to this repository or to an unmatched baseline; the current
workspace already contains these changes. Preserve unrelated working edits.

- `20261004-combat-discovery.patch.gz`: first-tranche procedural wolf attacks,
  humanoid capture/loop review, project indexing, and landing discovery changes.
- `20261005-kimodo-library.patch.gz`: generated-library browsing, validated
  SOMA retargeting, adaptive camera/root-height handling, source metadata search,
  tests, and rendered game-rig reference anchors. Reverse-apply validation passes
  against the current workspace. `source-files.json` records final file hashes.

The full 35-clip BVH dataset is deliberately excluded from Git. Its indexed
SHA-256 URLs are served by https://animflow.app.nz/motions/kimodo/index.json.
After restoring a matching workspace and its index, fetch only the immutable
runtime files:

```bash
node integrations/animflow/restore-runtime-assets.mjs /path/to/workspace
```

The helper verifies checksums, bounds response sizes and
never overwrites an existing file with different contents. The production
library is also reproducible from the retained Kimodo archive using
`python library_server.py --export ../animflow/public`.

These backups do not replace configuring an AnimFlow/games/Lonewolf Git remote.
The SimplexGen publication branch is
`lee101/hiresnz2:improve-motion-discovery-clean-20261005`; only unpublished
history is replayed without databases, local executables and training arrays.
No working database, staged user change, original branch or published history
is deleted or force-pushed.

All generated assets retain their NVIDIA model/license provenance. The anchors
are procedural renders, not generated-video or mocap evidence. No fresh model
generation, training, video submission or GPU rental is included in this backup.
