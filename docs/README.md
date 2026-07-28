# Editing the YASPS website

The numbered Markdown files are the main learning path:

| Chapter | Markdown source | Public page |
| --- | --- | --- |
| 01 | `01-getting-started.md` | `/getting-started/` |
| 02 | `02-scene-mesh-and-primitive-model.md` | `/concepts/` |
| 03 | `03-attributes-and-expressions.md` | `/attributes/` |
| 04 | `04-connectivity-and-join.md` | `/join/` |
| 05 | `05-primitive-unions.md` | `/union/` |
| 06 | `06-energies-and-minimization.md` | `/optimization/` |
| 07 | `07-dynamic-topology.md` | `/dynamic-scenes/` |
| 08 | `08-mixed-bodies-with-separated-assembly.md` | `/tutorials/mixed-separation/` |

`index.md` is the homepage. The files under `advanced/` and `reference/`, plus
`architecture.md`, `examples.md`, and `troubleshooting.md`, are supplementary
pages rather than numbered chapters.

## Publish an edit

The website deploys automatically from the `gh-pages` branch. A local commit is
not enough; the commit must also be pushed to GitHub.

```bash
git switch gh-pages

# Edit one or more files, then validate them.
python3 docs/scripts/check_docs.py

# Stage only the files you changed.
git add docs/03-attributes-and-expressions.md
git commit -m "docs: clarify attributes"
git push origin gh-pages
```

GitHub Actions will rebuild
<https://txstc55.github.io/yasps/> after the push. This normally takes about a
minute. If you edit a file directly on GitHub, commit it to the `gh-pages`
branch to trigger the same deployment.

The `permalink` in each file controls its public URL. Renaming a Markdown file
does not change the webpage as long as its `permalink` remains unchanged.
