## Overleaf Documentation

The thesis documentation is maintained in the following Overleaf project:

[Open the thesis project in Overleaf](https://www.overleaf.com/project/6a192de01f9877e35710e1d1)

### Sync changes from VS Code to Overleaf

After editing files inside `docs/thesis/`, copy and run:

```bash
git add docs/thesis
git commit -m "docs: update thesis documentation"
git push
git subtree push --prefix=docs/thesis overleaf main
```

After the push finishes, open Overleaf and recompile the document if automatic compilation does not start.

### Sync changes from Overleaf to VS Code

If changes were made directly in Overleaf, copy and run:

```bash
git subtree pull --prefix=docs/thesis overleaf main --squash
git push
``` 
