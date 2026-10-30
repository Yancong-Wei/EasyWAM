# Contributing to the RoboDojo development core

Keep changes focused on RoboDojo data, training, inference, evaluation and reproducible releases. Preserve the original EasyWAM attribution and license. Model/backbone research should share the existing abstractions rather than reintroducing unrelated benchmark integrations.

Before review:

```bash
python -m unittest discover -s experiments/robodojo/tests -v
git diff --check
```

Update user-facing documentation, preserve train/inference normalization and configuration contracts, and report actual verification levels. CPU/static tests do not establish GPU training or simulator success. Do not commit datasets, weights, generated results, local research snapshots, credentials or machine-local deployment files.

Use a separate evaluation directory whenever checkpoint content or inference settings change. Do not publish partial runs as complete scores or call a local result officially verified.
