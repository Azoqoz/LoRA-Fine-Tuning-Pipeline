# MODEL SHIFT

A static, prerecorded QLoRA experiment explorer. No backend, API, model runtime,
inference, or GPU. The original experiment inputs are read-only.

## Local development

Use Node.js 22 or newer. From `frontend/`:

```sh
npm install
npm run dev
```

Open http://localhost:3000. Data is generated before development and production
builds. Run `npm run data` again if approved source inputs change.

## Validation and static export

```sh
npm test
npm run lint
npm run build
```

The build produces `out/`, containing the static site. Serve this directory with
any static host. Interactive filtering and metric switching run in the browser.
The optional Google Fonts stylesheet uses a system sans-serif fallback offline.

## Vercel

Import the complete repository, select `frontend` as the Root Directory, enable
**Include source files outside of the Root Directory in the Build Step**, and use
the Next.js framework preset with build command `npm run build`. The data script
needs `../results/` and `../configs/` from the repository. No environment variables,
secrets, server functions, or GPU services are required. Nothing is deployed by
the local build.

## Data contract

`scripts/prepare-data.mjs` reads only:

- `../results/metrics.json`
- `../results/comparison.csv`
- `../results/TRAINING_RESULTS.md`
- `../configs/training_config.yaml`

It uses `csv-parse` for quoted/multiline CSV, validates numeric ranges, unique IDs,
row/category counts, and per-model aggregate scores and deltas within 1e-12.
It derives per-example deltas, Token F1 change labels, and exact-match counts.
Recorded runtime/hardware/counts are extracted from the validated report; missing
metadata fails the build. Generated `src/data/experiment.json` is ignored and
never replaces the source evidence. The lockfile pins installed dependencies.

The public interface displays all 70 original responses as React text nodes.
It distinguishes lexical improvement from correctness and surfaces two known
failure cases. It does not answer new questions or claim unseen-fact generalization.
