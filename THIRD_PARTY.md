# Third-party provenance

Project code uses [Apache 2.0](LICENSE). External inputs retain their own licenses and attribution requirements.

| Material | Source and treatment |
|---|---|
| CometBFT 0.38.26 and ABCI definitions | [CometBFT](https://github.com/cometbft/cometbft/tree/v0.38.26), derived from Tendermint; [Apache 2.0 license](licenses/CometBFT-LICENSE). The installer builds the engine separately. The minimal ABCI definitions and generated bindings preserve protocol compatibility. |
| Tiny Shakespeare | `docs/eval/data/input.txt` and its wheel copy match [Karpathy's char-rnn corpus](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt), SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. The upstream [README identifies its license as MIT](https://github.com/karpathy/char-rnn#license). This distribution attributes the corpus preparation to Andrej Karpathy and the underlying text to William Shakespeare. No private user corpus is included. |
| Model weights | Initialized and trained by this repository's prescribed profile; no external pretrained checkpoint is bundled. See the [model card](docs/MODEL_CARD.md). |
| IEEEtran | Existing `docs/IEEEtran.cls`; preserve its embedded authorship and LaTeX Project Public License notices. Paper text and claims belong to its authors. |
| Python/web dependencies | Installed separately. Native versions are pinned in `docs/demo-requirements.txt`; npm lockfiles record resolved web packages and licenses. |
| Icons and fonts | Lucide through npm; Instrument Sans, Space Grotesk, and JetBrains Mono through Google Fonts, under their upstream licenses. |
| NeuroShard logos | Existing project assets retained from the original repository; no new third-party illustration introduced. |

Preserve provenance and required notices when adding code, data, models, or assets. Historical materials retain their original attribution and do not establish production readiness.
