# Project governance

NeuroShard currently uses maintainer review. The initial maintainer is [Linir Zamir](https://github.com/0x00LZ), working through the [neuroshard-ai organization](https://github.com/neuroshard-ai). Add maintainers through public proposals describing responsibilities and demonstrated contributions.

Development happens in this public repository. Use issues for reproducible bugs and discussions for research questions. Propose consensus, verification, issuance, or compatibility changes through an RFC before implementation. Include assumptions, invariants, alternatives, migration implications, and falsifiable evaluation.

Maintainers review changes and publish named releases with validation results. Prefer independent review for consensus and accounting changes. Urgent fixes may be released with a public explanation and follow-up review. Disclose relevant conflicts of interest.

Repository permissions and native voting power are separate. Tokens do not grant merge rights. A merged change does not upgrade validators. Operators choose their compatible release and whether to accept a proposed new genesis. The current chain has no automatic software-upgrade or token-holder software governance mechanism.

The launch validators are controlled by one operator across two machines. Additional processes do not establish independent ownership. Independent node, worker, sponsor, and validator operators remain necessary.

Follow the [code of conduct](CODE_OF_CONDUCT.md) and [security policy](SECURITY.md).
