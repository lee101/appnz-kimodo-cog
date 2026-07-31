# Third-party models and code

The MIT license in this repository covers only this adapter and orchestration
code. It does not relicense downloaded dependencies or model weights.

| Component | Default identifier | Terms |
| --- | --- | --- |
| Kimodo source | `nv-tlabs/kimodo` | Apache-2.0 |
| Kimodo motion checkpoint | `nvidia/Kimodo-SOMA-RP-v1.1` | NVIDIA Open Model License |
| Quantized LLM2Vec encoder | `matbee/kimodo-llm2vec-nf4` | Meta Llama 3 Community License |
| LLM2Vec training adapters | McGill NLP LLM2Vec Llama 3 models | Their repository/model terms |

The default encoder is a public, pre-quantized derivative of Meta Llama 3.
Redistribution and use must comply with the Meta Llama 3 Community License.
The service downloads weights at runtime and does not include them in this
repository.

The SOMA-RP checkpoint was selected instead of the SMPL-X checkpoint because
the latter has separate non-commercial SMPL-X body-model constraints. Generated
assets retain provenance fields identifying the checkpoint and encoder.

