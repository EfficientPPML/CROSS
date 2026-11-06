# Post-freeze integration API note

The 135-test planning checkpoint and its33-source archive remain unchanged. After that freeze, the full masked-softmax owner completed476 ideal tests in `/tmp/bert-softmax-tile-assembly-20261103`, with109 exact sources and archive SHA256 `51bd2b90ff2938dfcf4998c4976de290058fbf46979e6ccbf6035a94739ea2b4`. This is public/ideal evidence, not encrypted execution or a repeated-refresh numerical pass.

`BertSoftmaxTile.continuation_levels` counts capacity **after** the mandatory mask following a final refresh. The selected attention tail consumes15 levels and reserves2, so the actual constructor argument is **17**; its refresh-output requirement is at least18. The older planning README's request18 used a before-mask convention. The numerical schedule is unchanged: refresh25→mask24→PV21→dense/residual20→LayerNorm9. There is no final-mask skip flag.

The isolated initial-plus-five-round owner is now implemented, so that portion of the README's missing-code status is historical. Joined ciphertext attention, actual common-profile B2 repeated refresh, model-bound8input/Toutput transport, v2 LayerNorm integration and complete model/file-stage work remain unvalidated/unimplemented as applicable. Valid query rows are compacted with original learned positions preserved; no padded-query nonlinear-domain claim is added.
