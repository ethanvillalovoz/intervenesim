# InterveneSim-Value visual pilot

A frozen ImageNet ResNet-18 visual encoder plus a learned value head uses **498** training images. Evaluation contains **512** images from a held-out policy seed.

The head is trained only on `frontview`. `agentview` is a zero-shot camera-transfer test. Proprioception includes robot state, control time, and task identity but excludes object and goal position.

| camera    |   AUROC |   average_precision |
|:----------|--------:|--------------------:|
| agentview |   0.696 |               0.586 |
| frontview |   0.712 |               0.582 |

This is a secondary failure-detection probe, not an end-to-end visual control policy.
