# InterveneSim benchmark report

The two augmentation conditions each received **4,820 additional labeled actions**.

## Headline result

Across 200 disturbed, seed-matched episodes, recovery-data training achieved **78.5%** success versus **58.0%** for additional clean demonstrations (**+20.5 percentage points**). It won 53 paired episodes and lost 12 (two-sided exact McNemar p=2.79e-07).

Against the unaugmented baseline, recovery-data training improved disturbed success from 60.5% to 78.5%. Nominal success was 92.0% versus 94.0% for additional clean demonstrations.

## Autonomous success

| disturbance   | baseline   | more_demos   | recovery_data   |
|:--------------|:-----------|:-------------|:----------------|
| nominal       | 100.0%     | 94.0%        | 92.0%           |
| object_shift  | 80.0%      | 76.0%        | 82.0%           |
| action_noise  | 64.0%      | 68.0%        | 86.0%           |
| action_delay  | 44.0%      | 52.0%        | 60.0%           |
| gripper_slip  | 54.0%      | 36.0%        | 86.0%           |

## Interpretation

This report describes a state-based simulation benchmark. Results do not establish real-world transfer, visual robustness, or safety. See `docs/benchmark.md` for the predeclared protocol and interpretation boundaries. The confidence intervals and paired test quantify evaluation uncertainty for one training run; they do not capture variance across independently trained policies.
