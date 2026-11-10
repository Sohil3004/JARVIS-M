# Ranking evaluation

Ranking inside each question's candidate subgraph (questions with gold evidence in the candidates).

## StateBench test

questions with evidence: 2667 | candidate recall: 0.999 | mean superseded fraction: 0.0335 | chains / 100 msgs: 13.58

| Scorer | n | R@5 | MRR | knowledge_update | multi_hop | single_hop | temporal | user_implicit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seeds | 2665 | 0.935 | 0.761 | 0.714 | 0.652 | 0.988 | 0.938 | 0.162 |
| fixed | 2665 | 0.940 | 0.799 | 0.745 | 0.991 | 0.953 | 0.921 | 0.240 |
| hgnn:hgnn_v1 | 2665 | 0.998 | 0.933 | 0.914 | 0.955 | 0.962 | 0.905 | 0.941 |
| hgnn:hgnn_v1_nogate | 2665 | 0.997 | 0.939 | 0.935 | 0.914 | 0.965 | 0.925 | 0.912 |

## LoCoMo

questions with evidence: 1536 | candidate recall: 0.967 | mean superseded fraction: 0.0045 | chains / 100 msgs: 9.54

| Scorer | n | R@5 | MRR | multi_hop | open_domain | single_hop | temporal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seeds | 1486 | 0.363 | 0.298 | 0.168 | 0.171 | 0.318 | 0.396 |
| fixed | 1486 | 0.341 | 0.276 | 0.161 | 0.167 | 0.290 | 0.370 |
| hgnn:hgnn_v1 | 1486 | 0.414 | 0.333 | 0.218 | 0.242 | 0.347 | 0.422 |
| hgnn:hgnn_v1_nogate | 1486 | 0.406 | 0.324 | 0.228 | 0.255 | 0.334 | 0.400 |

## LongMemEval-S test (stratified 20%)

questions with evidence: 94 | candidate recall: 1.000 | mean superseded fraction: 0.0104 | chains / 100 msgs: 10.25

| Scorer | n | R@5 | MRR | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seeds | 94 | 0.779 | 0.734 | 0.844 | 0.708 | 0.689 | 0.417 | 0.896 | 0.711 |
| fixed | 94 | 0.743 | 0.658 | 0.733 | 0.579 | 0.675 | 0.323 | 0.877 | 0.653 |
| hgnn:hgnn_v1 | 94 | 0.820 | 0.772 | 0.833 | 0.754 | 0.676 | 0.317 | 0.903 | 0.842 |
| hgnn:hgnn_v1_nogate | 94 | 0.788 | 0.764 | 0.880 | 0.823 | 0.704 | 0.323 | 0.839 | 0.738 |

