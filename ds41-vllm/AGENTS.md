Before making any patch, check the current heads of local-inference-lab/vllm
`dev/jovian-judgement` and local-inference-lab/b12x. Compare with the image pins;
record the checked revisions and whether upstream supersedes the proposed work.
Do not silently advance pins or claim an offline check is current. Keep local
source patches hash-guarded, with independent rollback switches and relevant
native/GPU checks. Distinguish source correctness from measured fleet speedups.
