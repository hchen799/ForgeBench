# Preserved logs

`csim/<precision>/<model>/verification.txt` contains the complete high-level
verifier log. Every current log ends with a **Final layer output comparison**
section containing output shape, fixed-code mismatches, maximum code error,
FP64 maximum absolute error, relative L2, the 5% limit, and verdicts.

`simulation.txt` is the Vitis C-simulation log for a ResNet. Llama directories
contain `simulation_call0000.txt` for prefill and one file for each decode call.
The ctx2048 directory is named `llama3_8b`; the ctx8192 directory is
`llama3_8b_ctx8192`.

Files are renamed from `.log` to `.txt` because the repository globally ignores
`*.log`. Reports under `../results/` are the authoritative machine-readable
records; these logs preserve compiler and execution evidence.
