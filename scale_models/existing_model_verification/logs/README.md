# Preserved logs

`pytest/production_tests.txt` is the fresh 42-test run for this package.
`pytest/full_verification_87_tests.txt` is the fresh combined regression run.
The adjacent JSON files record commands, runtimes, hashes, and exit status.

`csim/<precision>/<model>/verification.txt` is the verifier's complete
high-level log. `simulation.txt` is the corresponding Vitis C-simulation log
for ResNet. Llama has one `simulation_callNNNN.txt` for prefill and each decode
call. These files were copied verbatim from the completed full-model runs and
renamed from `.log` to `.txt` because the repository globally ignores `*.log`.

The reports under `../results/` remain the authoritative machine-readable
verdicts; logs provide supporting compiler and execution details.
