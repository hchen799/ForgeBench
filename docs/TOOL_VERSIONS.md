# Tool versions

Versions of the tools behind the R2 results, as reported by the tools themselves on the run server (chao-srv2.ece.gatech.edu).

## Vitis (all sweeps, operator / Table 7 verification)

| Tool | Version | Source |
|---|---|---|
| Vitis HLS | v2024.1.2 (64-bit), SW Build 5096458 (Sep 5 2024) | `vitis_hls -version` |
| Vivado | 2024.1.2 | impl runs (see `docs/revision_r2/CHANGELOG_R2.md`) |

## Catapult backend check (verification/paper_runs/run_catapult.sh)

| Tool | Version | Source |
|---|---|---|
| Catapult | Catapult Prime Synthesis 2026.2/1292347 (Production Release, May 14 2026), `/tools/software/siemens/catapult/2026.2` | session banner of every run |
| `ac_fixed` / `ac_std_float` / `ac_math` headers | shipped with Catapult 2026.2 (`Mgc_home/shared/include`) | header banner |
| C simulation | SCVerify `Verify_orig_cxx_osci.mk`, Catapult-bundled g++ (Calypto) 13.4.0 (`Mgc_home/pkgs/dcs_gcc`) | Catapult transcript |
| RTL simulation | Questa Sim-64 2026.1_2 (`/tools/software/siemens/questa/2026.1_2`), SCVerify `Verify_rtl_v_msim.mk` | `vsim -version` |
| Target library | `nangate-45nm_beh` (DesignCompiler flow, Nangate 045nm) + `ram_nangate-45nm-singleport_beh`; clock 10 ns | `backends/catapult.py` (`DEFAULT_LIBS`) |

Licenses: the Siemens license server named in `/tools/software/siemens/setup.csh` (`SALT_LICENSE_SERVER`, `MGLS_LICENSE_FILE`).
`run_catapult.sh` exports those two variables; with the `MGLS_LICENSE_FILE` set in `~/.bashrc` alone, Catapult 2026.2 waits indefinitely
after its banner.
