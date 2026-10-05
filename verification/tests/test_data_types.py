"""Lowering of `data_type` spellings by the tool backends (generic fixed<W,I[,q[,o]]>, raw ap_*/ac_*)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import backends  # noqa: E402
from backends.base import parse_fixed  # noqa: E402

V, C = backends.get_backend("vitis"), backends.get_backend("catapult")


def test_raw_tool_spellings_unchanged():
    # existing configs: Vitis passes ap_* through verbatim; Catapult keeps truncate+wrap for mode-less ap_fixed
    assert V.type_decl("ap_fixed<16, 5>") == "ap_fixed<16, 5>"
    assert V.type_decl("ap_fixed<16,5,AP_RND,AP_SAT>") == "ap_fixed<16,5,AP_RND,AP_SAT>"
    assert V.type_suffix("ap_fixed<16, 5>") == "ap_fixed_16_5_"
    assert C.type_decl("ap_fixed<16, 5>") == "ac_fixed<16,5,true,AC_TRN,AC_WRAP>"
    assert C.type_suffix("ap_fixed<16, 5>") == "fixed_16_5_"


def test_generic_defaults_to_round_saturate():
    assert parse_fixed("fixed<16,5>") == (16, 5, "rnd", "sat")
    assert V.type_decl("fixed<16,5>") == "ap_fixed<16, 5, AP_RND, AP_SAT>"
    assert C.type_decl("fixed<16,5>") == "ac_fixed<16,5,true,AC_RND,AC_SAT>"


def test_explicit_modes():
    assert V.type_decl("fixed<32,10,trn,wrap>") == "ap_fixed<32, 10, AP_TRN, AP_WRAP>"
    assert parse_fixed("fixed<16,5,RND>") == (16, 5, "rnd", "sat")
    assert parse_fixed("ap_fixed<16,5,AP_RND>") == (16, 5, "rnd", "wrap")          # tool spelling: missing mode = tool default
    assert parse_fixed("ac_fixed<16,5,true,AC_RND,AC_SAT>") == (16, 5, "rnd", "sat")
    assert parse_fixed("float") is None


def test_unsupported_modes_raise():
    for bad in ("fixed<16,5,ceil>", "fixed<16,5,rnd,clip>", "fixed<16,5,rnd,sat,x>", "ac_fixed<16,5,false>"):
        try:
            parse_fixed(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
