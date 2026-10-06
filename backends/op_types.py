"""Per-operator `data_type`.

An operator may carry its own `data_type` (same spellings as the design-wide one); unset means the design's `data_t`. The operator
function is generated with that type for its storage (arrays, locals, accumulators that default to it), and the call site converts
the operand arrays at the boundary:

    { <op type> _cv0[..]; copy-in each operand;  op(_cv0, ...);  copy-out the result operand (named output/out) to the design-wide BRAM array }

Operator text refers to the design type as `data_t`, so for an operator with its own type that name is rebound to a per-type typedef
(`data_t` -> `dt_<tag>`) in the operator's text and the function is renamed with the same tag so differently-typed instances of one
operator do not collide. Operators with the design type are emitted exactly as before (no change to existing designs).
"""
import re


def _tag(ctype, backend):
    return re.sub(r"[^0-9A-Za-z]+", "_", backend.type_suffix(ctype)).strip("_")


def _split_top(s):
    """Split on commas outside <> and () (types such as ap_fixed<16, 5, AP_TRN, AP_WRAP>)."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _params(text, fname):
    """[(name, dims or None)] of the function `fname` in `text`."""
    m = re.search(r"\bvoid\s+" + re.escape(fname) + r"\s*\(", text)
    k, depth = m.end(), 1
    while depth:
        depth += {"(": 1, ")": -1}.get(text[k], 0)
        k += 1
    res = []
    for p in _split_top(text[m.end():k - 1]):
        mm = re.match(r"^(?:const\s+)?.*?(\w+)((?:\s*\[\s*\d+\s*\])*)\s*$", p, re.S)
        dims = re.findall(r"\[\s*(\d+)\s*\]", mm.group(2)) if mm else []
        res.append((mm.group(1), [int(d) for d in dims] or None))
    return res


def design_type_text(op_info, design_ctype, backend):
    """-> (ctype, differs). The op's own type spelled for the backend, or the design's."""
    own = op_info.get("data_type")
    if not own:
        return design_ctype, False
    ctype = backend.type_decl(own)
    return ctype, ctype != design_ctype


def emit_op(op_info, design_ctype, backend, gen_def, gen_call):
    """-> (def_code, def_name, call_code) for one operator, honouring an operator-level `data_type`."""
    ctype, differs = design_type_text(op_info, design_ctype, backend)
    if not differs:
        code, name = gen_def(op_info, design_ctype)
        return code, name, gen_call(op_info, design_ctype)

    info = dict(op_info)
    fi = list(info.get("func_info") or [])
    if fi and isinstance(fi[-1], bool) and info["func_name"] in ("gemm", "vmm", "mmv", "dot_product"):
        fi[-1] = False                      # inline code works on the design-typed BRAMs directly; a typed operator is a function
        info["func_info"] = fi
    tag = _tag(ctype, backend)
    code, name = gen_def(info, ctype)
    call = gen_call(info, ctype)
    cname = re.search(r"^\s*(\w+)\s*\(", call, re.M).group(1)          # the C function actually called (the returned `name` may be a longer id)
    new_name = f"{cname}__{tag}"
    code = re.sub(r"\b" + re.escape(cname) + r"\b", new_name, code)
    code = re.sub(r"\bdata_t\b", f"dt_{tag}", code)
    code = f"typedef {ctype} dt_{tag};\n" + code
    call = re.sub(r"\b" + re.escape(cname) + r"\b", new_name, call)

    m = re.search(re.escape(new_name) + r"\s*\(([^;]*)\)\s*;", call)
    params = _params(code, new_name)
    args = _split_top(m.group(1))
    pre, post, new_args = [], [], []
    arrs = [n for n, (_, dims) in enumerate(params) if dims is not None]
    named = [n for n in arrs if params[n][0] in ("output", "out")]
    out_idx = named[-1] if named else arrs[-1]            # the result array: the operand named output/out, else the last array operand
    for n, ((pname, dims), arg) in enumerate(zip(params, args)):
        if dims is None:
            new_args.append(arg)
            continue
        loops = "".join(f"for (int _i{d} = 0; _i{d} < {s}; _i{d}++) " for d, s in enumerate(dims))
        idx = "".join(f"[_i{d}]" for d in range(len(dims)))
        cv = f"_cv{n}"
        pre.append(f"dt_{tag} {cv}{''.join(f'[{s}]' for s in dims)};")
        if n != out_idx:
            pre.append(f"{loops}{cv}{idx} = ({'dt_' + tag}){arg}{idx};")
        else:
            post.append(f"{loops}{arg}{idx} = (data_t){cv}{idx};")
        new_args.append(cv)
    body = call.replace(m.group(0), f"{new_name}({', '.join(new_args)});")
    call_code = "{\n" + "\n".join(pre) + "\n" + body + "\n" + "\n".join(post) + "\n}"
    return code, f"{name}__{tag}", call_code
