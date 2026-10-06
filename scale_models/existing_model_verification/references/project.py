"""Inspect and snapshot saved HLS projects; emit only an external testbench."""
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from .arithmetic import DATA, Format
from .resnet import RESNET_SPECS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def declarations(text, prefix):
    return {
        name: tuple(map(int, re.findall(r"\[(\d+)\]", dims)))
        for name, dims in re.findall(
            r"\b(?:data_t|int32_t|acc_t)\s+(" + prefix + r"\w+)\s*((?:\[\d+\])+)", text
        )
    }


def parameters(family, ports):
    if family.startswith("resnet"):
        return {
            n: s
            for n, s in ports.items()
            if n == "DRAM_input"
            or n == "DRAM_fc"
            or n.startswith(("DRAM_w_", "DRAM_bn_"))
        }
    names = {
        "DRAM_embedding",
        "DRAM_final_norm",
        "DRAM_lm_head",
        "DRAM_attn_norm",
        "DRAM_ffn_norm",
    }
    return {n: s for n, s in ports.items() if n in names or n.endswith("_proj")}


class Project:
    def __init__(self, path, family, config=None):
        self.path, self.family = Path(path).resolve(), family
        self.source = (self.path / "top.cpp").read_text()
        header = (self.path / "top.h").read_text()
        types = re.findall(r"typedef\s+(ap_fixed\s*<[^>]+>)\s+data_t\s*;", header)
        if len(types) != 1:
            raise ValueError("expected one explicit ap_fixed data_t in saved top.h")
        self.dtype = re.sub(r"\s+", "", types[0])
        match = re.fullmatch(r"ap_fixed<(16,5|32,10)(?:,AP_TRN,AP_WRAP)?>", self.dtype)
        if match is None:
            raise ValueError(
                "unsupported arithmetic variant: production references support Q16.5 and Q32.10 with AP_TRN/AP_WRAP"
            )
        self.format = Format(*map(int, match[1].split(",")))
        source_dtype = re.search(
            r"typedef\s+(ap_fixed\s*<[^>]+>)\s+data_t\s*;", self.source
        )
        if source_dtype is None or re.sub(r"\s+", "", source_dtype[1]) != self.dtype:
            raise ValueError("source/header datatype mismatch")
        self.top = re.search(r"void\s+(\w+)\s*\(([^;]+)\)\s*;", header, re.S)
        if self.top is None:
            raise ValueError("cannot parse saved top function declaration")
        self.top_name = self.top[1]
        self.ports = declarations(self.top[2], "DRAM_")
        self.port_types = {
            name: dtype
            for dtype, name in re.findall(
                r"\b(data_t|int32_t)\s+(DRAM_\w+)", self.top[2]
            )
        }
        source_sig = re.search(
            r"void\s+" + re.escape(self.top_name) + r"\s*\((.*?)\)\s*\{",
            self.source,
            re.S,
        )
        if (
            not self.ports
            or source_sig is None
            or list(declarations(source_sig[1], "DRAM_").items())
            != list(self.ports.items())
        ):
            raise ValueError("saved source/header port mismatch")
        if len(self.top[2].split(",")) != len(self.ports):
            raise ValueError("unsupported or duplicate top argument")
        source_port_types = dict(
            (n, t)
            for t, n in re.findall(r"\b(data_t|int32_t)\s+(DRAM_\w+)", source_sig[1])
        )
        if source_port_types != self.port_types:
            raise ValueError("saved source/header port type mismatch")
        self.brams = declarations(self.source[: source_sig.start()], "BRAM_")
        self.inputs = parameters(family, self.ports)
        self.config_path = Path(config).resolve() if config else None
        if self.config_path is None:
            candidate = (
                self.path.parent.parent
                / "auto_generated_configs"
                / (self.path.name + ".json")
            )
            if candidate.exists():
                self.config_path = candidate
            elif (self.path / "resolved_config.json").exists():
                self.config_path = self.path / "resolved_config.json"
        if self.config_path:
            self.config_sha256 = sha(self.config_path)
            self.config = json.loads(self.config_path.read_text())
            if list(self.ports.items()) != [
                (d["name"], tuple(d["dims"])) for d in self.config["drams"]
            ]:
                raise ValueError("production JSON/header port mismatch")
            if re.sub(r"\s+", "", self.config["data_type"]) != self.dtype:
                raise ValueError("production JSON/header datatype mismatch")
            if {
                d["name"]: d.get("dtype", "data_t") for d in self.config["drams"]
            } != self.port_types:
                raise ValueError("production JSON/header port type mismatch")
        self.findings = []
        if family.startswith("resnet"):
            self.variant = family + (
                "-tiled" if "DRAM_stem_feat" in self.ports else "-full"
            )
            self._resnet()
        else:
            self.variant = (
                "llama3-prefill"
                if "DRAM_prefill_len" in self.ports
                else "llama3-decode"
            )
            self._llama()
        body = self.source[source_sig.end() :]
        unused = [
            n
            for n in self.inputs
            if not re.search(
                r"\b" + n + r"\b(?![^\n]*#)",
                re.sub(r"^\s*#pragma[^\n]*", "", body, flags=re.M),
            )
        ]
        if unused:
            self.findings.append(dict(kind="unread_parameter_ports", ports=unused))
        self.files = {}
        self._dependencies("top.cpp")
        self._dependencies("top.h")
        contracts = json.loads(Path(__file__).with_name("contracts.json").read_text())
        base = contracts["variants"].get(self.variant, {})
        choices = ([base] if "sources" in base else []) + base.get("revisions", [])
        contract = next(
            (
                c
                for c in choices
                if all(
                    self.files.get(n) == digest for n, digest in c["sources"].items()
                )
            ),
            None,
        )
        if contract is None:
            raise ValueError(
                "unregistered HLS source version: "
                + self.variant
                + "; validate its arithmetic and register a matching reference pair in references/contracts.json"
            )
        self.contract_version = contract["reference_version"]

    def _dependencies(self, relative):
        if relative in self.files:
            return
        path = (self.path / relative).resolve()
        if self.path not in path.parents or not path.is_file():
            raise ValueError("missing or nonlocal quoted include: " + relative)
        self.files[relative] = sha(path)
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), re.M):
            self._dependencies(str(Path(relative).parent / include))

    def _resnet(self):
        match = re.fullmatch(r"resnet(18|34|50|101|152)", self.family)
        if match is None:
            raise ValueError("unsupported ResNet family: " + self.family)
        depth = int(match[1])
        self.block_counts, bottleneck = RESNET_SPECS[depth]
        expansion = 4 if bottleneck else 1
        expected = {
            "DRAM_input": (3, 224, 224),
            "DRAM_w_stem": (64, 3, 7, 7),
            "DRAM_bn_stem": (4, 64),
            "DRAM_fc": (1000, 512 * expansion, 1, 1),
        }
        channels = 64
        for stage in range(1, 5):
            width = 64 * 2 ** (stage - 1)
            out = width * expansion
            for block in range(self.block_counts[stage - 1]):
                name = f"s{stage}_b{block}"
                expected[f"DRAM_w_{name}_1"] = (
                    (width, channels, 1, 1) if bottleneck else (out, channels, 3, 3)
                )
                expected[f"DRAM_w_{name}_2"] = (width, width, 3, 3)
                expected[f"DRAM_bn_{name}_1"] = (4, width)
                expected[f"DRAM_bn_{name}_2"] = (4, width)
                if bottleneck:
                    expected[f"DRAM_w_{name}_3"] = (out, width, 1, 1)
                    expected[f"DRAM_bn_{name}_3"] = (4, out)
                if block == 0 and (stage > 1 or channels != out):
                    expected[f"DRAM_w_{name}_down"] = (out, channels, 1, 1)
                channels = out
        if self.inputs != expected or self.ports.get("DRAM_out") != (1000, 1, 1):
            raise ValueError(
                f"interface is not the supported production ResNet-{depth} graph"
            )
        self.outputs = {"logits": ("DRAM_out", (1000,))}
        if self.variant.endswith("tiled"):
            mapping = {
                "stem": "DRAM_stem_feat",
                "pool": "DRAM_stem_pool",
                "gap": "DRAM_gap",
            }
            mapping.update(
                {
                    f"s{s}_b{b}": f"DRAM_s{s}_b{b}"
                    for s in range(1, 5)
                    for b in range(self.block_counts[s - 1])
                }
            )
            available = self.ports
        else:
            mapping = {
                "stem": "BRAM_feat_stem",
                "pool": "BRAM_feat_pool",
                "gap": "BRAM_pool",
            }
            mapping.update(
                {
                    f"s{s}_b{b}": f"BRAM_feat_s{s}_b{b}"
                    for s in range(1, 5)
                    for b in range(self.block_counts[s - 1])
                }
            )
            available = self.brams
        for label, name in mapping.items():
            if name not in available:
                raise ValueError("unsupported checkpoint layout: " + name)
            self.outputs[label] = (
                name,
                (512 * expansion,) if label == "gap" else available[name],
            )

    def _llama(self):
        required = {
            "DRAM_embedding": (128256, 4096),
            "DRAM_lm_head": (128256, 4096),
            "DRAM_final_norm": (4096,),
            "DRAM_attn_norm": (32, 4096),
            "DRAM_ffn_norm": (32, 4096),
        }
        for name, out, ins in [
            ("q", 4096, 4096),
            ("k", 1024, 4096),
            ("v", 1024, 4096),
            ("o", 4096, 4096),
            ("gate", 14336, 4096),
            ("up", 14336, 4096),
            ("down", 4096, 14336),
        ]:
            required[f"DRAM_{name}_proj"] = (32, out, ins)
        if self.inputs != required:
            raise ValueError(
                "interface is not the supported full Llama 3 8B parameter layout"
            )
        cache = self.ports.get("DRAM_k_cache", ())
        if (
            len(cache) != 4
            or cache[0] != 32
            or cache[2:] != (8, 128)
            or self.ports.get("DRAM_v_cache") != cache
        ):
            raise ValueError("unsupported Llama cache layout")
        self.max_ctx = cache[1]
        acc = re.search(
            r"typedef\s+ap_fixed\s*<\s*(\d+)\s*,\s*(\d+)(?:,[^>]+)?>\s+acc_t",
            self.source,
        )
        allowed_acc = ((32, 10), (64, 42)) if self.format.word == 16 else ((80, 36),)
        if acc is None or tuple(map(int, acc.groups())) not in allowed_acc:
            raise ValueError("unsupported production Llama accumulator")
        dims = self.brams.get("BRAM_matrix_in", ())
        if len(dims) != 2:
            raise ValueError("cannot identify Llama linear reduction tile")
        self.tile_in = dims[1]
        self.hidden_chunk = self.brams["BRAM_hidden_a"][1]
        self.outputs = {}
        if (
            self.port_types.get("DRAM_token_ids", self.port_types.get("DRAM_token_id"))
            != "int32_t"
        ):
            self.findings.append(
                dict(
                    kind="control_range",
                    detail="data_t token IDs/positions support only integers 0..15",
                )
            )
        if "(acc_t)4096" in self.source:
            self.findings.append(
                dict(
                    kind="zero_rms_divisor",
                    detail="(ap_fixed<32,10>)4096 wraps to zero",
                )
            )
        if "hls::sqrt((data_t)128)" in self.source:
            self.findings.append(
                dict(
                    kind="zero_attention_divisor",
                    detail="(ap_fixed<16,5>)128 wraps to zero",
                )
            )

    def manifest(self):
        return dict(
            project=str(self.path),
            family=self.family,
            variant=self.variant,
            sources=self.files,
            ports=self.ports,
            port_types=self.port_types,
            arithmetic=self.format.describe(),
            reference_contract_version=self.contract_version,
            findings=self.findings,
            config=str(self.config_path) if self.config_path else None,
            config_sha256=self.config_sha256 if self.config_path else None,
        )

    def assert_unchanged(self):
        if self.config_path and sha(self.config_path) != self.config_sha256:
            raise ValueError("original production JSON changed during verification")
        for name, digest in self.files.items():
            if sha(self.path / name) != digest:
                raise ValueError(
                    "original HLS source changed during verification: " + name
                )

    def snapshot(self, destination):
        destination.mkdir(parents=True)
        for name in self.files:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path / name, target)
        if self.config_path:
            shutil.copy2(self.config_path, destination / "production_config.json")


def testbench(project, input_names, output_specs):
    """Only port allocation, raw-code I/O, top invocation, and tensor export."""
    fmt = getattr(project, "format", DATA)
    lines = [
        r"""#include "top.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <memory>
#include <algorithm>
static const int DATA_BITS = @DATA_BITS@;
static const int DATA_BYTES = DATA_BITS / 8;
static void load(const std::string &p, data_t *x, size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("missing input: " + p);
    for (size_t i=0;i<n;++i) {
        unsigned char b[DATA_BYTES]; f.read((char*)b,DATA_BYTES);
        if (!f) throw std::runtime_error("short input: " + p);
        uint32_t v=0;
        for(int j=0;j<DATA_BYTES;++j) v |= uint32_t(b[j]) << (8*j);
        x[i].range(DATA_BITS-1,0) = v;
    }
    if (f.peek()!=EOF) throw std::runtime_error("extra input bytes: " + p);
}
static void load(const std::string &p, int32_t *x, size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("missing input: " + p);
    for (size_t i=0;i<n;++i) {
        unsigned char b[4]; f.read((char*)b,4);
        if (!f) throw std::runtime_error("short input: " + p);
        uint32_t v=uint32_t(b[0]) | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
        x[i]=static_cast<int32_t>(v);
    }
    if (f.peek()!=EOF) throw std::runtime_error("extra input bytes: " + p);
}
static void save(const std::string &p, const data_t *x, size_t n) {
    std::ofstream f(p, std::ios::binary);
    for (size_t i=0;i<n;++i) {
        uint32_t v=x[i].range(DATA_BITS-1,0).to_uint();
        unsigned char b[DATA_BYTES];
        for(int j=0;j<DATA_BYTES;++j) b[j]=static_cast<unsigned char>(v>>(8*j));
        f.write((char*)b,DATA_BYTES);
    }
    if (!f) throw std::runtime_error("cannot write: " + p);
}"""
    ]
    lines[0] = lines[0].replace("@DATA_BITS@", str(fmt.word))
    for name in sorted({s[0] for s in output_specs.values()} - project.ports.keys()):
        dims = project.brams[name]
        lines.append("extern data_t " + name + "".join(f"[{d}]" for d in dims) + ";")
    lines.append(
        'int main(int argc,char **argv) { try { if(argc!=3) throw std::runtime_error("expected input and case directories");'
    )
    for name, dims in project.ports.items():
        dtype = project.port_types[name]
        lines.append(
            f"std::unique_ptr<{dtype}[]> {name}(new {dtype}[{math.prod(dims)}ULL]());"
        )
        if name in input_names:
            root = "argv[1]" if name in project.inputs else "argv[2]"
            lines.append(
                f'load(std::string({root})+"/{name}.bin",{name}.get(),{math.prod(dims)}ULL);'
            )
        else:
            # ap_fixed's default constructor does not initialize its bits.
            lines.append(f"std::fill_n({name}.get(),{math.prod(dims)}ULL,{dtype}(0));")
    args = []
    for name, dims in project.ports.items():
        suffix = "".join(f"[{d}]" for d in dims[1:])
        args.append(
            f"reinterpret_cast<{project.port_types[name]} (*){suffix}>({name}.get())"
            if suffix
            else name + ".get()"
        )
    lines.append(project.top_name + "(" + ",".join(args) + ");")
    for label, (name, shape) in output_specs.items():
        ptr = (
            name + ".get()"
            if name in project.ports
            else f"reinterpret_cast<data_t*>({name})"
        )
        lines.append(
            f'save(std::string(argv[2])+"/actual/{label}.bin",{ptr},{math.prod(shape)}ULL);'
        )
    lines.append(
        'std::cout<<"ACCELERATOR_EXECUTED"<<std::endl; return 0; } catch(const std::exception &e) {std::cerr<<e.what()<<std::endl;return 2;} }'
    )
    return "\n".join(lines) + "\n"
