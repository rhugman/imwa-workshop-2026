"""Write a size-scaled copy of a workshop notebook for fast CI testing.

The tutorial notebooks are left untouched.
This rewrites only a throwaway copy, shrinking the expensive ensemble/iteration
sizes so the whole 00->05 chain runs quickly in CI:

    ies_num_reals -> 4      (prior/posterior ensemble size)
    num_workers   -> 2      (parallel PEST agents)
    control_data.noptmax -> 1   (only positive values; -1/-2/0 modes are preserved)

Usage:  python scale_nb.py <source.ipynb> <dest.ipynb>

Substitutions are keyed on the setting name and match any integer, so they keep
working if the tutorial values change.
"""
import json
import re
import sys

SUBS = [
    # pst.pestpp_options["ies_num_reals"] = <n>
    (re.compile(r'(["\']ies_num_reals["\']\]\s*=\s*)\d+'), r"\g<1>4"),
    # num_workers = <n> OR = min(10, os.cpu_count() - 1)  -> 2
    # (won't touch `num_workers=num_workers`: no digit / min() there; the min() branch matches
    # one nested call so it stops at min()'s own ')', not a trailing def ')')
    (re.compile(r"(\bnum_workers\s*=\s*)(?:\d+|min\([^()]*\([^()]*\)[^()]*\))"), r"\g<1>2"),
    # control_data.noptmax = <positive n>  -> 1   (leaves -1, -2, 0 alone)
    (re.compile(r"(\bcontrol_data\.noptmax\s*=\s*)[1-9]\d*"), r"\g<1>1"),
]


def main(src, dst):
    # notebooks are UTF-8; be explicit so Windows doesn't default to cp1252
    with open(src, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        for rx, rep in SUBS:
            s = rx.sub(rep, s)
        cell["source"] = s.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
