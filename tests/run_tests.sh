#!/usr/bin/env bash
# Run the workshop notebooks end-to-end at small (CI) size, in order.
#
# The tutorial notebooks are NOT modified -- scaled throwaway copies (_ci_*.ipynb)
# are generated, executed with nbmake, and removed on success. On failure they are
# left in place (with their error outputs) for inspection / CI artifact upload.
# To run the notebooks at full size, just open and run them normally in Jupyter.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"          # tests/
cd "$here/../tutorial"

copies=()
for nb in 00-*.ipynb 01-*.ipynb 02-*.ipynb 03-*.ipynb 04-*.ipynb 05-*.ipynb; do
    out="_ci_${nb}"
    python "$here/scale_nb.py" "$nb" "$out"
    copies+=("$out")
done

python -m pytest -c "$here/pytest.ini" -q "${copies[@]}"   # set -e: leaves _ci_* on failure
rm -f _ci_*.ipynb                                          # only on success
