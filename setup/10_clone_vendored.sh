#!/usr/bin/env bash
# Clone the two third-party repos SATURN needs, at pinned commits.
#   tools/Orient-Anything-V2   SpatialVision/Orient-Anything-V2  (also vendors VGGT under vggt/)
#   tools/sam3/sam3            facebookresearch/sam3             (pip install -e)
set -euo pipefail; cd "$(dirname "$0")/.."
clone_pinned() {
    local url="$1" path="$2" sha="$3"
    if [ -d "$path/.git" ]; then
        local cur
        cur=$(git -C "$path" rev-parse HEAD)
        if [[ "$cur" == "$sha"* ]]; then
            echo "  [skip] $path already at $sha"
        else
            echo "  [skip] $path already cloned (HEAD=$cur, expected=$sha)"
        fi
        return 0
    fi
    echo "  [clone] $url -> $path @ $sha"
    git clone "$url" "$path"
    git -C "$path" checkout "$sha"
}
mkdir -p tools/sam3
clone_pinned https://github.com/SpatialVision/Orient-Anything-V2.git      tools/Orient-Anything-V2          6b1fa7aec18c
clone_pinned https://github.com/facebookresearch/sam3.git      tools/sam3/sam3                   86ed77094094
source .venv/bin/activate 2>/dev/null || true
pip install -e tools/sam3/sam3 --no-deps
echo "vendored clones done"
