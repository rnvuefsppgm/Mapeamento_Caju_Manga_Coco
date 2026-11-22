#!/usr/bin/env bash
cd "$(dirname "$0")"; mkdir -p logs
while read -r R || [ -n "$R" ]; do R=$(echo "$R"|tr -d '\r'); case $R in m*) ./export_one.sh mapbiomas2024 "$R";; esac; done < ../v8/data/aoi/regions.txt
../.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'.'); from pathlib import Path; import importlib.util as u; s=u.spec_from_file_location('m','03_mosaic_vrt.py'); m=u.module_from_spec(s); s.loader.exec_module(m); b=Path('data/ee_exports/2025/mapbiomas2024'); [m.build_vrt(d) for d in b.iterdir() if d.is_dir()]"
echo "MB_FIM $(date '+%F %T') | $(du -sh data/ee_exports/2025/mapbiomas2024 | cut -f1)" | tee -a logs/export.log
