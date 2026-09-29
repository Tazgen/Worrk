#!/bin/sh
# Regenerate every styled logo from the source images in logos/.
# One line per logo: <source> <output> [stylize.py options]. Runs 3 at a time to limit memory.
set -e
mkdir -p output
xargs -P 3 -L 1 sh -c 'python3 stylize.py "$@" || { echo "FAILED: $*" >&2; exit 255; }' _ <<'LIST'
logos/nip.png output/nip.png
logos/mouz.png output/mouz.png --color #A8FF0A
logos/falcons.png output/falcons.png
logos/vitality.png output/vitality.png
logos/new_logo.png output/new_logo.png
logos/virtuspro.png output/virtuspro.png --knockout-dark --color #FF6A1E
logos/w_logo.png output/w_logo.png --blend 0.10
logos/fut.png output/fut.png --color #E8102E
logos/mancity.png output/mancity.png --knockout-light 185 --knockout-thin-dark 0.014 --tint #5BA3DA --split-colors
logos/gentlemates.png output/gentlemates.png --color #E7A6E6
logos/cloud9.png output/cloud9.png
logos/dignitas.png output/dignitas.png --knockout-dark
logos/geng.png output/geng.png
logos/kru.png output/kru.png
logos/pink_s.png output/pink_s.png
logos/dignitas_pastel.png output/dignitas_pastel.png --knockout-dark
logos/geekay.png output/geekay.png --split-colors
logos/monkey.png output/monkey.png --knockout-thin-dark 0.014 --split-colors
logos/beaver.png output/beaver.png
logos/s2.png output/s2.png --knockout-light 200 --color #C8CDD2
LIST
