#!/bin/sh
# Regenerate every styled logo from the source images in logos/.
set -e
mkdir -p output
python3 stylize.py logos/nip.png output/nip.png &
python3 stylize.py logos/mouz.png output/mouz.png --color '#A8FF0A' &
python3 stylize.py logos/falcons.png output/falcons.png &
python3 stylize.py logos/vitality.png output/vitality.png &
python3 stylize.py logos/new_logo.png output/new_logo.png &
wait
