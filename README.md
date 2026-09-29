# Badge Forge

Turns flat team logos into textured metal badges.

## Run the web tool locally

The page loads a module worker, so it must be served over http (opening
`index.html` directly won't work). From this folder:

    cd web
    python -m http.server 8000

Then open http://localhost:8000 in your browser.

## Python scripts (optional)

Needs Python 3 and: `pip install pillow numpy scipy`

    python stylize.py logos/nip.png out.png [--color #A8FF0A] [--knockout-dark] ...
    python textlogo.py "TEAM NAME" out.png --color #4FA8E8
    ./run_all.sh          # regenerate every logo in output/

## Layout

- `web/src/stylize.js` - the styling engine used by the web tool
- `web/index.html` - the tool itself
- `stylize.py`, `textlogo.py` - the original Python versions
- `logos/` - source logos, `output/` - finished badges
