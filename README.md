<p align="center"><img src="https://github.com/nathanbronson/Skyground/blob/main/logo.jpg?raw=true" alt="logo" width="200"/></p>

_____
# Skyground
dynamic wallpaper of real-time air traffic

## About
Skyground is a dynamic wallpaper for macOS. It renders real-time United States air traffic on an elevation map. Data comes from the OpenSky Network's public API.

This codebase includes the code for creating GIFs of air traffic and for rendering air traffic as a dynamic wallpaper, updating at a set interval.

## Examples
<img src="https://github.com/nathanbronson/Skyground/blob/main/flights.GIF?raw=true" alt="flights" loop=infinite>

## Usage
Run `background.py` in the terminal to activate the wallpaper. `make_gif.py` can be used to output a gif of current air traffic as well. Parameters can be changed in each file to adjust the number of flights rendered and the time period watched.

## License
See `LICENSE`.
