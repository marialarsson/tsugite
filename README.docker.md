# Tsugite in the browser

Debian with Mesa libraries and Gallium drivers "llvmpipe" and "softpipe" provides OpenGL support inside a Docker container without the need for a GPU.

Should therefore run on most hardware configurations, albeit slower than when utilizing hardware with GPUs.

A web UI based on noVNC exposes the python app for usage from a web browser.


# Usage

After a local build, use:

		docker run --rm -port "8083:8083" tsugite 

Then open the browser at http://localhost:8083 and login with password "tsugite"

