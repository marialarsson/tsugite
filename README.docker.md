# Tsugite in the browser

Debian with Mesa libraries and Gallium drivers "llvmpipe" and "softpipe" provides OpenGL support inside a Docker container without the need for a GPU.

Should therefore run on most hardware configurations, albeit slower than when utilizing hardware with GPUs.

A web UI based on noVNC exposes the python app for usage from a web browser.


# Usage

Try this oneliner to use the image built by the GitHub Action and stored in GitHub Container Registry:

		docker run --rm -p "8083:8083" ghcr.io/marialarsson/tsugite

If you use "make build" to build locally instead, you can start the app with:

		docker run --rm -p "8083:8083" tsugite

Then open the browser at http://localhost:8083 and login with password "tsugite"

