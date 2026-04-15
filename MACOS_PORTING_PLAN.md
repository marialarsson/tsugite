# Tsugite — macOS Native Porting Plan

## Background

Tsugite is a Python-based interactive CAD tool for designing and CNC-fabricating wood joints. It uses a PyQt5 GUI with an embedded OpenGL 3D viewport.

The app currently runs inside Docker not because the application itself is complex, but because running OpenGL headlessly on Linux requires a virtual X11 display (Xvfb), software rendering via Mesa/llvmpipe, and a VNC stack to stream output to a browser. On macOS, all of that scaffolding becomes unnecessary — the OS provides native OpenGL, native windowing, and a native display out of the box.

---

## Phase 1 — Get It Running
**Estimated effort: 1–2 days**

The minimal viable port. The application logic is pure Python with cross-platform libraries and requires only minor fixes to launch natively on macOS.

### Environment Setup

- [ ] Install Python 3.10 or 3.11 via [python.org](https://python.org) or Homebrew (`brew install python`)
- [ ] Install Xcode Command Line Tools: `xcode-select --install`
- [ ] Create a virtual environment: `python3 -m venv venv`
- [ ] Activate it: `source venv/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify PyQt5 installed correctly: `python -c "from PyQt5.QtWidgets import QApplication; print('OK')"`
- [ ] Verify PyOpenGL installed correctly: `python -c "import OpenGL; print('OK')"`

### Code Fixes

- [ ] **Fix texture path resolution in `setup/Buffer.py`** (lines 20–24)
  - Currently uses bare relative paths like `"textures/end_grain.jpg"` which break if the app isn't launched from the `setup/` directory
  - Replace with `pathlib.Path(__file__).parent / "textures" / "end_grain.jpg"`

- [ ] **Fix UI file path in `setup/_mainWindow.py`** (line 261)
  - Currently: `loadUi('Tsugite.ui', self)`
  - Replace with: `loadUi(Path(__file__).parent / 'Tsugite.ui', self)`

- [ ] **Fix icon path in `setup/Tsugite_app.py`** (if applicable)
  - Any icon or asset loaded by the main app file should use `Path(__file__).parent` as the base

- [ ] **Fix filename string splitting in `setup/_mainWindow.py`** (lines 258, 269, 283)
  - Currently splits on hardcoded `"/"` e.g. `self.filename.split("/")[-1]`
  - Replace with `Path(self.filename).name` using pathlib

### First Launch Test

- [ ] Launch the app: `cd setup && python Tsugite_app.py`
- [ ] Confirm the main window opens with the Qt control panel and OpenGL viewport
- [ ] Confirm 3D joint geometry renders in the viewport
- [ ] Open the sample file `my_joint.tsu` and verify it loads correctly
- [ ] Interact with the joint (rotate view, double-click to open/close joint)
- [ ] Save a file and verify it writes to disk

### Create a Launcher Script

- [ ] Create a shell script `run_tsugite.sh` at the project root that activates the venv and launches from the correct directory:
  ```bash
  #!/bin/bash
  DIR="$(cd "$(dirname "$0")" && pwd)"
  source "$DIR/venv/bin/activate"
  cd "$DIR/setup"
  python Tsugite_app.py
  ```
- [ ] Make it executable: `chmod +x run_tsugite.sh`
- [ ] Confirm double-clicking the script (or running it from Terminal) launches the app

---

## Phase 2 — Polish and Package
**Estimated effort: 3–5 days**

Make the app feel native and distributable on macOS.

### Retina / HiDPI Display Fixes

- [ ] Verify that `QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)` is set before the QApplication is created in `Tsugite_app.py`
- [ ] Verify that `QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)` is also set
- [ ] Test on a Retina display — confirm UI text and controls are crisp (not blurry)
- [ ] Check mouse coordinates in the OpenGL viewport: Qt may report logical pixels while OpenGL expects physical pixels on Retina displays
  - If the mouse cursor doesn't align with 3D selection correctly, multiply mouse coords by `self.devicePixelRatio()` in the GL widget's mouse event handlers
- [ ] Verify the viewport resizes correctly when the window is dragged between standard and Retina monitors

### macOS Keyboard Shortcuts

- [ ] Map save shortcut to `Cmd+S` (Qt default may be `Ctrl+S` — on macOS these should be equivalent via `QKeySequence::Save`, verify it works)
- [ ] Map open shortcut to `Cmd+O`
- [ ] Verify `Cmd+W` closes the window (standard macOS behavior)
- [ ] Verify `Cmd+Q` quits the application
- [ ] Audit any other keyboard shortcuts in the app and ensure they follow macOS conventions (Cmd instead of Ctrl)

### File Dialogs

- [ ] Test the Save dialog — confirm it uses the native macOS file picker
- [ ] Test the Open dialog — confirm native macOS file picker
- [ ] Test exporting G-code and NC-code files via the export buttons
- [ ] Verify saved `.tsu` files can be reopened

### Build a `.app` Bundle with py2app

- [ ] Install py2app: `pip install py2app`
- [ ] Create a `setup_app.py` file at the project root:
  ```python
  from setuptools import setup
  APP = ['setup/Tsugite_app.py']
  DATA_FILES = [('', ['setup/Tsugite.ui']),
                ('textures', ['setup/textures/end_grain.jpg',
                              'setup/textures/friction_area.jpg',
                              'setup/textures/contact_area.jpg'])]
  OPTIONS = {'argv_emulation': False, 'packages': ['PyQt5', 'OpenGL', 'numpy', 'PIL', 'pyrr']}
  setup(app=APP, data_files=DATA_FILES, options={'py2app': OPTIONS}, setup_requires=['py2app'])
  ```
- [ ] Build the app bundle: `python setup_app.py py2app`
- [ ] Test the generated `dist/Tsugite_app.app` by double-clicking it in Finder
- [ ] Confirm all assets (textures, UI file) are bundled and load correctly
- [ ] Create a `.dmg` installer: `hdiutil create -volname Tsugite -srcfolder dist/Tsugite_app.app -ov -format UDZO Tsugite.dmg`
- [ ] Test installing from the `.dmg` by copying to `/Applications` and launching

---

## Phase 3 — Future-Proofing (Optional but Recommended)
**Estimated effort: 1–2 weeks**

These changes are not required to run on macOS today, but will prevent the app from breaking as macOS and Qt evolve.

### Migrate from `QGLWidget` to `QOpenGLWidget`

`QGLWidget` was deprecated in Qt 5.5 and removed entirely in Qt 6. The current code in `Tsugite_app.py` inherits directly from `QGLWidget`.

- [ ] Replace `from PyQt5.QtOpenGL import QGLWidget` with `from PyQt5.QtWidgets import QOpenGLWidget`
- [ ] Change `class GLWidget(QGLWidget)` to `class GLWidget(QOpenGLWidget)`
- [ ] Update the GL context initialization — `QOpenGLWidget` uses a different lifecycle:
  - `initializeGL()` is called once after context creation (same)
  - `resizeGL(w, h)` is called on resize (same)
  - `paintGL()` is called to render (same)
  - Remove any direct `QGLFormat` usage and replace with `QSurfaceFormat`
- [ ] Set surface format before creating QApplication:
  ```python
  fmt = QSurfaceFormat()
  fmt.setVersion(2, 1)
  fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
  QSurfaceFormat.setDefaultFormat(fmt)
  ```
- [ ] Test that all rendering (joint geometry, selection, textures, milling paths) still works after the migration

### Modernize the OpenGL Pipeline (Longer Term)

The app uses the OpenGL 1.2 fixed-function pipeline (immediate mode with `glLoadIdentity`, `glOrtho`, `glFrustum`, etc.). Apple deprecated this in macOS 10.14 Mojave and it is not supported on Apple Silicon beyond OpenGL 4.1 compatibility profile.

- [ ] Audit all OpenGL calls in `Show.py` — identify which are fixed-function and which are modern (VBOs/EBOs are already modern)
- [ ] Replace `glLoadIdentity` / `glOrtho` / `glFrustum` / `glMultMatrixf` with explicit uniform matrices passed to shaders
- [ ] Update GLSL shaders in `Show.py` from `#version 120` to `#version 330 core`
- [ ] Replace `gl_FragColor` (built-in) with an explicit `out vec4 fragColor` in fragment shaders
- [ ] Replace `attribute` / `varying` keywords with `in` / `out`
- [ ] Verify rendering output is visually identical to the original after shader migration
- [ ] Request an OpenGL 3.3 core profile context via `QSurfaceFormat` instead of compatibility profile

### Dependency Updates

- [ ] Pin `PyQt5` to a specific known-working version (e.g. `PyQt5==5.15.9`)
- [ ] Pin `PyOpenGL` to a specific version (e.g. `PyOpenGL==3.1.6`)
- [ ] Remove `six` from `requirements.txt` (Python 2/3 compat shim, no longer needed)
- [ ] Remove `testresources` if not used in tests
- [ ] Update `numpy` to a version with Apple Silicon wheels (1.23+)
- [ ] Update `Pillow` to latest stable (9.x or 10.x)

---

## Testing Checklist

Run through these after each phase to confirm nothing is broken.

### Core Functionality
- [ ] App launches without errors in Terminal
- [ ] Main window displays with control panel and 3D viewport
- [ ] Default joint geometry renders on first launch
- [ ] Can open `.tsu` file — sample file `my_joint.tsu` included in repo
- [ ] Can save `.tsu` file to a new location
- [ ] Geometry updates when spinboxes (resolution, angle, timber dimensions) are changed

### 3D Interaction
- [ ] Right-click drag rotates the view
- [ ] Scroll wheel zooms in/out
- [ ] Double-click opens/closes the joint animation
- [ ] Face selection highlighting works (hover over joint faces)
- [ ] Face pulling/editing works (click and drag a face)
- [ ] Shift-click multi-selection works

### Feedback & Validation
- [ ] Connectivity feedback colors update as joint is edited
- [ ] Durability feedback updates
- [ ] Fabricability feedback updates
- [ ] Toggling feedback checkboxes in the control panel works

### Export
- [ ] G-code export produces a valid `.gcode` file
- [ ] NC-code export works
- [ ] ShopBot `.sbp` export works
- [ ] Exported files are readable plain text

### Platform-Specific (macOS)
- [ ] UI is crisp on Retina display (not blurry)
- [ ] Mouse cursor aligns with 3D selection on Retina display
- [ ] `Cmd+S` saves the file
- [ ] `Cmd+O` opens a file
- [ ] `Cmd+Q` quits cleanly
- [ ] App appears correctly in the macOS Dock and menu bar
- [ ] Window resize is smooth
- [ ] No console errors or warnings during normal use

---

## Dependency Reference

| Package | Current Version | macOS Status | Notes |
|---------|----------------|--------------|-------|
| PyQt5 | unversioned | ✅ Works | Pin to 5.15.9+ |
| PyOpenGL | unversioned | ✅ Works | Pin to 3.1.6+ |
| numpy | 1.20.3 | ⚠️ Old | Update to 1.23+ for Apple Silicon wheels |
| Pillow | 8.3.2 | ✅ Works | Update to 9.x+ recommended |
| pyrr | 0.10.3 | ✅ Works | Pure Python, no issues |
| multipledispatch | 0.6.0 | ✅ Works | Pure Python |
| six | 1.15.0 | ➖ Remove | Not needed for Python 3 |
| testresources | 2.0.1 | ➖ Remove | Not used |
