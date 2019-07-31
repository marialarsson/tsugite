# DISCO JOINT

DISCrete Optimization of JOINT geometries, and automatic milling path generation.

Developed with python 3.6.0 / OpenGL / GLFW

Run setup/main.py

## Interface

### Model Joint
Edit joint geometry by pushing/pulling on faces:
<p float="left">
  <img src="/Screenshots/screenshot_edit_pulled.png" width="430" />
  <img src="/Screenshots/screenshot_edit_result.png" width="430" />
</p>
Randomize joint geometry with key R, and clear joint geometry key C

### Joint types
Change joint type with keys I L T X:
<p float="left">
  <img src="/Screenshots/screenshot_type_I.png" width="210" />
  <img src="/Screenshots/screenshot_type_L.png" width="210" />
  <img src="/Screenshots/screenshot_type_T.png" width="210" />
  <img src="/Screenshots/screenshot_type_X.png" width="210" />
</p>

### Sliding Directions
Change main sliding direction with arrow keys up and right:
<p float="left">
  <img src="/Screenshots/screenshot_slide_up.png" width="210" />
  <img src="/Screenshots/screenshot_slide_rt.png" width="210" />
</p>

Arrows show the current sliding directions. Dashed arrows indicated slides additional to the main sliding direction:
<p float="left">
  <img src="/Screenshots/screenshot_slide_2.png" width="210" />
  <img src="/Screenshots/screenshot_slide_3.png" width="210" />
  <img src="/Screenshots/screenshot_slide_4.png" width="210" />
  <img src="/Screenshots/screenshot_slide_5.png" width="210" />
</p>

### Joint Resolution
Edit voxel cube dimension with keys 2, 3 (default), 4 and 5:
<p float="left">
  <img src="/Screenshots/screenshot_dim_2.png" width="210" />
  <img src="/Screenshots/screenshot_dim_3.png" width="210" />
  <img src="/Screenshots/screenshot_dim_4.png" width="210" />
  <img src="/Screenshots/screenshot_dim_5.png" width="210" />
</p>

### Feedback

Unconnected voxels and unbridged components are colored:
<p float="left">
  <img src="/Screenshots/screenshot_unconnected.png" width="210" />
  <img src="/Screenshots/screenshot_unbridged.png" width="210" />
</p>

Structural evaluation (to be implemented).

### Fabrication

Switch to view of fabricated geometry with F (to be implemented).

Turn on/off milling path display with key M (to be improved).

### Preview Options

Rotate view with right mouse button.

Open/close joint with key O:
<p float="left">
  <img src="/Screenshots/screenshot_type_I.png" width="210" />
  <img src="/Screenshots/screenshot_open.png" width="210" />
</p>

Turn on/off component view with keys A and B:
<p float="left">
  <img src="/Screenshots/screenshot_A.png" width="210" />
  <img src="/Screenshots/screenshot_B.png" width="210" />
</p>

Turn on/off hidden lines with key H:
<p float="left">
  <img src="/Screenshots/screenshot_hidden_show.png" width="210" />
  <img src="/Screenshots/screenshot_hidden_hide.png" width="210" />
</p>

Press key P to save a screenshot.

Save joint geometry with key S, and load saved geometry with G

Hit ESC key to quit.
