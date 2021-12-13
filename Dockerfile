FROM debian:bullseye as builder

RUN apt-get update && apt-get install -y \
	libtool-bin \
	autoconf \
	python3-pip \
	libx11-dev \
	libxext-dev \
	x11proto-core-dev \
	x11proto-gl-dev \
	libglew-dev \
	freeglut3-dev \
	bison \
	flex \
	wget \
	pkg-config \
	zlib1g-dev \
	llvm-dev \
	meson \
	libxcb-randr0-dev \
	libunwind-dev \
	x11-xserver-utils \
	libxrandr-dev \
	mesa-utils

RUN pip3 install mako
RUN wget -O mesa.tar.xz https://archive.mesa3d.org/mesa-21.3.1.tar.xz
RUN tar xf mesa.tar.xz
RUN mkdir mesa-21.3.1/build
WORKDIR mesa-21.3.1/build

RUN meson \
	-D glx=gallium-xlib \
	-D gallium-drivers=swrast \
	-D platforms=x11 \
	-D dri3=disabled \
	-D dri-drivers="" \
	-D vulkan-drivers="" \
	-D buildtype=release \
	-D optimization=3

RUN ninja

RUN ninja install

RUN tar -cf /softpipe.tar /usr/local/lib/x86_64-linux-gnu/libglapi.so.0.0.0 /usr/local/lib/x86_64-linux-gnu/libGLESv1_CM.so.1.1.0 /usr/local/lib/x86_64-linux-gnu/libGLESv2.so.2.0.0 /usr/local/lib/x86_64-linux-gnu/libGL.so.1.5.0 /usr/local/include/KHR/khrplatform.h /usr/local/include/GLES/egl.h /usr/local/include/GLES/gl.h /usr/local/include/GLES/glext.h /usr/local/include/GLES/glplatform.h /usr/local/include/GLES2/gl2.h /usr/local/include/GLES2/gl2ext.h /usr/local/include/GLES2/gl2platform.h /usr/local/include/GLES3/gl3.h /usr/local/include/GLES3/gl31.h /usr/local/include/GLES3/gl32.h /usr/local/include/GLES3/gl3ext.h /usr/local/include/GLES3/gl3platform.h /usr/local/include/GL/gl.h /usr/local/include/GL/glcorearb.h /usr/local/include/GL/glext.h /usr/local/include/GL/glx.h /usr/local/include/GL/glxext.h /usr/local/share/drirc.d/00-mesa-defaults.conf /usr/local/lib/x86_64-linux-gnu/pkgconfig/glesv1_cm.pc /usr/local/lib/x86_64-linux-gnu/pkgconfig/glesv2.pc /usr/local/lib/x86_64-linux-gnu/pkgconfig/gl.pc

# multistage build, second phase (remove build deps, image goes from 1.7GB to a third of that)

FROM debian:bullseye

COPY --from=builder /softpipe.tar /

RUN cd / && tar -xf /softpipe.tar && rm /softpipe.tar 

RUN apt -y update && apt install -y --no-install-recommends \
	python3-pyqt5.qtopengl \
	python3-pip \
	libunwind-dev

WORKDIR /code
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

COPY setup/ .
COPY my_joint* ./

ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":0.0" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="softpipe"

# add supervisord and a novnc websockified UI to the app

RUN apt-get install -y --no-install-recommends git && \
	git clone https://github.com/kanaka/noVNC.git /root/noVNC \
	&& git clone https://github.com/kanaka/websockify /root/noVNC/utils/websockify \
	&& rm -rf /root/noVNC/.git \
	&& rm -rf /root/noVNC/utils/websockify/.git \
	&& apt-get remove -y git 
	
RUN cp /root/noVNC/vnc.html /root/noVNC/index.html

RUN apt-get install -y --no-install-recommends \
	x11vnc \
	tini \
	supervisor \
	socat \
	autocutsel \
	fluxbox \
	xvfb \
	procps

#RUN cp /etc/X11/xinit/xinitrc /root/.xinitrc
RUN mkdir ~/.vnc && x11vnc -storepasswd tsugite ~/.vnc/passwd

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
EXPOSE 8083
ENTRYPOINT [ "tini", "--" ]

ENV DISPLAY_WIDTH="1920" \
	DISPLAY_HEIGHT="1080"

CMD /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf

#CMD python3 Tsugite_app.py

