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

RUN mkdir installdir && \
	DESTDIR=installdir ninja install && \
	cd installdir && \
	tar -cf /softpipe.tar . 

# multistage build, second phase (remove build deps, image goes from 1.7GB to a third of that)

FROM debian:bullseye

# install mesa llvmpipe and softpipe GALLIUM drivers for non-GPU OpenGL
COPY --from=builder /softpipe.tar /
RUN cd / && tar -xf /softpipe.tar

# install tsugite app
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

# add supervisord and a novnc websockified UI in front of the app
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

# set up default password for noVNC login
#RUN cp /etc/X11/xinit/xinitrc /root/.xinitrc
RUN mkdir ~/.vnc && x11vnc -storepasswd tsugite ~/.vnc/passwd

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
EXPOSE 8083
ENTRYPOINT [ "tini", "--" ]

ENV XVFB_WHD="1200x768x24"\
    DISPLAY=":0.0" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe"

ENV DISPLAY_WIDTH="1600" \
	DISPLAY_HEIGHT="1024"

CMD /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf

#CMD python3 Tsugite_app.py

