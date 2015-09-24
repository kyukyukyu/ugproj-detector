# Undergraduate Project

## Dependencies

- [CMake](http://www.cmake.org/) 3.1+
- [Boost](http://www.boost.org/). Used modules are:
    - filesystem
    - math\_tr1
    - program\_options
    - system
- [Eigen](http://eigen.tuxfamily.org/) 3
- [OpenCV](http://opencv.org/) >=2.4.9, <3.0

## How to Build

Build using GCC or LLVM is highly recommended.

```shell
mkdir build && cd $_    # Make a new directory for build artifacts.
cmake ..                # Generate Makefile.
make
```

## How to Run

Executable files will be generated in a directory named `bin` which is under
the root directory.

```shell
cd bin
```

There are two executable files: `tracker`, and `clusterer`. The first one is
supposed to be run first, then the second one. For both executables, a
configuration file is required. A sample configuration file can be found in
`conf` directory.

`tracker` should be executed like this:

```shell
./tracker -c ../conf/apink.mrchu.720p.cfg -v ../sample/apink.mrchu.720p.mp4 -o output.apink.mrchu.720p.20150925
```

- `-c` or `--config-file` – Path to configuration file.
- `-v` or `--video-file` – Path to input video file.
- `-o` or `--output-dir` – Path to directory where output files will be
created.

Once `tracker` has been run and exited without any error, several files will be
generated in the directory whose path is given as the value for `--output-dir`
option.

- `%d.png` (PNG files whose name starts with number) – Result image for face
tracking. Detected or restored faces in single video frame is expressed with
boundary box and the face's number. The number in filename is position of the
frame.
- `result.avi` – Moving picture which is composed of `%d.png` images.
Compressed with Xvid codec.
- `tracklet_%d.png` – Image for single tracklet. Faces in one tracklet are
listed in this image. The face from smaller frame position comes first.
- `tracklet.yaml` – Metadata for tracklets. Frame positions for faces in single
tracklet are stored.
- `mapping.yaml` – Global metadata for face tracking. Mapping between frame
indices and frame positions are stored here. For definitions of these two
words, see issue #32.

`clusterer` should be executed like this:

```shell
./clusterer -c ../conf/apink.mrchu.720p.cfg -m output.apink.mrchu.720p.20150925/tracklet.yaml -p output.apink.mrchu.720p.20150925/mapping.yaml -i output.apink.mrchu.720p.20150925/ -o output.apink.mrchu.720p.20150925
```

- `-c` or `--config-file` – Path to configuration file.
- `-m` or `--metadata-file` – Path to metadata file for tracklets.
- `-p` or `--mapping-file` – Path to global metadata file for face tracking.
- `-i` or `--input-dir` – Path to directory which contains input files.
- `-o` or `--output-dir` – Path to directory where output files will be
created.

Once `clusterer` has been run and exited without any error, several files will
be generated in the directory whose path is given as the value for
`--output-dir` option.

- `clusterer_%d.png` – Result image for face clustering. Faces in one cluster
is listed in this image. The face from smaller frame position comes first.

## Notes on Cloning This Project

This project has hanjianwei/cmake-modules as its submodule to load CMake
modules which are needed to make use of external libraries. So, when you clone
this project, you should run one of these operations:

1) Cloning with `--recursive` option

```shell
git clone --recursive https://github.com/kyukyukyu/ugproj-detector.git
```

2) Init and update git submodule after cloning

```shell
git clone https://github.com/kyukyukyu/ugproj-detector.git
git submodule init
git submodule update
```

For more information on git submodule, check
[this](http://git-scm.com/book/en/v2/Git-Tools-Submodules) out.
