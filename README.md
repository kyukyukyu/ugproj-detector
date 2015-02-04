# Undergraduate Project

## Notes on cloning this project

This project has hanjianwei/cmake-modules as its submodule to load CMake
modules which are needed to make use of external libraries. So, when you clone
this project, you should run one of these operations:

1) Cloning with `--recursive` option

```shell
$ git clone --recursive https://github.com/kyukyukyu/ugproj-detector.git
```

2) Init and update git submodule after cloning

```shell
$ git clone https://github.com/kyukyukyu/ugproj-detector.git
$ git submodule init
$ git submodule update
```

For more information on git submodule, check
[this](http://git-scm.com/book/en/v2/Git-Tools-Submodules) out.
