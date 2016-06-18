#-------------------------------------------------
#
# Project created by QtCreator 2016-03-30T16:38:02
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = ImageStereo
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

INCLUDEPATH += E://opencv-mingw//install//include
INCLUDEPATH += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\include"
INCLUDEPATH += "include"

LIBS += E:\opencv-mingw\install\x64\mingw\bin\libopencv_core2411.dll
LIBS += E:\opencv-mingw\install\x64\mingw\bin\libopencv_highgui2411.dll
LIBS += E:\opencv-mingw\install\x64\mingw\bin\libopencv_imgproc2411.dll
LIBS += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\Win32\opencl.lib"

SOURCES += \
    include/EdgeMatch/cwz_edge_detect.cpp \
    include/EdgeMatch/cwz_edge_match.cpp \
    include/EdgeMatch/cwz_img_proc.cpp \
    include/GuidedFilter/cwz_integral_img.cpp \
    include/PxlMatch/cwz_cl_cpp_functions.cpp \
    include/PxlMatch/cwz_cl_data_type.cpp \
    include/TreeFilter/cwz_disparity_generation.cpp \
    include/TreeFilter/cwz_mst.cpp \
    include/common_func.cpp \
    qt_main.cpp

HEADERS += \
    include/EdgeMatch/cwz_edge_cl_config.h \
    include/EdgeMatch/cwz_edge_detect.h \
    include/EdgeMatch/cwz_edge_match.h \
    include/EdgeMatch/cwz_img_proc.h \
    include/GuidedFilter/cwz_integral_img.h \
    include/PxlMatch/cwz_cl_cpp_functions.h \
    include/PxlMatch/cwz_cl_data_type.h \
    include/TreeFilter/cwz_disparity_generation.h \
    include/TreeFilter/cwz_mst.h \
    include/common_func.h \
    include/cwz_config.h \
    include/cwz_tree_filter_loop_ctrl.h \
    include/Vessel/math_3d.h
