CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -g
PKG_CONFIG ?= pkg-config
GST_CFLAGS := $(shell $(PKG_CONFIG) --cflags gstreamer-1.0 gstreamer-app-1.0)
GST_LIBS := $(shell $(PKG_CONFIG) --libs gstreamer-1.0 gstreamer-app-1.0)

TARGET := point_cloud_collimation
SRC := point_cloud_collimation.cpp

.PHONY: all run viewer clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(GST_CFLAGS) $< -o $@ $(GST_LIBS)

run: $(TARGET)
	./$(TARGET)

viewer: $(TARGET)
	./$(TARGET) --viewer

clean:
	rm -f $(TARGET)
