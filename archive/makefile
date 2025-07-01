CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -funroll-loops -ffast-math -DNDEBUG
OPENCV = `pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv 2>/dev/null || echo "-I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_objdetect -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio"`

fast_track: fast_track.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV)

clean:
	rm -f fast_track

