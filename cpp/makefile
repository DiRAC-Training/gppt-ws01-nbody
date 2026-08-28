nbody: main.cpp util.hpp
	g++ -O3 -march=native main.cpp -o nbody

all: nbody

fmt:
	clang-format -i --style=llvm *.cpp

clean:
	rm -f nbody *.csv *.png

.PHONY: all fmt
