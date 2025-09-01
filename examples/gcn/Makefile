.PHONY: build

all: build run

./zig-out/bin/main: *.zig *.zon src/*.zig
	zig build -Doptimize=ReleaseFast

build: ./zig-out/bin/main

run:
	./zig-out/bin/main
