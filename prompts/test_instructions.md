## Build
Go to Ruzino/build and run:
```
ninja.exe
```

But if there are only shader changes, rebuilding is not needed.

Adding new source files or test files requires re-running cmake and they will be automatically added to the build. Command is 
```
cmake .. -DRUZINO_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
```

Added test cpp file will be scanned into <filename>_test.exe automatically.

## Test Instructions
Run the following command in Ruzino/Binaries/Release:
```
.\.\headless_render.exe -u ..\..\Assets\Ground.usdc -j ..\..\Assets\render_nodes_save.json -o profile_sponza.png -w 1920 -h 1080 -s 16 -v
```

Or
```
.\Ruzino.exe somestage.usdc
```

If it is some test, just use the cpp name plus _test.exe, e.g.,
```
.\geom_algorithms_test.exe
```