# AutoDiff
My attempt to perform automatic differentiation in C++

## Build Instructions

### Prerequisites
```bash
sudo apt update
sudo apt install gcc-14 g++-14 cmake
```

### Build Commands
```bash
mkdir build && cd build

# Configure (with debug symbols)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build all targets
make -j$(nproc)

# Build specific targets
make example1
make autodiff_test

# Run tests
ctest --output-on-failure

# Install system-wide (optional)
sudo make install
```

### Key Options
| CMake Option          | Default | Description                  |
|-----------------------|---------|------------------------------|
| BUILD_TESTING         | OFF     | Enable test builds           |
| BUILD_EXAMPLES        | ON      | Build example programs       |

