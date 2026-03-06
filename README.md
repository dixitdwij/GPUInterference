# GPUInterference


## ROCm Systems Profiler setup

> The complete installation guide of ROCm Systems Profiler can be found [here](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/docs-6.3.1/install/install.html). 

For setting up ROCm Systems Profiler:

```bash
# Get the wanted ROCm Systems Profiler version
wget https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-7.0.2/rocprofiler-systems-1.1.1-ubuntu-20.04-ROCm-60300-PAPI-OMPT-Python3.sh
#  Make the binary executable
chmod +x rocprofiler-systems-1.1.1-ubuntu-20.04-ROCm-60300-PAPI-OMPT-Python3.sh
# Create the folder where you want ROCm System Profiler source code to be
mkdir rocprofiler-system
# Execute the binary 
./rocprofiler-systems-1.1.1-ubuntu-20.04-ROCm-60300-PAPI-OMPT-Python3.sh --prefix=rocprofiler-system --exclude-subdir
```
For setting up the environment:
```bash
source rocprofiler-system/share/rocprofiler-systems/setup-env.sh
```
For testing that the installation succesfully finished:
```bash
rocprof-sys-instrument --help
rocprof-sys-avail --help
```

> In case some libraries are not found, navigate to ***rocprofiler-system/share/rocprofiler-systems/setup-env.sh*** and set ***ROCM_PATH = /opt/rocm-6.3.3***. 

For instrumenting a file:
```bash
rocprof-sys-instrument -M sampling -o multistream.instr -- ./multistream
```

For running an instrumented file:
```bash
rocprof-sys-run -- ./multistream.instr
```

>Running an instrumented file will generate a folder containing some JSON files and a ***.proto*** file. In order to analyse the data, use [Perfetto Trace Viewer](https://ui.perfetto.dev/).

## ROCMPROFILER-COMPUTE setup

For setting up rocprofiler-compute

```bash
module load rocm/7.2.0
python3 -m venv rocprof_env

pip install --upgrade pip # The pip on hacc-boxes is old, so pls do
source rocprof_env/bin/activate
# Find the requirements file and install the dependencies
REQ_FILE="$(dirname $(which rocprof-compute))/../libexec/rocprofiler-compute/requirements.txt"
pip install -r $REQ_FILE

```

For using the installed rocprofiler-compute:
```bash
module load rocm/7.2.0
source ./rocprof_env/bin/activate
```

For attaching rocprof, compile with debug symbols (-g flag on hipcc)
```bash
rocprof-compute profile --name <name> -- ./<binary>

# On terminal output
rocprof-compute analyze -p workloads/vecAddProf/MI210 

# Output to a file
rocprof-compute analyze -p workloads/vecAddProf/MI210 > vecAddProfDump

# lanuches a GUI depiction, can be viewed using port forwarding
rocprof-compute analyze -p workloads/vecAddProf/MI210 --gui
```

## GPU details
We use a MI210 GPU (4 GPUs per node).

#### Obtaining GPU Arch Code (gfx90a)
```
$ /opt/rocm/llvm/bin/amdgpu-arch
gfx90a

ARCH="gfx90a"
```

#### Isolating execution to a single GPU (here, to the first)
```
export HIP_VISIBLE_DEVICES=0
```


## Using util.cpp tools

### 1. Error Handling with HipErrorCheck

HipErrorCheck is a global singleton that tracks the state of HIP API calls.

### Manual Usage
Wrap any HIP function call to update the internal error state:
```cpp
HipErrorCheck(hipMalloc(&ptr, size));
if (HipErrorCheck) {
    printf("Error: %s\n", HipErrorCheck.what());
}
```

### Implicit Kernel Checking
Call it without arguments to automatically fetch the result of the last asynchronous operation (like a kernel launch):

```C++
myKernel<<<grid, block>>>(d_ptr);
if (HipErrorCheck()) { // Fetches hipGetLastError() internally
    throw std::runtime_error(HipErrorCheck.what());
}
```

### 2. Stream and Event Management
The HipStream and HipStartStop structs handle the lifecycle of HIP resources.

Timing Kernels
```C++
HipStream computeStream;
HipStartStop timer;

timer.startTiming();
// Launch work on computeStream.stream
timer.stopTiming(&computeStream); // Synchronizes the stream automatically

printf("Kernel execution time: %f ms\n", timer.elapsedTime());
```

### 3. Memory Management with HipVector
HipVector automates the allocation of dual buffers (Host and Device).

#### Workflow
1. Allocation: Constructor allocates float* on CPU and hipMalloc on GPU.
2. To Device: Move data to the GPU for processing.
3. To Host: Retrieve results back to the CPU.
4. Cleanup: Destructor handles all deallocations automatically.

```C++
{
    HipVector data(1024); // Allocates both sides
    
    // Fill host data...
    data.toDevice(); 

    // Use data.d_vec in your HIP kernel
    
    data.toHost(); 
} // All memory is safely freed here
```