

# Vulkan Compute / Cuda / Cpu Benchmark
 - Only benchmark is a naive gaussian blur function. Cuda impl could be made significantly faster.
 - Originally used the 'Kompute' library to remove Vulkan boilerplate, but then did it by hand. It was several times faster with my code, so something in Kompute was messed up?
 - I had to copy the some code from nvidia's vulkan samples, all that stuff in the `vk::su::*` namespace.
 - Requires the `Vulkan-HPP` repo to be built and installed.
 - Optionally requires `Vulkan-ValidationLayers` to be installed for easier debugging.


## Results
#####  On laptop (i7-7700HQ @ 2.8Ghz, GeForce GTX 1060 + Cuda 11.4):
#####  On desktop (i7-4770K @ 3.5Ghz, GeForce GTX 1070 Ti + Cuda 10.2):
<pre>
 - Task <font color="#FFC0CB">    cpu4Blur</font> took <font color="#4682B4">   61.633s </font> (avg <font color="#FFA500">    6.163ms</font> ± <font color="#90EE90">  616.300ms</font>)
 - Task <font color="#FFC0CB">    cpu1Blur</font> took <font color="#4682B4">   65.256s </font> (avg <font color="#FFA500">    6.526ms</font> ± <font color="#90EE90">  652.531ms</font>)
 - Task <font color="#FFC0CB">    vlknBlur</font> took <font color="#4682B4">    1.068s </font> (avg <font color="#FFA500">  106.836μs</font> ± <font color="#90EE90">   10.683ms</font>)
 - Task <font color="#FFC0CB">    cudaBlur</font> took <font color="#4682B4">  866.306ms</font> (avg <font color="#FFA500">   86.631μs</font> ± <font color="#90EE90">    8.663ms</font>)</pre>
 - Vulkan was about as fast as CUDA! If the CUDA impl were to gather to shared memory it'd be several times faster. Not sure if vulkan supports local memory.
