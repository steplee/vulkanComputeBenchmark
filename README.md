

# Vulkan Compute / Cuda / Cpu Benchmark
 - Only benchmark is a naive gaussian blur function. Cuda impl could be made significantly faster.
 - Originally used the 'Kompute' library to remove Vulkan boilerplate, but then did it by hand. It was several times faster with my code, so something in Kompute was messed up?
 - I had to copy the some code from nvidia's vulkan samples, all that stuff in the `vk::su::*` namespace.
 - Requires the `Vulkan-HPP` repo to be built and installed.
 - Optionally requires `Vulkan-ValidationLayers` to be installed for easier debugging.


## Results
Ignore stddev measurmenets, they are wrong
#####  On laptop (i7-7700HQ @ 2.8Ghz, GeForce GTX 1060 + Cuda 11.4):
#####  On desktop (i7-4770K @ 3.5Ghz, GeForce GTX 1070 Ti + Cuda 10.2):
 <pre>
 - Task <font color="#FFC0CB">    cpu4Blur</font> took <font color="#4682B4">   16.278s </font> (avg <font color="#FFA500">    1.628ms</font> ± <font color="#90EE90">  162.773ms</font>)
 - Task <font color="#FFC0CB">    cpu1Blur</font> took <font color="#4682B4">   63.732s </font> (avg <font color="#FFA500">    6.373ms</font> ± <font color="#90EE90">  637.285ms</font>)
 - Task <font color="#FFC0CB">    vlknBlur</font> took <font color="#4682B4">    1.058s </font> (avg <font color="#FFA500">  105.793μs</font> ± <font color="#90EE90">   10.579ms</font>)
 - Task <font color="#FFC0CB">    cudaBlur</font> took <font color="#4682B4">  860.417ms</font> (avg <font color="#FFA500">   86.042μs</font> ± <font color="#90EE90">    8.604ms</font>)
</pre>
 - Vulkan was about as fast as CUDA! If the CUDA impl were to gather to shared memory it'd be several times faster. Not sure if vulkan supports local memory.
#####  On Jetson AGX Xavier
<pre>
 - Task <font color="#FFC0CB">    cpu4Blur</font> took <font color="#4682B4">   38.569s </font> (avg <font color="#FFA500">    3.857ms</font> ± <font color="#90EE90">  385.669ms</font>)
 - Task <font color="#FFC0CB">    cpu1Blur</font> took <font color="#4682B4">   76.585s </font> (avg <font color="#FFA500">    7.658ms</font> ± <font color="#90EE90">  765.809ms</font>)
 - Task <font color="#FFC0CB">    vlknBlur</font> took <font color="#4682B4">    4.817s </font> (avg <font color="#FFA500">  481.681μs</font> ± <font color="#90EE90">   48.166ms</font>)
 - Task <font color="#FFC0CB">    cudaBlur</font> took <font color="#4682B4">    4.465s </font> (avg <font color="#FFA500">  446.494μs</font> ± <font color="#90EE90">   44.647ms</font>)
</pre>
  - Similar story. Funny the 8-core ARM v8.2 processor only was able to get a 2x speed up with four threads, whereas the i7 got an expected nearly 4x.
