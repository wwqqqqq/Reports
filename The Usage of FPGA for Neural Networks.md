# FPGA-accelerated AI

## Background

[TODO]

### AI accelerator

As Moore's Law approaches to its ending, we are at the end of the performance scaling that we are used to. To accelerate computing, we cannot depend on CPU processors to get faster in high speed as they used to be, but have to design alternative approaches, for example, using specialized processors. In the industry, Google has its TPU (Tensor Processing Unit), with one core per chip and software-cotrolled memory instead of caches; NVidia's GPU has 80-plus cores; and Microsoft is taking an FPGA approach.

## FPGA introduction

A FPGA (field-programmable gate array) is an integrated circuit designed to be configured after manufacturing. The FPGA configuration is generally specified using a hardware description language (HDL).

FPGAs contain an array of programmable logic blocks to be "wired together", like many logic gates that can be inter-wired in different configurations. In most FPGAs, logic blocks also include memory elements, which may be simple flip-flops or more complete blocks of memory.

A FPGA can be used to solve any problem which is computable. Their advantage lies in that they are sometimes significantly faster for some applications because of their parallel nature and optimality in terms of the number of gates used for a certain process. 

FPGAs are commonly used in hardware acceleration, where one can use FPGA to accelerate certain parts of an algorithm and share part of the computation between the FPGA and a generic processor.

[TODO] 
https://www.zdnet.com/article/ai-chips-for-big-data-and-machine-learning-gpus-fpgas-and-hard-choices-in-the-cloud-and-on-premise/

### Advantages

1. FPGAs provide a combination of programmability and performance comparing to ASIC and CPU/GPU. 

    ASIC is known for its performance, but it can cost huge amount of time to design the specific circuit to achieve best performance. Meanwhile, deap learning frameworks are still evolving, making it hard to design or update custom hardware. FPGAs are reconfigurable, and far easier to design and program comparing to ASIC, making them convenient to extend to a range of applications, e.g. several different types of nueral networks. FPGA structure can be considered as an alternative to software, but is implemented directly on hardware.

    Comparing to GPU/CPU, FPGA is customized and specialized in specific application, making it faster and more energy-efficient, since general-purpose processors invest excessive hardware resources flwxibly support various workloads. Also, due to the intrinsic property of FPGAs, they have high compute throughput and networking throughput, and low compute or network latency.

2. Parallelism 

    There are several specific types of parallelism in neural networks:
    * _Training parallelism:_ Different training sessions can be run in parallel.
    * _Layer parallelism:_ In a multilayer network, different layers can be processed in parallel.
    * _Node parallelism:_ Node parallelism matches FPGAs very well, since a typical FPGA basically consists of a large number of "cells" that can operate in parallel, on which neurons can be mapped.
    * _Weight parallelism:_ In the computation of an output $y=\Phi(\sum_{i=1}^n w_i x_i)$, where $x_i$ is an input and $w_i$ is a weight, the products $x_iw_i$ can all be computed in parallel, and the sum of these products can also be computed with high parallelism (e.g. by using an adder-tree of logarithmic depth).

    Typically, conventional (i.e. sequential) general-purpose processors do not fully exploit the parallelism inherent in neural network models. Although some SIMD features are already implemented in CPU to offset this problem, proving instruction level parallelism, the specificly designed dataflow and memory pattern in FPGAs may provide better parallelism.
    


### FPGAs for neural networks

1. Heterogenous computation

    Comparing to CPU/GPU, there are some drawbacks of FPGAs, too. Specific hardware programming is needed to implement applications on FPGAs. However, the toolchain and DHL of FPGA programming is hard to use and sometimes buggy, requiring extra time for developer to learn their usage. Also, developers are required to have deep understanding of FPGA's architecture and its design to develop high efficient programs.


2. Networking (data transfering between CPU and FPGA)



3. Low precision computing

    Quatization is a simple and efficient approach to accelerating inference and a sophisticatedly designed quantization method does not damage prediction accuracy of neural networks. Some frequently-used quantization methods, such as fixed-point quantization and low-precision floating point, can be easily implemented in FPGA, making quantization and de-quantization process, along with low-precision numbers' arithmetics faster.

4. Application-specific architecture

    There are several aspects of computer arithmetic that need to be considered in the design of neurocomputers; these include data representation, inner-product computation, implementation of activation functions, storage and update of weights, and the nature of learning algorithms.


5. Domain-specific language(???)/ISA.

6. FPGA-specific neural network model






## Industry

[TODO] industry existing projects

1. [BrainChip](https://www.brainchipinc.com/)'s FPGA-based Neuromorphic Accelerator

BrainChip in September 2017 introduced a commercial PCI Express card with a Xilinx Kintex Ultrascale FPGA running neuromorphic neural cores applying pattern recognition on 600 video images per second using 16 watts of power. 

BrainChip's accelerator is a PCIe server-accelerator card that simultaneously processes 16 channels of video in a variety of video formats using spiking neural networks rather than convolutional neural networks. The BrainChip Accelerator card is based on a 6-core implementation BrainChip's Spiking Neural Network processor instantiated in an on-board Xilinx Kintex UltraScale FPGA.

Each BrainChip core performs fast, user-defined image scaling, spike generation, and SNN comparison to recognize objects. The SNNs can be trained using low-resolution images as small as 20x20 pixels.

The processing is done by six BrainChip Accelerator cores in a Xilinx Kintex Ultrascale field-programmable gate array (FPGA). Each core performs fast, user-defined image scaling, spike generation, and spiking neural network comparison to recognize objects. Scaling images up and down increases the probability of finding objects, and due to the low-power characteristics of spiking neural networks, each core consumes approximately one watt while processing up to 100 frames per second. In comparison to GPU-accelerated deep learning classification neural networks like GoogleNet and AlexNet, this is a 7x improvement of frames/second/watt.


<img src="images/BrainChip-effective frames per sec per watt.png" alt="BrainChip performance compared to GoogLeNet and AlexNet on GPU">

[TODO] The architecture of BrainChip, or is it closed source?

2. Microsoft's [BrainWave](https://www.microsoft.com/en-us/research/uploads/prod/2018/03/mi0218_Chung-2018Mar25.pdf) project ([aml-real-time-ai]((https://github.com/Azure/aml-real-time-ai)))

https://www.top500.org/news/microsoft-launches-fpga-powered-machine-learning-for-azure-customers/


3. Baidu's [SDA](https://www.hotchips.org/wp-content/uploads/hc_archives/hc26/HC26-12-day2-epub/HC26.12-5-FPGAs-epub/HC26.12.545-Soft-Def-Acc-Ouyang-baidu-v3--baidu-v4.pdf): Software-Defined Accelerator for Large-Scale DNN Systems

* Software-defined
    - Reconfigure active functions by user-space API
    - Support very fast iteration of internet services
* Combine small requests to big one
    - Improve the QPS while batch size is small
    - The batch size of real workload is small
* CUBLAS-compatiable APIs
* Performance
    - Provide higher performance in the DNN prediction system than GPU and CPU server
    - Leverage mid-end FPGA to achieve about 380Gflops
    - 10~20W power in real production system
* Can be deployed in any types of servers
* Demonstrate that FPGA is a good choice for large-scale DNN systems

4. Xilinx's [XDNN](https://www.nextplatform.com/2018/08/27/xilinx-unveils-xdnn-fpga-architecture-for-ai-inference/) FPGA Architecture for AI Inference

https://www.nextplatform.com/2018/07/18/fpga-maker-snaps-up-deep-learning-chip-startup/

5. Alibaba

https://www.computerweekly.com/blog/Eyes-on-APAC/An-inside-look-at-Alibabas-deep-learning-processor

6. Intel (?)

[Intel FPGAS Powering Real-Time AI Inferencing](https://ai.intel.com/intel-fpgas-powering-real-time-ai-inferencing/?utm_campaign=2018-Q3-US-AI-Always-On-IntelAI_FB&utm_source=facebook&utm_medium=social&utm_content=157_Static_DMT_CSTM&utm_keyword=read-more&cid=2018-Q3-US-AI-Always-On-IntelAI_FB&spredfast-trk-id=sf195620923)

https://www.nextplatform.com/2018/07/31/intel-fpga-architecture-focuses-on-deep-learning-inference/

### Other AI accelerator (ASIC)

## Acdemia

[TODO] related papers

1. [A Genral Neural Network Hardware Architecture on FPGA](https://arxiv.org/ftp/arxiv/papers/1711/1711.05860.pdf)

2. [DLAU](https://arxiv.org/pdf/1605.06894.pdf): A Scalable Deep Learning Accelerator Unit on FPGA

[LeCun'09] [Farabet'10] [Aysegui'13] [Gokhale'15] [Zhang'15], etc. 

http://cadlab.cs.ucla.edu/~cong/slides/HALO15_keynote.pdf

## Conclusion

[TODO] What? Why? How? Exsiting projects.


## Reference
1. [FPGA introduction on Wikipedia](https://en.wikipedia.org/wiki/Field-programmable_gate_array)
2. [FPGA Implementations of Neural Networks](https://link.springer.com/book/10.1007/0-387-28487-7)

4. [BrainWave repo](); [BrainWave on Azure (aml-real-time-ai)](https://github.com/Azure/aml-real-time-ai)
