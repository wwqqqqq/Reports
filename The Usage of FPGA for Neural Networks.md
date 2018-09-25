# FPGA-accelerated AI

## Background

<p style="color:#FF0000;">[TODO] short introduction of NN and its application. The importance of NN speedup.</p>

### AI accelerator

As Moore's Law approaches to its ending, we are at the end of the performance scaling that we are used to. To accelerate computing, we cannot depend on CPU processors to get faster in high speed as they used to be, but have to design alternative approaches, for example, using specialized processors. In the industry, Google has its TPU (Tensor Processing Unit), with one core per chip and software-cotrolled memory instead of caches; NVidia's GPU has 80-plus cores; and Microsoft is taking an FPGA approach.

## FPGA introduction

A FPGA (field-programmable gate array) is an integrated circuit designed to be configured after manufacturing. The FPGA configuration is generally specified using a hardware description language (HDL).

FPGAs contain an array of programmable logic blocks to be "wired together", like many logic gates that can be inter-wired in different configurations. In most FPGAs, logic blocks also include memory elements, which may be simple flip-flops or more complete blocks of memory.

A FPGA can be used to solve any problem which is computable. Their advantage lies in that they are sometimes significantly faster for some applications because of their parallel nature and optimality in terms of the number of gates used for a certain process. 

FPGAs are commonly used in hardware acceleration, where one can use FPGA to accelerate certain parts of an algorithm and share part of the computation between the FPGA and a generic processor.

<p style="color:#FF0000;">[TODO]
https://www.zdnet.com/article/ai-chips-for-big-data-and-machine-learning-gpus-fpgas-and-hard-choices-in-the-cloud-and-on-premise/</p> 

### Advantages

1. FPGAs provide a combination of programmability and performance comparing to ASIC and CPU/GPU. 

    ASIC is known for its performance, but it can cost huge amount of time to design the specific circuit to achieve best performance. Meanwhile, deap learning frameworks are still evolving, making it hard to design or update custom hardware. FPGAs are reconfigurable, and far easier to design and program comparing to ASIC, making them convenient to extend to a range of applications, e.g. several different types of nueral networks. FPGA structure can be considered as an alternative to software, but is implemented directly on hardware.

    Comparing to GPU/CPU, FPGA is customized and specialized in specific application, making it faster and more energy-efficient, since general-purpose processors invest excessive hardware resources flwxibly support various workloads. Also, due to the intrinsic property of FPGAs, they have high compute throughput and networking throughput, and low compute or network latency.

2. Parallelism 

    There are several specific types of parallelism in neural networks:
    * _Training parallelism:_ Different training sessions can be run in parallel.
    * _Layer parallelism:_ In a multilayer network, different layers can be processed in parallel.
    * _Node parallelism:_ Node parallelism matches FPGAs very well, since a typical FPGA basically consists of a large number of "cells" that can operate in parallel, on which neurons can be mapped.
    * _Weight parallelism:_ Common implementations of neural networks rely heavily on dense matrix multiplication, which is natural suitable for parallel computing. In the computation of an output $y=\Phi(\sum_{i=1}^n w_i x_i)$, where $x_i$ is an input and $w_i$ is a weight, the products $x_iw_i$ can all be computed in parallel, and the sum of these products can also be computed with high parallelism (e.g. by using an adder-tree of logarithmic depth).

    Typically, conventional (i.e. sequential) general-purpose processors do not fully exploit the parallelism inherent in neural network models. Although some SIMD features are already implemented in CPU to offset this problem, proving instruction level parallelism, the specificly designed dataflow and memory pattern in FPGAs may provide better parallelism.

    


### FPGAs for neural networks

1. Heterogenous computation

    Comparing to CPU/GPU, there are some drawbacks of FPGAs, too. Specific hardware programming is needed to implement applications on FPGAs. However, the toolchain and DHL of FPGA programming is hard to use and sometimes buggy, requiring extra time for developer to learn their usage. Also, developers are required to have deep understanding of FPGA's architecture and its design to develop high efficient programs.


2. Networking (data transfering between CPU and FPGA)

    In Microsoft's Project Brainwave, FPGA communicates with CPUs and other FPGAs via LTL (Lightweight Transport Layer).

    <p style="color:#FF0000;">[TODO] about LTL.</p>

    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/Cloud-Scale-Acceleration-Architecture.pdf

3. Low precision computing

    Quatization is a simple and efficient approach to accelerating inference and a sophisticatedly designed quantization method does not damage prediction accuracy of neural networks. Some frequently-used quantization methods, such as fixed-point quantization and low-precision floating point, can be easily implemented in FPGA, making quantization and de-quantization process, along with low-precision numbers' arithmetics faster than their implementation on CPU, since some specific data formats are not naturally supported by general-purpose CPUs.

    For example, in Microsoft, we have a special quantization format called ms-fp9, which includes 1 bit sign, 5 bits exponent and 3 bits mantissa. It can be very hard to implement ms-fp9 arithmetics on CPUs, but on FPGA, we can store ms-gp9 numbers in 9-bit register, and easily design specific arithmetics units for it.

    Accuracy of three models using different quantization formats:

    <img src="images/model's accuracy of different quantization formats.png">

    Peak performance of the Brainwave DPU across three generations of Intel FPGAs:

    <img src="images/peak performance using different quantization formats.png">

4. Application-specific architecture

    There are several aspects of computer arithmetic that need to be considered in the design of neurocomputers; these include data representation, inner-product computation, implementation of activation functions, storage and update of weights, and the nature of learning algorithms. 

5. Domain-specific ISA

    If the architecture is designed as an instruction system with a decoder, new ISA has to be designed. Although it is not necessary to apply the idea of CPU architecture on FPGA, but the architecture including ISA gives FPGA more flexibility to implement different types of neural networks.

    Although it is not a FPGA system but an ASIC one, [Cambricon's ISA](https://ieeexplore.ieee.org/document/7551409/) is a good example for this. Based on a comprehensive analysis of existing neural networks techniques, the developers designed a domain-specific instruction set for their chips, which integrates scalar, vector, matrix, logical, data transfer and control instructions.

6. FPGA-specific neural network model

    <p style="color:#FF0000;">[TODO] BrainChip's spike NN?</p>






## Industry

<p style="color:#FF0000;">[TODO] industry existing projects</p>

1. [BrainChip](https://www.brainchipinc.com/)'s FPGA-based Neuromorphic Accelerator

    BrainChip in September 2017 introduced a commercial PCI Express card with a Xilinx Kintex Ultrascale FPGA running neuromorphic neural cores applying pattern recognition on 600 video images per second using 16 watts of power. 

    BrainChip's accelerator is a PCIe server-accelerator card that simultaneously processes 16 channels of video in a variety of video formats using spiking neural networks rather than convolutional neural networks. The BrainChip Accelerator card is based on a 6-core implementation BrainChip's Spiking Neural Network processor instantiated in an on-board Xilinx Kintex UltraScale FPGA.

    Each BrainChip core performs fast, user-defined image scaling, spike generation, and SNN comparison to recognize objects. The SNNs can be trained using low-resolution images as small as 20x20 pixels.

    The processing is done by six BrainChip Accelerator cores in a Xilinx Kintex Ultrascale field-programmable gate array (FPGA). Each core performs fast, user-defined image scaling, spike generation, and spiking neural network comparison to recognize objects. Scaling images up and down increases the probability of finding objects, and due to the low-power characteristics of spiking neural networks, each core consumes approximately one watt while processing up to 100 frames per second. In comparison to GPU-accelerated deep learning classification neural networks like GoogleNet and AlexNet, this is a 7x improvement of frames/second/watt.


    <img src="images/BrainChip-effective frames per sec per watt.png" alt="BrainChip performance compared to GoogLeNet and AlexNet on GPU">

    <p style="color:#FF0000;">[TODO] The architecture of BrainChip, or is it closed source?</p>

    Akida Architecture: https://globenewswire.com/news-release/2018/09/10/1568247/0/en/BrainChip-Announces-the-Akida-Architecture-a-Neuromorphic-System-on-Chip.html

2. Microsoft's [BrainWave](https://www.microsoft.com/en-us/research/uploads/prod/2018/03/mi0218_Chung-2018Mar25.pdf) project ([aml-real-time-ai]((https://github.com/Azure/aml-real-time-ai)))

    https://www.top500.org/news/microsoft-launches-fpga-powered-machine-learning-for-azure-customers/

    https://www.microsoft.com/en-us/research/project/project-brainwave/

    Project Brainwave is a deep learning acceleration platform for cloud customers. Deployed on Intel FPGA, Project Brainwave allows customers to access dedicated hardware that can accelerate real-time AI calculations giving a competitive cost benefit with low latency. In addition, Microsoft developed and tested a custom 9-bit floating-point format (FP9) before settling on an 8-bit format (FP8) that doubles performance over standard INT8. To meet the needs of its data centers, Microsoft also optimized Brainwave for low latency, maintaining high efficiency even with small numbers of requests.

    **Brainwave Architecture**

    Brainwave consists of three main layers:

    1. a tool flow and runtimes for low-friction deployment of trained models,
    2. a distributed system architecture mapped onto CPUs and hardware microservices, and
    3. a high-performance soft DNN processing unit synthesized onto FPGAs.

    <img src="images/three major layers of the Brainwave.png">

    First in this architecture, DNN models are exported into a common graph intermediate representation (IR). Tool flow optimizes the IR and partitions it into sub-graphs assigned to different CPUs and FPGAs. Device specific backends generate device assembly and are linked together by a federated runtime that gets deployed into a deployable live FPGA hardware microservice.

    On each FPGA, a Brainwave soft NPU is implemented.

    <img src="images/Brainwave soft NPU.png">

    The Brainwave NPU is a "mega-SIMD" vector processor architecture. A sequentially programmed control processor asynchronously controls the neighboring neural function unit (NFU) optimized for fast DNN operations. The heart of the NFU is a dense matrix vector multiplication unit (MVU) capable of processing single DNN requests at low batch with high utilization. The MVU is joined to secondary multifunctional units (MFU) that perform element-wise vector-vector operations and activation functions.

    To achive high parallelism in this architecture, the MVU consists of tens of thousands of parallel multiply accumulators organized into parallel multi-lane vector dot product units.

    <img src="images/multi-lane vector dot unit.png">

    Brainwave has been applied in some of Bing productions. Improvement is significant:

    <img src="images/Brainwave performance.png">


    <p style="color:#FF0000;">[TODO] Azure machine learning - real time AI program.</p>

    https://www.microsoft.com/en-us/research/wp-content/uploads/2014/06/HC26.12.520-Recon-Fabric-Pulnam-Microsoft-Catapult.pdf
    <img src="images/multi-lane vector dot unit.png">

    Brainwave has been applied in some of Bing productions. Improvement is significant:

    <img src="images/Brainwave performance.png">


    <p style="color:#FF0000;">[TODO] Azure machine learning - real time AI program.</p>

    https://www.microsoft.com/en-us/research/wp-content/uploads/2014/06/HC26.12.520-Recon-Fabric-Pulnam-Microsoft-Catapult.pdf

    Usage in ClickNP: https://www.microsoft.com/en-us/research/publication/clicknp-highly-flexible-high-performance-network-processing-reconfigurable-hardware/

    Usage in Bing:
    https://www.researchgate.net/profile/Ningyi_Xu/publication/255596450_FPGA-based_Accelerators_for_Learning_to_Rank_in_Web_Search_Engines/links/55d1f22c08aec1b0429dc9e1/FPGA-based-Accelerators-for-Learning-to-Rank-in-Web-Search-Engines.pdf

    More about it:
    https://www.msra.cn/zh-cn/news/features/fpga-20170111



3. Baidu's [SDA](https://www.hotchips.org/wp-content/uploads/hc_archives/hc26/HC26-12-day2-epub/HC26.12-5-FPGAs-epub/HC26.12.545-Soft-Def-Acc-Ouyang-baidu-v3--baidu-v4.pdf): Software-Defined Accelerator for Large-Scale DNN Systems

    Baidu has deployed deep nerual networks to accelerate many critical services, such as speech recognition, image search, ads, web page search and natural language processing. DNN usually achieves higher performance in certain jobs, but demands more compute power, and sometimes runs significantly slowlier. Baidu's large-scale DNN system costs large amount of servers and time to train, and online prediction also have seconds of latency for large models. To deal with these problems with low budget, Baidu proposes a software-defined accelerator (SDA) using FPGA as low-level hardware.

    SDA is designed to:
    - Supports major workloads, including training and prediction with floating point numbers.
    - Achieve acceptable performance to about 400 Gflops, higher than 16-core x86 server.
    - Cut budget. Medium-end FPGA is at low cost.
    - Remain the existent data center environments.
    - Support fast iteration.

    The major functions that SDA implements are floating point matrix multiplication and floating point activation functions. The customized FP MUL and ADD units reduce about 50% resource compared to standard IPs. The software-defined activation function unit supports tens of activation functions, including sigmoid, tanh, softsign, etc. It is implemented using lookup table and linear fitting, the lookup table in which can be reconfigured by user-space API.

    SDA APIs includes computation APIs, which are similar to and compatiable with CUBLAS APIs, and reconfiguration API for activation functions.

    **Performance**

    The matrix multiplication's GFlops of CPU, GPU and FPGA are shown below:

    <img src="images/Baidu FPGA GFlops.png">

    And the power consumptions are:

    **Processor** | CPU | FPGA | GPU
    -- | -- | -- | --
    **Gflops/W** | 4 | 12.6 | 8.5

    With online prediction workload, input batch size is small (typically 8 or 16), and the size of hidden layer is from several hundreds to several thousands. Two production workload scenes are set:
    1. Workload 1: batch size = 8, layer = 8, hidden layer size = 512.
    2. Workload 2: batch size = 8, layer = 8, hidden layer size = 2048.

    Under these two circumstances, the requests processed per second are measured:

    <img src="images/Baidu FPGA requests per sec.png">

    We can conclude from that FPGA can merge small requests to improve performance and achieve higher throughput, providing higher performance in the DNN prediction system than GPU and CPU server.

    <p style="color:#FF0000;">[TODO] XPU: </p>
    
    https://www.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.40-Processors-Pub/HC29.21.410-XPU-FPGA-Ouyang-Baidu.pdf

    https://www.nextplatform.com/2017/08/22/first-look-baidus-custom-ai-analytics-processor/

4. Xilinx's [XDNN](https://www.nextplatform.com/2018/08/27/xilinx-unveils-xdnn-fpga-architecture-for-ai-inference/) FPGA Architecture for AI Inference

    Xilinx's xDNN architecture focus on inference part of deep learning.


    The xDNN configurable overlay processor maps a range of neural network frameworks onto the VU9P Virtex UltraScale+ FPGA with options to beef up memory, work with custom applications, and tap into a compiler and runtime primed for this on either Amazon's cloud with the F1 instance or in-house.

    FPGAs are data parallel and support data reuse as well as compression and sparsity by nature and with the xDNN processor's 2D array of MACs, flexible on-chip memory access with high bandwidth and several ways to get to it, data movement is more efficient. xDNN also supports flexible data types (i.e. FP32/FP16 and INT 16/8/4/2, etc.).

    xDNN is a configurable overlay processor, which means it gets mapped onto the FPGA without need to reprogram after. Xilinx has also provided a DNN specific instruction set (convolutions, max pool, etc.) and can work with any network or image size and can also compile and run new networks. In other words, it can work with TensorFlow without requiring reprogramming or changing the FPGA.

    The Virtex hardware can be tricked out with several types of memory; from the basic distributed RAM that sits next to the DSP blocks to UltraRAM, High Bandwidth Memory, and external DDR4. This allows for optimization for efficiency or performance.

    <img src="images/Xilinx xDNN architecture.png">

    The processing elements are mapped onto the DSP blocks along with the weights, which are held in fast but low-capacity distributed RAM next to the processing. In other words, these distributed RAMs are weight caches.

    xDNN's "Tensor Memory" sits next to the systolic array and hold input and output feature maps. This is also channel parallel so each of those rows in that array are associated with a row of memory. This means xDNN can multitask, computing on the array while also bringing in a new network layer if needed, for instance.

    xDNN also includes instruction-level parallelism, and automatic intra-layer tiling for when feature map sizes exceed on-chip memory, enabling xDNN to work on any feature map size.

    <p style="color:#FF0000;">[TODO] Further investigation needed.</p>

    https://www.nextplatform.com/2018/07/18/fpga-maker-snaps-up-deep-learning-chip-startup/

    https://github.com/Xilinx/ml-suite

    <p style="color:#FF0000;">[TODO] Xilinx Gemx: http://www.ispd.cc/slides/2018/s2_3.pdf</p>

5. Alibaba DLP

    Alibaba has developed a FPGA-based ultra-low latency and high performance deep learning processor (DLP). Alibaba said its DLP can support sparse convolution and low precision data computing at the same time, while a customized ISA was defined to meet the requirements for flexibility and user experience. Latency test results with ResNet-18 showed that Alibaba's DLP has a delay of only 0.174 ms.

    The supported computing of DLP includes convolution, batch normalization, activation and other calculations. The architecture is illustrated below:

    <img src="images/DLP architecture.jpg">

    The Protocal Engine (PE) in the DLP supports Int4 data type input, Int32 data type output, Int16 quantization. Quantization in PE offers over 90% higher efficiency.

    Low precision computing is used in DLP. In the training process, the model is first trained with full accuracy in FP32, then prunning is included, cutting weight to 85%. Re-training is needed on the sparse model to develop an accurate model. Next, weights are quantized with first ADMM (Alternating Direction Method Multipliers) method and then statistical-based feature map quantization. Sparse rate can be as high as 85% using this process with 4% accuracy loss.

    As FPGA development can take weeks or months, Alibaba designed an industry standard architecture (ISA) and compiler to reduce model upgrade time to just a few minutes. Alibaba's software-hardware co-development platform is shown below:

    <img src="images/Alibaba DLP sw-hw co-development platform.png">

    The DLP was implemented on an Alibaba-designed FPGA card, which has PCIe and DDR4 memory. The DLP, combined with this FPGA card, can benefit applications such as online image searches.

    The latency and QPS of ResNet18 using several different batchsize on DLP FPGA and GPU are illustrated below:

    <img src="images/FPGA vs GPU with Resnet18 (Alibaba).png">

    https://www.computerweekly.com/blog/Eyes-on-APAC/An-inside-look-at-Alibabas-deep-learning-processor

6. Intel (?)

    [Intel FPGAS Powering Real-Time AI Inferencing](https://ai.intel.com/intel-fpgas-powering-real-time-ai-inferencing/?utm_campaign=2018-Q3-US-AI-Always-On-IntelAI_FB&utm_source=facebook&utm_medium=social&utm_content=157_Static_DMT_CSTM&utm_keyword=read-more&cid=2018-Q3-US-AI-Always-On-IntelAI_FB&spredfast-trk-id=sf195620923)

    https://www.nextplatform.com/2018/07/31/intel-fpga-architecture-focuses-on-deep-learning-inference/

### Other AI accelerator (ASIC)

1. Google TPU (Tensor Processing Unit)

2. Cambricon

3. Intel Nervana NNP (Neural Network Processor)

4. TrueNorth

## Acdemia

<p style="color:#FF0000;">[TODO] related papers</p>

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
