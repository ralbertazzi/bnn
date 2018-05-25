# Training and Implementation of a CNN for image classification with binary weights and activations on FPGA with HLS tools

This is the repository for the project work of [Reconfigurable Embedded Systems](http://vision.deis.unibo.it/~smatt/Site/Courses.html) at the University of Bologna - Professor [Stefano Mattoccia](http://vision.deis.unibo.it/~smatt/Site/Home.html)

In this work I first learnt about binary networks with binary weights and activations; I then trained a network on the MNIST dataset, achieving 96% accuracy with a small network (460k weights).

However, the core of the project is the implementation of a Binary Network on a FPGA device (Zynq) using High-Level Synthesis Tools (Vivado HLS). I implemented the core modules (convolution, dense, max pooling, padding), optimizing resources, evaluating results and discussing trade-offs between timing and resources. The whole implementation is pipelined and can achieve a throughput of thousands of images/sec by using less than half the resources available on the FPGA.

You can find additional informations inside the provided slides.

# References

* Reference paper for Binary Networks: [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
* [Keras implementation of Binary Net](https://github.com/DingKe/nn_playground): I have adapted my training code from this project
* Papers about implementations of BNNs on FPGA:
  * [FINN: A Framework for Fast, Scalable Binarized Neural Network Inference](https://arxiv.org/abs/1612.07119)
  * [Accelerating Binarized Convolutional Neural Networks with Software-Programmable FPGAs](https://dl.acm.org/citation.cfm?id=3021741)
  
# Author

[Riccardo Albertazzi](https://www.linkedin.com/in/riccardo-albertazzi-03b5aa133/) - University of Bologna - May 2018
