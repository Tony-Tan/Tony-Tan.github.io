---
title: 'An Introduction to Neural Networks'
tags:
 - Artificial Neural Networks
 - Artificial Intelligence
 - Neural Networks
 - Biological Inspiration
 - Dendrites
 - Cell body
 - Axon
 - History of Artificial Neural Networks
 - structure of a neuron
keywords:
 - Artificial Neural Networks
 - Artificial Intelligence
 - Neural Networks
 - Biological Inspiration
 - Dendrites
 - Cell body
 - Axon
 - History of Artificial Neural Networks
 - structure of a neuron
categories:
 - Artificial Neural Networks
date: 2019-12-08 19:01:32
lastmod: 2022-04-29 14:00:00
markup: pdc
draft: false
images: ""
url: "/An-Introduction-to-Neural-Networks"
---
## Preliminaries
1. Nothing

## Neural Networks[^1]
Neural Networks are a model of our brain that is built with neurons and is considered the source of intelligence. There are almost $10^{11}$ neurons in the human brain and $10^4$ connections of each neuron to other neurons. Some of these brilliant structures were given when we were born. Some other structures could be established by experience, and this progress is called learning. Learning is also considered as the establishment or modification of the connections between neurons.

A biological Neural Network is a system of intelligence. Memories and other neural functions are stored in the neurons and their connections. Up to now, neurons and their connections are taken as the main direction of research on intelligence.


**Artificial Neural network**(ANN for short) is the name of a mathematical model which is a tool for studying and simulating biological neural networks, and what we do here is to build a small neural network and observe their performance. However, these small models have amazing capacities for solving difficult problems which are hard or impossible to achieve by traditional methods. Traditional methods are not the old ones but the ones without learning progress or the ones dealing with traditional problems like sorting, solving equations, etc. What we say small model here is the model with much fewer neurons and connections than the human brain because a small model can be investigated easily and efficiently. However, the bigger models are constructed with small ones. So when we gain an insight into the smaller building blocks, we can predict the bigger ones' performance precisely.


Another fundamental distinction between ANNs and biological neural networks is that ANNs are built of silicon.


## Biological Inspiration

![](https://raw.githubusercontent.com/Tony-Tan/picgo_images_bed/master/2022_04_29_15_46_boogical_inspiration.gif)

This figure represents the abstraction of two neurons. Although it looks humble, it has all the components of our best performance ANNs. This is the strong evidence that tells us that real intelligence is not so easy to simulate.

Let's look at this simplified structure. Three principal components:

- Dendrites
- Cell body
- Axon

Dendrites are tree-like receptive networks of nerve fibers, that carry electrical signals into the cell body. The cell body, sums, and thresholds these incoming signals. Axon is a single long fiber carrying electrical signals to other neurons.

The contact between dendrites and axons in the structure is called **synapse**. This is an interesting structure for **its properties largely influence the performance of the whole network**.

More details of biological neural science should be found in their subject textbooks. However, in my personal opinion, we can never build artificial intelligence by just studying ANNs, what we should do is investigate our brain and neural science. In other words. to find artificial intelligence, go to biological intelligence. However, until today, our models are far from any known brains on earth.

But there are still two similarities between artificial neural networks and biological ones:

1. building blocks of both networks are simple computational devices
2. connection between neurons determines the function of the networks 

> 'there is also the superiority of ANNs(or more rigorous of the computer) that is the speed. Biological neurons are slower than electrical circuits($10^{-3}$ to $10^{-10}$).' 

However, I don't agree with this point, for we don't even know what computation has been done during the period of $10^{-3}$ seconds in the biological neurons. So this comparison made no sense. But the parallel structure in brains is beyond the reach of any computer right now.



## A Brief History of Artificial Neural Networks
This is just a brief history of ANNs because so many researchers had finished so many works during the last 100 years. The following timeline is just some big events in the last 50 years.

'Neurocomputing: foundations of research is a book written by John Anderson. It contains 43 papers on neural networks representing special historical interest.

ANNs come from the building background of physics, psychology, and neurophysiology:

- From the late 19th to early 20th: general theories of learning, vision, and conditioning were built, but there was no mathematical model of neuron operation
- 1943: Warren McCulloch and Walter Pitts found neurons could compute any arithmetic or logic function and this is considered the origin of the neural network field
- 1949: Donald Hebb proposed that classical conditioning is presented because of an individual neuron. He proposed a mechanism for learning in biological neurons.
- 1958: First practical application of ANN which is perceptron was proposed by Rosenblatt. This model was able to perform pattern recognition.
- 1960: Bernard Widrow and Ted Hoff developed a new learning algorithm and train adaptive linear neuron networks which are similar to Rosenblatt's perceptron in both structure and capability.
- 1969: Marvin Minsky and Seymour Papert proved the limitation of Rosenblatt's perceptron and Bernard Widrow and Ted Hoff's learning algorithm. And they thought further research on neural networks is a dead end. This coursed a lot of researchers gave up.
- 1972: Teuvo Kohonen and James Anderson built the neural networks acting as memories independently.
- 1976: Stephen Grossberg built a self-organizing network
- 1982: Statistical mechanics was used to explain the recurrent network by John Hopfield which was also known as an associative memory
- 1986: Backpropagation is proposed by David Rumelhart and James McClelland which broke the limitation given by Minsky

This history ended in 1990. This is just the beginning of the neural network to us now, however, what we do today is also the beginning of the future. This progress is not "slow but sure", it was dramatic sometimes but almost stop most of the time.

New concepts of neural networks come from the following aspects:

- innovative architectures
- training rules



## Conclusion

1. We took a look at the structure of a neuron, ANN is a simple model of a biological neural network.
2. A brief history of neural network 


## References
[^1]: Demuth, Howard B., Mark H. Beale, Orlando De Jess, and Martin T. Hagan. Neural network design. Martin Hagan, 2014.