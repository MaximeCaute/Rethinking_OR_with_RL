# Research Logs
__________________
## March 04th 2020

### DONE

#### Bibliography
- Consultation of 14 variously related papers
- Detailed reading of a 2007 problem presentation & state-of-the-art survey (Laporte & Cordeau)
- Detailed reading of a summary of the incentive project  (Stocco & Alahi)
- Reading of recent, closely related article: Deep Pool (Al-Abbasi & al.)
- Reading of a genetic algorithm 2007 approach for DaRP (Bergvinsdottir & al.)

#### Code
- Code setup
- Code running for trouble shooting
- Minor code reading
- Pytorch training

### TODO

#### Generic
- Preparing a presentation of the project (understanding exercise)

#### Bibliography
- Further search for closely related articles (e.g. "*Imitating OR with RL*")
  - Especially, checking if no recent article are similar to our goals

#### Code
- Playing with the code to understand it
  - Trying to retrieve the precedent results
  - Trying different parameters configurations (ex: architecture)

#### Report
- Writing the "Related Work" section
__________________
# March 11th 2020

### DONE

#### Generic
- Prepared a generic presentation of the project (6 slides)

#### Bibliography
- Consultation of several closely related papers
- Found a 1997 MIT Report for RL in DaRP

#### Code
- Several neural Networks trained and tested

#### Report
- Planned "Related works section"

### TODO

#### Generic
- Testing servers for tests

#### Code
- Continue testing imitation learning networks

#### Report
- Continuing the "Related Work" section

__________________
# March 17th 2020

**CoVID-19 pandemic measures hampering organization**
__________________
# March 24th 2020

### DONE

#### Code
- Reproduction of some precedent results with neural networks.
*NB: Strangely, slightly lower accuracy*
- Tested several minor architecture changes: added convolutional, linear layers, changed layer sizes, convolutional kernel size...
  - not very conclusive, generally worse results

#### Bibliography
- Lecture of Vinyals et al.
- Detailed lecture of Bongiovanni et al.,
  - Brief analysis of policy: presence of randomness in inputs, sequential algorithm.
- Detailed lecture of Stocco and Alahi for problem framing.

#### Report
- Wrote a first "Related Works" version, mentioning the most closely related works

### TODO

#### Generic
- Testing servers for tests (access received)

#### Theory
- Ensure policy's learnability
  - How to limit randomness?
  - Can we capture the whole information?
  - Aiming for the best modelization of states
- Find methods for RL improvement (with reward for efficiency...)

#### Code
- Explore RNNs architecture (to model iterations of the algorithm)
- Further results reproduction

__________________
# April 1st 2020

### DONE

#### Generic
- SSH access (through VPN) tested for servers

#### Theory
- Isolated essential components for problem encapsulation(e.g. time windows)
  -> Still difficult learnability
- Considered theoretical methods for RL improvement.

#### Code
- Tried several RNN architecture (low results)

### TODO

#### Code
- Test network on simple, deterministic policies(e.g nearest neighbour)
- Generate new data with time-windows

__________________
# April 8th 2020

### DONE

#### Code
- Tested network on numerous policies with different architectures

### TODO

#### Code
- Reimplement training and testing from scratch
- Try out several architectures on basic

__________________
# April 15th 2020

### DONE

#### Code
- Reimplemented training and testing from scratch

#### Experiments
- Tested simple networks on Top Corner and Nearest Neighbour Policy

### TODO

#### Experiments
- Fine tune accuracy results on larger images
- Look for Atari games learning networks

__________________
# April 22nd 2020

### DONE

#### Code
- Implemented AtariNet
- Implemented training curves studies

#### Experiments
- Identified effective learning rate

### TODO

#### Experiments
- Fine tune accuracy results
  - Identify most important parameters (e.g. learning rate)
  - Try much larger data sets
  - Try data normalization
  - Try replacing zeros in images

__________________
# April 29st 2020

### DONE

#### Experiments
- Image changes

### TODO

#### Code
- Remove dropoff.
