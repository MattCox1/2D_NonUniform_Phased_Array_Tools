# 2D_NonUniform_Phased_Array_Tools


This python notebook allows a user to create an optimised 2D non-uniform phased array design based on a determined scoring function. In its current state, it performs beamforming via interference pattern generation. 

### NOTE: This repository is in its very early days and as of now, is incomplete.


## Phased arrays

Phased arrays are antenna systems that can electronicly control directionality of either an emitted or recieved signal. Rather than having to physically move a single antenna element to direct a signal, a phased array antenna has multiple antenna elements that one can apply phase shifts (or time delays) to in order to change interference of their signals at a given point in space. To put this another way, simply by applying a set of time delays all the antenna elements you can create a beam of directed radiation (electromagnetic, acoustic and others). This beam can be steered by altering these time delays. If the phased array is recieving information rather than transmitting, the principle is exactly the same but in reverse. The "beam" that was before directed outwards can now be thought of as the array's sensitivity to a signal originating from that location. 



## Beamforming

Overview

Beamforming is a technique used in phased array antennas to control the directionality of the emitted or received electromagnetic waves. This is achieved by adjusting the relative phase of the signals at each element of the antenna array, allowing the radiation pattern to be shaped in a specific way. The concept of beamforming is based on the fact that the directionality of an electromagnetic wave emitted by an antenna is determined by the phase and amplitude of the signal at the antenna elements. When the phases and amplitudes of the signals at the different elements are adjusted in a specific manner, the resulting wavefront can be made to propagate in the desired direction. This allows the radiation pattern of the antenna to be steered, allowing it to be focused on a specific target or to avoid interference from other sources. Beamforming is performed by using a beamforming algorithm, which calculates the required phase and amplitude adjustments for each element of the antenna array. The phase and amplitude adjustments are then applied to the signals at each element using phase shifters and amplitude modulators. One can use beamforming to result in an increased gain of the antenna, improving its range and sensitivity. It is common to assume that all of the elements in a phased array antenna are emitters rather than receivers when analyzing its performance. This assumption simplifies the analysis and allows us to create an interference pattern, which can be used to determine the antenna's key characteristics. By understanding these characteristics, we can evaluate the antenna's performance and make improvements as necessary. In this experiment I have, thus far, made all the antenna elements emitters. The interference pattern is being analysed and captured on a 2D screen that is 2m by 2m and 1m away from the array, this yields approximately a 45 degree view and appears to capture most of the side lobes. The gain of an antenna is a measure of its efficiency in transmitting or receiving a signal, and it is typically expressed in decibels (dB), although I have not used dB. The gain of an antenna can be determined by its radiation pattern. An antenna with a highly directional radiation pattern, such as a dish antenna, will have a high gain because it is able to concentrate the emitted signal in a specific direction. An antenna with a less directional radiation pattern, such as an omnidirectional antenna, will have a lower gain because it radiates the signal in all directions. 
