# ResNet-for-JAX

[![Project Status: Working](https://img.shields.io/badge/status-working-brightgreen.svg)](https://github.com/yourusername/Audio-VQVAE-for-JAX)

A JAX-based implementation of ResNet, heavily based on XTTS. This model is part of the JAXTTS (eXtended Text-To-Speech) series, where I rewrite XTTS in JAX to understand how it works from A to Z, and learn JAX along the way.

## Overview

This project leverages **JAX** and **Equinox** for a HiFiGAN model focused on audio data, copying [XTTS](https://github.com/coqui-ai/TTS)'s structure.

## Features

- A convolutional network mapping Mel Spectrograms to a latent representation of 512 dim vectors, that can then be fed to HiFiGAN. 
- Comprehensive JAX and Equinox integration
- Documentation with step-by-step tutorials and explanations for each module
- A notebook with code to map the weights of the Coqui-ai's to the JAX implementation. Outputs have very high similarity with the model weights of XTTS.

## Getting Started

To get started, clone the repository and follow the tutorial on https://tugdual.fr/ResNet-for-JAX/