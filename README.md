# Applying Neural Controlled Differential Equations (NCDE) with LEAP and Neural Flow for Stock Price Prediction

This repository contains the implementation of a framework for stock price prediction using **Neural Controlled Differential Equations (NCDE)**, **Learnable Path (LEAP)**, and **Neural Flow**. The proposed model improves the prediction accuracy by better capturing temporal dependencies in stock price data.

## Introduction

This work references the paper **"Learnable Path in Neural Controlled Differential Equations"**, which introduces the concept of a 'learnable path' in NCDE. The goal is to use this framework to predict stock prices by capturing the dynamic nature of the market over time. The primary objectives are:

1. **Modeling stock price dynamics** with NCDE to understand how stock prices evolve over time.
2. **Improving prediction accuracy** by integrating **LEAP** and **Neural Flow**, which better capture the temporal dependencies in the data.

### Key Concepts

- **NCDE**: Models continuous-time data and captures temporal dependencies. Unlike traditional models (e.g., RNNs), NCDE learns from continuous flows, making it ideal for irregularly sampled time-series data.
- **LEAP**: Converts input sequences into latent representations, which are then used as initial conditions for the NCDE model, improving its performance.
- **Neural Flow**: A model that uses differential equations to simulate the evolution of states over time, enhancing stock price predictions by modeling the dynamic process.

## Workflow

1. **LEAP** processes the raw input sequences and converts them into initial hidden states.
2. **Neural Flow** applies learnable differential equations to simulate how these hidden states evolve over time.
3. The **odeint solver** integrates the system’s equations, generating the trajectory of the state.
4. The final state is passed through a fully connected layer to generate the predicted stock price.

## Training Process

- Normalize the stock data and create sliding window sequences for both the input and target values.
- LEAP converts the data into initial hidden states for Neural Flow.
- Neural Flow, combined with odeint, calculates how the hidden states evolve over time.
- The final state is used to predict the stock price, and **Mean Squared Error (MSE)** is computed to evaluate the model's performance.
- The model is trained using **backpropagation** and the **Adam optimizer** to improve prediction accuracy.

## Model Structure

The model consists of:

1. **Learnable Path (LEAP)**: Transforms raw input data into smooth paths.
2. **Neural Flow**: Uses these paths to evolve hidden states over time with odeint.
3. **Linear Output Layer**: Converts the final state into predictions (e.g., stock prices).

## Dataset

The dataset used focuses on the **NIFTY 50 Healthcare Sector Stocks** from August 29, 2022, to November 8, 2024, with minute-level stock price data. It includes 203,621 data points with the following price information:

- **Open**: The price at market opening.
- **High**: The highest price during the time interval.
- **Low**: The lowest price.
- **Close**: The closing price.

This dataset allows the model to capture real-time fluctuations in stock prices, which is crucial for continuous-time modeling.

## Performance Comparison

- **Simple NCDE**: MSE = 0.0025
- **Proposed Model**: MSE = 0.0015

The proposed model outperforms the simple NCDE by combining the LEAP module with flow dynamics, improving its ability to model nonlinear temporal patterns and predict stock prices accurately.

## Installation

To run the code, you'll need the following Python libraries:

- `torch`
- `odeint`
- `numpy`
- `matplotlib`
- `pandas`

You can install these dependencies using:

```bash
pip install torch odeint numpy matplotlib pandas
