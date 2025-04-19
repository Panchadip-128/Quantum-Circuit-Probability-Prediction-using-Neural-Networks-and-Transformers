# Quantum-Circuit-Probability-Prediction-using-ML

The Quantum Circuit Probability Predictor is a machine learning-based application designed to predict the probability of measuring a specific quantum state after applying a series of quantum gates to a qubit. Leveraging the principles of quantum mechanics and classical machine learning, this project aims to create a robust model that accurately estimates the probabilities associated with different quantum states resulting from varied input parameters.

The core functionalities include:

Quantum Circuit Simulation: 
-----------------------------
Utilizing Qiskit's advanced quantum simulation capabilities, the project creates quantum circuits that implement rotations around the X-axis based on user-defined angles.

State Probability Calculation:
-------------------------------
The application computes the probabilities of measuring the |0⟩ and |1⟩ states for various angles, using statevector sampling to retrieve the state vector of the quantum circuit after the operations are performed.

Model Training: 
----------------
A machine learning model is trained on the computed probabilities to predict outcomes for angles not seen during training, enabling the model to generalize well to new inputs.

Interactive Visualization: 
--------------------------
The project features an intuitive interface that allows users to input angles and visualize the resulting probabilities and model predictions, enhancing the understanding of quantum state dynamics.

Educational Tool:
-----------------
This project serves as an educational resource for students and enthusiasts interested in quantum computing and machine learning, demonstrating the intersection of these fields through hands-on experience.

Technologies Used:
------------------
Quantum Computing Framework: Qiskit
Machine Learning: Python, NumPy, and relevant ML libraries (e.g., scikit-learn, TensorFlow, or PyTorch)
Data Visualization: Matplotlib or similar libraries for plotting probabilities and predictions
User Interface: Streamlit or Flask for creating a web application interface (to be deployed soon after making model more optimized)

Sample Output of predicted probability 
![1000034342](https://github.com/user-attachments/assets/bc3fc538-ef41-47bb-a685-a2ff63d942cc)
![1000034344](https://github.com/user-attachments/assets/dc0d666e-8417-4c9c-88eb-2ae33b85ccc0)

# Transformers performance visualizations:

This project demonstrates how a Transformer neural network can learn to predict quantum measurement probabilities for a single-qubit RX rotation circuit. The workflow integrates quantum theory, machine learning, and comprehensive evaluation with clear visualizations.

Workflow Overview
Data Generation

Compute the true quantum probability of measuring ∣1⟩ after an RX(θ) rotation:
P(∣1⟩)
= sin2(θ/2) P(∣1⟩)=sin 2(θ/2)

Generate a dataset of 1,000 θ values from 0 to π and their corresponding probabilities.

Model Architecture

Use a Transformer-based regression model with positional encoding to map input angles (θ) to output probabilities.

The model is implemented in PyTorch and consists of an input linear layer, positional encoding, Transformer encoder layers, and an output linear layer.

Training

The model is trained for 1,000 epochs using the Adam optimizer and mean squared error (MSE) loss.

Training data: θ as input, true quantum probability as target.

Evaluation

After training, the model’s predictions are compared to the true quantum probabilities across the full range of θ


Key metrics and visualizations are generated to assess model performance.

Key Visualizations
Quantum Measurement Prediction Curve:
Plots the true quantum probability and the Transformer’s prediction as functions of θ, showing close agreement and highlighting any deviations.

Correlation Heatmap:
Shows high correlation between θ, true probabilities, predicted probabilities, and absolute errors (mainly for completeness; limited insight here).

True vs. Predicted Scatter Plot:
Each point compares the model’s prediction to the true value. Points close to the diagonal indicate high accuracy.

Absolute Error Histogram:
Displays the distribution of prediction errors, demonstrating that most errors are small (often below 0.07).

![trans_hist_err](https://github.com/user-attachments/assets/392b6d08-8372-45d2-a964-53d14d6077f0)
![true_vs_pred_trans](https://github.com/user-attachments/assets/84766c42-2dfa-4226-af6d-22db7bb8d6d0)
![transformer_pred_quantum](https://github.com/user-attachments/assets/7cca0e85-78d4-440b-9d91-a3eea736d46c)

