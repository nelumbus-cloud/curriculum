#Sampling strategy for curriculum learning
# define difficulty levels: let beta be fog/weather level, then f(beta) is difficulty level provided f is monotonically increasing function.ArithmeticError
# pacing function: d_t+1 = g(d_t, l_theta_t) where l_theta_t is the loss of the model at time t
#
# Set the Current Difficulty:
# The current allowed maximum degradation difficulty, epsilon, is set to the current level d_k.
#
# Create the Training Batch:
# Start with a mini-batch of clean images and their labels, {(x_i, y_i)}.
# For each clean image x_i:
#   Select a Degradation (t_i):
#   With probability alpha (Adversarial Selection): Find the degradation t (where d(t) <= epsilon) that makes the current detector (h_{k-1}) perform the worst (i.e., maximizes its loss). This creates a challenging example.
#   With probability 1 - alpha (Uniform Selection): Pick a random degradation t from all allowed degradations (where d(t) <= epsilon).
# Generate the Degraded Sample: Create the training image x_tilde_i by applying the selected degradation t_i to the clean image x_i.
# Update the Detector:
# Use the batch of newly generated degraded images {x_tilde_i} and their correct labels {y_i} to update the detector from h_{k-1} to h_k. The goal is to minimize the loss on these degraded images using training techniques like Stochastic Gradient Descent (SGD).


#simple utils to test
def difficulty(beta):
    return beta

def pacing_function(d_t, l_theta_t):
    return d_t + l_theta_t

import mmengine

@mmengine.register_module()
def sampling(cfg):
    pass