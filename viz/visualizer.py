import torch
import matplotlib.pyplot as plt


class Visualizer:
    """
    A class that
    1. takes a pretrained patient-level classifier
    2. takes data (tensors)
    3. forwards the tensor through the classifier
    4. Gets the attention & gradient info or other labels, depending on the classifier
    5. Uses the labels from 4. to display the top and bottom images, or shows a heatmap
    """
    def __init__(self):
        self.possible_classifiers = ['deepmil']

    def visualize_first_patient(self, loader, model, method):
        assert method in self.possible_classifiers, f"{method} has not been implemented yet. Please choose one of {self.possible_classifiers}"

        for step, data in enumerate(loader):
            x = data[0]
            y = data[1]

            Y_out, Y_hat, A = model.forward(x)


            print(x.shape, A.shape)
            return

    @staticmethod
    def funcname(parameter_list):
        pass

