import numpy as np
import torch

class Generate_Dataset:
    """
    A class to generate synthetic datasets for training models.

    This class generates a dataset of input-output pairs based on specified 
    limits. The inputs are random integers within a defined range, and the 
    outputs are computed based on specific rules, such as multiplying inputs 
    by 1111 or generating alternative numbers.
    """
    def __init__(self, datasetSize, input_lowerLimit, input_upperLimit, output_lowerLimit, output_upperLimit):
        """
        Initialize the Generate_Dataset class.

        Sets up the parameters for dataset generation.

        Parameters:
        datasetSize (int): The number of input-output pairs to generate.
        input_lowerLimit (int): The minimum value for the input numbers.
        input_upperLimit (int): The maximum value for the input numbers.
        output_lowerLimit (int): The minimum value for the output numbers.
        output_upperLimit (int): The maximum value for the output numbers.
        """
        self.datasetSize = datasetSize
        self.input_lowerLimit = input_lowerLimit
        self.input_upperLimit = input_upperLimit
        self.output_lowerLimit = output_lowerLimit
        self.output_upperLimit = output_upperLimit

    def __getOtherNumber(self)->int:
        """
        Generate an alternative output number.

        This method generates a random integer within the output limits, 
        avoiding multiples of 1111 that result in a number greater than the 
        upper limit. It adds a calculated value to avoid conflicts.

        Returns:
        int: A generated alternative number for the output.
        """
        randNum = np.random.randint(self.output_lowerLimit, self.output_upperLimit-9) # -9 to avoid 9*1111
        return randNum + randNum//1111

    def getX_Y(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate input-output pairs for the dataset.

        This method creates random input values within the specified limits and 
        computes their corresponding output values according to defined rules.

        Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two tensors:
            - X: The input data tensor.
            - Y: The output data tensor. 
        """
        X = np.random.randint(self.input_lowerLimit, self.input_upperLimit + 1, self.datasetSize)
        Y = np.zeros((self.datasetSize,), dtype=int)
        for i in range(len(X)):
            if 1 <= X[i] <= 9:
                Y[i] = X[i] * 1111
            else:
                Y[i] = self.__getOtherNumber()
        
        return torch.tensor(X,dtype=torch.float32).view(-1,1), torch.tensor(Y,dtype=torch.float32).view(-1,1)
