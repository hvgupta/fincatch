import numpy as np

class Generate_Dataset:
    def __init__(self, datasetSize, input_lowerLimit, input_upperLimit, output_lowerLimit, output_upperLimit):
        self.datasetSize = datasetSize
        self.input_lowerLimit = input_lowerLimit
        self.input_upperLimit = input_upperLimit
        self.output_lowerLimit = output_lowerLimit
        self.output_upperLimit = output_upperLimit
        

    def __getOtherNumber(self)->int:
        randNum = np.random.randint(self.output_lowerLimit, self.output_upperLimit-9) # -9 to avoid 9*1111
        return randNum + randNum//1111

    def getX_Y(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.random.randint(self.input_lowerLimit, self.input_upperLimit + 1, self.datasetSize)
        Y = np.zeros((self.datasetSize,), dtype=int)
        for i in range(len(X)):
            if 1 <= X[i] <= 9:
                Y[i] = X[i] * 1111
            else:
                Y[i] = self.__getOtherNumber()
        
        return X, Y