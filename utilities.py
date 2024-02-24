import numpy as np
import pickle




def calculate_angle(a, b, c):
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radian * 180/np.pi)
    
    return angle


def load_model(title):
    with open(title, "rb") as f:
        model = pickle.load(f)
        
    return model