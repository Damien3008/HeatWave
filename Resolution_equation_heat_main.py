# ===========================================================================================================================================================
# Resolution of the heat equation in one and two dimensions: Main code.
# ===========================================================================================================================================================

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#Importing Libraries

import numpy as np
from Resolution_equation_heat import EqHeat as H

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Examples for the 1D resolution:
    
#%% Example 1: vector approach

obj1 = H(L = 10 , T = 100 , M = 50 , N = 50 , gamma = 1)
obj1.Resolution1D_vect(a = lambda t : 0,
                        b = lambda t : 0, 
                        f = lambda x , t : 0,
                        u0 = lambda x : x * (obj1.L - x))


#%% Example 1: matrix approach

obj1 = H(L = 10 , T = 100 , M = 50 , N = 50 , gamma = 1)
obj1.Resolution1D_mat(a=lambda t : 0,
                        b=lambda t : 0, 
                        f=lambda x,t : 0,
                        u0 = lambda x : x*(obj1.L-x))



#%% Example 2:

obj1 = H(L = 10 , T = 100 , M = 50 , N = 50 , gamma = 1)
obj1.Resolution1D_vect(a = lambda t : 10 * (1 - np.cos(15 * np.pi * t / obj1.T)),
                        b = lambda t : 10 * (1 - np.cos(15 * np.pi * t / obj1.T)), 
                        f = lambda x , t : 0,
                        u0 = lambda x : 0)

#%% Example 3:
  
obj1 = H(L = 10 , T = 100 , M = 50 , N = 50 , gamma = 1)
obj1.Resolution1D_vect(a = lambda t : 5 * np.cos(5 * np.pi * t / obj1.T),
                        b = lambda t : 0, 
                        f = lambda x , t : 0,
                        u0 = lambda x : 0)


#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Examples for the 2D resolution:
    
#%% Example 1:
    
obj2 = H(L = 5 , T = 5 , M = 50 , N = 50 , gamma = 1)
obj2.Resolution2D(f = lambda x , y , t : 0 , u0 = lambda x , y : 1)

#%% Example 2:

obj2 = H(L = 5 , T = 5 , M = 50 , N = 50 , gamma = 1)
obj2.Resolution2D(f = lambda x , y , t : 0 , u0 = lambda x , y : np.exp(-x ** 2))

#%% Example 3:

obj2 = H(L = 5 , T = 5 , M = 50 , N = 50 , gamma = 1)
obj2.Resolution2D(f = lambda x , y , t : np.exp(-x ** 2) , u0 = lambda x , y : 5 * np.cos(5 * np.pi))

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Bonus: creating videos

#%% Example with a video

obj3 = H(L = 5 , T = 5 , M = 30 , N = 30 , gamma = 1)
obj3.Resolution2D(f = lambda x , y , t : np.cos(np.pi * t + x) , u0 = lambda x , y : (1 / 2) * np.exp(-x ** 2) , vid = True, filename = "video_u0_exp_f_cos")



