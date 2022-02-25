# =================================================================================================================================================================================
# Resolution of the heat equation in one and two dimensions: Library file
# =================================================================================================================================================================================

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import ImageGrab
import cv2
import time

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creation of the classe Eqchaleur:

class EqHeat:
    
    def __init__(self,L,T,M,N,gamma):
        
        """
        Parameters
        ----------
        L: int (Length).
        T: int (Time).
        M: int (Space step).
        N: int (Time step).
        gamma: float (coefficient).
        """
        
        self.M = M
        self.N = N
        self.T = T
        self.L = L
        self.dx = self.L/self.M
        self.dt = self.T/self.N
        self.gamma = gamma

        self.l = self.gamma * (self.dt / (self.dx ** 2))
          
    
    def Resolution1D_vect(self , a , b , f , u0):

        """
        Parameters
        ----------
        a: anonymous function (alpha).
        b: anonymous function (Beta).
        f: anonymous function (f).
        """
        
        # Initialization of initial conditions
        self.a = a
        self.b = b
        # Definition of intervals
        x = np.linspace(0 , self.L , self.M)
        t = np.linspace(0 , self.T , self.N + 1)
        
        # Construction of matrix B and inversion
        B = ((3 / 2) + 2 * self.l) * np.diag(np.ones(self.M)) - self.l * np.diag(np.ones(self.M - 1) , 1) - self.l * np.diag(np.ones(self.M - 1) , -1)
        Binv = np.linalg.inv(B)
           
        # Anonymous function u0 for t=0
        U0 = np.zeros(self.M).reshape(self.M , 1)
        for i in range(0 , self.M , 1):
            U0[i] = u0(x[i])
        
        # Construction of the matrix C with the conditions in x=0 and x=L at time t
        C = np.zeros(self.M).reshape(self.M , 1)
        C[0] , C[-1] = self.a(t[1]) , self.b(t[1])
        C = self.l * C
        
        # We build the matrix F: 2nd member of the PDE
        F = np.zeros((self.M,self.N))
        for i in range(0 , self.M , 1):
            for j in range(0 , self.N , 1):
                F[i , j] = f(x[i] , t[j])
            
        # We calculate U1 with U0
        K = (1 + 2 * self.l) * np.diag(np.ones(self.M)) - self.l * np.diag(np.ones(self.M - 1) , 1) - self.l * np.diag(np.ones(self.M - 1) , -1)
        Kinv = np.linalg.inv(K)
        U1 = Kinv @ U0 + self.dt * (Kinv @ F[: , 1].reshape(self.M , 1))

            
# =============================================================================
#                           GRAPHIC DISPLAY
# =============================================================================
        
        # Construction of the graphic display
        fig,ax = plt.subplots()
        ax.set_title("Evolution of temperature on a string")
        ax.set_xlabel("x")
        ax.set_ylabel("Temperature")
        courbe, = ax.plot(x , U0)
        plt.show(block = False)
        fig.canvas.draw()
        ax.set_xlim(0 , self.L)
        ax.set_ylim(-5 , 20)
    
        
        # Loop to display the solution vector U in real time
        for i in range(2 , self.N , 1):

            U = 2 * Binv @ U1 - (1 / 2) * Binv @ U0 + Binv @ (self.dt * F[: , i].reshape(self.M , 1)) + Binv @ C
            U0 = U1
            U1 = U
            
            courbe.set_ydata(U0)
            ax.draw_artist(ax.patch)
            ax.draw_artist(courbe)
            fig.canvas.update()
            fig.canvas.flush_events()
            plt.pause(0.1)
            
            # Conditions in x=0 and x=L at the following time
            C[0] , C[-1] = self.a(t[i + 1]), self.b(t[i + 1])
            C = self.l * C
            
            
            
        
    def Resolution1D_mat(self , a , b , f , u0):

        """
        Parameters
        ----------
        a: anonymous function (alpha).
        b: anonymous function (Beta).
        f: anonymous function (f).
        """
        
        # Initialization of initial conditions
        self.a = a
        self.b = b
        
        # Definition of intervals
        x = np.linspace(0 , self.L , self.M)
        t = np.linspace(0 , self.T,self.N + 2)
        
        # Construction of the matrix by block A
        B = ((3 / 2) + 2 * self.l) * np.diag(np.ones(self.M)) - self.l * np.diag(np.ones(self.M - 1) , 1) - self.l * np.diag(np.ones(self.M - 1) , -1)
        IM = np.eye(self.M)
        IN = np.eye(self.N)
        A = -2 * np.kron(IN , IM) + np.kron(np.diag(np.ones(self.N - 1) , 1) , B) + (1 / 2) * np.kron(np.diag(np.ones(self.N - 1) , -1) , IM)

        P = np.zeros((self.N , self.N))
        P[-1 , -1] = 1
        
        # We calculate U1 with U0
        K = (1 + 2 * self.l) * np.eye(self.M) - self.l * np.diag(np.ones(self.M - 1) , 1) - self.l * np.diag(np.ones(self.M - 1) , -1)
        Kinv = np.linalg.inv(K)
        
        A += np.kron(P , B @ Kinv) 
        
               
        # We build the matrix F: 2nd member of the PDE
        F = np.zeros((self.M,self.N + 1))
        for i in range(0 , self.M , 1):
            for j in range(0 , self.N + 1 , 1):
                F[i , j] = f(x[i] , t[j])
                     
        # Creation U0
        U0 = np.zeros(self.M)
        for i in range(0 , self.M , 1):
            U0[i] = u0(x[i])
            
        d = np.zeros(self.M * self.N)
        d[:self.M] = (-1 / 2) * U0
        
        # Construction of the matrix C with the conditions in x=0 and x=L at time t
        C = np.zeros(self.M * self.N)
        for i in range(0 , self.N , 1):
            C[i * self.M] = self.l * self.a(t[i + 2])
            C[(i * self.M) - 1] = self.l * self.b(t[i + 2])
            
        # Construction of F 
        F = np.zeros(self.M * self.N)
        for i in range(0 , self.M , 1):
            for j in range(0 , self.N , 1):
                F[j * self.M + i] = self.dt * f(x[i] , t[j + 2])
        
        d += C + F
        # Inversion of A, A is not invertible so pinv
        Ainv = np.linalg.pinv(A)
        
        # Calculation of the solution matrix U
        U = Ainv @ d
                   
# =============================================================================
#                              GRAPHIC DISPLAY
# =============================================================================
        
        # Construction of the graphic display
        fig , ax = plt.subplots()
        ax.set_title("Evolution of temperature on a string")
        ax.set_xlabel("x")
        ax.set_ylabel("Temperature")
        plt.plot(x , U0 , '*')

        
        # Loop to display in real time the solution vector U at instant i U[:,i]
        for i in range(1 , self.N , 1):
            plt.plot(x , U[i * self.M: (i + 1) * self.M])
            ax.set_title("Evolution of temperature on a string")
            ax.set_xlabel("x")
            ax.set_ylabel("Temperature")
            plt.pause(0.1)
            
    
    def Resolution2D(self , f , u0 ,vid = False ,filename = ""):

        # Definition of intervals
        x = np.linspace(0 , self.L , self.M).reshape(self.M , 1)
        y = np.linspace(0 , self.L , self.M).reshape(self.M , 1)
        
        # Construction of matrix B and block matrix A
        B = ((3 / 2) + 4 * self.l) * np.eye(self.M) - self.l * np.diag(np.ones(self.M - 1) , -1) - self.l * np.diag(np.ones(self.M - 1) , 1)
        A = np.kron(np.eye(self.M) , B) - np.kron(self.l * np.diag(np.ones(self.M - 1) , -1) , np.eye(self.M))- np.kron(self.l * np.diag(np.ones(self.M - 1) , 1) , np.eye(self.M))
        # Inversion of A
        Ainv = np.linalg.inv(A)

        # We build the matrix F: 2nd member of the PDE depending on (x,y,t)
        # We build the matrix U0 depending on (x,y)
        U0 = np.zeros(self.M ** 2).reshape(self.M ** 2 , 1)
        F = np.zeros((self.M ** 2 , 1,self.N))
        compt = 0
        for i in range(0 , self.M , 1):
            for j in range(0 , self.M , 1):
                for k in range(0 , self.N , 1):
                    F[compt , 0 , k] = f(x[i] , y[j] , k)
                U0[compt] = u0(x[i] , y[j])
                compt += 1
        
        # We calculate U1 from U0
        Z = (1 + 4 * self.l) * np.eye(self.M) - np.diag(self.l * np.ones(self.M - 1) , -1) - np.diag(self.l * np.ones(self.M - 1) , 1)
        W = np.kron(np.eye(self.M) , Z) - np.kron(np.diag(self.l * np.ones(self.M - 1) , -1),np.eye(self.M)) - np.kron(np.diag(self.l * np.ones(self.M - 1) , 1) , np.eye(self.M))
        Winv = np.linalg.inv(W)
        U1 = Winv @ U0 + self.dt * Winv @ F[: , : , 1]
        
        # We create the solution matrix Zf in which we will implement U
        Zf = np.zeros((self.M + 2 , self.M + 2))
        
        # A grid is made for the display of Zf
        x1 = np.linspace(0 , self.L , self.M + 2)
        y1 = np.linspace(0 , self.L , self.M + 2)
        X1 , Y1 = np.meshgrid(x1 , y1)
        
        Zf[1:-1 , 1:-1] = U0.reshape((self.M , self.M)).T
        
# =============================================================================
#                              GRAPHIC DISPLAY
# =============================================================================

        # Creation of the graphic display
        fig = plt.figure("3D heat equation" , figsize = plt.figaspect(1./2.))
        ax1 = fig.add_subplot(1 , 2 , 1 , projection = '3d')
        ax2 = fig.add_subplot(1 , 2 , 2)
        ax1.set_xlim3d([0 , self.L])
        ax1.set_xlabel('x')
        ax1.set_ylim3d([0 , self.L])
        ax1.set_ylabel('y')
        ax1.set_zlim3d([0 , 1])
        ax1.set_zlabel('T')
        plt.suptitle("Temperature evolution on a flat plate for T={}".format(self.T))
        ax2.set_xlim([0 , self.L])
        ax2.set_xlabel('x')
        ax2.set_ylim([0 , self.L])
        ax2.set_ylabel('y')
        imageNum = 0
        
        if vid == True:
            fps = self.T
            curScreen = ImageGrab.grab() 
            height, width = curScreen.size
            video = cv2.VideoWriter('{}.avi'.format(filename), cv2.VideoWriter_fourcc(*'XVID') , fps , (height , width))
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            
        for z in range(0 , self.N , 1):

            ax1.cla()
            ax2.cla()
    
            ax1.set_xlim3d([0 , self.L])
            ax1.set_xlabel('x')
            ax1.set_ylim3d([0 , self.L])
            ax1.set_ylabel('y')
            ax1.set_zlim3d([0 , 1])
            ax1.set_zlabel('T')
            plt.suptitle("Temperature evolution on a flat plate for T={}".format(self.T))
            
            ax2.set_xlim([0 , self.L])
            ax2.set_xlabel('x')
            ax2.set_ylim([0 , self.L])
            ax2.set_ylabel('y')
            
            U = 2 * Ainv @ U1 - (1 / 2) * Ainv @ U0 + self.dt * Ainv @ F[: , : , z]
            U0 = U1
            U1 = U
            
            Zf[1:-1 , 1:-1] = U.reshape((self.M , self.M)).T
            ax1.plot_surface(X1 , Y1 , Zf , cmap = cm.jet)
            ax2.contourf(X1 , Y1 , Zf , cmap = cm.jet)
            plt.pause(0.01)

            if vid == True:
                imageNum += 1
                captureImage = ImageGrab.grab()  
                frame = cv2.cvtColor(np.array(captureImage) , cv2.COLOR_RGB2BGR)
             
                video.write(frame)
                
        if vid == True:
            video.release()
            cv2.destroyAllWindows()
                        
       
             





