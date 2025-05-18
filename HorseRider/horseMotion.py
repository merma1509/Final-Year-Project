import numpy as np
import matplotlib.pyplot as plt

class Control_Force:
    def __init__(self, v0, H1, A1, m1, w1, j1, pi, ci2, g):
        self.v0 = v0
        self.H1 = H1
        self.A1 = A1
        self.m1 = m1
        self.j1 = j1
        self.w1 = w1
        self.pi = pi
        self.ci2 = ci2
        self.g = g    
    # this function contains all of our constraints  
    def constraint(self, q, t):
        f1 = q[0] - self.v0*t
        f2 = q[1] - (self.H1 + self.A1*np.sin(q[2]))
        f3 = q[2] - np.cos(self.w1*t) 
        return np.array([f1, f2, f3])
    
    # this function is the first derivative of all our contraints
    def dconstraint(self, q, v, t): 
        Df1 = v[0] - self.v0
        Df2 = v[1] - self.A1*v[2]*np.cos(q[2])
        Df3 = v[2] + self.w1*np.sin(self.w1*t)
        return np.array([Df1, Df2, Df3])
    
    def get_ControlForce(self, q, v, t):
        df = self.dconstraint(q, v, t)
        f = self.constraint(q, t)
        U1 = - self.m1*(2*self.pi*df[0] + self.ci2*f[0])
        U3 = - self.j1*(2*self.pi*df[2] + self.ci2*f[2] + (self.w1**2)*np.cos(self.w1*t))
        w = self.g + self.A1*((1/self.j1)*U3*np.cos(q[2]) - (v[2]**2)*np.sin(q[2]))
        U2 = self.m1*(w - (2*self.pi*df[1] + self.ci2*f[1]))
        return np.array([U1, U2, U3])
    
    def dv(self, q, v, t):
        U = self.get_ControlForce(q, v, t)
        dv1 = (1/self.m1)*(U[0])
        dv2 = (1/self.m1)*(U[1]) - self.g
        dv3 = (1/self.j1)*(U[2])
        return np.array([dv1, dv2, dv3])
        
    def dq(self, q, v, t):
        return v
    
q = np.array([0, 0, 0]) # initial coordinates x1, y1, w1
v = np.array([0, 0, 0]) # initial velocity vx1, vy1, vx2, vy2, w1, w2 
ControlForce = Control_Force(5, 0, 0.06, 500, 0.0766, 10, 0.2, 0.03, 10)
dv = ControlForce.dv
dq = ControlForce.dq
U = ControlForce.get_ControlForce
T = 10*100 # time for motion
dt = 10/T # steps

print(dv(q, v, 0), dq(q, v, 0), U(q,v,0))

x1, y1, O1, vx, vy, O2, time, dConstr = [], [], [], [], [], [], [], []
F1, F2, F3, F4, F5, F6 = [], [], [], [], [], []
KE, PE = [], []
U1, U2, U3 = [], [], []
    
def rk(q, v, t):
    k1 = dt*dv(q, v, t)
    l1 = dt*dq(q, v, t)
    k2 = dt*dv(q+(l1/2), v+(k1/2), t+(dt/2))
    l2 = dt*dq(q+(l1/2), v+(k1/2), t+(dt/2))
    k3 = dt*dv(q+(l2/2), v+(k2/2), t+(dt/2))
    l3 = dt*dq(q+(l2/2), v+(k2/2), t+(dt/2))
    k4 = dt*dv(q+l3, v+k3, t+dt)
    l4 = dt*dq(q+l3, v+k3, t+dt)
    #print("V_init= ", v,"\n")
    v = v + (k1 + 2*k2 + 2*k3 + k4)/6
    q = q + (l1 + 2*l2 + 2*l3 + l4)/6
    #print("V_fin= ", v,"\n")
    return [q, v, t]

def plotter():
    fig, ax = plt.subplots(3,3)
    ax[0,0].plot(time, F1)
    ax[0,0].set_title("constraint F1 to time")

    ax[0,1].plot(time, F2)
    ax[0,1].set_title("constraint F2 to time")

    ax[0,2].plot(time, F3)
    ax[0,2].set_title("constraint F3 to time")

    ax[1,0].plot(x1, y1)
    ax[1,0].set_title("trajectory x vertical axis to y horizontal axis")

    ax[1,1].plot(time, x1)
    ax[1,1].set_title("change of x over time")

    ax[1,2].plot(time, y1)
    ax[1,2].set_title("change of y over time")

    ax[2,0].plot(time, U1)
    ax[2,0].set_title("force Fx to time")
    
    ax[2,1].plot(time, U2)
    ax[2,1].set_title("force Fy to time")

    ax[2,2].plot(time, U3)
    ax[2,2].set_title("Moment M to time")

    """ax[2,0].plot(time, KE)
    ax[2,0].set_title("Kinetic energy")

    ax[2,1].plot(time, PE)
    ax[2,1].set_title("Potential energy")"""
    fig.tight_layout()
    plt.show()

def main(q, v):
    t = 0
    while t < T:
        fullStep = rk(q, v, t)
        F = ControlForce.constraint(fullStep[0], t)
        dF = ControlForce.dconstraint(fullStep[0], fullStep[1], t)
        forces = ControlForce.get_ControlForce(q, v, t)
        x1.append(fullStep[0][0])
        y1.append(fullStep[0][1])
        vx.append(fullStep[1][0])
        vy.append(fullStep[1][1])
        O1.append(fullStep[0][2])
        F1.append(F[0])
        F2.append(F[1])
        F3.append(F[2])
        U1.append(forces[0])
        U2.append(forces[1])
        U3.append(forces[2])
        #KE.append(500*()/2)
        time.append(t)
        q = fullStep[0]
        v = fullStep[1]
        t += dt

main(q, v)

plotter()
