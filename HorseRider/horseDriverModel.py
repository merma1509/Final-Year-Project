import numpy as np
import matplotlib.pyplot as plt
from Graphing.graphing import Graphing as gr

class Control_Force:
    def __init__(self, v0, H1, A1, w1, m1, J1, H2, A2, L2, w2, m2, J2, pi, ci2, E, N, g):
        self.v0 = v0
        self.H1 = H1
        self.A1 = A1
        self.H2 = H2
        self.A2 = A2
        self.m1 = m1
        self.J1 = J1
        self.m2 = m2
        self.J2 = J2
        self.L2 = L2
        self.w1 = w1
        self.w2 = w2
        self.pi = pi
        self.ci2 = ci2
        self.E = E
        self.N = N
        self.g = g
    
    # this function contains all of our constraints  
    def constraint(self, q, t):
        f1 = q[0] - self.v0*t
        f2 = q[1] - (self.H1 + self.A1*np.sin(q[4]))
        f3 = (q[0]-q[2]) + self.E*np.cos(q[4]) + self.N*np.sin(q[4]) - (self.L2 + self.A2*np.sin(q[5]))
        f4 = (q[1]-q[3]) + self.E*np.sin(q[4]) - self.N*np.cos(q[4]) - (self.H2 + self.A2*np.cos(q[5]))
        f5 = q[4] - np.cos(self.w1*t)
        f6 = q[5] - np.cos(self.w2*t)
        return np.array([f1, f2, f3, f4, f5, f6])
    
    def P1_P2(self, q, v):
        Px = self.E*np.cos(q[4]) + self.N*np.sin(q[4])
        Py = self.E*np.sin(q[4]) - self.N*np.cos(q[4])
        P2x = q[0] - q[2] + Px
        P2y = q[1] - q[3] + Py
        return np.array([Px, Py, P2x, P2y])
    
    def Matrices(self, q, v, t):
        P1x, P1y, P2x, P2y = self.P1_P2(q,v)
        B_U = np.array([[(1/self.m1), 0, (1/self.m1), 0, 0, 0],
                        [0, (1/self.m1), 0, (1/self.m1), 0, 0],
                        [0, 0, (1/self.m2), 0, 0, 0],
                        [0, 0, 0, (1/self.m2), 0, 0],
                        [0, 0, (P1y/self.J1), (-P1x/self.J1), (1/self.J1), 0],
                        [0, 0, (-P2y/self.J2), (P2x/self.J2), 0, (1/self.J2)]])

        S_Qk = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, (-self.A1*np.cos(q[4])), 0],
                        [1, 0, -1, 0, (-self.E*np.sin(q[4]) - self.N*np.cos(q[4])), (-self.A2*np.cos(q[5]))],
                        [0, 1, 0, -1, (self.E*np.cos(q[4]) - self.N*np.sin(q[4])), (self.A2*np.sin(q[5]))],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])

        S2_Qk2 = np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, (self.A1*np.sin(q[4])), 0],
                           [0, 0, 0, 0, (-self.N*np.sin(q[4]) + self.E*np.cos(q[4])), (self.A2*np.sin(q[5]))],
                           [0, 0, 0, 0, (-self.E*np.sin(q[4]) - self.N*np.cos(q[4])),(self.A2*np.cos(q[5]))],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

        S2_t2 = np.array([0, 0, 0, 0, ((self.w1**2)*np.cos(self.w1*t)), ((self.w2**2)*np.cos(self.w2*t))])
        S_t = np.array([-self.v0, 0, 0, 0, (self.w1*np.sin(self.w1*t)), (self.w2*np.sin(self.w2*t))])
        Gk = np.array([0, -self.g, 0, -self.g, 0, 0])
        return [B_U, S_Qk, S2_Qk2, S_t, S2_t2, Gk]        

    # this function is the first derivative of all our contraints
    def dconstraint(self, q, v, t, Matr): 
        df = np.dot(Matr[1],v) + Matr[3]
        return df
    
    def Forces(self, q, v, t, Matr):
        w = np.dot(Matr[2], v**2) + Matr[4]
        Y = -(2*self.pi*self.dconstraint(q, v, t, Matr) + self.ci2*self.constraint(q, t) + w + np.dot(Matr[1],Matr[-1]))
        a = np.array(np.dot(Matr[1],Matr[0]))
        aInv = np.linalg.inv(a)
        U = np.dot(aInv, Y)
        return U
    
Data = np.array([500, 50, 10, 12]) # m1, m2, j1, j2
q = np.array([0, 0, (0.1-0.1), (0.3-0.5), 0, 0]) # initial coordinates x1, y1, x2, y2, w1, w2
v = np.array([0, 0, 0, 0, 0, 0]) # initial velocity vx1, vy1, vx2, vy2, w1, w2 
ControlForce = Control_Force(5, 0, 0.06, 7.66, Data[0], Data[2], 0.5, 0.12, 0.1, 3.8, Data[1], Data[3], 0.2, 0.03, 0.1, 0.3, 10)
T = 500 # time for motion
dt = 0.1 # steps
x1, y1, x2, y2, time = [], [], [], [], [] 
Constr, ControlForces = [[],[],[],[],[],[]], [[],[],[],[],[],[]]
def dv(q, v, t):
    Matr = ControlForce.Matrices(q, v, t)
    U = ControlForce.Forces(q, v, t, Matr)
    Q = Matr[-1]
    B = Matr[0]
    dv = Q + np.dot(B, U)
    #print("B= ",B,"\ndv= ",dv, "\nU= ",U, "\n")
    return dv

def dq(q,v,t):
    return v
    
def rk(t, q, v, dt):
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
    
    return np.array([q, v])

def main(q, v):
    t = 0
    while t < T:
        fullStep = rk(t, q, v, dt/2)
        HalfStep = rk(t+(dt/2), q, v, dt/2)

        error = np.linalg.norm((np.array([fullStep - HalfStep])))
        #print(error)
        #x11 = np.power(np.e, (-0.2*t))*((-1/0.01)*np.power(np.e, (np.sqrt(0.01)*t)) + (1/0.03)*np.power(np.e, -(np.sqrt(0.01)*t))) + (2/0.03) + (0.15*(t**2))/(0.4+(0.03*t))
        #fullStep[0][0] = x11

        F = ControlForce.constraint(fullStep[0], t)
        x1.append(fullStep[0][0])
        y1.append(fullStep[0][1])
        x2.append(fullStep[0][2])
        y2.append(fullStep[0][3])
        Constr[0].append(F[0])
        Constr[1].append(F[1])
        Constr[2].append(F[2])
        Constr[3].append(F[3])
        Constr[4].append(F[4])
        Constr[5].append(F[5])
        time.append(t)

        Matr = ControlForce.Matrices(q,v,t)
        U = ControlForce.Forces(q, v, t, Matr)
        ControlForces[0].append(U[0])
        ControlForces[1].append(U[1])
        ControlForces[2].append(U[2])
        ControlForces[3].append(U[3])
        ControlForces[4].append(U[4])
        ControlForces[5].append(U[5])
        q = fullStep[0]
        v = fullStep[1]
        t += dt/2

main(q, v)

graph = gr()
graph.NewGraph("Constraint Fx1", time, Constr[0])
graph.NewGraph("Constraint Fy1", time, Constr[1])
graph.NewGraph("Constraint Fx2", time, Constr[2])
graph.NewGraph("Constraint Fy2", time, Constr[3])
graph.NewGraph("Constraint FO1", time, Constr[4])
graph.NewGraph("Constraint FO2", time, Constr[5])
graph.NewGraph("Horse trajectory", x1, y1)
graph.NewGraph("driver trajectory", x2, y2)
graph.NewGraph("Control Force Ux1", time, ControlForces[0])
graph.NewGraph("Control Force Uy1", time, ControlForces[1])
graph.NewGraph("Control Force Ux2", time, ControlForces[2])
graph.NewGraph("Control Force Uy2", time, ControlForces[3])
graph.NewGraph("Control Force UO1", time, ControlForces[4])
graph.NewGraph("Control Force UO2", time, ControlForces[5])

NameList = ["Horse trajectory", "driver trajectory"]
graph.graphing(NameList, 2)

NameList = ["Constraint Fx1", "Constraint Fy1", "Constraint Fx2", "Constraint Fy2", "Constraint FO1", "Constraint FO2"]
graph.graphing(NameList, 6)

NameList = ["Control Force Ux1", "Control Force Uy1", "Control Force Ux2", "Control Force Uy2", "Control Force UO1", "Control Force UO2"]
graph.graphing(NameList, 6)

