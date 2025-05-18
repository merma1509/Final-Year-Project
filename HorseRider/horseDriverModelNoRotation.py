import numpy as np
import matplotlib.pyplot as plt

class Control_Force:
    def __init__(self, v0, H1, A1, m1, w1, j1, H2, A2, L2, m2, w2, j2, E, N, pi, ci2, g):
        self.v0 = v0
        self.H1 = H1
        self.A1 = A1
        self.m1 = m1
        self.j1 = j1
        self.H2 = H2
        self.A2 = A2
        self.L2 = L2
        self.m2 = m2
        self.j2 = j2
        self.w1 = w1
        self.w2 = w2
        self.E = E
        self.N = N
        self.pi = pi
        self.ci2 = ci2
        self.g = g    
    # this function contains all of our constraints  
    def constraint(self, q, t):
        f1 = q[0] - self.v0*t
        f2 = q[1]
        f3 = q[0] - q[2] + self.E - self.L2
        f4 = q[1] - q[3] + self.N - self.H2
        f5 = q[4] 
        f6 = q[5]
        return np.array([f1, f2, f3, f4, f5, f6])
    
    # this function is the first derivative of all our contraints
    def dconstraint(self, q, v, t): 
        Df1 = v[0] - self.v0
        return np.array([Df1, v[1], v[2], v[3], v[4], v[5]])
    
    def get_ControlForce(self, q, v, t):
        U3 = (-1)*self.m2 * self.ci2 * (q[0] - q[2] + self.E - self.L2)
        U4 = self.m2 * (self.g + self.ci2 * (q[1] - q[3] + self.N - self.H2))
        U1 = U3 - self.m1 * (2 * self.pi * (v[0] - self.v0) + (self.ci2) * (q[0] - self.v0 * t))
        U2 = U4 - self.m1 * self.g
        U5 = (self.E * U4 - self.N * U3)
        U6 = (q[1] - q[3] + self.N - self.H2) * U3 - (q[0] - q[2] + self.E - self.L2) * U4
        return np.array([U1, U2, U3, U4, U5, U6])
    
    def dv(self, q, v, t):
        U = self.get_ControlForce(q, v, t)
        dv1 = (1/self.m1)*(U[0] - U[2])
        dv2 = (1/self.m1)*(U[1] - U[3]) - self.g
        dv3 = (1/self.m2)*(U[2])
        dv4 = (1/self.m2)*(U[3]) - self.g
        dv5 = (1/self.j1)*(self.E*U[3] - self.N*U[2] + U[4])
        dv6 = (1/self.j2)*((q[1] - q[3] + self.N - self.H2)*U[2] - (q[0] - q[2] + self.E - self.L2)*U[3] + U[5])
        return np.array([dv1, dv2, dv3, dv4, dv5, dv6])
        
    def dq(self, q, v, t):
        return v
    
    
Data = np.array([10, 500, 50, 10, 12]) # g, m1, m2, j1, j2
q = np.array([0, 0, (0.1-0.1), (0.3-0.5), 0, 0]) # initial coordinates x1, y1, x2, y2, w1, w2
v = np.array([0, 0, 0, 0, 0, 0]) # initial velocity vx1, vy1, vx2, vy2, w1, w2 
ControlForce = Control_Force(5, 0, 0.06, 500, 0, 10, 2, 0.1, 0, 50, 0, 12, 0.1, 0.3, 0.2, 0.03, 10)
dv = ControlForce.dv
dq = ControlForce.dq
U = ControlForce.get_ControlForce
T = 10*50 # time for motion
dt = 10/T # steps

print(dv(q, v, 0), dq(q, v, 0), U(q,v,0))

x1, y1, O1, x2, y2, O2, time, dConstr = [], [], [], [], [], [], [], []
F1, F2, F3, F4, F5, F6 = [],[],[],[],[],[]
    
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

def main(q, v):
    t = 0
    while t < T:
        fullStep = rk(q, v, t)
        F = ControlForce.constraint(fullStep[0], t)
        dF = ControlForce.dconstraint(fullStep[0], fullStep[1], t)
        x1.append(fullStep[0][0])
        y1.append(fullStep[0][1])
        x2.append(fullStep[0][2])
        y2.append(fullStep[0][3])
        O1.append(fullStep[0][4])
        O2.append(fullStep[0][5])
        F1.append(F[0])
        F2.append(F[1])
        F3.append(F[2])
        F4.append(F[3])
        F5.append(F[4])
        F6.append(F[5])
        dConstr.append(dF)
        time.append(t)
        #print(fullStep[0][0],"\n")
        q = fullStep[0]
        v = fullStep[1]
        t += dt
        #print(t)

main(q, v)

#plt.plot(time, O2)
plt.plot(time, F1)
plt.show()
