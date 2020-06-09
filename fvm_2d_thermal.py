import numpy as np
import matplotlib.pyplot as plt

# input parameter =====================================================
#-- geometry
lx = 1.0
ly = 1.0
nx = 50
ny = 50

dx = lx / nx
dy = ly / ny 
ncell = nx * ny 

#-- solver
itermax = 1000
eps = 1.0e-5

#-- monitor
pmon = np.array([0.5, 0.5])

# condition ====================================================
#-- temperature
temp = np.full(ncell+1, 300.0)

#-- thermal conductivity
cond = np.full_like(temp, 100.0)
for i in range(1, nx+1):
    for j in range(1, ny+1):
        icell = i + (j - 1) * nx
        x = (i - 0.5) * dx
        y = (j - 0.5) * dy
        if y >= 0.8:
            cond[icell] = 10.0

#-- heat source
qc = np.zeros_like(temp)

#-- find monitor cell
imon = 0
flag = False
for i in range(1, nx+1):
    for j in range(1, ny+1):
        icell = i + (j - 1) * nx
        x = (i - 1) * dx
        y = (j - 1) * dy
        if pmon[0] >= x and pmon[0] <= (x+dx) and pmon[1] >= y and pmon[1] <= (y+dy):
            imon = icell
            flag = True
            break
    if flag:
        break          

# algebraic equations ==============================================  
def boundary(icell, iface, btype, tb=300.0, qb=0.0, ta=300.0, ht=10.0):
    condf = cond[icell]
    dcb = 0.5 * dcf[iface]

    if btype == 'dirichlet': # Dirichlet
        a_b = -condf * s[iface] / dcb
        a_f = 0.0
        a_c = -a_b
        b_c = -a_b * tb
    elif btype == 'neumann': # Neumann
        a_f = 0.0
        a_c = 0.0
        b_c = -qb * s[iface]
    elif btype == 'mixed': # Mixed                     
        htc = condf / dcb
        htf = ht * htc / (ht + htc) 
        a_b = -htf * s[iface] 
        a_f = 0.0
        a_c = -a_b
        b_c = -a_b * ta   

    return a_f, a_c, b_c  
    
s = [dy, dy, dx, dx]
dcf = [dx, dx, dy, dy] 
volume = dx * dy

ac = np.zeros_like(temp)
af = np.zeros((ncell+1, 4))
bc = np.zeros_like(temp)
cnbor = np.zeros_like(af, dtype='int')

for i in range(1, nx+1):
    for j in range(1, ny+1):
        icell = i + (j - 1) * nx
        x = (i - 0.5) * dx
        y = (j - 0.5) * dy

        ac[icell] = 0.0
        bc[icell] = qc[icell] * volume

        #-- face loop
        for iface in range(4):

            #-- face type
            if iface == 0 and i == nx: 
                ftype = 'right'
            elif iface == 1 and i == 1: 
                ftype = 'left'
            elif iface == 2 and j == ny: 
                ftype = 'top'
            elif iface == 3 and j == 1: 
                ftype = 'bottom'
            else:
                ftype = 'internal'
            
            #-- neighbor cell
            if ftype == 'internal':
                if iface == 0: inbor = icell + 1 # East
                elif iface == 1: inbor = icell - 1 # West
                elif iface == 2: inbor = icell + nx # North
                elif iface == 3: inbor = icell - nx # South
            else:
                inbor = 0    
            cnbor[icell][iface] = inbor                
            
            #-- matrix
            if ftype == 'internal': # internal face
                gf = 0.5
                condf = cond[icell] * cond[inbor] / ((1.0 - gf) * cond[icell] + gf * cond[inbor])
                a_f = -condf * s[iface] / dcf[iface]
                a_c = -a_f
                b_c = 0.0
            else: # boundary face
                if ftype == 'right' and y <= 0.2:
                    a_f, a_c, b_c = boundary(icell, iface, 'dirichlet', tb=500.0)
                elif ftype == 'top' and x <= 0.5: 
                    a_f, a_c, b_c = boundary(icell, iface, 'dirichlet', tb=300.0)  
                elif ftype == 'left':
                    a_f, a_c, b_c = boundary(icell, iface, 'mixed', ta=400.0, ht=100.0)
                else:
                    a_f, a_c, b_c = boundary(icell, iface, 'neumann', qb=0.0)

            ac[icell] += a_c
            af[icell][iface] = a_f
            bc[icell] += b_c   

# CG solver ===================================================      
iter_plt = []
resd_plt = []
temp_plt = []
d = np.zeros_like(temp)
r = np.zeros_like(temp)
ad = np.zeros_like(temp)

#-- initial residual
for icell in range(1, ncell+1):
    r[icell] = bc[icell] - ac[icell] * temp[icell]
    for iface in range(4):
        inbor = cnbor[icell][iface]
        r[icell] -= af[icell][iface] * temp[inbor]  
d = np.copy(r)
rr = np.dot(r, r)
bnrm = np.linalg.norm(bc)

#-- iteration
for iter in range(1, itermax+1):
    #-- A*d
    for icell in range(1, ncell+1):
        ad[icell] = ac[icell] * d[icell]
        for iface in range(4):
            inbor = cnbor[icell][iface]
            ad[icell] += af[icell][iface] * d[inbor]     

    #-- alpha                    
    alpha = np.dot(d, r) / np.dot(d, ad)

    #-- temperature
    temp += alpha * d

    #-- residual
    r -= alpha * ad    
    resd = np.linalg.norm(r) / bnrm

    #-- plot data
    iter_plt.append(iter)
    resd_plt.append(resd)
    temp_plt.append(temp[imon])
    print('iter {:6d} , resd {:12.4e} , temp {:12.4e}'.format(iter, resd, temp[imon]))

    #-- convergence check
    if resd <= eps:
        break  

    #-- beta
    beta = np.dot(r, r) / rr
    rr = np.dot(r, r)

    #-- d
    d = r + beta * d

# post processing ====================================================
#-- residual and monitor graph
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.plot(iter_plt, resd_plt)
ax1.set(ylabel='Residual')
ax1.set_yscale('log')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.plot(iter_plt, temp_plt)
ax2.set(xlabel='Iteration', ylabel='Monitor')      

#-- contour plot
fig2 = plt.figure()
ax3 = fig2.add_subplot(1, 1, 1)

gx = np.arange(0.0, lx+0.5*dx, dx)
gy = np.arange(0.0, ly+0.5*dy, dy)
X, Y = np.meshgrid(gx, gy)

val = np.zeros((ny, nx))
for i in range(1, nx+1):
    for j in range(1, ny+1):
        icell = i + (j - 1) * nx 
        val[j-1][i-1] = temp[icell]

im = ax3.pcolormesh(X, Y, val, cmap='jet')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_aspect('equal')
fig2.colorbar(im)
plt.show()