import numpy as np

# input parameter =====================================================
#-- geometry
lx = 1.0
ly = 1.0
nx = 50
ny = 50

dx = lx / nx
dy = ly / ny 
ncell = nx * ny 

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

        print('cell {:d} neighbor {:d} {:d} {:d} {:d}'.format(icell, cnbor[icell][0], cnbor[icell][1], cnbor[icell][2], cnbor[icell][3]))
        print('ac {:6.3f} af {:6.3f} {:6.3f} {:6.3f} {:6.3f}'.format(ac[icell], af[icell][0], af[icell][1], af[icell][2], af[icell][3]))
        print('bc {:6.3f}'.format(bc[icell]))
        print(' ')

