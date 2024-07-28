#
# Notice: This module contains slightly adapted version of optimization functionalities
#         from Frankestein code written by Hong-Zhou Ye.
#

import numpy,sys, h5py


def line_search_LF(func, xold, fold, dx, iter_):
    """Adapted from D.-H. Li and M. Fukushima, Optimization Metheods and Software, 13, 181 (2000)
    """
    beta = 0.1
    rho = 0.9
    sigma1 = 1E-3
    sigma2 = 1E-3
    eta = (iter_+1)**-2.

    xk = xold + dx
    lcout = 0
    
    fk = func(xk)        
    lcout += 1
    
    norm_dx = numpy.linalg.norm(dx)
    norm_fk = numpy.linalg.norm(fk)
    norm_fold = numpy.linalg.norm(fold)
    alp = 1.
    
    if norm_fk > rho*norm_fold - sigma2*norm_dx**2.:
        while norm_fk > (1.+eta)*norm_fold - sigma1*alp**2.*norm_dx**2.:
            
            alp *= beta
            xk = xold + alp * dx
            
            
            fk = func(xk)
                       
            lcout += 1
            norm_fk = numpy.linalg.norm(fk)
            if lcout == 20:
                break

    print(' No. of line search steps in QN opt :', lcout, flush=True)
    print(flush=True)
    return alp, xk, fk

def opt_QN(self):

    J0 = self.get_be_error_jacobian(solver)


class FrankQN:
    """ Quasi Newton Optimization

    Performs quasi newton optimization. Interfaces many functionalities of the
    frankestein code originaly written by Hong-Zhou Ye
    

    """
    
    def __init__(self, func, x0, f0, J0, trust=0.3, max_space=500):
              
        self.x0 = x0
        self.n = x0.size        
        self.f0 = f0
        self.func = func
        
        self.B0 = numpy.linalg.pinv(J0)

        self.iter_ = 0

        self.tol_gmres = 1.e-6
        self.xnew = None           # new errvec
        self.xold = None           # old errvec
        self.fnew = None           # new jacobian?
        self.fold = None           # old jacobian?
        self.max_subspace = max_space   
        self.dxs = numpy.empty([self.max_subspace,self.n])
        self.fs =  numpy.empty([self.max_subspace,self.n])
        self.us =  numpy.empty([self.max_subspace,self.n])  # u_m = B_m @ f_m
        self.vs =  numpy.empty([self.max_subspace,self.n])  # v_m = B_0 @ f_{m+1}
        self.B = None
        self.trust = trust
        
    def next_step(self):

        if self.iter_ == 0:            
            self.xnew = self.x0            
            self.fnew = self.func(self.xnew) if self.f0 is None else self.f0   
            self.fs[0] = self.fnew.copy()
            self.us[0] = numpy.dot(self.B0, self.fnew)
            self.Binv = self.B0.copy()

        # Book keeping
        if not self.iter_ == 0:
            dx_i = self.xnew - self.xold
            df_i = self.fnew - self.fold
        
        self.xold = self.xnew.copy()
        self.fold = self.fnew.copy()
                        
        if not self.iter_ ==0:
                        
            tmp__ = numpy.outer(dx_i - self.Binv@df_i, dx_i@self.Binv)/\
                (dx_i@self.Binv@df_i)
            self.Binv += tmp__            
            us_tmp = self.Binv@self.fnew
                
        self.us[self.iter_] = self.get_Bnfn(self.iter_)
                
        alp, self.xnew, self.fnew = line_search_LF(
            self.func, self.xold, self.fold, -self.us[self.iter_], self.iter_)
        
        # udpate vs, dxs, and fs        
        self.vs[self.iter_] = numpy.dot(self.B0, self.fnew)
        self.dxs[self.iter_] = self.xnew - self.xold
        self.fs[self.iter_+1] = self.fnew.copy()

        self.iter_ += 1

    def get_Bnfn(self, n):

        # self.us; self.dxs; self.vs
        if n == 0: return self.us[0]

        vs = [None] * n
        for i in range(n): vs[i] = self.vs[n-i-1]
        for i in range(1,n+1):
            un_ = self.us[i-1]
            dxn_ = self.dxs[i-1]
            vps = [None] * (n-i+1)
            for j in range(n-i+1):
                a = vs[j]
                b = vs[n-i] - un_
                
                vps[j] = a + (dxn_@a)/(dxn_@b) * (dxn_ - b)
                
            vs = vps
           
        return vs[0]

def get_be_error_jacobian(self,jac_solver='HF'):

    Jes = [None] *    self.Nfrag
    Jcs = [None] *    self.Nfrag
    xes = [None] *    self.Nfrag
    xcs = [None] *    self.Nfrag
    ys = [None] *     self.Nfrag
    alphas = [None] * self.Nfrag

    if jac_solver =='MP2':
        res_func = mp2res_func
    elif jac_solver =='CCSD':
        res_func = ccsdres_func
    elif jac_solver=='HF':
        res_func = hfres_func
        
    Ncout = [None] * self.Nfrag
    for A in range(self.Nfrag):

        Jes[A], Jcs[A], xes[A], xcs[A], ys[A], alphas[A], Ncout[A] = \
            get_atbe_Jblock_frag(self.Fobjs[A], res_func)
                
    alpha = sum(alphas)
    
    # build Jacobian
    ''' ignore!
    F0-M1 F1-M2M2 F2-M3M3 F3-M4
       M1   M2  M2 M3  M3 M4
    M1 E0   C1-1 
    M2 C0-0 E1  E1
    M2      E1  E1 C2-2 
    M3      C1-1   E2  E2
    M3             E2  E2 C3-3
    M4             C2-2   E3
    '''
    N_ = sum(Ncout)
    J = numpy.zeros((N_+1, N_+1))
    cout = 0
    
    for findx, fobj in enumerate(self.Fobjs):
        J[cout:Ncout[findx]+cout, cout:Ncout[findx]+cout] = Jes[findx]
        J[cout:Ncout[findx]+cout, N_:] = numpy.array(xes[findx]).reshape(-1,1)
        J[N_:, cout:Ncout[findx]+cout] = ys[findx]

        coutc = 0
        coutc_ = 0
        for cindx, cens in enumerate(fobj.center_idx):
            coutc +=  Jcs[fobj.center[cindx]].shape[0]
            start_ = sum(Ncout[:fobj.center[cindx]])
            end_ = start_ + Ncout[fobj.center[cindx]]
            J[cout+coutc_: cout+coutc, start_:end_] += Jcs[fobj.center[cindx]]
            J[cout+coutc_: cout+coutc, N_:] += numpy.array(
                xcs[fobj.center[cindx]]).reshape(-1,1)
            coutc_ = coutc
        cout += Ncout[findx]        
    J[N_:,N_:] = alpha
    
    return J

def get_atbe_Jblock_frag(fobj, res_func):
    from molbe.helper import get_scfObj, get_eri

    vpots = get_vpots_frag(fobj.nao, fobj.edge_idx,
                           fobj.fsites)
    eri_ = get_eri(fobj.dname, fobj.nao, eri_file=fobj.eri_file)           
    dm0 = numpy.dot( fobj._mo_coeffs[:,:fobj.nsocc],
                     fobj._mo_coeffs[:,:fobj.nsocc].T) *2.  
    mf_ = get_scfObj(fobj.fock+fobj.heff, eri_, fobj.nsocc, dm0=dm0)
    
    dPs, dP_mu = res_func(mf_, vpots, eri_, fobj.nsocc) 
    
    Je = []
    Jc = []
    y=[]
    xe = []
    xc = []
    cout = 0

    for edge in fobj.edge_idx:
        
        for j_ in range(len(edge)):
            for k_ in range(len(edge)):
                if j_>k_:
                    continue
                ## response w.r.t matching pot
                # edges 
                tmpje_ = []
                
                for edge_ in fobj.edge_idx:
                    lene = len(edge_)
                
                    for j__ in range(lene):
                        for k__ in range(lene):
                            if j__>k__:
                                continue
                            
                            tmpje_.append(dPs[cout][edge_[j__],edge_[k__]])                            
                y_ = 0.
                for fidx, fval in enumerate(fobj.fsites):
                    
                    if not any(fidx in sublist
                               for sublist in fobj.edge_idx):
                        y_ += dPs[cout][fidx, fidx]
                    
                y.append(y_)
                
                tmpjc_ = []                
                # center on the same fragment
                #for cen in fobj.efac[1]:                    
                for j__ in fobj.centerf_idx:
                    for k__ in fobj.centerf_idx:
                        if j__>k__:
                            continue                            
                        tmpjc_.append(-dPs[cout][j__,k__])
                
                Je.append(tmpje_)
                
                Jc.append(tmpjc_)
                            
                ## response w.r.t. chem pot
                # edge
                xe.append(dP_mu[edge[j_],edge[k_]])                                            
                cout += 1    
    Je = numpy.array(Je).T
    Jc = numpy.array(Jc).T
    
    alpha = 0.
    for fidx, fval in enumerate(fobj.fsites):        
        if not any(fidx in sublist
                   for sublist in fobj.edge_idx):
            alpha += dP_mu[fidx, fidx]        
    
    for j__ in fobj.centerf_idx:
        for k__ in fobj.centerf_idx:
            if j__>k__:
                continue
            xc.append(-dP_mu[j__,k__])    
    return Je, Jc, xe, xc, y, alpha, cout


def get_be_error_jacobian_selffrag(self,jac_solver='HF'):

    Jes = [None] *    self.Nfrag
    Jcs = [None] *    self.Nfrag
    xes = [None] *    self.Nfrag
    xcs = [None] *    self.Nfrag
    ys = [None] *     self.Nfrag
    alphas = [None] * self.Nfrag

    if jac_solver =='MP2':
        res_func = mp2res_func
    elif jac_solver =='CCSD':
        res_func = ccsdres_func
    elif jac_solver=='HF':
        res_func = hfres_func
        
    Jes, Jcs, xes, xcs, ys, alphas, Ncout = \
        get_atbe_Jblock_frag(self.Fobjs[0], res_func)
    
    N_ = Ncout    
    J = numpy.zeros((N_+1, N_+1))
        
    J[:Ncout, :Ncout] = Jes
    J[:Ncout, N_:] = numpy.array(xes).reshape(-1,1)
    J[N_:, :Ncout] = ys
    J[:Ncout, N_:] += numpy.array([*xcs, *xcs]).reshape(-1,1)
    J[N_:,N_:] = alphas    
    
    return J


def hfres_func(mf, vpots, eri, nsocc):
    from molbe.external.cphf_utils import cphf_kernel_batch,get_rhf_dP_from_u
    C = mf.mo_coeff
    moe = mf.mo_energy
    eri = mf._eri
    no = nsocc

    us = cphf_kernel_batch(C, moe, eri, no, vpots)
    dPs = [get_rhf_dP_from_u(C, no, us[I]) for I in range(len(vpots)-1)]
    dP_mu = get_rhf_dP_from_u(C, no, us[-1])
    
    return dPs, dP_mu

def mp2res_func(mf, vpots, eri, nsocc):
    from molbe.external.cpmp2_utils import get_dPmp2_batch_r
         
    C = mf.mo_coeff
    moe = mf.mo_energy
    eri = mf._eri
    no = nsocc
    
    dPs_an = get_dPmp2_batch_r(C, moe, eri, no, vpots, aorep=True)
    dPs_an = numpy.array([dp_*0.5 for dp_ in dPs_an])
    dP_mu = dPs_an[-1]
    
    return dPs_an[:-1], dP_mu
    

def ccsdres_func(mf, vpots, eri, nsocc):
    from molbe.external.jac_utils import get_dPccsdurlx_batch_u

    C = mf.mo_coeff
    moe = mf.mo_energy
    eri = mf._eri
    no = nsocc

    dPs_an = get_dPccsdurlx_batch_u(C, moe, eri, no, vpots)

    dP_mu = dPs_an[-1]

    return dPs_an[:-1], dP_mu


def get_vpots_frag(nao, edge_idx, fsites):
    vpots = []
        
    for edge_ in edge_idx:
        lene = len(edge_)
        for j__ in range(lene):
            for k__ in range(lene):
                if j__>k__:
                    continue

                tmppot = numpy.zeros((nao,nao))
                tmppot[edge_[j__],edge_[k__]] = tmppot[edge_[k__],edge_[j__]] = 1
                vpots.append(tmppot)

    # only the centers
    # outer edges not included
    tmppot = numpy.zeros((nao,nao))
    for fidx, fval in enumerate(fsites):
        
        if not any(fidx in sublist
                   for sublist in edge_idx):
            tmppot[fidx, fidx] = -1
        

    vpots.append(tmppot)
    return vpots
              
