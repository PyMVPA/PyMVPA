import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
from scipy.stats import chi2
from mvpa2.suite import Mapper, Dataset
from scipy.spatial.distance import pdist
from operator import itemgetter
import matplotlib as mpl
import sys

class StatisMapper(Mapper):
    """Implementation of STATIS.
    Compromise matrices are the optimal linear combination of
    cross-product matrices, the general eigen decomposition of
    which yield same results as left generalized singular vectors
    """
    def __init__(self,tables_attr='tables', bootstrap_iter=0,
                col_center=False, col_norm=None,
                row_center=False, row_norm=None, table_norm=None,
                stack = 'h', **kwargs):

        self._col_center = col_center
        self._col_norm = col_norm
        self._row_center = row_center
        self._row_norm = row_norm
        self._chunks_attr = tables_attr
        self._table_norm = table_norm
        self._bootstrap_iter = bootstrap_iter
        self._stack = stack

        Mapper.__init__(self, **kwargs)

    def _train(self, dataset):


        self.dataset = dataset
        self.samples = samples = dataset.samples
        if self._stack == 'v':
            self.chunks = chunks = np.array(dataset.sa[self._chunks_attr])
        if self._stack == 'h':
            self.chunks = chunks = np.array(dataset.fa[self._chunks_attr])

        self.ntables = ntables = len(np.unique(chunks))
        i,J = samples.shape

        self._subtable_stats = {}


        X = None # tables matrix with subtables h-stacked
        self.subtable_idx = np.array([])
        for ch, chunk in enumerate(np.unique(chunks)):
            # compute cross-product for each table, Eq. 7
            if self._stack == 'v':
                ntargets = i/ntables
                t = samples[chunks==chunk,:]
                if dataset.sa.has_key('targets'):
                    self.targets = dataset.sa['targets'][chunks==chunk]
                else:
                    self.targets = range(ntargets)

            if self._stack == 'h':
                ntargets = i
                t = samples[:,chunks==chunk]
                if dataset.sa.has_key('targets'):
                    self.targets = dataset.sa['targets']
                else:
                    self.targets = range(ntargets)

            t,c_m,c_n,r_m,r_n,t_n = self.center_and_norm_table(t)
            self._subtable_stats[ch] = {'row_mean':r_m, 'row_norm':r_n,
                        'col_mean':c_m, 'col_norm':c_n, 'table_norm':t_n}
            self.subtable_idx=np.hstack((self.subtable_idx,ch*np.ones((t.shape[1]))))


            if X is None:
                X = t
            else:
                X = np.hstack((X,t))

        self.I = I = ntargets
        self.M = M = (1.0/I)*np.eye(I) # masses, eq. 2
        self.X = X
        (A, alpha,C,G) = inter_table_Rv_analysis(X,self.subtable_idx)
        self.A = A
        self.alpha = alpha
        self.Rv = C
        self.inter_table_structure = G

        self.compromise = (X.T*np.sqrt(np.diag(M))).T*np.sqrt(A)

        Pt,delta,Qt = np.linalg.svd(self.compromise, full_matrices=0)     #Eq.42
        
        self.P = np.sqrt(I)*Pt # duh. this replacs line above
        self.Qt = Qt
        self.Pt = Pt
        self.Q = (Qt*np.sqrt(A)).T
        self.F = self.P*delta

        self.eigv=delta**2
        self.inertia=delta**2/np.sum(delta**2)

    def _forward_dataset(self, ds):
        targ = np.copy(ds.targets)
        mapped = None
        X = None
        i,j = ds.shape
        ntables = len(np.unique(self.chunks))

        for ch,chunk in enumerate(np.unique(self.chunks)):
            if self._stack == 'v':
                table = ds.samples[ds.sa[self._chunks_attr].value==chunk,:]
                nrows = i/ntables
                targ = ds.targets[:nrows]
            if self._stack == 'h':
                table = ds.samples[:,ds.fa[self._chunks_attr].value==chunk]
                nrows = i
            table = self.center_and_norm_table(table,
                    col_mean=self._subtable_stats[ch]['col_mean'],
                    col_norm=self._subtable_stats[ch]['col_norm'],
                    table_norm = self._subtable_stats[ch]['table_norm'])[0]


            m = 1.0/self.I
            

            part = Dataset(np.dot(table,self.Q[self.subtable_idx==ch,:]))
            part.sa['chunks'] = np.ones((len(part.samples,)))*ch
            part.sa['targets'] = targ
            if mapped is None:
                mapped = part
                X = table
            else:
                mapped.append(part)
                X = np.hstack((X,table))
        mapped.a['X'] = X

        return mapped

    #def _reverse_data(self, data):
        


    def center_and_norm_table(self,table,col_mean=None, col_norm=None,
                    row_mean=None, row_norm=None, table_norm=None):
        """
        Using the norming parameters set in the constructor preprocess each
        subtable.

        Parameters
        ----------
        table       : any subtable
        col_mean    : An optional row vector of column means
        col_norm    : An optional row vector of row norms
        row_mean    : an optional column vector of row means
        row_norm    : an optional column vector of row norms
        table_norm  : optional value to divide entire table by for normalization
        """

        t = table
        # first columns
        if self._col_center:
            if col_mean is not None:
                pass
            else:
                col_mean = np.mean(t, 0)
            t = t - col_mean
        if self._col_norm:
            if col_norm is not None:
                pass
            elif self._col_norm=='l2':
                col_norm = np.apply_along_axis(np.linalg.norm,0,t)
            elif self._col_norm=='std':
                col_norm = np.apply_along_axis(np.std,0,t)
            t = t / col_norm
        # then rows
        if self._row_center:
            if row_mean is not None:
                pass
            else:
                row_mean = np.mean(t.T, 0)
            t = (t.T - row_mean).T
        if self._row_norm:
            if row_norm is not None:
                pass
            elif self._row_norm=='l2':
                row_norm = np.apply_along_axis(np.linalg.norm,0,t.T)
            elif self._row_norm=='std':
                row_norm = np.apply_along_axis(np.std,0,t.T)
            t = (t.T / row_norm).T

        # whole table norm
        if self._table_norm:
            if table_norm is not None:
                pass
            elif self._table_norm == 'l2':
                table_norm = np.linalg.norm(t)
            elif self._table_norm == 'std':
                table_norm = np.std(t)
            t = t / table_norm

        return t, col_mean, col_norm, row_mean, row_norm, table_norm


def run_bootstrap(ds, sts, niter=1000):
    """
       dataset: Input fmri dataset with targets and chunks, should be preprocessed
       Q:       The loading matrix from statis
       niter:   number of iteration for bootstrap
       This function iteratively samples random chunks from dataset with
       replacement and computes each statis solution.
        The factor scores from each statis are then projected into original compromise
       matrix space using Q
       OUTPUT: FBoot collects all projected factor scores
    """
    ntables = sts.ntables
    rows, nfactors = ds.shape
    nrows = rows/ntables
    boot = np.zeros((nrows,nfactors,niter))
    X = ds.a['X'].value

    for i in range(niter):
        idx = np.floor(ntables*np.random.random_sample(ntables))
        Y = None
        Y_idx = None
        fselect = np.zeros((nrows,nfactors,ntables))

        for k,j in enumerate(idx):

            Y_t = X[:,sts.subtable_idx==j]
            if Y_idx is None:
                Y_idx = np.ones((Y_t.shape[1]))*k
            else:
                Y_idx = np.hstack((Y_idx,np.ones((Y_t.shape[1]))*k))

            if Y == None:
                Y = Y_t
            else:
                Y = np.hstack((Y,Y_t))

            fselect[:,:,k] = ds.samples[ds.chunks==j,:]

        (A,alpha,C,G) = inter_table_Rv_analysis(Y,Y_idx)
        boot[:,:,i] = np.sum(fselect*alpha.flatten(),2)

        if i%100==0:
            sys.stdout.write("iter:%s/%s\r"% (i,niter))
            sys.stdout.flush()

    boot = Dataset(boot)
    boot.sa['targets'] = ds.targets[:nrows]
    return boot

def inter_table_Rv_analysis(X_, subtable_idx):

    Z = None
    ntables = len(np.unique(subtable_idx))
    for idx in np.unique(subtable_idx):
        t = X_[:,subtable_idx==idx]
        S_t = np.dot(t,t.T).flatten()
        S_t = S_t / np.linalg.norm(S_t)
        # K by I^2 matrix Z, eq.10
        if Z is None:
            Z = S_t
        else:
            Z = np.vstack((Z,S_t))

    C = np.dot(Z,Z.T) # Inner-product matrix C, eq.11
    w,v = eigh(C) # Sorted eigen decomposition of C,
    eins = np.ones((ntables,1))
    u = v[:,0].reshape(ntables,1)
    alpha = np.dot(u,np.dot(u.T,eins).__pow__(-1)) # eq.15
    a = None
    for k,val in enumerate(alpha):

        alph = val*np.ones(sum(subtable_idx==np.unique(subtable_idx)[k]))
        if a is None:
            a = alph

        else:
            a = np.hstack((a,alph))

    G = v.dot(np.diag(np.sqrt(w))) # eq. 13
    #G = G/np.linalg.norm(G[:,0])
    return a,alpha,C,G



def plot_inter_table_map(sts, prefix='T_', labels=None, axes='off',hw=.03,lw=2):
    ax = plt.figure();

    G = sts.inter_table_structure[:,[0,1]]
    G[:,0] = np.abs(G[:,0])
    xmx = np.max(G[:,0])
    w = xmx*.005
    hw = xmx*.03
    ymx = np.max(abs(G[:,1]))
    for i,idx in enumerate(np.unique(sts.chunks)):
        s = prefix + str(idx)
        plt.text(G[i,0],G[i,1],s)
    #mx = np.max(np.abs(G[:,1]))
    plt.arrow(0,0,xmx*1.1,0,color='gray',alpha=.7,width=w,
                head_width=hw,length_includes_head=True)
    plt.arrow(0,-ymx*1.1,0,2*(ymx*1.1),color='gray',alpha=.7,width=w,
            head_width=hw,length_includes_head=True)
    plt.axis((-.1*xmx,xmx*1.1,-ymx*1.1,ymx*1.1))
    plt.axis(axes)
    plt.text(-.1*xmx,ymx,'$2$', fontsize=20)
    plt.text(xmx,-.2*ymx,'$1$',fontsize=20)
    

def plot_partial_factors(ds,sts,x=0,y=1,cmap=None,axes='off',
                        nude=False):

    mx = np.max(np.abs(ds.samples))
    xmx = mx*1.1
    hw = .05*xmx
    w = .01*xmx
    plt.arrow(-xmx,0,2*xmx,0,color = 'gray',alpha=.7,width=w,
                head_width=hw,length_includes_head=True)
    plt.arrow(0,-mx,0,2*mx,color = 'gray',alpha=.7, width=w,
                head_width=hw,length_includes_head=True)
    ntables = len(np.unique(ds.chunks))
    if cmap is None:
        cmap = cm.spectral(np.linspace(.2,.85,ntables))
    m,ncol = ds.shape
    nrows = m/ntables
    data = ds.samples.T.reshape((ncol,nrows,ntables),order='F')

    centers = np.mean(data,2).T[:,[x,y]]
    plt.scatter(centers[:,0],centers[:,1])

    for t in range(ntables):
        tab = data[:,:,t].T[:,[x,y]]
        for r in range(nrows):
            a,b = centers[r,:]
            j,k = tab[r,:]
            plt.plot([a,j],[b,k],c=cmap[t],lw=2,alpha=.5)
    plt.axis('equal')
    plt.axis((-mx,mx,-mx,mx))
    #plt.axis('equal')
    plt.axis(axes)
    if not nude:
        for t in range(nrows):
            plt.annotate(ds.targets[t],xy = (centers[t,0], centers[t,1]))

        plt.text(-xmx,.05*mx,'$\lambda = %s$'%np.round(sts.eigv[x],2))
        plt.text(mx*.05,mx*.9,'$\lambda = %s$'%np.round(sts.eigv[y],2))
        tau = '$\\tau = $'
        perc = '$\%$'
        mpl.rcParams['text.usetex'] = False
        plt.text(-xmx,-.1*mx, '%s $%s$%s' %
                (tau,np.round(100*sts.inertia[x],0),perc))
        plt.text(xmx*.05,mx*.8, '%s $%s$%s' %
                (tau,np.round(100*sts.inertia[y],0),perc))
        plt.text(-.15*xmx,.8*mx,'$%s$'%(y+1), fontsize=20)
        plt.text(xmx*.85,-mx*.2,'$%s$'%(x+1),fontsize=20)

    plt.axis('scaled')
    plt.axis([-xmx,xmx,-mx,mx])


def plot_ellipses(ds, sts, x=0, y=1, ci=.95, labels=None,
                   cmap=None,scat=False,linestyle=None, nude=False,
                    axes='off', fig=None, **kwargs):

    """
    center: should be the factor scores from original compromise matrix
    points: should be factor scores from bootstrap
    """

    f = plt.figure(fig)
    ax = f.gca()
    boot = ds.samples
    i,j,k = boot.shape

    if cmap==None:
        cmap = cm.spectral(np.linspace(.2,.85,i))
    if linestyle is None:
        if ds.sa.has_key('linestyle'):
            linestyle = list(ds.sa['linestyle'])
        else:
            linestyle = ['solid']*i
    if labels is None:
        labels = list(ds.targets)


    mx = np.max(abs(boot[:,[x,y],:]))
    xmx = mx*1.2
    w = .01*xmx
    hw = .05*xmx
    #plt.plot([-mx,mx],[0,0],c = 'gray',alpha=.7, lw=2)
    #plt.plot([0,0],[-mx*.8,mx*.8],c = 'gray',alpha=.7, lw=2)
    plt.arrow(-xmx,0,2*xmx,0,color = 'gray',alpha=.7,width=w,
            head_width=hw,length_includes_head=True)
    plt.arrow(0,-mx,0,mx*2,color = 'gray',alpha=.7, width=w,
            head_width=hw,length_includes_head=True)
 
    for l in range(i):
        points = np.hstack((boot[l,x,:].reshape(-1,1),
                                boot[l,y,:].reshape(-1,1)))
        center = np.mean(points,0)
        w, rot = np.linalg.eigh(np.cov(points.T))

        # get size corresponding to level
        a = np.sqrt(w[0] * chi2.ppf(ci, 2))
        b = np.sqrt(w[1] * chi2.ppf(ci, 2))

        j = np.linspace(0,2*np.pi,128)
        coords = np.hstack((    (np.cos(j)*a).reshape((-1,1)),
                        (np.sin(j)*b).reshape((-1,1))))
        coords = np.mat(coords.dot(rot.T) + center)

        plt.plot(np.vstack((coords[:,0], coords[0,0])),
                    np.vstack((coords[:,1], coords[0,1])),
                    c=cmap[l], ls=linestyle[l], **kwargs)
        if scat:
            plt.scatter(points[:,0],points[:,1],c=cmap[l])

        if not nude:
            plt.annotate(labels[l],xy = (center[0], center[1]))
            mpl.rcParams['text.usetex'] = True

    if not nude:
        plt.text(-xmx,.05*mx,'$\lambda = %s$'%np.round(sts.eigv[x],2))
        plt.text(xmx*.05,mx*.9,'$\lambda = %s$'%np.round(sts.eigv[y],2))
        tau = '$\\tau = $'
        perc = '$\%$'
        mpl.rcParams['text.usetex'] = False
        plt.text(-xmx,-.1*mx, '%s $%s$%s' %
                (tau,np.round(100*sts.inertia[x],0),perc))
        plt.text(xmx*.05,mx*.8, '%s $%s$%s' %
                (tau,np.round(100*sts.inertia[y],0),perc))
        plt.text(-.15*xmx,.8*mx,'$%s$'%(y+1), fontsize=20)
        plt.text(xmx*.85,-mx*.2,'$%s$'%(x+1),fontsize=20)
    #plt.axis('equal')
    #plt.axis([-mx,mx,-mx,mx])
    plt.axis('scaled')
    plt.axis([-xmx,xmx,-mx,mx])
    plt.axis(axes)
    return f.number



def plot_mds(self,x=0,y=1,labels=None):
    """
    This plots MDS solution from the compromise matrix
    """
    plt.figure()
    if labels is None:
        labels = self.targets[0:self.I]
    mx = max(max(np.abs(self.F[:,x])),max(np.abs(self.F[:,y])))
    for i in range(self.I):
        plt.scatter(self.F[i,x],self.F[i,y])
        plt.text(self.F[i,x],self.F[i,y],labels[i])
    plt.xlim((-mx,mx))
    plt.ylim((-mx,mx))

def plot_partial_factor(self,x=0,y=1,labels=None,colors=None):
    """
    This plots partial factor scores from each subject
    """
    if colors==None:
        colors=['r','r','b','r','g','g','b','b','b','r','b','b']
    if labels==None:
        labels=self.targets[0:12]
    self.plot_mds(x=x,y=y)
    plt.hold(True)
    for chunk in range(self.ntables):
        f=self.factors[chunk]
        plt.scatter(f[:,x],f[:,y])
        point=np.hstack((f[:,x].reshape(-1,1),f[:,y].reshape(-1,1)))
        center=np.hstack((self.F[:,x].reshape(-1,1),self.F[:,y].reshape(-1,1)))
        for label, xc, yc in zip(labels, f[:,x], f[:,y]):
            plt.annotate(label[0:3]+'-'+str(chunk),xy=(xc,yc))
        for i in range(self.ntargets):
            plt.plot([point[i,0],center[i,0]],[point[i,1],center[i,1]],'%s--'%colors[i],linewidth=1)

def remove_pc(self,pc_num,method):
    pc_all=None
    res_all=None
    q_all=None
    f_all=[]
    for chunk in range(self.ntables):
        X_k,Q_k,F_k=self.partial_k(chunk)
        if method=='acc':
            f=F_k[:,0:pc_num+1]
            q=Q_k.T[0:pc_num+1,:]
            pc=np.dot(f,q)
        #this makes animate positive, inanimate negative for q
        elif method=='con':
            f=F_k[:,pc_num]
            q=Q_k.T[pc_num,:]
            pc=np.dot(f.reshape(-1,1),q.reshape(1,-1))
        res=X_k-pc
        if pc_all==None:
            q_all=q
            f_all.append(f)
            pc_all=dataset_wizard(pc,targets=np.unique(self.targets),chunks=np.repeat(chunk+1,self.ntargets))
            res_all=dataset_wizard(res,targets=np.unique(self.targets),chunks=np.repeat(chunk+1,self.ntargets))
        else:
            q_all=np.vstack((q_all,q))
            f_all.append(f)
            pc_all.append(dataset_wizard(pc,targets=np.unique(self.targets),
                            chunks=np.repeat(chunk+1,self.ntargets)))
            res_all.append(dataset_wizard(res,targets=np.unique(self.targets),
                            chunks=np.repeat(chunk+1,self.ntargets)))
    return pc_all, res_all, q_all, f_all


def eigh(X):
    """Convenience function that returns the output of np.linalg.eigh
    (i.e., eigen decomposition of Hermitian matrices) after ordering
    the output in descending eigen value order
    """
    w,v = np.linalg.eigh(X)
    w_s = np.array(sorted(w,reverse=True))
    v_s = None
    for i in range(len(w)):
        if v_s is None:
            v_s = v[:,w==w_s[i]]
        else:
            v_s = np.hstack((v_s,v[:,w==w_s[i]]))
    return w_s,v_s





