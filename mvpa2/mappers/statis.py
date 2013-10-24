import numpy as np
import matplotlib.pyplot as plt
from mvpa2.suite import Mapper, Dataset
from scipy.spatial.distance import pdist
from operator import itemgetter


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

def paq(X, k=None):
    epsilon = 10**-15
    i,j = X.shape
    m = np.min([i,j])
    if k is None:
        k = m
    elif k > m:
        k = m

    flip = False
    if i < j:
        X = X.T
        flip = True
    l, Q = eigh(np.dot(X.T,X))
    
    ll = np.max(l.shape)
    if k > ll:
        k = len(l)
    Q = Q[:,0:k+1]
    l = l[0:k+1]
    a = np.sqrt(l)
    niq, njq = Q.shape
    a_ = 1./(a.reshape(1,-1)).squeeze()
    a_ = np.tile(a_,niq).reshape(niq,len(a_))
    P = np.dot(X,Q*a_)
    if flip:
        X = X.T
        tmp = Q
        Q = P
        P = tmp 
    return P,a,Q


def _make_ellipse(mean, cov, ax, level=0.95, color=None):
    """Support function for scatter_ellipse."""
    from matplotlib.patches import Ellipse

    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1]/u[0])
    angle = 180 * angle / np.pi # convert to degrees
    v = 2 * np.sqrt(v * stats.chi2.ppf(level, 2)) #get size corresponding to level
    ell = Ellipse(mean[:2], v[0], v[1], 180 + angle, facecolor='none',
                  edgecolor=color,
                  #ls='dashed',  #for debugging
                  lw=1.5)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)


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
        self.targets = targets = dataset.sa.targets
        if self._stack == 'v':
            self.chunks = chunks = np.array(dataset.sa[self._chunks_attr])
        if self._stack == 'h':
            self.chunks = chunks = np.array(dataset.fa[self._chunks_attr])

        self.ntables = ntables = len(np.unique(chunks))
        self.ntargets= ntargets= len(np.unique(targets))
        i,J = samples.shape
        self.I = I = ntargets
        self.M = M = (1.0/I)*np.eye(I) # masses, eq. 2
        self._subtable_stats = {}
        self._supp_obs = None
        
    
        X = None # tables matrix with subtables h-stacked
        self.subtable_idx = np.array([])
        for ch, chunk in enumerate(np.unique(chunks)):
            # compute cross-product for each table, Eq. 7
            if self._stack == 'v':
                t = samples[chunks==chunk,:]
            if self._stack == 'h':
                t = samples[:,chunks==chunk]
            
            t,c_m,c_n,r_m,r_n,t_n = self.center_and_norm_table(t)
            self._subtable_stats[ch] = {'row_mean':r_m, 'row_norm':r_n,
                        'col_mean':c_m, 'col_norm':c_n, 'table_norm':t_n}
            self.subtable_idx=np.hstack((self.subtable_idx,ch*np.ones((t.shape[1]))))
            

            if X is None:
                X = t
            else:
                X = np.hstack((X,t))

        self.X = X
        (A, alpha,C) = self.inter_table_Rv_analysis(X,self.subtable_idx)
        self.A = A
        self.alpha = alpha
        self.C = C
               
        #self.compromise1 = np.dot(np.sqrt(self.M),np.dot(X,np.sqrt(self.A)))
        self.compromise = (X.T*np.sqrt(np.diag(M))).T*np.sqrt(np.diag(A))
   
        P,delta,Qt = np.linalg.svd(self.compromise, full_matrices=0)     #Eq.42
        self.P = np.dot(np.linalg.inv(np.sqrt(M)),P)
        self.Q = np.dot(np.linalg.inv(np.sqrt(self.A)),Qt.T)

        self.Delta = Delta = np.diag(delta)
        self.F = F = np.dot(self.P,Delta)     #Eq. 43
        self.eigv=delta**2
        self.inertia=delta**2/np.sum(delta**2)
        self.partial_factors = self._forward_dataset(dataset)
        if self._bootstrap_iter > 0:
            self.run_bootstrap(self._bootstrap_iter)

    def _forward_dataset(self, ds, supp=None):

        # The idea here is that given a dataset with the same set of chunks as
        # the training dataset, preprocess each unique chunk table the same way
        # as the original set of tables.

        # first check to see that the chunks match.
        if supp is None:
            mapped_ds = None
            for ch in np.unique(self.subtable_idx):
                
                partF = Dataset(np.dot(self.X[:,self.subtable_idx==ch],
                            self.Q[self.subtable_idx==ch,:]))
                partF.sa['chunks'] = np.ones((len(partF.samples,)))*ch

                if mapped_ds is None:
                    mapped_ds = partF
                else:
                    mapped_ds.append(partF)

        
            return mapped_ds

        if supp is 'obs':
       
            Y = None
            targets = None

            for ch in np.unique(self.chunks):
                if self._stack == 'v':
                    t = ds.samples[ds.sa[self._chunks_attr]==ch,:]
                if self._stack == 'h':
                    t = ds.samples[:,ds.fa[self._chunks_attr]==ch]
                t = self.center_and_norm_table(t,
                        col_mean=self._subtable_stats[ch]['col_mean'],
                        col_norm=self._subtable_stats[ch]['col_norm'],
                        table_norm = self._subtable_stats[ch]['table_norm'])[0]        

                if Y is None:
                    Y = t
                else:
                    Y = np.hstack((Y,t))
            self._supp_obs = {}
            self._supp_obs['targets'] = ds.targets[:len(Y)]
            self._supp_obs['Y'] = Y
            return np.dot(Y,np.dot(self.A,self.Q))


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
            elif self._row_norm=='std':
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

   

    def partial_k(self,k):
        X_k=self.X[:,k*self.J:(k+1)*self.J]
        Q_k=self.Q[k*self.J:(k+1)*self.J,:]*(1/np.sqrt(self.alpha[k]))      #Eq.44
        F_k=np.dot(X_k,Q_k)
        return X_k,Q_k,F_k

    def run_bootstrap(self, niter): 
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
        nrows = len(np.unique(self.targets))
        self.FBoot = np.zeros((nrows,nrows,niter))
        if self._supp_obs:
            n,m = self._supp_obs['Y'].shape
            self._supp_obs['boot'] = np.zeros((n,nrows,niter))
        
        for i in range(niter):
            idx = np.floor(self.ntables*np.random.random_sample(self.ntables))   
            Y = None
            Y_idx = None
            Y_supp = None
            fselect = np.zeros((nrows,nrows,self.ntables))

            for k,j in enumerate(idx):

                Y_t = self.X[:,self.subtable_idx==j]
                if self._supp_obs:
                    Y_supp_t = self._supp_obs['Y'][:,self.subtable_idx==j]
                if Y_idx is None:
                    Y_idx = np.ones((Y_t.shape[1]))*k
                else:
                    Y_idx = np.hstack((Y_idx,np.ones((Y_t.shape[1]))*k))

                if Y == None:
                    Y = Y_t
                    if self._supp_obs:
                        Y_supp = Y_supp_t
                else:
                    Y = np.hstack((Y,Y_t))
                    if self._supp_obs:
                        Y_supp = np.hstack((Y_supp,Y_supp_t))


                fselect[:,:,k] = self.partial_factors.samples[self.partial_factors.chunks==j,:]
            
            """
            ds_i = Dataset(Y)
            ds_i.fa['tables'] = Y_idx
            ds_i.sa['targets'] = self.targets[0:nrows]
            stat_i = StatisMapper()
            stat_i.train(ds_i)
            """
                
            (A,alpha,C) = self.inter_table_Rv_analysis(Y,Y_idx)
            self.FBoot[:,:,i]=np.sum(fselect*alpha.flatten(),2)
            #self.FBoot[:,:,i] = np.dot(Y,np.dot(stat_i.A,stat_i.Q))
            if self._supp_obs:
                self._supp_obs['boot'][:,:,i] = np.dot(Y_supp,np.dot(A,self.Q))
            if i%100==0:
                print "iter:%s"%i

    def inter_table_Rv_analysis(self, X_, subtable_idx):

        Z = None
        ntables = len(np.unique(subtable_idx))
        for idx in np.unique(subtable_idx):
            t = X_[:,subtable_idx==idx]
            S_t = np.dot(t,t.T).flatten()

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

        A = np.diag(a)
        return A,alpha,C



       


    def draw_ellipses(self, x=0, y=1, level=.95, labels=None,
                        colors=None,scat=False,linestyle=None, **kwargs):
        """
        center: should be the factor scores from original compromise matrix
        points: should be factor scores from bootstrap
        """
        from matplotlib.patches import Ellipse
        import scipy.stats as stats

        fig = plt.figure()
        ax = fig.gca()
        boot = self.FBoot
        if self._supp_obs:
            boot = np.vstack((boot,self._supp_obs['boot']))

        i,j,k = boot.shape

        if colors==None:
            colors = ['b','g','r','c','m','y','k']
            while len(colors) < len(self.targets):
                colors = colors + colors
        if linestyle is None:
            linestyle = ['-']*j + ['--']*(i-j)
        if labels is None:
            labels = list(self.targets[:len(self.X)])
            if self._supp_obs:
                labels = labels + list(self._supp_obs['targets'])

        mx = []
        mn = []
        
        for l in range(i):
            points = np.hstack((boot[l,x,:].reshape(-1,1),
                                    boot[l,y,:].reshape(-1,1)))
            center = np.mean(points,0)
           

            ####### FROM statsmodels _make_ellipse 
            v, w = eigh(np.cov(points))
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1]/u[0])
            angle = 180 * angle / np.pi 
            v = 2 * np.sqrt(v * stats.chi2.ppf(level, 2)) 
            ell = Ellipse(center, v[0], v[1], 180 + angle, facecolor='none',
                    edgecolor=colors[l],
                    #ls='dashed',  #for debugging
                    lw=1.5)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            #######################################


            dist = np.array([ np.sqrt((points[m,0]-center[0])**2+(points[m,1]-center[1])**2) 
                            for m in range(points.shape[0])])

            dp=zip(dist,points)
            
            # Trim the 5% of points with the largest distances from the center
            #don't do this
            #dp=sorted(dp, key=itemgetter(0))[0:int(points.shape[0]*.95)]
            
            # update set of points and new center
            orig_points = points
            points = np.array([dp[n][1] for n in range(len(dp))])
            center = np.mean(points,0)
            
            if scat:
                plt.scatter(orig_points[:,0],orig_points[:,1],c=colors[l]) # uncomment to see all points
            # PCA on set of centered points
            #p,a,q = np.linalg.svd(points - center, full_matrices=0)
            p,a,q = paq(points - center)
            F_ = p.dot(np.diag(a))

            

            F_hull=np.asarray(abs(F_))

            prop=a[0]/a[1]   #proportion
            y_length = max(np.sqrt((F_hull[:,0]**2)/(prop**2) + (F_hull[:,1]**2)))
            x_length = y_length*prop


            j = np.array(range(1,129))*np.pi/64

            coords = np.array([np.cos(j.T)*x_length,np.sin(j.T)*y_length]).T
            coords = np.mat(coords)*np.mat(q.T) + np.mat(np.ones((len(j),1)))*np.mat(center)
            mx.append(np.max(coords))
            mn.append(np.min(coords))
            #plt.plot(np.vstack((coords[:,0],coords[0,0])),
            #            np.vstack((coords[:,1],coords[0,1])),
            #            colors[l], linestyle=linestyle[l],**kwargs)

            #plt.annotate(labels[l],xy = (center[0], center[1]))
        mn = min(mn)
        mx = max(mx)
        pad = (mx-mn)/30.
        mn = mn - pad
        mx = mx + pad
        plt.axis([mn, mx, mn, mx])

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






