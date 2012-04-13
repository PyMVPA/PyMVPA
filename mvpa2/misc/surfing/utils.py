'''
Created on Jan 30, 2012

@author: nick
'''
import os, subprocess, time, datetime, collections

def as_list(v):
    '''makes this a singleton list if the input is not a list'''
    if type(v) not in [list,tuple]:
        v=[v]
    return v

def file_contains_string(fn,s):
    '''Returns whether a file contains a string
    
    Parameters
    ----------
    fn: str
        filename
    s: str
        string
        
    Returns
    -------
    True iff the file name fn contains s
    '''
    f=open(fn)
    c=f.read()
    return c.find(s)>=0
    

def afni_fileparts(fn):
    '''File parts for afni filenames.
    
    Returns a tuple with these four parts.
    
    Also works for .nii files, in which case the third part is the empty
    
    Not tested for other file types
    
    Parameters
    ----------
    whole filename
      PATH/TO/FILE/NAME+orig.HEAD
      
    
    Returns
    -------
    fullpath: str
      PATH/TO/FILE
    rootname: str
      NAME
    orientation: str
      +orig
    extensions: str
      .HEAD
    
    '''
    
    tail,head=os.path.split(fn)

    s=head.split('+')
    name=s[0]
    orient='+'+s[1] if len(s)==2  else ''
    
    afniorients=['+orig','+tlrc','+acpc']
    ext=None
    for a in afniorients:
        if orient.startswith(a):
            #ext=orient[len(a):]
            orient=a
            ext=".HEAD"
            #if ext=='.':
            #    ext=''
            
    if ext is None:
        s=name.split(".")
        if len(s)>1:
            ext="."+".".join(s[1:])
            name=s[0]
        else:
            ext=''
        
    return tail,name,orient,ext

def afni_fileexists(fn):
    '''
    Parameters
    ----------
    fn : str
        AFNI filename (possibly without .HEAD or .BRIK extension)
    
    Returns
    -------
    bool
        True iff fn exists as AFNI file
    '''
    p,n,o,e=afni_fileparts(fn)
        
    if o:
        return os.path.isfile('%s/%s%s.HEAD'%(p,n,o))
    else:
        return (e in ['.nii','.nii.gz']) and os.path.isfile(fn) 
    
def run_matlabcommand(matlabcmd,env=None,indir='.'):
    '''wrapper to run a single matlab command in the shell'''
    
    if env is None:
        env=os.environ

    # ensure matlabcmd ends with semicolon        
    lastchar=matlabcmd.strip()[-1];
    if lastchar not in [',',';']:
        matlabcmd+=';'
    
    cmd='cd %s;matlab -nosplash -nodisplay -r "%sexit"' % (indir,matlabcmd)
    run_cmds(cmd,env)

def run_cmds(cmds,env=None,dryrun=False):
    '''exectute a list of commands in the shell'''
    if env is None:
        env=os.environ
    
    # if cmds is just one command, make a singleton list    
    cmds=as_list(cmds)
    
    # run each command    
    for cmd in cmds:
        print("** Will execute the following commands:")
        for c in cmd.split(';'):
            print '** - %s' % c
        if not dryrun:
            print("**>> Starting now:")
            
            subprocess.call(cmd,env=os.environ,shell=True)
            
            print("**<< ... completed execution")
            """
            [this doesn't work well']
            r=os.system(cmd)
            
            if r:
                print("**<< ... completed execution")
            else:
                raise Exception("Error occured running the process, return code %d" % p.returncode)
            
            [the following works but it seems it may keep a child running if the parent is killed]
            
            p=subprocess.Popen(cmd,shell=True,executable='/bin/bash',env=env)
            p.communicate() # wait until command is finished
            if p.returncode!=0:
                raise Exception("Error occured running the process, return code %d" % p.returncode)
            else:
                print("**<< ... completed execution")
            """
            
    return

def which(f,env=None):
    '''Finds the full path to a file in the path
    
    Parameters
    ----------
    f: str
        Filename of executable
    env (optional): 
        Environment in which path is found.
        By default this is the environment in which python runs
        
    
    Returns
    str
        Full path of 'f' if 'f' is executable and in the path, 'f' itself 
        if 'f' is a path, None otherwise
    '''
    if env==None:
        env=os.environ
    
    def is_executable(fullpath):
        return os.path.exists(fullpath) and os.access(fullpath,os.X_OK)
    
    [p,n]=os.path.split(f)
    if p: 
        return f
    else:
        for path in env['PATH'].split(os.pathsep):
            fullfn=os.path.join(path,n)
            if is_executable(fullfn):
                return fullfn
        return None

def tictoc():
    '''Measure time a la matlab
    
    Returns
    -------
    timer: Timer
        class with methods 'tic()' and 'toc()' to be called before and 
        after a task is run. 'toc()' shows a message of the time elapsed
        since the last time that 'tic()' was called
    '''
    class Timer():
        def __init__(self):
            self.tic()
        def tic(self):
            self.t=time.time()
        def toc(self,msg=None):
            print "Time elapsed: %.3f s" % float((time.time()-self.t))
        def tt(self):
            self.toc()
            self.tic()
    return Timer()

def hist(vs):
    '''Histogram function
    
    Parameters
    ----------
    vs : iterable
        values that should be counted
    
    Returns
    -------
    h : dict
        'h[i]==j' means that 'i' occurs 'j' times in 'vs' (with 'j'>0)
    '''
    v2count=collections.defaultdict(lambda:0)
    for v in vs:
        v2count[v]+=1
    return v2count

def flatten(l, ltypes=(list, tuple)):
    '''
    Flattens a list or tuple, recursively
    
    Note
    ----
    From http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html'''
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def eta(starttime,progress,msg=None,show=True):
    '''Simple linear extrapolation to estimate how much time is needed 
    to complete a task.
    
    Parameters
    ----------
    starttime
        Time the tqsk started, from 'time.time()'
    progress: float
        Between 0 (nothing completed) and 1 (fully completed)
    msg: str (optional)
        Message that describes progress
    show: bool (optional, default=True)
        Show the message and the estimated time until completion
    
    Returns
    -------
    eta
        Estimated time until completion
    
    Note
    ----
    ETA refers to estimated time of arrival
    '''  
    if msg is None:
        msg=""
    
    now=time.time()
    took=now-starttime
    eta=-1 if progress==0 else took*(1-progress)/progress
    
    f=lambda t:str(datetime.timedelta(seconds=t))
    
    fullmsg='%s, took %s, remaining %s' % (msg, f(took), f(eta))
    if show:
        print fullmsg

    return fullmsg


def _get_fingerdata_dir():
    
    dirs=['/Users/nick/Downloads',
          '/apps/nicksurfing/testdata']
    
    indir='fingerdata-0.2'
    
    for d in dirs:
        fullpath='%s/%s/' % (d, indir)
        
        if os.path.exists(fullpath):
            return fullpath
        
    raise ValueError('Directory for fingerdata not found')
    
        
    
    
    
if __name__ == '__main__':
    fn="my/dir/file.nii"
    fn='/Users/nick/Downloads/fingerdata-0.2/sef/anat_al+orig.HEAD'
    p=afni_fileparts(fn)
    print afni_fileexists(fn)
    
    