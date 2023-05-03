""" Concorde base layer.

Cython wrappers around Concorde API.

"""
import numpy as np
cimport numpy as np

cdef extern from "concorde.h":
    struct CCdatagroup:
        double *x
        double *y
        double *z

    struct CCrandstate:
        pass

    int CCutil_gettsplib(char *datname, int *ncount, CCdatagroup *dat)

    int CCtsp_solve_dat(int ncount, CCdatagroup *indat, int *in_tour,
        int *out_tour, double *in_val, double *optval, int *success,
        int *foundtour, char *name, double *timebound, int *hit_timebound,
        int silent, CCrandstate *rstate)

    double CCutil_real_zeit() # note that I have got errors beause of diactances
    
    void CCutil_sprand(int seed, CCrandstate *r)
    
    void CCutil_init_datagroup(CCdatagroup *dat)
    
    void CCutil_freedatagroup(CCdatagroup *dat)
    
    int CCutil_graph2dat_matrix (int ncount, int ecount, int *elist,
        int *elen, int defaultlen, CCdatagroup *dat)
    
    double CCutil_zeit ()
    
    

cdef class _CCdatagroup:

    cdef CCdatagroup c_data
    cdef bint initialized
    cdef int ncount

    def __cinit__(self):
        self.initialized = False

    #def __dealloc__(self):
        #if self.initialized:
            #CCutil_freedatagroup(&(self.c_data))

    @property
    def x(self):
        cdef double[:] x_data
        if self.initialized:
            x_data = <double[:self.ncount]>self.c_data.x
            return np.asarray(x_data)
        else:
            return np.array([])

    @property
    def y(self):
        cdef double[:] y_data
        if self.initialized:
            y_data = <double[:self.ncount]>self.c_data.y
            return np.asarray(y_data)
        else:
            return np.array([])

    @property
    def z(self):
        cdef double[:] y_data
        if self.initialized:
            z_data = <double[:self.ncount]>self.c_data.z
            return np.asarray(z_data)
        else:
            return np.array([])


def _CCutil_gettsplib(str fname):
    cdef int ncount, retval
    cdef _CCdatagroup dat

    dat = _CCdatagroup()

    retval = CCutil_gettsplib(fname.encode('utf-8'), &ncount, &dat.c_data)
    if retval == 0:
        dat.initialized = True
        dat.ncount = ncount
        return ncount, dat
    else:
        return -1, None
        
#--------------------@Enas---------------------------------------------------

def _CCtsp_solve_mat(
        int ncount, str name, double timebound, int silent, int seed=0, dist=np.empty(0)):
		
		
		
    cdef:
        int *in_tour = NULL
        double *in_val = NULL      # initial upper bound
        double opt_val = 0         # value of the optimal tour
        int success = 0            # set to 1 if the run finishes normally
        int foundtour = 0          # set to 1 if a tour has been found
        double *_timebound = NULL  # NULL if no timebound, >= 0 otherwise
        int hit_timebound = 0
        int retval
        double szeit; #Measure cpu time
        
        # Random state used by the solver
        CCrandstate rstate

        # Output tour
        np.ndarray[int, ndim=1] out_tour
        int ecount = <int>(ncount * (ncount - 1) / 2) # Number of edges
        int[:] elist 
        int[:] elen = dist[np.triu_indices_from(dist,k=1)] 
      
        int edge = 0
        
        int edgeWeight = 0
        
        CCdatagroup dat
 
    a = np.triu_indices_from(dist,k=1)
    elist = np.array((a[0],a[1]),dtype=np.int32).T.reshape(1,int(2*ecount)).flatten() 
    #Initialize a CCdatagroup
    CCutil_init_datagroup (&dat)
 
    retval = CCutil_graph2dat_matrix (ncount, ecount, &elist[0], &elen[0], 1, &dat)
	
    out_tour = np.zeros(ncount, dtype=np.int32)
    if seed != 0:
        seed = <int>CCutil_real_zeit()
    CCutil_sprand (seed, &rstate)

    if timebound > 0:
        _timebound = &timebound

    retval = CCtsp_solve_dat(ncount, &dat, in_tour, &out_tour[0],
                             in_val, &opt_val, &success, &foundtour,
                             name.encode('utf-8'), _timebound, &hit_timebound,
                             silent, &rstate)

    return out_tour, opt_val, bool(success), bool(foundtour), bool(hit_timebound)


#--------------------------------------------------------------------------
def _CCtsp_solve_dat(
        int ncount, _CCdatagroup ingroup,
        str name, double timebound, int silent, int seed=0):

    cdef:
        int *in_tour = NULL
        double *in_val = NULL      # initial upper bound
        double opt_val = 0         # value of the optimal tour
        int success = 0            # set to 1 if the run finishes normally
        int foundtour = 0          # set to 1 if a tour has been found
        double *_timebound = NULL  # NULL if no timebound, >= 0 otherwise
        int hit_timebound = 0
        int retval

        # Random state used by the solver
        CCrandstate rstate

        # Output tour
        np.ndarray[int, ndim=1] out_tour

    out_tour = np.zeros(ncount, dtype=np.int32)

    if seed != 0:
        seed = <int>CCutil_real_zeit()
    CCutil_sprand (seed, &rstate)

    if timebound > 0:
        _timebound = &timebound

    retval = CCtsp_solve_dat(ncount, &ingroup.c_data, in_tour, &out_tour[0],
                             in_val, &opt_val, &success, &foundtour,
                             name.encode('utf-8'), _timebound, &hit_timebound,
                             silent, &rstate)
    szeit = CCutil_zeit()
    #CCIFREE(elist,int)
    #CCIFREE(elen,int)
    #CCIFREE(out_tour,int)
	
    return out_tour, opt_val, bool(success), bool(foundtour), bool(hit_timebound)
