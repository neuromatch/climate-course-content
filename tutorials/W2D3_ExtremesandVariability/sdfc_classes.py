# We credit 2020 Yoann Robin
# and his free SDFC (Statistical Distribution Fit with Covariates) software: 
# https://github.com/yrobink/SDFC-python
# you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.

# We removed the QuantileRegression implementation in C++ and replaced it with a scikitlearn implementation https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html
# to be dependent on python only (cf. quantile() function).

import warnings
import numpy as np          
import scipy.optimize as sco
import scipy.stats as sc
import scipy.special as scs
import texttable as tt
import scipy.linalg   as scl
from sklearn.linear_model import QuantileRegressor

###############
## Class(es) ##
###############

class LHS:
	def __init__( self , names : list , n_samples : int ):
		self.names     = names
		self.n_lhs     = len(self.names)
		self.n_samples = n_samples
		self._values   = { n : None for n in self.names }
		self.jacobian_ = None
		self._fixed    = { n : False for n in self.names }
	
	def is_fixed( self , name ):
		return self._fixed.get(name)
	
	@property
	def values_( self ):
		return self._values
	
	@values_.setter
	def values_( self , values ):
		for n,v in zip(self.names,values):
			self._values[n] = v

class UnivariateLink:##{{{
	"""
	SDFC.link.UnivariateLink
	========================
	base class for univariate link
	"""
	def __init__(self): pass
	
	def __call__( self , x ):
		return self.transform(x)
##}}}

class ULIdentity(UnivariateLink): ##{{{
	"""
	SDFC.link.ULIdentity
	====================
	
	Identity link function, i.e.:
		f(x) = x
		f^{-1}(x) = x
		df(x) = 1
	"""
	
	def __init__( self ):
		UnivariateLink.__init__(self)
	
	def transform( self , x ):
		return x
	
	def inverse( self , x ):
		return x
	
	def jacobian( self , x ):
		return np.ones_like(x)
##}}}

class MultivariateLink:##{{{
	"""
	SDFC.link.MultivariateLink
	==========================
	Base class for MultivariateLink
	
	"""
	
	def __init__( self , *args , **kwargs ):##{{{
		self._special_fit_allowed = False
		self._n_features = kwargs.get("n_features")
		self._n_samples  = kwargs.get("n_samples")
	##}}}
	
	def transform( self , coef , X ):##{{{
		pass
	##}}}
	
	def jacobian( self , coef , X ):##{{{
		pass
	##}}}
	
	@property
	def n_features(self):##{{{
		return self._n_features
	##}}}
	
	@property##{{{
	def n_samples(self):
		return self._n_samples
	##}}}
	
##}}}

class MLConstant(MultivariateLink):##{{{
	"""
	SDFC.link.MLConstant
	====================
	Link used for fixed parameters f_<param>
	
	"""
	
	def __init__( self , value , *args , **kwargs ):##{{{
		kwargs["n_features"] = 0
		MultivariateLink.__init__( self , *args , **kwargs )
		self.value_ = np.array([value]).reshape(-1)
		if self.value_.size == 1:
			self.value_ = np.repeat( value , self.n_samples )
	##}}}
	
	def transform( self , *args , **kwargs ):##{{{
		return self.value_
	##}}}
	
##}}}

class MLLinear(MultivariateLink): ##{{{
	"""
	SDFC.link.MLLinear
	==================
	Link function which contains an univariate link function. The idea is to
	chain a linear map with an univariate transform
	
	"""
	
	def __init__( self , *args , **kwargs ):##{{{
		self._l     = kwargs.get("l")
		self._c     = kwargs.get("c")
		if self._l is None: self._l = ULIdentity()
		if self._c is not None: kwargs["n_samples"] = self._c.shape[0]
		MultivariateLink.__init__( self , *args , **kwargs )
		
		self.design_ = np.ones( (self.n_samples,1) )
		if self._c is not None:
			self.design_ = np.hstack( (self.design_,self._c) )
		
	##}}}
	
	def _linear_transform( self , coef , X ): ##{{{
		return self.design_ @ coef
	##}}}
	
	def transform( self , coef , X ):##{{{
		out = self._l.transform( self._linear_transform(coef,X) )
		return out
	##}}}
	
	def jacobian( self , coef , X ): ##{{{
#		jac = np.zeros( (self.n_samples,self.n_features) )
#		jac[:,0]  = 1
#		jac[:,1:] = X
		return self._l.jacobian( self._linear_transform( coef , X ).reshape(-1,1) ) * self.design_
	##}}}
	
##}}}

class MLTensor(MultivariateLink):##{{{
	"""
	SDFC.link.MLTensor
	==================
	Link function used to build the product of univariate link function
	
	
	"""
	
	def __init__( self , l_p , s_p , *args , **kwargs ):##{{{
		kwargs["n_features"] = np.sum(s_p)
		MultivariateLink.__init__( self , *args , **kwargs )
		self._l_p = l_p
		self._s_p = s_p
		self._special_fit_allowed = np.all( [isinstance(l,(MLLinear,MLConstant)) for l in self._l_p] )
	##}}}
	
	def transform( self , coef , X ): ##{{{
		list_p = []
		ib,ie = 0,0
		for s,l,x in zip(self._s_p,self._l_p,X):
			ie += s
			list_p.append( l.transform( coef[ib:ie] , x ) )
			ib += s
		return list_p
	##}}}
	
	def jacobian( self , coef , X ): ##{{{
		list_jac = []
		ib,ie = 0,0
		jac = np.zeros( (np.nonzero(self._s_p)[0].size,self.n_samples,self.n_features) )
		i = 0
		for s,l,x in zip(self._s_p,self._l_p,X):
			if s > 0:
				ie += s
				jac[i,:,ib:ie] = l.jacobian( coef[ib:ie] , x )
				ib += s
				i += 1
		return jac
	##}}}
	
##}}}


class RHS:
	"""
	Class that contains coef_ to fit, covariates and link function. This class
	contains:
	- self.lhs_ : the Left Hand Side part, updated by this class
	- self.c_global : list of covariate
	- self.l_global : global link function
	- self.s_global : length of coef per params of lhs
	
	The parameter self.coef_ is a property, and when it is set the LHS is
	updated accordingly
	
	"""
	def __init__( self , lhs_ : LHS ): ##{{{
		"""
		d
		"""
		self.lhs_     = lhs_
		self.c_global = None
		self.l_global = None
		self.s_global = None
		self._coef_   = None
	##}}}
	
	def build( self , **kwargs ):##{{{
		"""
		Here five kinds of arguments can be passed:
		- c_<param> : covariate of the param,
		- l_<param> : link function of the param,
		- f_<param> : fixed values of the LHS
		- c_global  : list of all covariates, sorted by lhs order
		- l_global  : global link function generated the LHS
		If c_global is set, all arguments (except l_global) are ignored
		"""
		
		## If global covariate and link functions are defined, just set it
		##================================================================
		if kwargs.get("c_global") is not None or kwargs.get("l_global") is not None:
			self.c_global = kwargs.get("c_global")
			self.l_global = kwargs.get("l_global" , MLTensor( [] , [1 for _ in range(self.lhs_.n_lhs)] , n_features = self.lhs_.n_lhs ) )
			self.l_global._n_samples = self.lhs_.n_samples
			return
		
		## Else loop on lhs to find global parameters
		##===========================================
		self.c_global = []
		self.s_global = []
		l_global      = [] ## This list will be passed to MLTensor
		for lhs in self.lhs_.names:
			## Start with covariate
			if kwargs.get("c_{}".format(lhs)) is not None:
				c = kwargs["c_{}".format(lhs)].squeeze()
				if c.ndim == 1: c = c.reshape(-1,1)
				self.c_global.append(c)
				self.s_global.append(1 + c.shape[1])
			## No covariate, two choices : lhs is 1d or fixed
			else:
				self.c_global.append(None)
				if kwargs.get("f_{}".format(lhs)) is not None:
					self.s_global.append(0)
					self.lhs_._fixed[lhs] = True
				else:
					self.s_global.append(1)
			
			## Now the link functions
			if kwargs.get("f_{}".format(lhs)) is not None:
				l_global.append( MLConstant( kwargs["f_{}".format(lhs)] , n_samples = self.lhs_.n_samples ) )
			else:
				l = kwargs.get("l_{}".format(lhs))
				if l is None or issubclass(l.__class__,UnivariateLink):
					l = MLLinear( c = self.c_global[-1] , l = l , n_samples = self.lhs_.n_samples )
				l_global.append(l)
		
		self.l_global = MLTensor( l_global , self.s_global , n_features = np.sum(self.s_global) , n_samples = self.lhs_.n_samples )
	##}}}
	
	## Properties
	## {{{
	
	@property
	def n_features(self):
		return self.l_global.n_features
	
	@property
	def coef_(self):
		return self._coef_
	
	@coef_.setter
	def coef_( self , coef_ ):
		self._coef_         = np.array( [coef_] ).squeeze().reshape(-1)
		self.lhs_.values_   = self.l_global.transform( self.coef_ , self.c_global )
		self.lhs_.jacobian_ = self.l_global.jacobian( self.coef_ , self.c_global )
	
	##}}}

class AbstractLaw:
	##{{{
	"""
	
	Attributes
	==========
	<param> : np.array
		Value of param fitted, can be loc, scale, name of param of law, etc.
	method : string
		method used to fit
	coef_  : numpy.ndarray
		Coefficients fitted
	info_  : AbstractLaw.Info
		Class containing info of the fit
	
	
	Fit method
	==========
	
	The method <law>.fit is generic, and takes arguments of the form
	<type for param>_<name of param>, see below.
	In case of Bayesian fit, some others optional parameters are available.
	
	Arguments
	---------
	Y         : numpy.ndarray
		Data to fit
	c_<param> : numpy.ndarray or None
		Covariate of a param to fit
	f_<param> : numpy.ndarray or None
		Fix value of a param
	l_<param> : SDFC.tools.LinkFct (optional)
		Link function of a param
	c_global  : list
		List of covariates, sorted by parameters. *_<param> are ignored if set
	l_global  : Inherit of SDFC.link.MultivariateLink
		Global link function. *_<param> are ignored if set
	
	
	Fit with bootstrap
	==================
	The method <law>.fit_bootstrap takes the same arguments that <law>.fit, with
	two new parameters:
	
	n_bootstrap : int
		Number of resampling
	
	
	Possible attributes of <law>.info_
	==================================
	<law>.info_.cov_         : numpy.array
		Covariance matrix, if MLE is used
	<law>.info_.mle_optim_result : scipy.optimize.OptimizeResult
		Result of the optimization of the MLE, if MLE is used
	<law>.info_.n_bootstrap  : int
		Number of resampling
	<law>.info_.coefs_bs_    : numpy.array[shape = (n_bootstrap,n_features)
		Coef fitted with the bootstrap
	
	
	Optional arguments for MLE fit
	==============================
	init : numpy.array
		Init value, optional
	mle_n_restart : integer
		The MLE needs to find a starting point before optimization, this
		parameter control how many time the fit is restarted with a new random
		starting point
	
	
	Optional arguments for Bayesian fit
	===================================
	prior : None or law or prior
		Prior for Bayesian fit, if None a Multivariate Normal law assuming
		independence between parameters is used, if you set it, this must be a
		class which implement the method logpdf(coef), returning the log of
		probability density function
	mcmc_init: None or vector of initial parameters
		Starting point of the MCMC algorithm. If None, prior.rvs() is called.
	transition: None or function
		Transition function for MCMC algorithm, if None is given a normal law
		N(0,0.1) is used.
	n_mcmc_drawn : None or integer
		Number of drawn for MCMC algorithm, if None, the value 10000 is used.
	
	
	Examples
	========
	Example with a Normal law:
	>>> _,X_loc,X_scale,_ = SDFC.Dataset.covariates(2500)
	>>> loc   = 1. + 0.8 * X_loc
	>>> scale = 0.08 * X_scale
	>>> 
	>>> Y = numpy.random.normal( loc = loc , scale = scale )
	>>> 
	>>> ## Define the Normal law estimator, with the MLE method
	>>> law = SDFC.Normal( method = "MLE" )
	>>>
	>>> ## Now perform the fit, c_loc is the covariate of loc, and c_scale the
	>>> ## covariate of scale, and we pass a link function to scale
	>>> law.fit( Y , c_loc = X_loc , c_scale = X_scale , l_scale = SDFC.link.Exponential() )
	>>> print(law.coef_)
	>>>
	>>> ## But we can assume that scale is stationary, so no covariates are given:
	>>> law.fit( Y , c_loc = X_loc , l_scale = SDFC.link.Exponential() )
	>>> print(law.coef_)
	>>>
	>>> ## Or the loc can be given, so we need to fit only the scale:
	>>> law.fit( Y , f_loc = loc , c_scale = X_scale )
	>>> print(law.coef_)
	"""
	##}}}
	#####################################################################
	
	class _Info(object):##{{{
		def __init__(self):
			pass
	##}}}
	
	
	## Init function
	##==============
	
	def __init__( self , names : list , method : str ): ##{{{
		"""
		Initialization of AbstractLaw
		
		Parameters
		----------
		names  : list
			List of names of the law, e.g. ["loc","scale"] for Normal law.
		method : string
			Method called to fit parameters
		
		"""
		self._method = method.lower()
		self._lhs    = LHS(names,0)
		self._rhs    = RHS(self._lhs)
		self.info_   = AbstractLaw._Info()
	##}}}
	
	
	## Properties
	##===========
	
	@property
	def method(self):##{{{
		return self._method
	##}}}
	
	@property
	def coef_(self):##{{{
		return self._rhs.coef_
	##}}}
	
	@coef_.setter
	def coef_( self , coef_ ): ##{{{
		self._rhs.coef_ = coef_
	##}}}
	
	@property
	def cov_(self):##{{{
		return self.info_.cov_
	##}}}
	
	
	## Fit functions
	##==============
	
	def _random_valid_point(self):##{{{
		"""
		Try to find a valid point in the neighborhood of self.coef_
		"""
		coef_ = self.coef_.copy()
		cov_  = 0.1 * np.identity(coef_.size)
		
		p_coef = coef_.copy()
		n_it   = 1
		while not (np.isfinite(self._negloglikelihood(p_coef)) and np.all(np.isfinite(self._gradient_nlll(p_coef))) ):
			if n_it % 100 == 0: cov_ *= 2
			p_coef = np.random.multivariate_normal( coef_ , cov_ )
			n_it += 1
		self.coef_ = p_coef
	##}}}
	
	def _fit_MLE( self , **kwargs ): ##{{{
		
		try:
			self.coef_ = np.array(kwargs["init"]).ravel()
		except:
			self._init_MLE()
            
		coef_ = self.coef_.copy()
		
		is_success = False
		n_test     = 0
		max_test   = kwargs.get( "mle_n_restart" )
		if max_test is None: max_test = 2
		
		while not is_success and n_test < max_test:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				self.info_.mle_optim_result = sco.minimize( self._negloglikelihood , self.coef_ , jac = self._gradient_nlll , method = "BFGS" )
			self.info_.cov_             = self.info_.mle_optim_result.hess_inv
			self.coef_                  = self.info_.mle_optim_result.x
			is_success                  = self.info_.mle_optim_result.success
			if not is_success:
				self._random_valid_point()
			n_test += 1
		
		if not is_success:
			self.coef_ = coef_
		
	##}}}
	
	def _fit_Bayesian( self , **kwargs ):##{{{
		## Find numbers of features
		##=========================
		n_features = self._rhs.n_features
		
		## Define prior
		##=============
		prior = kwargs.get("prior")
		if prior is None:
			prior = sc.multivariate_normal( mean = np.zeros(n_features) , cov = 10 * np.identity(n_features) )
		
		## Define transition
		##==================
		transition = kwargs.get("transition")
		if transition is None:
			transition = lambda x : x + np.random.normal( scale = np.sqrt(np.diag(prior.cov)) / 5 )
		
		## Define numbers of iterations of MCMC algorithm
		##===============================================
		n_mcmc_drawn = kwargs.get("n_mcmc_drawn")
		if n_mcmc_drawn is None:
			n_mcmc_drawn = 10000
		
		## MCMC algorithm
		##===============
		draw   = np.zeros( (n_mcmc_drawn,n_features) )
		accept = np.zeros( n_mcmc_drawn , dtype = bool )
		
		## Init values
		##============
		init = kwargs.get("mcmc_init")
		if init is None:
			init = prior.rvs()
		
		draw[0,:]     = init
		lll_current   = -self._negloglikelihood(draw[0,:])
		prior_current = prior.logpdf(draw[0,:]).sum()
		p_current     = prior_current + lll_current
		
		for i in range(1,n_mcmc_drawn):
			draw[i,:] = transition(draw[i-1,:])
			
			## Likelihood and probability of new points
			lll_next   = - self._negloglikelihood(draw[i,:])
			prior_next = prior.logpdf(draw[i,:]).sum()
			p_next     = prior_next + lll_next
			
			## Accept or not ?
			p_accept = np.exp( p_next - p_current )
			if np.random.uniform() < p_accept:
				lll_current   = lll_next
				prior_current = prior_next
				p_current     = p_next
				accept[i] = True
			else:
				draw[i,:] = draw[i-1,:]
				accept[i] = False
		
		self.coef_ = np.mean( draw[int(n_mcmc_drawn/2):,:] , axis = 0 )
		
		## Update information
		self.info_.draw         = draw
		self.info_.accept       = accept
		self.info_.n_mcmc_drawn = n_mcmc_drawn
		self.info_.rate_accept  = np.sum(accept) / n_mcmc_drawn
		self.info_._cov         = np.cov(draw.T)
	##}}}
	
	def fit( self , Y , **kwargs ): ##{{{
		"""
		<law>.fit
		=========
		
		See global documentation for parameters
		"""
		
		## Add Y
		self._Y = Y.reshape(-1,1)
		
		## Init LHS/RHS
		self._lhs.n_samples = Y.size
		self._rhs.build(**kwargs)
		
		if self._rhs.n_features == 0:
			raise ValueError("All parameters are fixed (n_features == 0), no fit")
		
		self.coef_ = np.zeros(self._rhs.n_features)
		## Now fit
		if self._method not in ["mle","bayesian","bayesian-experimental"] and self._rhs.l_global._special_fit_allowed:
			self._special_fit()
		elif self._method == "mle" :
			self._fit_MLE(**kwargs)
            #print('MLE triggered')
		else:
			self._fit_Bayesian(**kwargs)
		
		
	##}}}
	
	def fit_bootstrap( self , Y , n_bootstrap , **kwargs ):##{{{
		"""
		<law>.fit_bootstrap
		===================
		
		See global documentation for parameters
		"""
		
		atleast2d = lambda x : x.reshape(-1,1) if x.ndim == 1 else x
		
		Y_       = atleast2d(Y)
		n_sample = Y_.size
		idxs     = np.random.choice( n_sample , n_sample * (n_bootstrap + 1) , replace = True ).reshape((n_bootstrap+1,n_sample))
		idxs[0,:] = range(n_sample)
		
		coefs_bs = []
		kwargs_bs = kwargs.copy()
		for i in range(n_bootstrap):
			
			idx = idxs[i,:]
			
			if "l_global" in kwargs:
				kwargs_bs["l_global"] = kwargs["l_global"]
				if kwargs.get("c_global") is not None:
					kwargs_bs["c_global"] = []
					for c in kwargs["c_global"]:
						if c is None:
							kwargs_bs["c_global"].append(None)
						else:
							
							kwargs_bs["c_global"].append( atleast2d(c)[idx,:])
			else:
				for lhs in self._lhs.names:
					if kwargs.get( "c_{}".format(lhs) ) is not None:
						kwargs_bs["c_{}".format(lhs)] = atleast2d(kwargs[f"c_{lhs}"])[idx,:]
					if kwargs.get( "f_{}".format(lhs) ) is not None:
						kwargs_bs["f_{}".format(lhs)] = atleast2d(kwargs[f"f_{lhs}"])[idx,:]
			
			if len(coefs_bs) > 0:
				kwargs_bs["init"] = coefs_bs[0]
			
			self.fit( Y_[idx,:] , **kwargs_bs )
			coefs_bs.append(self.coef_.copy())
		
		self.info_.n_bootstrap  = n_bootstrap
		self.info_.coefs_bs_    = np.array(coefs_bs)
		
		self.fit( Y , **kwargs )
	##}}}

class GEV(AbstractLaw):
	"""
	Class to fit a GEV law with covariates, available methods are:
	
	moments  : use empirical estimator of mean and standard deviation to find
	           loc and scale, possibly with least square regression if
	           covariates are given
	lmoments : Use L-Moments estimation, only in stationary context
	lmoments_experimental: Use non-stationary L-Moments with Quantile
	           Regression, experimental and not published, only
	           used to find an initialization of MLE
	bayesian : Bayesian estimation, i.e. the coefficient fitted is the mean of
	           n_mcmc_iteration sample draw from the posterior P(coef_ | Y)
	mle      : Maximum likelihood estimation
	
	Parameters
	==========
	loc   : location parameter
	scale : scale parameter
	shape : shape parameter
	
	Warning
	=======
	The shape parameter is the opposite of the shape parameter from scipy:
	GEV ~ scipy.stats.genextreme( loc = loc , scale = scale , c = - shape )
	"""
	__doc__ += AbstractLaw.__doc__
	
	def __init__( self , method = "MLE" ):##{{{
		"""
		Initialization of GEV law
		
		Parameters
		----------
		method         : string
			Method called to fit parameters
		"""
		AbstractLaw.__init__( self , ["loc","scale","shape"] , method )
	##}}}
	
	## Properties
	##===========
	
	@property
	def loc(self):##{{{
		return self._lhs.values_["loc"]
	##}}}
	
	@property
	def scale(self):##{{{
		return self._lhs.values_["scale"]
	##}}}
	
	@property
	def shape(self):##{{{
		return self._lhs.values_["shape"]
	##}}}
	
	@property
	def upper_bound(self):##{{{
		return np.where( self.shape < 0 , self.loc - self.scale / self.shape , np.inf )
	##}}}
	
	@property
	def lower_bound(self):##{{{
		return np.where( self.shape < 0 , - np.inf , self.loc - self.scale / self.shape )
	##}}}
	
	## Fit methods
	##============
	
	def _fit_moments(self):##{{{
		
		coefs = np.zeros(self._rhs.n_features)
		m = np.mean(self._Y)
		s = np.sqrt(6) * np.std(self._Y) / np.pi
		
		iloc   = m - 0.57722 * s
		iscale = max( 0.1 , np.log(s) )
		ishape = 1e-8
		
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		
		## Fit scale
		if not self._lhs.is_fixed("scale"):
			coefs[isc_b] = self._rhs.l_global._l_p[1]._l.inverse(iscale)
		
		## Fit loc
		if not self._lhs.is_fixed("loc"):
			if self._lhs.is_fixed("scale"):
				iloc = m - 0.57722 * np.exp(self.scale)
				coefs[il_b:il_e] = mean( iloc , self._rhs.c_global[0] , value = False , link = self._rhs.l_global._l_p[0]._l )
			else:
				coefs[il_b] = self._rhs.l_global._l_p[1]._l.inverse(iloc)
		
		## Fit shape
		if not self._lhs.is_fixed("shape"):
			coefs[ish_b] = self._rhs.l_global._l_p[2]._l.inverse(ishape)
		
		self.coef_ = coefs
	##}}}
	
	def _fit_lmoments( self ): ##{{{
		
		coefs = np.zeros(self._rhs.n_features)
		
		lmom = lmoments( self._Y )
		
		tau3  = lmom[2] / lmom[1]
		co    = 2. / ( 3. + tau3 ) - np.log(2) / np.log(3)
		kappa = 7.8590 * co + 2.9554 * co**2
		g     = scs.gamma( 1. + kappa )
		
		
		iscale = lmom[1] * kappa / ( (1 - np.power( 2 , - kappa )) * g )
		iloc   = lmom[0] - iscale * (1 - g) / kappa
		ishape = - kappa
		
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		
		## Fit scale
		if not self._lhs.is_fixed("scale"):
			coefs[isc_b] = self._rhs.l_global._l_p[1]._l.inverse(iscale)
		
		## Fit loc
		if not self._lhs.is_fixed("loc"):
			if self._lhs.is_fixed("scale"):
				iloc = lmom[0] - self.scale.squeeze() * (1 - g) / kappa
				coefs[il_b:il_e] = mean( iloc , self._rhs.c_global[0] , value = False , link = self._rhs.l_global._l_p[0]._l )
			else:
				coefs[il_b] = self._rhs.l_global._l_p[1]._l.inverse(iloc)
		
		## Fit shape
		if not self._lhs.is_fixed("shape"):
			coefs[ish_b] = self._rhs.l_global._l_p[2]._l.inverse(ishape)
		
		self.coef_ = coefs
		
	##}}}
	
	def _fit_lmoments_experimental(self):##{{{
		
		## First step, find lmoments
		try:
			c_Y  = np.ones((self._Y.size,1))
			for c in self._rhs.c_global:
				if c is None: continue
				if c.ndim == 1: c = c.reshape(-1,1)
				for i in range(c.shape[1]):
					c_Y2 = np.hstack( [c_Y,c[:,i].reshape(-1,1)] )
					if np.linalg.matrix_rank(c_Y2) > c_Y.shape[1]:
						c_Y = c_Y2
			if c_Y.shape[1] > 1:
				c_Y = c_Y[:,1:]
			else:
				c_Y = None
		except:
			c_Y = None
		if c_Y is None or c_Y.size == 0:
			self._fit_lmoments()
			return
		
		lmom = lmoments( self._Y , c_Y )
		
		## Find shape
		def uni_shape_solver(tau):
			bl,bu=-1,1
			fct = lambda x : 3 / 2 + tau / 2 - ( 1 - 3**x ) / (1 - 2**x )
			while fct(bl) * fct(bu) > 0:
				bl *= 2
				bu *= 2
			opt = sco.root_scalar( fct , method = "brenth" , bracket = [bl , bu] )
			return opt.root
		shape_solver = np.vectorize(uni_shape_solver)
		tau3 = lmom[:,2] / lmom[:,1]
		try:
			shape = shape_solver(tau3)
		except:
			co    = 2. / ( 3. + tau3 ) - np.log(2) / np.log(3)
			shape = - 7.8590 * co - 2.9554 * co**2
		
		## Find scale
		gshape = scs.gamma( 1 - shape )
		scale = - lmom[:,1] * shape / ( gshape * ( 1 - 2**shape ) )
		
		if not ~(scale.min() > 0):
			idx = ~(scale > 0)
			scale[idx] = 1e-3
		
		## Find loc
		loc = lmom[:,0] - scale * ( gshape - 1 ) / shape
		
		## And now find coefs
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		coefs = np.array([])
		if not self._lhs.is_fixed("loc"):
			coefs = np.hstack( (coefs,mean( loc , self._rhs.c_global[0] , value = False , link = self._rhs.l_global._l_p[0]._l )) )
		if not self._lhs.is_fixed("scale"):
			coefs = np.hstack( (coefs,mean( scale , self._rhs.c_global[1] , value = False , link = self._rhs.l_global._l_p[1]._l )) )
		if not self._lhs.is_fixed("shape"):
			coefs = np.hstack( (coefs,mean( shape , self._rhs.c_global[2] , value = False , link = self._rhs.l_global._l_p[2]._l )) )
		
		self.coef_ = coefs
	##}}}
	
	def _fit_last_chance(self): ##{{{
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		coefs = np.zeros(ish_e)
		if not self._lhs.is_fixed("scale"):
			coefs[isc_b] = 0.5
		if not self._lhs.is_fixed("shape"):
			coefs[ish_b] = -0.1
		self.coef_ = coefs
	##}}}
	
	def _special_fit( self ):##{{{
		if self.method == "moments":
			self._fit_moments()
		elif self.method == "lmoments":
			self._fit_lmoments()
		elif self.method == "lmoments-experimental":
			self._fit_lmoments_experimental()
		elif self.method == "last-chance":
			self._fit_last_chance()
	##}}}
	
	def _init_MLE( self ): ##{{{
		if self._rhs.l_global._special_fit_allowed:
			try:
				self._fit_lmoments_experimental()
			except:
				self._fit_last_chance()
		else:
			self.coef_ = self._rhs.l_global.valid_point( self )
	##}}}
	
	
	def _logZafun( self, Z , alpha ):##{{{
		return alpha * np.log( 1. + self.shape * Z )
	##}}}
	
	def _Zafun( self , Z , alpha ):##{{{
		return np.exp( self._logZafun( Z , alpha ) )
	##}}}
	
	def _negloglikelihood( self , coef ): ##{{{
		
		self.coef_ = coef
		
		## Impossible scale
		if not np.all( self.scale > 0 ):
			return np.inf
		
		## Remove exponential case
		zero_shape = ( np.abs(self.shape) < 1e-10 )
		shape = self.shape
		if np.any(zero_shape):
			shape[zero_shape] = 1e-10
		
		dshape = self._Y.shape
		loc    = self.loc.reshape(dshape)
		scale  = self.scale.reshape(dshape)
		shape  = shape.reshape(dshape)
		
		##
		Z = 1 + shape * ( self._Y - loc ) / scale
		
		if not np.all(Z > 0):
			return np.inf
		
		res = np.sum( ( 1. + 1. / shape ) * np.log(Z) + np.power( Z , - 1. / shape ) + np.log(scale) )
		
		
		return res if np.isfinite(res) else np.inf
	##}}}
	
	def _gradient_nlll( self , coef ): ##{{{
		self.coef_ = coef
		
		## Parameters
		dshape = self._Y.shape
		loc   = self.loc.reshape(dshape)
		scale = self.scale.reshape(dshape)
		shape = self.shape.reshape(dshape)
		shc   = 1 + 1 / shape
		Z     = ( self._Y - loc ) / scale
		ZZ    = 1 + shape * Z
		ZZi   = np.power( ZZ ,  - 1 / shape )
		ZZim1 = np.power( ZZ ,  - shc )
		
		## Compute gradient
		T0 = ZZim1 / scale - shc * shape / ( ZZ * scale )
		T1 = 1 / scale + ZZim1 * Z / scale - shc * shape * Z / ( ZZ * scale )
		T2 = np.log(ZZ) * ZZi / shape**2 - ZZim1 * Z / shape - np.log(ZZ) / shape**2 + shc * Z / ZZ
		
		for T in [T0,T1,T2]:
			if not np.isfinite(T).all():
				return np.zeros_like(self.coef_) + np.nan
		
		
		jac = self._lhs.jacobian_
		p = 0
		if not self._lhs.is_fixed("loc"):
			jac[p,:,:] *= T0
			p += 1
		if not self._lhs.is_fixed("scale"):
			jac[p,:,:] *= T1
			p += 1
		if not self._lhs.is_fixed("shape"):
			jac[p,:,:] *= T2
		
		return jac.sum( axis = (0,1) )
	##}}}

###############
## Functions ##
###############

def quantile( Y , ltau , c_Y = None , value = True ):
	"""
	SDFC.NonParametric.quantile
	===========================
	
	Estimate quantile given a covariate (or not)
	
	Parameters
	----------
	Y       : np.array
		Dataset to fit the quantile
	ltau    : np.array
		The quantile to fit, between 0 and 1
	c_Y   : np.array or None
		Covariate(s)
	link  : class based on SDFC.tools.Link
		Link function, default is identity
	value : bool
		If true return value fitted, else return coefficients of fit
	
	Returns
	-------
	The quantiles
	"""
	
	ltau = np.array( [ltau] ).ravel()
	q    = None
	coef = None
	
	if c_Y is None:
		q    = np.percentile( Y , 100 * ltau )
		coef = q.copy()
	else:
        # Sklearn Quantile Regressor 
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html
		reg = QuantileRegressor( quantile = ltau )
        # C++ in SDFC
		#reg  = QuantileRegression( ltau = ltau )
		reg.fit( Y , c_Y )
		#q    = reg.quantiles
		coef = reg.coef_
	return q if value else coef

def lmoments_matrix( size ):##{{{
	"""
		SDFC.NonParametric.lmoments_matrix
		==================================
		
		Build a matrix to infer L-Moments in stationary case. If M = lmoments_matrix(Y.size), then
		the fourth first L-Moments are just M.T @ np.sort(Y)
		
	"""
	C0 = scs.binom( range( size ) , 1 )
	C1 = scs.binom( range( size - 1 , -1 , -1 ) , 1 )
	
	## Order 3
	C2 = scs.binom( range( size ) , 2 )
	C3 = scs.binom( range( size - 1 , -1 , -1 ) , 2 )
	
	## Order 4
	C4 = scs.binom( range( size ) , 3 )
	C5 = scs.binom( range( size - 1 , -1 , -1 ) , 3 )
	
	M = np.zeros( (size,4) )
	M[:,0] = 1. / size
	M[:,1] = ( C0 - C1 ) / ( 2 * scs.binom( size , 2 ) )
	M[:,2] = ( C2 - 2 * C0 * C1 + C3 ) / ( 3 * scs.binom( size , 3 ) )
	M[:,3] = ( C4 - 3 * C2 * C1 + 3 * C0 * C3 - C5 ) / ( 4 * scs.binom( size , 4 ) )
	
	return M
##}}}

def _lmoments_stationary( Y ):##{{{
	Ys = np.sort(Y.squeeze())
	M = lmoments_matrix( Y.size )
	return M.T @ Ys
##}}}

def lmoments( Y , c_Y = None , order = None , lq = np.arange( 0.05 , 0.96 , 0.01 ) ):##{{{
	"""
	SDFC.NonParametric.lmoments
	===========================
	
	Estimate the lmoments of orders 1 to 4. If a covariate is given, a quantile regression is performed
	and the instantaneous L-Moments are estimated from the quantile fitted.
	
	Parameters
	----------
	Y     : np.array
		Dataset to fit the lmoments
	c_Y   : np.array or None
		Covariate
	order : integer, list of integer or None
		Integers between 1 and 4
	lq    : np.array
		Quantiles for quantile regression, only used if a covariate is given. Default is np.arange(0.05,0.96,0.01)
	
	Returns
	-------
	The lmoments.
	"""
	
	order = order if order is None else np.array( [order] , dtype = np.int ).squeeze() - 1
	
	if c_Y is None:
		lmom = _lmoments_stationary(Y)
		return lmom if order is None else lmom[order]
	else:
		Y = Y.reshape(-1,1)
		if c_Y.ndim == 1: c_Y = c_Y.reshape(-1,1)
		Yq = quantile( Y , lq , c_Y )
		M  = lmoments_matrix(Yq.shape[1])
		lmom = np.transpose( M.T @ Yq.T )
		if order is None:
			return lmom
		else:
			return lmom[:,order]
##}}}


def lmoments_matrix( size ):##{{{
	"""
		SDFC.NonParametric.lmoments_matrix
		==================================
		
		Build a matrix to infer L-Moments in stationary case. If M = lmoments_matrix(Y.size), then
		the fourth first L-Moments are just M.T @ np.sort(Y)
		
	"""
	C0 = scs.binom( range( size ) , 1 )
	C1 = scs.binom( range( size - 1 , -1 , -1 ) , 1 )
	
	## Order 3
	C2 = scs.binom( range( size ) , 2 )
	C3 = scs.binom( range( size - 1 , -1 , -1 ) , 2 )
	
	## Order 4
	C4 = scs.binom( range( size ) , 3 )
	C5 = scs.binom( range( size - 1 , -1 , -1 ) , 3 )
	
	M = np.zeros( (size,4) )
	M[:,0] = 1. / size
	M[:,1] = ( C0 - C1 ) / ( 2 * scs.binom( size , 2 ) )
	M[:,2] = ( C2 - 2 * C0 * C1 + C3 ) / ( 3 * scs.binom( size , 3 ) )
	M[:,3] = ( C4 - 3 * C2 * C1 + 3 * C0 * C3 - C5 ) / ( 4 * scs.binom( size , 4 ) )
	
	return M
##}}}

def _lmoments_stationary( Y ):##{{{
	Ys = np.sort(Y.squeeze())
	M = lmoments_matrix( Y.size )
	return M.T @ Ys
##}}}

def mean( Y , c_Y = None , link = ULIdentity() , value = True ):
	"""
	SDFC.NonParametric.mean
	=======================
	
	Estimate the mean
	
	Parameters
	----------
	Y     : np.array
		Dataset to fit the mean
	c_Y   : np.array or None
		Covariate(s)
	link  : class based on SDFC.tools.Link
		Link function, default is identity
	value : bool
		If true return value fitted, else return coefficients of fit
	
	Returns
	-------
	The mean or the coefficients of the regression
	"""
	out,coef = None,None
	if c_Y is None:
		out = np.mean(Y)
		coef = link.inverse(out)
	else:
		size = c_Y.shape[0]
		if c_Y.ndim == 1:
			c_Y = c_Y.reshape(-1,1)
		design = np.hstack( ( np.ones((Y.size,1)) , c_Y ) )
		coef,_,_,_ = scl.lstsq( design , link.inverse(Y) )
		out = link( design @ coef )
	return out if value else coef	