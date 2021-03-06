#
#   Copyright 2007-2016 The OpenMx Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# -----------------------------------------------------------------------------
# Program: TwoLocusLikelihood.R  
# Author: Michael Neale
# Date: 2009.11.23 
#
# ModelType: Locus Likelihood
# DataType: Frequencies
# Field: Human Behavior Genetics
#
# Purpose:
#      Two Locus Likelihood model to estimate allele frequencies
#      Bernstein data on ABO blood-groups
#      c.f. Edwards, AWF (1972)  Likelihood.  Cambridge Univ Press, pp. 39-41
#
# RevisionHistory:
#      Hermine Maes -- 2010.02.22 updated & reformatted
#      Ross Gore -- 2011.06.15 added Model, Data & Field
#      Hermine Maes -- 2014.11.02 piecewise specification
# -----------------------------------------------------------------------------


require(OpenMx)
# Load Library
# -----------------------------------------------------------------------------
	
# Matrices for allele frequencies, p and s
matP         <- mxMatrix( type="Full", nrow=1, ncol=1, 
                          free=TRUE, values=c(.3333), name="P")
matS         <- mxMatrix( type="Full", nrow=1, ncol=1, 
                          free=TRUE, values=c(.3333), name="S")
# Matrix of observed data    
obsFreq      <- mxMatrix( type="Full", nrow=4, ncol=1, 
                          values=c(211,104,39,148), name="ObservedFreqs")
matQ         <- mxAlgebra( expression=1-P, name="Q")
matT         <- mxAlgebra( expression=1-S, name="T")
# Algebra for predicted proportions
expFreq      <- mxAlgebra( rbind ((P*P+2*P*Q)*T*T, (Q*Q)*(S*S+2*S*T), 
                          (P*P+2*P*Q)*(S*S+2*S*T), (Q*Q)*(T*T)), name="ExpectedFreqs")
# Algebra for -2logLikelihood
m2ll         <- mxAlgebra( expression=-(sum(log(ExpectedFreqs) 
                          * ObservedFreqs)), name="NegativeLogLikelihood")
# User-defined objective
funAl        <- mxFitFunctionAlgebra("NegativeLogLikelihood") 

TwoLocusModel <- mxModel("TwoLocus",
                          matP, matS, matQ, matT, obsFreq, expFreq, m2ll, funAl)

# Create an MxModel object
# -----------------------------------------------------------------------------

TwoLocusFit<-mxRun(TwoLocusModel, suppressWarnings=TRUE)
TwoLocusFit$matrices
TwoLocusFit$algebras


estimates <- c(
    TwoLocusFit$matrices$P$values, 
    TwoLocusFit$matrices$S$values, 
    TwoLocusFit$algebras$NegativeLogLikelihood$result)
Mx1Estimates<-c(0.2915,0.1543,647.894)

omxCheckCloseEnough(estimates,Mx1Estimates,.01)
# Compare OpenMx estimates to original Mx's estimates
# -----------------------------------------------------------------------------
