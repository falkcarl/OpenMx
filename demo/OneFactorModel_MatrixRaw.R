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
# Program: OneFactorModel_MatrixRaw.R  
# Author: Ryne Estabrook
# Date: 2009.08.01 
#
# ModelType: Factor
# DataType: Continuous
# Field: None
#
# Purpose: 
#      One Factor model to estimate factor loadings, residual variances and means
#      Matrix style model input - Raw data input
#
# RevisionHistory:
#      Hermine Maes -- 2009.10.08 updated & reformatted
#      Ross Gore -- 2011.06.06 added Model, Data & Field metadata
#      Hermine Maes -- 2014.11.02 piecewise specification
# -----------------------------------------------------------------------------

require(OpenMx)
# Load Library
# -----------------------------------------------------------------------------


data(myFADataRaw)
# Prepare Data
# -----------------------------------------------------------------------------

manifestVars <- c("x1","x2","x3","x4","x5","x6")
latentVars <- "F1"

myFADataRaw <- myFADataRaw[,manifestVars]

dataRaw      <- mxData( observed=myFADataRaw, type="raw" )
matrA        <- mxMatrix( type="Full", nrow=7, ncol=7,
                          free=  c(F,F,F,F,F,F,F,
                                   F,F,F,F,F,F,T,
                                   F,F,F,F,F,F,T,
                                   F,F,F,F,F,F,T,
                                   F,F,F,F,F,F,T,
                                   F,F,F,F,F,F,T,
                                   F,F,F,F,F,F,F),
                          values=c(0,0,0,0,0,0,1,
                                   0,0,0,0,0,0,1,
                                   0,0,0,0,0,0,1,
                                   0,0,0,0,0,0,1,
                                   0,0,0,0,0,0,1,
                                   0,0,0,0,0,0,1,
                                   0,0,0,0,0,0,0),
                          labels=c(NA,NA,NA,NA,NA,NA,"l1",
                                   NA,NA,NA,NA,NA,NA,"l2",
                                   NA,NA,NA,NA,NA,NA,"l3",
                                   NA,NA,NA,NA,NA,NA,"l4",
                                   NA,NA,NA,NA,NA,NA,"l5",
                                   NA,NA,NA,NA,NA,NA,"l6",
                                   NA,NA,NA,NA,NA,NA,NA),
                          byrow=TRUE, name="A" )
matrS        <- mxMatrix( type="Symm", nrow=7, ncol=7, 
                          free=  c(T,F,F,F,F,F,F,
                                   F,T,F,F,F,F,F,
                                   F,F,T,F,F,F,F,
                                   F,F,F,T,F,F,F,
                                   F,F,F,F,T,F,F,
                                   F,F,F,F,F,T,F,
                                   F,F,F,F,F,F,T),
                          values=c(1,0,0,0,0,0,0,
                                   0,1,0,0,0,0,0,
                                   0,0,1,0,0,0,0,
                                   0,0,0,1,0,0,0,
                                   0,0,0,0,1,0,0,
                                   0,0,0,0,0,1,0,
                                   0,0,0,0,0,0,1),
                          labels=c("e1",NA,  NA,  NA,  NA,  NA,  NA,
                                   NA, "e2", NA,  NA,  NA,  NA,  NA,
                                   NA,  NA, "e3", NA,  NA,  NA,  NA,
                                   NA,  NA,  NA, "e4", NA,  NA,  NA,
                                   NA,  NA,  NA,  NA, "e5", NA,  NA,
                                   NA,  NA,  NA,  NA,  NA, "e6", NA,
                                   NA,  NA,  NA,  NA,  NA,  NA, "varF1"),
                          byrow=TRUE, name="S" )
matrF        <- mxMatrix( type="Full", nrow=6, ncol=7,
                          free=FALSE,
                          values=c(1,0,0,0,0,0,0,
                                   0,1,0,0,0,0,0,
                                   0,0,1,0,0,0,0,
                                   0,0,0,1,0,0,0,
                                   0,0,0,0,1,0,0,
                                   0,0,0,0,0,1,0),
                          byrow=TRUE, name="F" )
matrM        <- mxMatrix( type="Full", nrow=1, ncol=7,
                          free=c(T,T,T,T,T,T,F),
                          values=c(1,1,1,1,1,1,0),
                          labels=c("meanx1","meanx2","meanx3",
                                   "meanx4","meanx5","meanx6",NA),
                          name="M" )
exp          <- mxExpectationRAM("A","S","F","M", 
                                  dimnames=c(manifestVars, latentVars))
funML        <- mxFitFunctionML()
oneFactorModel <- mxModel("Common Factor Model Matrix Specification", 
                          dataRaw, matrA, matrS, matrF, matrM, exp, funML)
# Create an MxModel object
# -----------------------------------------------------------------------------
      
oneFactorFit<-mxRun(oneFactorModel)

summary(oneFactorFit)
oneFactorFit$output$estimate

omxCheckCloseEnough(oneFactorFit$output$estimate[["l2"]], 0.999, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["l3"]], 0.959, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["l4"]], 1.028, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["l5"]], 1.008, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["l6"]], 1.021, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["varF1"]], 0.645, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["e1"]], 0.350, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["e2"]], 0.379, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["e3"]], 0.389, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["e4"]], 0.320, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["e5"]], 0.370, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["e6"]], 0.346, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["meanx1"]], 2.988, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["meanx2"]], 3.011, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["meanx3"]], 2.986, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["meanx4"]], 3.053, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["meanx5"]], 3.016, 0.01)
omxCheckCloseEnough(oneFactorFit$output$estimate[["meanx6"]], 3.010, 0.01)
# Compare OpenMx results to Mx results 
# -----------------------------------------------------------------------------