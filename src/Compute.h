/*
 *  Copyright 2013 The OpenMx Project
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _OMX_COMPUTE_H_
#define _OMX_COMPUTE_H_

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include "omxDefines.h"
#include "Eigen/SparseCore"
#include "glue.h"

// See R/MxRunHelperFunctions.R npsolMessages
enum ComputeInform {
	INFORM_UNINITIALIZED = -1,
	INFORM_CONVERGED_OPTIMUM = 0,
	INFORM_UNCONVERGED_OPTIMUM = 1,
	// The final iterate satisfies the optimality conditions to the accuracy requested,
	// but the sequence of iterates has not yet converged.
	// Optimizer was terminated because no further improvement
	// could be made in the merit function (Mx status GREEN).
	INFORM_LINEAR_CONSTRAINTS_INFEASIBLE = 2,
	// The linear constraints and bounds could not be satisfied.
	// The problem has no feasible solution.
	INFORM_NONLINEAR_CONSTRAINTS_INFEASIBLE = 3,
	// The nonlinear constraints and bounds could not be satisfied.
	// The problem may have no feasible solution.
	INFORM_ITERATION_LIMIT = 4,
	// The major iteration limit was reached (Mx status BLUE).
	INFORM_NOT_AT_OPTIMUM = 6,
	// The model does not satisfy the first-order optimality conditions
	// to the required accuracy, and no improved point for the
	// merit function could be found during the final linesearch (Mx status RED)
	INFORM_BAD_DERIVATIVES = 7,
	// The function derivates returned by funcon or funobj
	// appear to be incorrect.
	INFORM_INVALID_PARAM = 9
	// An input parameter was invalid'
};

enum ComputeInfoMethod {
	INFO_METHOD_DEFAULT,
	INFO_METHOD_HESSIAN,
	INFO_METHOD_SANDWICH,
	INFO_METHOD_BREAD,
	INFO_METHOD_MEAT
};

struct HessianBlock {
	std::vector<int> vars;  // global freeVar ID in order
	Eigen::MatrixXd mat;    // vars * vars, only upper triangle referenced

	HessianBlock() {}
	HessianBlock *clone();
	bool posDefinite();
};

// The idea of FitContext is to eventually enable fitting from
// multiple starting values in parallel.

class FitContext {
	static omxFitFunction *RFitFunction;

	FitContext *parent;

	std::vector<HessianBlock*> allBlocks;
	//bool overlap;

	bool haveSparseHess;
	Eigen::SparseMatrix<double> sparseHess;
	bool haveSparseIHess;
	Eigen::SparseMatrix<double> sparseIHess;

	bool haveDenseHess;
	Eigen::MatrixXd hess;
	bool haveDenseIHess;
	Eigen::MatrixXd ihess;

 public:
	FreeVarGroup *varGroup;
	size_t numParam;               // cached from varGroup
	std::vector<int> mapToParent;
	double mac;
	double fit;
	double *est;
	// We need some protocol to manage flavor assignment
	// when the multigroup fitfunction is involved. TODO
	int *flavor;
	Eigen::VectorXd grad;
	int infoDefinite;
	double infoCondNum;
	double *stderrs;   // plural to distinguish from stdio's stderr
	enum ComputeInfoMethod infoMethod;
	double *infoA; // sandwich, the bread
	double *infoB; // sandwich, the meat
	std::vector<double> caution;
	int iterations;
	enum ComputeInform inform;
	int wanted;

	void init();
	FitContext(std::vector<double> &startingValues);
	FitContext(FitContext *parent, FreeVarGroup *group);
	void allocStderrs();
	void copyParamToModel(omxState* os, double *at);
	void copyParamToModel(omxState *os);
	void copyParamToModel(omxMatrix *mat, double *at);
	void copyParamToModel(omxMatrix *mat);
	double *take(int want);
	void maybeCopyParamToModel(omxState* os);
	void updateParent();
	void updateParentAndFree();
	void log(const char *where);
	void log(const char *where, int what);
	~FitContext();
	
	// deriv related
	void clearHessian();
	void negateHessian();
	void queue(HessianBlock *hb);
	void refreshDenseHess();
	void refreshDenseIHess();
	Eigen::VectorXd ihessGradProd();
	double *getDenseHessUninitialized();
	double *getDenseIHessUninitialized();
	double *getDenseHessianish();  // either a Hessian or inverse Hessian, remove TODO
	void copyDenseHess(double *dest);
	void copyDenseIHess(double *dest);
	Eigen::VectorXd ihessDiag();
	void preInfo();
	void postInfo();

	static void cacheFreeVarDependencies();
	static void setRFitFunction(omxFitFunction *rff);
};

typedef std::vector< std::pair<int, MxRList*> > LocalComputeResult;

class omxCompute {
	int computeId;
 protected:
        virtual void reportResults(FitContext *fc, MxRList *slots, MxRList *glob) {};
	void collectResultsHelper(FitContext *fc, std::vector< omxCompute* > &clist,
				  LocalComputeResult *lcr, MxRList *out);
	static enum ComputeInfoMethod stringToInfoMethod(const char *iMethod);
 public:
	FreeVarGroup *varGroup;
	omxCompute();
        virtual void initFromFrontend(SEXP rObj);
        virtual omxFitFunction *getFitFunction() { return NULL; }
        void compute(FitContext *fc);
        virtual void computeImpl(FitContext *fc) {}
	virtual void collectResults(FitContext *fc, LocalComputeResult *lcr, MxRList *out);
	virtual double getOptimizerStatus() { return NA_REAL; }  // backward compatibility
        virtual ~omxCompute();
};

class Ramsay1975 {
	// Ramsay, J. O. (1975). Solving Implicit Equations in
	// Psychometric Data Analysis.  Psychometrika, 40(3), 337-360.

	FitContext *fc;
	size_t numParam;
	int flavor;
	int verbose;
	int boundsHit;
	double minCaution;
	double highWatermark;
	std::vector<int> vars;
	std::vector<double> prevEst;
	std::vector<double> prevAdj1;
	std::vector<double> prevAdj2;
	bool goingWild;

public:
	double maxCaution;
	double caution;

	Ramsay1975(FitContext *fc, int flavor, double caution, int verbose, double minCaution);
	void recordEstimate(int px, double newEst);
	void apply();
	void recalibrate(bool *restart);
	void restart(bool myFault);
};

class omxCompute *omxNewCompute(omxState* os, const char *type);

class omxCompute *newComputeGradientDescent();
class omxCompute *newComputeNumericDeriv();
class omxCompute *newComputeNewtonRaphson();

void omxApproxInvertPosDefTriangular(int dim, double *hess, double *ihess, double *stress);
void omxApproxInvertPackedPosDefTriangular(int dim, int *mask, double *packedHess, double *stress);

#endif
