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

#include "omxDefines.h"
#include "omxState.h"
#include "omxFitFunction.h"
#include "omxNPSOLSpecific.h"
#include "omxExportBackendState.h"
#include "omxCsolnp.h"
#include "nloptcpp.h"
#include "Compute.h"
#include "npsolswitch.h"
#include "glue.h"
#include "ComputeSD.h"

enum OptEngine {
	OptEngine_NPSOL,
	OptEngine_CSOLNP,
    OptEngine_NLOPT,
    OptEngine_SD
};

class omxComputeGD : public omxCompute {
	typedef omxCompute super;
	const char *engineName;
	enum OptEngine engine;
	const char *gradientAlgoName;
	enum GradientAlgorithm gradientAlgo;
	int gradientIterations;
	double gradientStepSize;
	omxMatrix *fitMatrix;
	int verbose;
	double optimalityTolerance;
	int maxIter;

	bool useGradient;
	SEXP hessChol;
	bool nudge;

	int warmStartSize;
	double *warmStart;

public:
	omxComputeGD();
	virtual void initFromFrontend(omxState *, SEXP rObj);
	virtual void computeImpl(FitContext *fc);
	virtual void reportResults(FitContext *fc, MxRList *slots, MxRList *out);
};

class omxCompute *newComputeGradientDescent()
{
	return new omxComputeGD();
}

omxComputeGD::omxComputeGD()
{
	hessChol = NULL;
	warmStart = NULL;
}

void omxComputeGD::initFromFrontend(omxState *globalState, SEXP rObj)
{
	super::initFromFrontend(globalState, rObj);

	SEXP slotValue;
	fitMatrix = omxNewMatrixFromSlot(rObj, globalState, "fitfunction");
	setFreeVarGroup(fitMatrix->fitFunction, varGroup);
	omxCompleteFitFunction(fitMatrix);

	ScopedProtect p1(slotValue, R_do_slot(rObj, Rf_install("verbose")));
	verbose = Rf_asInteger(slotValue);

	ScopedProtect p2(slotValue, R_do_slot(rObj, Rf_install("tolerance")));
	optimalityTolerance = Rf_asReal(slotValue);
	if (!std::isfinite(optimalityTolerance)) {
		optimalityTolerance = Global->optimalityTolerance;
	}

	ScopedProtect p3(slotValue, R_do_slot(rObj, Rf_install("engine")));
	engineName = CHAR(Rf_asChar(slotValue));
	if (strEQ(engineName, "CSOLNP")) {
		engine = OptEngine_CSOLNP;
	} else if (strEQ(engineName, "SLSQP")) {
		engine = OptEngine_NLOPT;
	} else if (strEQ(engineName, "NPSOL")) {
#if HAS_NPSOL
		engine = OptEngine_NPSOL;
#else
		Rf_error("NPSOL is not available in this build. See ?omxGetNPSOL() to download this optimizer");
#endif
	} else if(strEQ(engineName, "SD")){
		engine = OptEngine_SD;
	} else {
		Rf_error("%s: engine %s unknown", name, engineName);
	}

	ScopedProtect p5(slotValue, R_do_slot(rObj, Rf_install("useGradient")));
	if (Rf_length(slotValue)) {
		useGradient = Rf_asLogical(slotValue);
	} else {
		useGradient = Global->analyticGradients;
	}

	ScopedProtect p4(slotValue, R_do_slot(rObj, Rf_install("nudgeZeroStarts")));
	nudge = Rf_asLogical(slotValue);

	ScopedProtect p6(slotValue, R_do_slot(rObj, Rf_install("warmStart")));
	if (!Rf_isNull(slotValue)) {
		SEXP matrixDims;
		Rf_protect(matrixDims = Rf_getAttrib(slotValue, R_DimSymbol));
		int *dimList = INTEGER(matrixDims);
		int rows = dimList[0];
		int cols = dimList[1];
		if (rows != cols) Rf_error("%s: warmStart matrix must be square", name);

		warmStartSize = rows;
		warmStart = REAL(slotValue);
	}

	ScopedProtect p7(slotValue, R_do_slot(rObj, Rf_install("maxMajorIter")));
	if (Rf_length(slotValue)) {
		maxIter = Rf_asInteger(slotValue);
	} else {
		maxIter = -1; // different engines have different defaults
	}

	ScopedProtect p8(slotValue, R_do_slot(rObj, Rf_install("gradientAlgo")));
	gradientAlgoName = CHAR(Rf_asChar(slotValue));
	if (strEQ(gradientAlgoName, "forward")) {
		gradientAlgo = GradientAlgorithm_Forward;
	} else if (strEQ(gradientAlgoName, "central")) {
		gradientAlgo = GradientAlgorithm_Central;
	} else {
		Rf_error("%s: gradient algorithm '%s' unknown", name, gradientAlgoName);
	}

	ScopedProtect p9(slotValue, R_do_slot(rObj, Rf_install("gradientIterations")));
	gradientIterations = std::max(Rf_asInteger(slotValue), 1);

	ScopedProtect p10(slotValue, R_do_slot(rObj, Rf_install("gradientStepSize")));
	gradientStepSize = Rf_asReal(slotValue);
}

void omxComputeGD::computeImpl(FitContext *fc)
{
	size_t numParam = varGroup->vars.size();
	if (numParam <= 0) {
		omxRaiseErrorf("%s: model has no free parameters", name);
		return;
	}

	for (int px = 0; px < int(numParam); ++px) {
		omxFreeVar *fv = varGroup->vars[px];
		if (nudge && fc->est[px] == 0.0) {
			fc->est[px] += 0.1;
		}
		if (fv->lbound > fc->est[px]) {
			fc->est[px] = fv->lbound + 1.0e-6;
		}
		if (fv->ubound < fc->est[px]) {
			fc->est[px] = fv->ubound - 1.0e-6;
		}
        }

	omxFitFunctionPreoptimize(fitMatrix->fitFunction, fc);

	fc->createChildren();

	int beforeEval = Global->computeCount;

	if (verbose >= 1) mxLog("%s: engine %s (ID %d) gradient=%s tol=%g",
				name, engineName, engine, gradientAlgoName, optimalityTolerance);

	//if (fc->CI) verbose=3;
	GradientOptimizerContext rf(verbose);
	rf.fc = fc;
	rf.fitMatrix = fitMatrix;
	rf.ControlTolerance = optimalityTolerance;
	rf.useGradient = useGradient;
	rf.gradientAlgo = gradientAlgo;
	rf.gradientIterations = gradientIterations;
	rf.gradientStepSize = gradientStepSize;
	if (maxIter == -1) {
		rf.maxMajorIterations = -1;
	} else {
		rf.maxMajorIterations = fc->iterations + maxIter;
	}
	if (warmStart) {
		if (warmStartSize != int(numParam)) {
			Rf_warning("%s: warmStart size %d does not match number of free parameters %d (ignored)",
				   warmStartSize, numParam);
		} else {
			// Not sure if this code path works, need test TODO
			Eigen::Map< Eigen::MatrixXd > hessWrap(warmStart, numParam, numParam);
			rf.hessOut = hessWrap;
			rf.warmStart = true;
		}
	}

	switch (engine) {
        case OptEngine_NPSOL:{
#if HAS_NPSOL
		omxNPSOL(fc->est, rf);
		fc->wanted |= FF_COMPUTE_GRADIENT;
		if (rf.hessOut.size() && fitMatrix->currentState->conList.size() == 0) {
			if (!hessChol) {
				Rf_protect(hessChol = Rf_allocMatrix(REALSXP, numParam, numParam));
			}
			Eigen::Map<Eigen::MatrixXd> hc(REAL(hessChol), numParam, numParam);
			hc = rf.hessOut;
			Eigen::Map<Eigen::MatrixXd> dest(fc->getDenseHessUninitialized(), numParam, numParam);
			dest.noalias() = rf.hessOut.transpose() * rf.hessOut;
			fc->wanted |= FF_COMPUTE_HESSIAN;
		}
#endif
		break;}
        case OptEngine_CSOLNP:
		rf.avoidRedundentEvals = true;
		omxCSOLNP(fc->est, rf);
		if (rf.gradOut.size()) {
			fc->grad = rf.gradOut.tail(numParam);
			Eigen::Map< Eigen::MatrixXd > hess(fc->getDenseHessUninitialized(), numParam, numParam);
			hess = rf.hessOut.bottomRightCorner(numParam, numParam);
			fc->wanted |= FF_COMPUTE_GRADIENT | FF_COMPUTE_HESSIAN;
		}
		break;
        case OptEngine_NLOPT:
		if (rf.maxMajorIterations == -1) rf.maxMajorIterations = Global->majorIterations;
		omxInvokeNLOPT(fc->est, rf);
		fc->wanted |= FF_COMPUTE_GRADIENT;
		break;
        case OptEngine_SD:{
		SteepestDescent(rf);
            break;
        }
        default: Rf_error("Optimizer %d is not available", engine);
	}

	fc->inform = rf.informOut;
	if (fc->inform <= 0 && Global->computeCount - beforeEval == 1) {
		fc->inform = INFORM_STARTING_VALUES_INFEASIBLE;
	}

	// Optimizers can terminate with inconsistent fit and parameters
	fc->copyParamToModel();
	ComputeFit(name, fitMatrix, FF_COMPUTE_FIT, fc);

	if (verbose >= 2) {
		mxLog("%s: final fit is %2f", name, fc->fit);
		fc->log(FF_COMPUTE_ESTIMATE);
	}

	fc->wanted |= FF_COMPUTE_BESTFIT;
}

void omxComputeGD::reportResults(FitContext *fc, MxRList *slots, MxRList *out)
{
	omxPopulateFitFunction(fitMatrix, out);

	if (engine == OptEngine_NPSOL && hessChol) {
		out->add("hessianCholesky", hessChol);
	}
}

// -----------------------------------------------------------------------

class ComputeCI : public omxCompute {
	typedef omxCompute super;
	omxCompute *plan;
	omxMatrix *fitMatrix;
	int verbose;
	SEXP intervals, intervalCodes, detail;
	const char *ctypeName;
	bool useInequality;
	bool useEquality;

public:
	ComputeCI();
	virtual void initFromFrontend(omxState *, SEXP rObj);
	virtual void computeImpl(FitContext *fc);
	virtual void reportResults(FitContext *fc, MxRList *slots, MxRList *out);
};

omxCompute *newComputeConfidenceInterval()
{
	return new ComputeCI();
}

ComputeCI::ComputeCI()
{
	intervals = 0;
	intervalCodes = 0;
	detail = 0;
	useInequality = false;
	useEquality = false;
}

void ComputeCI::initFromFrontend(omxState *globalState, SEXP rObj)
{
	super::initFromFrontend(globalState, rObj);

	SEXP slotValue;
	{
		ScopedProtect p1(slotValue, R_do_slot(rObj, Rf_install("verbose")));
		verbose = Rf_asInteger(slotValue);
	}
	{
		ScopedProtect p1(slotValue, R_do_slot(rObj, Rf_install("constraintType")));
		ctypeName = CHAR(Rf_asChar(slotValue));
		if (strEQ(ctypeName, "ineq")) {
			useInequality = true;
		} else if (strEQ(ctypeName, "eq")) {
			useEquality = true;
		} else if (strEQ(ctypeName, "both")) {
			useEquality = true;
			useInequality = true;
		} else if (strEQ(ctypeName, "none")) {
			// OK
		} else {
			Rf_error("%s: unknown constraintType='%s'", name, ctypeName);
		}
	}

	fitMatrix = omxNewMatrixFromSlot(rObj, globalState, "fitfunction");
	setFreeVarGroup(fitMatrix->fitFunction, varGroup);
	omxCompleteFitFunction(fitMatrix);

	Rf_protect(slotValue = R_do_slot(rObj, Rf_install("plan")));
	SEXP s4class;
	Rf_protect(s4class = STRING_ELT(Rf_getAttrib(slotValue, Rf_install("class")), 0));
	plan = omxNewCompute(globalState, CHAR(s4class));
	plan->initFromFrontend(globalState, slotValue);
}

extern "C" { void F77_SUB(npoptn)(char* string, int Rf_length); };

class ciConstraintIneq : public omxConstraint {
 private:
	typedef omxConstraint super;
	omxMatrix *fitMat;
 public:
	ciConstraintIneq(omxMatrix *fitMat) : super("CI"), fitMat(fitMat)
	{ size=1; opCode = LESS_THAN; };

	virtual void refreshAndGrab(FitContext *fc, Type ineqType, double *out) {
		omxFitFunctionCompute(fitMat->fitFunction, FF_COMPUTE_FIT, fc);
		const double fit = totalLogLikelihood(fitMat);
		double diff = std::max(fit - fc->targetFit, 0.0);
		if (diff > 100) diff = nan("infeasible");
		if (ineqType != opCode) diff = -diff;
		//mxLog("fit %f diff %f", fit, diff);
		out[0] = diff;
	};
};

class ciConstraintEq : public omxConstraint {
 private:
	typedef omxConstraint super;
	omxMatrix *fitMat;
 public:
	ciConstraintEq(omxMatrix *fitMat) : super("CI"), fitMat(fitMat)
	{ size=1; opCode = EQUALITY; };

	virtual void refreshAndGrab(FitContext *fc, Type ineqType, double *out) {
		omxFitFunctionCompute(fitMat->fitFunction, FF_COMPUTE_FIT, fc);
		const double fit = totalLogLikelihood(fitMat);
		double diff = fit - fc->targetFit;
		diff *= diff;
		if (fabs(diff) > 100000) diff = nan("infeasible");
		//mxLog("fit %f diff %f", fit, diff);
		out[0] = diff;
	};
};

// Optimization: For profile CIs of free parameters, the gradient for
// the fit is trivial.  Many evaluations could be saved.

void ComputeCI::computeImpl(FitContext *mle)
{
	if (intervals) Rf_error("Can only compute CIs once");
	if (!Global->intervals) {
		if (verbose >= 1) mxLog(name, "%s: mxRun(..., intervals=FALSE), skipping");
		return;
	}

	Global->unpackConfidenceIntervals();

	// Not strictly necessary, but makes it easier to run
	// mxComputeConfidenceInterval alone without other compute
	// steps.
	ComputeFit(name, fitMatrix, FF_COMPUTE_FIT, mle);

	int numInts = (int) Global->intervalList.size();
	if (verbose >= 1) mxLog("%s: %d intervals of '%s' (ref fit %f %s)",
				name, numInts, fitMatrix->name(), mle->fit, ctypeName);
	if (!numInts) return;

	if (!std::isfinite(mle->fit)) Rf_error("%s: reference fit is not finite", name);

	// I'm not sure why INFORM_NOT_AT_OPTIMUM is okay, but that's how it was.
	if (mle->inform >= INFORM_LINEAR_CONSTRAINTS_INFEASIBLE && mle->inform != INFORM_NOT_AT_OPTIMUM) {
		// TODO: allow forcing
		Rf_warning("Not calculating confidence intervals because of optimizer status %d", mle->inform);
		return;
	}

	Rf_protect(intervals = Rf_allocMatrix(REALSXP, numInts, 3));
	Rf_protect(intervalCodes = Rf_allocMatrix(INTSXP, numInts, 2));

	int totalIntervals = 0;
	for(int j = 0; j < numInts; j++) {
		omxConfidenceInterval *oCI = Global->intervalList[j];
		totalIntervals += std::isfinite(oCI->lbound) + std::isfinite(oCI->ubound);
	}

	Rf_protect(detail = Rf_allocVector(VECSXP, 4 + mle->numParam));
	SET_VECTOR_ELT(detail, 0, Rf_allocVector(STRSXP, totalIntervals));
	SET_VECTOR_ELT(detail, 1, Rf_allocVector(REALSXP, totalIntervals));
	SET_VECTOR_ELT(detail, 2, Rf_allocVector(INTSXP, totalIntervals));
	for (int cx=0; cx < 1+int(mle->numParam); ++cx) {
		SET_VECTOR_ELT(detail, 3+cx, Rf_allocVector(REALSXP, totalIntervals));
	}

	SEXP detailCols;
	Rf_protect(detailCols = Rf_allocVector(STRSXP, 4 + mle->numParam));
	Rf_setAttrib(detail, R_NamesSymbol, detailCols);
	SET_STRING_ELT(detailCols, 0, Rf_mkChar("parameter"));
	SET_STRING_ELT(detailCols, 1, Rf_mkChar("value"));
	SET_STRING_ELT(detailCols, 2, Rf_mkChar("lower"));
	SET_STRING_ELT(detailCols, 3, Rf_mkChar("fit"));
	for (int nx=0; nx < int(mle->numParam); ++nx) {
		SET_STRING_ELT(detailCols, 4+nx, Rf_mkChar(mle->varGroup->vars[nx]->name));
	}

	SEXP detailRowNames;
	Rf_protect(detailRowNames = Rf_allocVector(INTSXP, totalIntervals));
	markAsDataFrame(detail);

	FitContext fc(mle, mle->varGroup);
	FreeVarGroup *freeVarGroup = fc.varGroup;

	const int n = int(freeVarGroup->vars.size());
	Eigen::Map< Eigen::VectorXd > Mle(mle->est, n);

	ciConstraintEq constrEq(fitMatrix);
	ciConstraintIneq constrIneq(fitMatrix);

	int detailRow = 0;
	for(int i = 0; i < (int) Global->intervalList.size(); i++) {
		omxConfidenceInterval *currentCI = Global->intervalList[i];

		std::string &matName = currentCI->matrix->nameStr;

		if (useInequality || useEquality) {
			currentCI->varIndex = freeVarGroup->lookupVar(currentCI->matrix, currentCI->row, currentCI->col);
		}

		for (int lower=0; lower <= 1; ++lower) {
			if (lower  && !std::isfinite(currentCI->lbound)) continue;
			if (!lower && !std::isfinite(currentCI->ubound)) continue;

			// Reset to previous optimum
			Eigen::Map< Eigen::VectorXd > Est(fc.est, n);
			Est = Mle;

			double *store = lower? &currentCI->min : &currentCI->max;

			Global->checkpointMessage(mle, mle->est, "%s[%d, %d] %s CI",
						  matName.c_str(), currentCI->row + 1, currentCI->col + 1,
						  lower? "lower" : "upper");

			if (useInequality) mle->state->conList.push_back(&constrIneq);
			if (useEquality)   mle->state->conList.push_back(&constrEq);

			fc.CI = currentCI;
			fc.compositeCIFunction = (!useInequality && !useEquality);
			fc.lowerBound = lower;
			fc.fit = mle->fit;
			plan->compute(&fc);

			if (useInequality) mle->state->conList.pop_back();
			if (useEquality)   mle->state->conList.pop_back();

			omxRecompute(currentCI->matrix, &fc);
			double val = omxMatrixElement(currentCI->matrix, currentCI->row, currentCI->col);

			// We check the fit again so we can report it
			// in the detail data.frame.
			fc.CI = NULL;
			ComputeFit(name, fitMatrix, FF_COMPUTE_FIT, &fc);

			double dist = lower? currentCI->lbound : currentCI->ubound;
			bool better = (fc.inform != INFORM_STARTING_VALUES_INFEASIBLE &&
				       ((!useInequality && !useEquality) || fabs(fc.fit - fc.targetFit) < (dist * .05)) &&
				       ((!std::isfinite(*store) ||
					 (lower && val < *store) || (!lower && val > *store))));

			if (better) {
				*store = val;
			}

			int inform = fc.inform;
			if (lower) currentCI->lCode = inform;
			else       currentCI->uCode = inform;

			if(verbose >= 1) {
				mxLog("CI[%d,%s] %s[%d,%d] val=%f fit-target=%f accepted=%d",
				      1+i, (lower?"lower":"upper"), matName.c_str(), 1+currentCI->row, 1+currentCI->col,
				      val, fc.fit - fc.targetFit, better);
			}

			INTEGER(detailRowNames)[detailRow] = 1 + detailRow;
			SET_STRING_ELT(VECTOR_ELT(detail, 0), detailRow, Rf_mkChar(currentCI->name.c_str()));
			REAL(VECTOR_ELT(detail, 1))[detailRow] = val;
			INTEGER(VECTOR_ELT(detail, 2))[detailRow] = lower;
			REAL(VECTOR_ELT(detail, 3))[detailRow] = fc.fit;
			for (int px=0; px < int(fc.numParam); ++px) {
				REAL(VECTOR_ELT(detail, 4+px))[detailRow] = Est[px];
			}

			++detailRow;
		}
	}

	Rf_setAttrib(detail, R_RowNamesSymbol, detailRowNames);

	mle->copyParamToModel();

	Eigen::Map< Eigen::ArrayXXd > interval(REAL(intervals), numInts, 3);
	interval.fill(NA_REAL);
	int* intervalCode = INTEGER(intervalCodes);
	for(int j = 0; j < numInts; j++) {
		omxConfidenceInterval *oCI = Global->intervalList[j];
		omxRecompute(oCI->matrix, mle);
		interval(j, 1) = omxMatrixElement(oCI->matrix, oCI->row, oCI->col);
		if (1) {
			interval(j, 0) = std::min(oCI->min, interval(j, 1));
			interval(j, 2) = std::max(oCI->max, interval(j, 1));
		} else {
			interval(j, 0) = oCI->min;
			interval(j, 2) = oCI->max;
		}
		intervalCode[j] = oCI->lCode;
		intervalCode[j + numInts] = oCI->uCode;
	}
}

void ComputeCI::reportResults(FitContext *fc, MxRList *slots, MxRList *out)
{
	if (!intervals) return;

	int numInt = (int) Global->intervalList.size();

	SEXP dimnames;
	SEXP names;
	Rf_protect(dimnames = Rf_allocVector(VECSXP, 2));
	Rf_protect(names = Rf_allocVector(STRSXP, 3));
	SET_STRING_ELT(names, 0, Rf_mkChar("lbound"));
	SET_STRING_ELT(names, 1, Rf_mkChar("estimate"));
	SET_STRING_ELT(names, 2, Rf_mkChar("ubound"));
	SET_VECTOR_ELT(dimnames, 1, names);

	Rf_protect(names = Rf_allocVector(STRSXP, numInt)); //shared between the two matrices
	for (int nx=0; nx < numInt; ++nx) {
		omxConfidenceInterval *ci = Global->intervalList[nx];
		SET_STRING_ELT(names, nx, Rf_mkChar(ci->name.c_str()));
	}
	SET_VECTOR_ELT(dimnames, 0, names);

	Rf_setAttrib(intervals, R_DimNamesSymbol, dimnames);

	out->add("confidenceIntervals", intervals);

	Rf_protect(dimnames = Rf_allocVector(VECSXP, 2));
	SET_VECTOR_ELT(dimnames, 0, names);

	Rf_protect(names = Rf_allocVector(STRSXP, 2));
	SET_STRING_ELT(names, 0, Rf_mkChar("lbound"));
	SET_STRING_ELT(names, 1, Rf_mkChar("ubound"));
	SET_VECTOR_ELT(dimnames, 1, names);

	Rf_setAttrib(intervalCodes, R_DimNamesSymbol, dimnames);

	out->add("confidenceIntervalCodes", intervalCodes);

	MxRList output;
	output.add("detail", detail);
	slots->add("output", output.asR());
}
