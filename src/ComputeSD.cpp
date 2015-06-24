/* Steepest Descent optimizer for unconstrained problems*/

#include <valarray>
#include <math.h>
#include "omxState.h"
#include "omxFitFunction.h"
#include "omxExportBackendState.h"
#include "Compute.h"
#include "matrix.h"
#include "ComputeSD.h"
#include "finiteDifferences.h"

namespace SteepestDescentNamespace {

struct SDcontext {
        Eigen::VectorXd grad;
        double shrinkage;
        GradientOptimizerContext &rf;
        //FitContext *fc;
        int ineq_size;
        int eq_size;
        double rho;
        double tau;
        double gam;
        double lam_min;
        double lam_max;
        double mu_max;
        Eigen::VectorXd mu;
        Eigen::VectorXd lambda;
        Eigen::VectorXd V;
        double ICM_tol;

        // two methods for optimization
        void optimize();
        void linesearch();
        // constructor
        SDcontext(GradientOptimizerContext &goc);
};

SDcontext::SDcontext(GradientOptimizerContext &goc): rf(goc){
            shrinkage = 0.7;
            ineq_size = 0;
            eq_size = 0;
            rho = 0;
            tau = 0.5;
            gam = 10;
            lam_min = -1e20;
            lam_max = 1e20;
            mu_max = 1e20;
	    double fudgeFactor = 0.002;
	    ICM_tol = Global->feasibilityTolerance * fudgeFactor;
}

struct fit_functional {
	SDcontext &sd;

	fit_functional(SDcontext &sd) : sd(sd) {};

	template <typename T1>
	double operator()(Eigen::MatrixBase<T1>& x) const {
		int mode = 0;
		double fit = sd.rf.solFun(x.derived().data(), &mode);
		sd.rf.solEqBFun();
		sd.rf.myineqFun();
		double al = 0;
		for (int i = 0; i < sd.eq_size; ++i) {
			al += 0.5 * sd.rho * (sd.rf.equality[i] + sd.lambda[i] / sd.rho) * (sd.rf.equality[i] + sd.lambda[i] / sd.rho);
		}

		for (int i = 0; i < sd.ineq_size; ++i) {
			al += 0.5 * sd.rho * std::max(0.0,(sd.rf.inequality[i] + sd.mu[i] / sd.rho)) * std::max(0.0,(sd.rf.inequality[i] + sd.mu[i] / sd.rho));
		}
		return fit + al;
	}
};

void SDcontext::linesearch()
{
    rf.informOut = INFORM_UNINITIALIZED;
    double priorSpeed = 1.0;
    Eigen::Map< Eigen::VectorXd > currEst(rf.fc->est, rf.fc->numParam);
    Eigen::VectorXd majorEst = currEst;

    fit_functional ff(*this);
    double refFit = ff(currEst);
    if (!std::isfinite(refFit)) {
	    Rf_error("Moved into infeasible region"); // should be impossible
    }

    grad.resize(rf.fc->numParam);

    double relImprovement = 0;
    int maxIter = 1000;
    int iter = 0;
    while(++iter < maxIter && !isErrorRaised()) {
	    rf.fc->iterations += 1;
	    gradient_with_ref(rf.gradientAlgo, rf.gradientIterations, rf.gradientStepSize,
			      ff, refFit, majorEst, grad);

	    if (rf.verbose >= 4) mxPrintMat("grad", grad);

        if(grad.norm() == 0)
        {
            rf.informOut = INFORM_CONVERGED_OPTIMUM;
            if(rf.verbose >= 2) mxLog("After %i iterations, gradient achieves zero!", iter);
            break;
        }

        double speed = std::min(priorSpeed, 1.0);
        double bestSpeed = speed;
        bool foundBetter = false;
        Eigen::VectorXd bestEst(majorEst.size());
        Eigen::VectorXd prevEst(majorEst.size());
        Eigen::VectorXd searchDir = grad;
        searchDir /= searchDir.norm();
        prevEst.setConstant(nan("uninit"));
	int retries = 300;
        while (--retries > 0 && !isErrorRaised()){
            Eigen::VectorXd nextEst = majorEst - speed * searchDir;
            nextEst = nextEst.cwiseMax(rf.solLB).cwiseMin(rf.solUB);

            if (nextEst == prevEst) break;
            prevEst = nextEst;

            rf.checkActiveBoxConstraints(nextEst);

            double fit = ff(nextEst);
            if (fit < refFit) {
                foundBetter = true;
		relImprovement = (refFit - fit) / (1+fabs(refFit));
		refFit = fit;
                bestSpeed = speed;
                bestEst = nextEst;
                break;
            }
            speed *= shrinkage;
        }

	double fudgeFactor = 1.0;
	if (!foundBetter || relImprovement < rf.ControlTolerance * fudgeFactor) {
		if(rf.verbose >= 2) {
			mxLog("After %i iterations, cannot find better estimation along the gradient direction", iter);
		}
		rf.informOut = INFORM_CONVERGED_OPTIMUM;
		break;
	}

        if (rf.verbose >= 2) mxLog("major fit %f bestSpeed %g", refFit, bestSpeed);
        majorEst = bestEst;
        priorSpeed = bestSpeed * 1.1;
    }
    currEst = majorEst;
    if ((grad.array().abs() > 0.1).any()) {
	    // wrong condition, see other optimizers
	    // box constraints need special handling
	    // Also, this check should only happen once at the end of optimize()
	    rf.informOut = INFORM_NOT_AT_OPTIMUM;
    }
    if (iter >= maxIter - 1) {
            rf.informOut = INFORM_ITERATION_LIMIT;
            if(rf.verbose >= 2) mxLog("Maximum iteration achieved!");
    }
    if(rf.verbose >= 1) mxLog("Status code : %i", rf.informOut);

}

void SDcontext::optimize()
{
    bool constrained = TRUE;
    rf.fc->copyParamToModel();
    ComputeFit("SD", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    if (!std::isfinite(rf.fc->fit)) {
	    rf.informOut = INFORM_STARTING_VALUES_INFEASIBLE;
	    return;
    }

    rf.setupIneqConstraintBounds();
    rf.solEqBFun();
    rf.myineqFun();

    ineq_size = rf.inequality.size();
    eq_size = rf.equality.size();
    if(ineq_size == 0 && eq_size == 0) constrained = FALSE;

    double eq_norm = 0, ineq_norm = 0;
    for(int i = 0; i < eq_size; i++)
    {
        eq_norm += rf.equality[i] * rf.equality[i];
    }
    for(int i = 0; i < ineq_size; i++)
    {
        ineq_norm += std::max(0.0, rf.inequality[i]) * std::max(0.0, rf.inequality[i]);
    }
    if(eq_norm + ineq_norm != 0){
        rho = std::max(1e-6, std::min(10.0, (2 * std::abs(rf.fc->fit) / (eq_norm + ineq_norm))));
    }
    else{
        rho = 0;
    }

    if (rf.verbose >= 1) {
	    mxLog("Welcome to SD, constrained=%d ICM_tol=%f", constrained, ICM_tol);
    }
    if (!constrained) {
            rf.fc->wanted |= FF_COMPUTE_GRADIENT;
    }

    mu.resize(ineq_size);
    mu.setZero();
    lambda.resize(eq_size);
    lambda.setZero();
    V.resize(ineq_size);

    switch (constrained) {
        case FALSE:{
            // unconstrained problem
            linesearch();
            break;
        }
        case TRUE:{
            double ICM = HUGE_VAL;
            int iter = 0;
            // initialize penalty parameter rho and the Lagrange multipliers lambda and mu
            while (!isErrorRaised()) {
                iter++;
                double prev_ICM = ICM;
                ICM = 0;
		if (rf.verbose >= 3) {
			mxLog("prev_ICM=%f, solve subproblem with rho=%f", prev_ICM, rho);
		}
                linesearch();
                rf.fc->copyParamToModel();
                rf.solEqBFun();
                rf.myineqFun();

                for(int i = 0; i < eq_size; i++){
                    lambda[i] = std::min(std::max(lam_min, (lambda[i] + rho * rf.equality[i])), lam_max);
                    ICM = std::max(ICM, std::abs(rf.equality[i]));
                }

                for(int i = 0; i < ineq_size; i++){
                    mu[i] = std::min(std::max(0.0, (mu[i] + rho * rf.inequality[i])),mu_max);
                }

                for(int i = 0; i < ineq_size; i++){
                    V[i] = std::max(rf.inequality[i], (-mu[i] / rho));
                    ICM = std::max(ICM, std::abs(V[i]));
                }

		rho *= gam; // why not do this every time? TODO

                if (ICM < ICM_tol) {
                    if(rf.verbose >= 1) mxLog("Augmented lagrangian coverges!");
                    return;
                }
                if (iter >= 10) {
                    rf.informOut = INFORM_ITERATION_LIMIT;
                    break;
                }
            }
            break;
        }
    }
}

}

void SteepestDescent(GradientOptimizerContext &rf)
{
	SteepestDescentNamespace::SDcontext sd(rf);
	sd.optimize();
}
