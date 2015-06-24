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
        int maxIter;
        double priorSpeed;
        double shrinkage;
        int retries;
        GradientOptimizerContext &rf;
        //FitContext *fc;
        size_t ineq_size;
        size_t eq_size;
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
            maxIter = 50000;
            priorSpeed = 1.0;
            shrinkage = 0.7;
            retries = 300;
            ineq_size = 0;
            eq_size = 0;
            rho = 0;
            tau = 0.5;
            gam = 10;
            lam_min = -1e20;
            lam_max = 1e20;
            mu_max = 1e20;
            ICM_tol = 1e-4;
}

struct fit_functional {
	SDcontext &sd;

	fit_functional(SDcontext &sd) : sd(sd) {};

	template <typename T1>
	double operator()(Eigen::MatrixBase<T1>& x) const {
        int mode = 0;
        double al = 0;
        for (size_t i = 0; i < unsigned(sd.eq_size); ++i)
        {
            al += 0.5 * sd.rho * (sd.rf.equality[i] + sd.lambda[i] / sd.rho) * (sd.rf.equality[i] + sd.lambda[i] / sd.rho);
        }

        for (size_t i = 0; i < unsigned(sd.ineq_size); ++i)
        {
            al += 0.5 * sd.rho * std::max(0.0,(sd.rf.inequality[i] + sd.mu[i] / sd.rho)) * std::max(0.0,(sd.rf.inequality[i] + sd.mu[i] / sd.rho));
        }
		return sd.rf.solFun(x.derived().data(), &mode) + al;
	}
};

void SDcontext::linesearch() {
    int iter = 0;
    Eigen::Map< Eigen::VectorXd > currEst(rf.fc->est, rf.fc->numParam);
    Eigen::VectorXd majorEst = currEst;

    fit_functional ff(*this);
    double refFit = ff(currEst);
    if (!std::isfinite(refFit)) {
	    rf.informOut = INFORM_STARTING_VALUES_INFEASIBLE;
	    return;
    }

    grad.resize(rf.fc->numParam);

    while(++iter < maxIter && !isErrorRaised()) {
	    rf.fc->iterations += 1;
	    gradient_with_ref(rf.gradientAlgo, rf.gradientIterations, rf.gradientStepSize,
			      ff, refFit, majorEst, grad);

	    if (rf.verbose >= 3) mxPrintMat("grad", grad);

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
        while (--retries > 0 && !isErrorRaised()){
            Eigen::VectorXd nextEst = majorEst - speed * searchDir;
            nextEst = nextEst.cwiseMax(rf.solLB).cwiseMin(rf.solUB);

            if (nextEst == prevEst) break;
            prevEst = nextEst;

            rf.checkActiveBoxConstraints(nextEst);

            double fit = ff(nextEst);
            if (fit < refFit) {
                foundBetter = true;
                refFit = fit;
                bestSpeed = speed;
                bestEst = nextEst;
                break;
            }
            speed *= shrinkage;
        }

        if (!foundBetter) {
            rf.informOut = INFORM_CONVERGED_OPTIMUM;
            if(rf.verbose >= 2) mxLog("After %i iterations, cannot find better estimation along the gradient direction", iter);
            break;
        }

        if (rf.verbose >= 2) mxLog("major fit %f bestSpeed %g", refFit, bestSpeed);
        majorEst = bestEst;
        priorSpeed = bestSpeed * 1.1;
    }
    currEst = majorEst;
    if ((grad.array().abs() > 0.1).any()) {
	    rf.informOut = INFORM_NOT_AT_OPTIMUM;
    }
    if (iter >= maxIter - 1) {
            rf.informOut = INFORM_ITERATION_LIMIT;
            if(rf.verbose >= 2) mxLog("Maximum iteration achieved!");
    }
    if(rf.verbose >= 1) mxLog("Status code : %i", rf.informOut);

}

void SDcontext::optimize() {
    bool constrained = TRUE;
    rf.fc->copyParamToModel();
    ComputeFit("SD", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    rf.setupSimpleBounds();
    rf.solEqBFun();
    rf.myineqFun();

    ineq_size = rf.inequality.size();
    eq_size = rf.equality.size();
    if(ineq_size == 0 && eq_size == 0) constrained = FALSE;
    double eq_norm = 0, ineq_norm = 0;
    for(size_t i = 0; i < eq_size; i++)
    {
        eq_norm += rf.equality[i] * rf.equality[i];
    }
    for(size_t i = 0; i < ineq_size; i++)
    {
        ineq_norm += std::max(0.0, rf.inequality[i]) * std::max(0.0, rf.inequality[i]);
    }
    if(eq_norm + ineq_norm != 0){
        rho = std::max(1e-6, std::min(10.0, (2 * std::abs(rf.fc->fit) / (eq_norm + ineq_norm))));
    }
    else{
        rho = 0;
    }

    mu.resize(ineq_size);
    mu.setZero();
    lambda.resize(eq_size);
    lambda.setZero();
    V.resize(ineq_size);

    rf.informOut = INFORM_UNINITIALIZED;

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
            while (1) {
                iter++;
                ICM_tol = 1e-4;
                double prev_ICM = ICM;
                ICM = 0;
                linesearch();
                if(rf.informOut == INFORM_STARTING_VALUES_INFEASIBLE) return;
                rf.fc->copyParamToModel();
                rf.solEqBFun();
                rf.myineqFun();

                for(size_t i = 0; i < eq_size; i++){
                    lambda[i] = std::min(std::max(lam_min, (lambda[i] + rho * rf.equality[i])), lam_max);
                    ICM = std::max(ICM, std::abs(rf.equality[i]));
                }

                for(size_t i = 0; i < ineq_size; i++){
                    mu[i] = std::min(std::max(0.0, (mu[i] + rho * rf.inequality[i])),mu_max);
                }

                for(size_t i = 0; i < ineq_size; i++){
                    V[i] = std::max(rf.inequality[i], (-mu[i] / rho));
                    ICM = std::max(ICM, std::abs(V[i]));
                }

                if(!(iter == 1 || ICM <= tau * prev_ICM))
                {
                    rho *= gam;
                }

                if(ICM < ICM_tol)
                {
                    if(rf.verbose >= 1) mxLog("Augmented lagrangian coverges!");
                    return;
                }
                if (iter >= maxIter) {
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
