#ifndef __SteepDescent_H_
#define __SteepDescent_H_

#include <valarray>
#include <math.h>
#include "omxState.h"
#include "omxFitFunction.h"
#include "omxExportBackendState.h"
#include "Compute.h"
#include "matrix.h"

class SDcontext {
    public:
        double fit;
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

#endif
