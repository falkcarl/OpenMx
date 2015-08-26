/*
   Distributed Gradient optimizer for unconstrained problems
   Implement "Distributed Subgradient Methods for Multi-Agent Optimization"
*/

#include <valarray>
#include <math.h>
#include "omxState.h"
#include "omxFitFunction.h"
#include "omxExportBackendState.h"
#include "Compute.h"
#include "matrix.h"
#include "ComputeDG.h"
#include "finiteDifferences.h"

namespace DistributedGradientNamespace {

class DGcontext {
public:
    Eigen::VectorXd grad;
    GradientOptimizerContext &rf;
    //Eigen::VectorXd weight;
    //Eigen::MatrixXd value;    // neighbours' estimation
    double stepsize;

    void update();

    // constructor
    DGcontext(GradientOptimizerContext &, double);
};

DGcontext::DGcontext(GradientOptimizerContext &goc, double speed): rf(goc), stepsize(speed) {};

struct fit_functional {
	DGcontext &dg;
    //constructor
	fit_functional(DGcontext &context) : dg(context) {};

	template <typename T1>
	double operator()(Eigen::MatrixBase<T1>& x) {
        if (dg.rf.fc->est != x.derived().data()) memcpy(dg.rf.fc->est, x.derived().data(), sizeof(double) * dg.rf.fc->numParam);
        dg.rf.fc->copyParamToModel();
        ComputeFit("DG", dg.rf.fitMatrix, FF_COMPUTE_FIT, dg.rf.fc);
        return -2*log(*(dg.rf.fitMatrix->data));
	}
};



void DGcontext::update(){
    rf.setupSimpleBounds();
    rf.fc->copyParamToModel();

    Eigen::Map< Eigen::VectorXd > currEst(rf.fc->est, rf.fc->numParam);
    Eigen::VectorXd majorEst = currEst;
    grad.resize(rf.fc->numParam);
    mxLog("DG update");
    fit_functional ff(*this);
    double refFit = ff(majorEst);
    gradient_with_ref(rf.gradientAlgo, rf.gradientIterations, rf.gradientStepSize, ff, refFit, majorEst, grad);
    majorEst = majorEst - stepsize * grad;
    majorEst = majorEst.cwiseMax(rf.solLB).cwiseMin(rf.solUB);
    rf.checkActiveBoxConstraints(majorEst);
    currEst = majorEst;
}

}

void DistributedGradient(GradientOptimizerContext &rf, double stepsize)
{
	DistributedGradientNamespace::DGcontext dg(rf, stepsize);
	dg.update();
}










