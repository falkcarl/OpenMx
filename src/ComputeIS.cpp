/* Incremental Subgradient optimizer for unconstrained problems*/

#include <valarray>
#include <math.h>
#include "omxState.h"
#include "omxFitFunction.h"
#include "omxExportBackendState.h"
#include "Compute.h"
#include "matrix.h"
#include "ComputeIS.h"
#include "finiteDifferences.h"

namespace IncrementalGradientNamespace {

struct IGcontext {
    Eigen::VectorXd grad;
    GradientOptimizerContext &rf;
    double speed;
    void optimize();
    // constructor
    IGcontext(GradientOptimizerContext &, double);
//private:
  //  double llwrapper(int i, GradientOptimizerContext &goc) const;
};

IGcontext::IGcontext(GradientOptimizerContext &goc, double stepsize): rf(goc), speed(stepsize) {}

/*
double IGcontext::llwrapper(int i, GradientOptimizerContext &goc){
    goc.setupSimpleBounds();
    goc.fc->copyParamToModel();
    ComputeFit("IS", goc.fitMatrix, FF_COMPUTE_FIT, goc.fc);
    return -2*log(*(goc.fitMatrix->data + (i -1)));
}
*/
struct rowfit_functional {
	IGcontext &ig;
    int row;
    //constructor
	rowfit_functional(IGcontext &context, int i) : ig(context), row(i) {};

	template <typename T1>
	double operator()(Eigen::MatrixBase<T1>& x) {
        if (ig.rf.fc->est != x.derived().data()) memcpy(ig.rf.fc->est, x.derived().data(), sizeof(double) * ig.rf.fc->numParam);
        ig.rf.fc->copyParamToModel();
        ComputeFit("IS", ig.rf.fitMatrix, FF_COMPUTE_FIT, ig.rf.fc);
        return -2*log(*(ig.rf.fitMatrix->data + row));
	}
};


void IGcontext::optimize(){
    rf.setupSimpleBounds();
    rf.fc->copyParamToModel();

    //Eigen::VectorXd ll(rf.fitMatrix->rows);


/*
    for(int i = 0; i< rf.fitMatrix->rows; i++){
        ll[i] = -2*log(*(rf.fitMatrix->data + i));
    }*/
    Eigen::Map< Eigen::VectorXd > currEst(rf.fc->est, rf.fc->numParam);
    Eigen::VectorXd majorEst = currEst;
    grad.resize(rf.fc->numParam);
    for(int i = 0; i< rf.fitMatrix->rows; i++){
        rowfit_functional ff(*this, i);
        double refFit = ff(majorEst);
        gradient_with_ref(rf.gradientAlgo, rf.gradientIterations, rf.gradientStepSize, ff, refFit, majorEst, grad);
        majorEst = majorEst - speed * grad;
        majorEst = majorEst.cwiseMax(rf.solLB).cwiseMin(rf.solUB);
        rf.checkActiveBoxConstraints(majorEst);
    }
    currEst = majorEst;
}

}

void IncrementalGradient(GradientOptimizerContext &rf, double speed)
{
	IncrementalGradientNamespace::IGcontext ig(rf, speed);
	ig.optimize();
}










