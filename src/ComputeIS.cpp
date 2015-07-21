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

class IGcontext {
public:
    Eigen::VectorXd grad;
    GradientOptimizerContext &rf;
    int speedmode;
    /*
    mode 1: Constant step size,
    mode 2: Constant step length,
    mode 3: Square summable but not summable
    mode 4: Nonsummable diminishing
    reference: Boyd, Stephen, Lin Xiao, and Almir Mutapcic. "Subgradient methods."
    lecture notes of EE392o, Stanford University, Autumn Quarter 2004 (2003): 2004-2005.
    */
    void optimize();
    //int getIter();
    // constructor
    IGcontext(GradientOptimizerContext &, int);
private:
    static int iterations;
};

int IGcontext::iterations = 0;

IGcontext::IGcontext(GradientOptimizerContext &goc, int stepsize): rf(goc), speedmode(stepsize) {
    iterations++;
}

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
        //int iter = getIter();
        printf("iterations: %i", iterations);
        double speed;
        switch (speedmode){
            case 1:{
                speed = 0.01;
                break;
            }
            case 2:{
                speed = 0.01/grad.norm();
                break;
            }
            case 3:{
                speed = 0.1/iterations;
                break;
            }
            case 4:{
                speed = 0.1/sqrt(iterations);
                break;
            }
            default:{
                break;
            }
        }
        majorEst = majorEst - speed * grad;
        majorEst = majorEst.cwiseMax(rf.solLB).cwiseMin(rf.solUB);
        rf.checkActiveBoxConstraints(majorEst);
    }
    currEst = majorEst;
}

}

void IncrementalGradient(GradientOptimizerContext &rf, int speedmode)
{
	IncrementalGradientNamespace::IGcontext ig(rf, speedmode);
	ig.optimize();
}










