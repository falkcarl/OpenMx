/*
 *  Copyright 2007-2016 The OpenMx Project
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

#ifndef _OMXALGEBRA_H_
#define _OMXALGEBRA_H_

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h> 
#include <R_ext/Rdynload.h> 
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include "omxDefines.h"

typedef struct omxAlgebra omxAlgebra;

#include "omxSymbolTable.h"
#include "omxState.h"
#include "algebraOp.h"

struct omxAlgebra {						// A matrix
										//TODO: Improve encapsulation
/* Fields unique to Algebras */
	algebra_op_t funWrapper;					// Wrapper for the algebra function itself
	omxMatrix** algArgs;
	int numArgs;						// Length of args

	omxMatrix* matrix;				// The matrix populated by this algebra
    SEXP sexpAlgebra;               // The SEXP MxAlgebra object

	const omxAlgebraTableEntry *oate;

	omxAlgebra();
};

/* Initialize and Destroy */
	omxMatrix* omxInitAlgebra(omxAlgebra *oa, omxState* os);			// Constructor 
void omxAlgebraAllocArgs(omxAlgebra *oa, int numArgs);
void omxFillAlgebraFromTableEntry(omxAlgebra *oa, const omxAlgebraTableEntry* oate, const int realNumArgs);
	void omxInitAlgebraWithMatrix(omxAlgebra *oa, omxMatrix* om);		// Constructor (with matrix)
	void omxFreeAlgebraArgs(omxAlgebra* algebra);						// Frees all args
void omxFillMatrixFromMxAlgebra(omxMatrix* om, SEXP algebra, std::string &name, SEXP dimnames);
	omxMatrix* omxMatrixLookupFromState1(SEXP matrix, omxState* os);		// Create a matrix/algebra from a matrix pointer
	omxMatrix* omxNewAlgebraFromOperatorAndArgs(int opCode, omxMatrix** args, int numArgs, omxState* os); // For constraints.

/* Other Functions */
	 																	// Adjust an algebra for a table entry
	void omxAlgebraCopyAlgebra(omxAlgebra *dest, omxAlgebra *src);		// Copy across another element.  
																		// NOTE: Duplicates.

void omxAlgebraRecompute(omxMatrix *mat, int want, FitContext *fc);

	void omxDuplicateAlgebra(omxMatrix *tgt, omxMatrix* src, omxState* tgtState);

void omxAlgebraPrint(omxAlgebra *source, const char* d);

#endif /* _OMXALGEBRA_H_ */


