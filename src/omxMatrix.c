/***********************************************************
* 
*  omxMatrix.cc
*
*  Created: Timothy R. Brick 	Date: 2008-11-13 12:33:06
*
*	Contains code for the omxMatrix class
*   omxDataMatrices hold necessary information to simplify
* 	dealings between the OpenMX back end and BLAS.
*
**********************************************************/
#include "omxMatrix.h"

const char omxMatrixMajorityList[3] = "Tn";		// BLAS Column Majority.

void omxMatrixPrint(omxMatrix *source, char* header) {
	int j, k;
	
	Rprintf("%s: (%d x %d)\n", header, source->rows, source->cols);
	if(OMX_DEBUG) {Rprintf("Matrix Printing is at %0x\n", source);}
	
	if(source->colMajor) {
		for(j = 0; j < source->rows; j++) {
			for(k = 0; k < source->cols; k++) {
				Rprintf("\t%3.6f", source->data[k*source->rows+j]);
			}
			Rprintf("\n");
		}
	} else {
		for(j = 0; j < source->cols; j++) {
			for(k = 0; k < source->rows; k++) {
				Rprintf("\t%3.6f", source->data[k*source->cols+j]);
			}
			Rprintf("\n");
		}
	}
}

omxMatrix* omxInitMatrix(omxMatrix* om, int nrows, int ncols, unsigned short isColMajor) {
	
	if(om == NULL) om = (omxMatrix*) R_alloc(1, sizeof(omxMatrix));
	if(OMX_DEBUG) { Rprintf("Initializing 0x%0x to (%d, %d).\n", om, nrows, ncols); }

	om->rows = nrows;
	om->cols = ncols;
	om->colMajor = (isColMajor?1:0);

	om->originalRows = om->rows;
	om->originalCols = om->cols;
	om->originalColMajor=om->colMajor;
	
	if(om->rows == 0 || om->cols == 0) {
		om->data = NULL;
		om->localData = FALSE;
	} else {
		om->data = (double*) Calloc(nrows * ncols, double);
		om->localData = TRUE;
	}

	om->aliasedPtr = NULL;
	om->algebra = NULL;
	om->objective = NULL;
	
	omxMatrixCompute(om);
	
	return om;
	
}

void omxCopyMatrix(omxMatrix *dest, omxMatrix *orig) {
	/* Duplicate a matrix.  NOTE: Matrix maintains its algebra bindings. */
	
	if(OMX_DEBUG) { Rprintf("omxCopyMatrix"); }
	
	omxFreeMatrixData(dest);

	dest->rows = orig->rows;
	dest->cols = orig->cols;
	dest->colMajor = orig->colMajor;
	dest->originalRows = dest->rows;
	dest->originalCols = dest->cols;
	dest->originalColMajor = dest->colMajor;

	if(dest->rows == 0 || dest->cols == 0) {
		dest->data = NULL;
		dest->localData=FALSE;
	} else {
		dest->data = (double*) Calloc(dest->rows * dest->cols, double);
		memcpy(dest->data, orig->data, dest->rows * dest->cols * sizeof(double));
		dest->localData = TRUE;
	}

	dest->aliasedPtr = NULL;

	omxMatrixCompute(dest);
	
}

void omxAliasMatrix(omxMatrix *dest, omxMatrix *src) {
	omxCopyMatrix(dest, src);
	dest->aliasedPtr = src->data;						// Interesting Aside: back matrix can change without alias
	dest->algebra = NULL;
	dest->objective = NULL;
}

void omxFreeMatrixData(omxMatrix * om) {
 
	if(om->localData && om->data != NULL) {
		if(OMX_DEBUG) { Rprintf("Freeing 0x%0x. Localdata = %d.\n", om->data, om->localData); }
		Free(om->data);
		om->data = NULL;
		om->localData = FALSE;
	}

}

void omxFreeAllMatrixData(omxMatrix *om) {
	
	if(OMX_DEBUG) { Rprintf("Freeing 0x%0x with data = %0x and algebra %0x.\n", om, om->data, om->algebra); }
	
	if(om->localData && om->data != NULL) {
		Free(om->data);
		om->data = NULL;
		om->localData = FALSE;
	}
	
	if(om->algebra != NULL) {
		omxFreeAlgebraArgs(om->algebra);
		om->algebra = NULL;
	}
	
	if(om->objective != NULL) {
		omxFreeObjectiveArgs(om->objective);
		om->objective = NULL;
	}

}

void omxResizeMatrix(omxMatrix *om, int nrows, int ncols, unsigned short keepMemory) {
	if(OMX_DEBUG) { Rprintf("Resizing matrix from (%d, %d) to (%d, %d) (keepMemory: %d)", om->rows, om->cols, nrows, ncols, keepMemory); }
	if(keepMemory == FALSE) { 
		if(OMX_DEBUG) { Rprintf(" and regenerating memory to do it"); }
		omxFreeMatrixData(om);
		om->data = (double*) Calloc(nrows * ncols, double);
		om->localData = TRUE;
	} else if(om->originalRows * om->originalCols < nrows * ncols) {
		warning("Upsizing an existing matrix may cause undefined behavior.\n"); // TODO: Define this behavior?
	}

	if(OMX_DEBUG) { Rprintf(".\n"); }
	om->rows = nrows;
	om->cols = ncols;
	if(keepMemory == FALSE) {
		om->originalRows = om->rows;
		om->originalCols = om->cols;
	}
	
	omxMatrixCompute(om);
}

void omxResetAliasedMatrix(omxMatrix *om) {
	om->rows = om->originalRows;
	om->cols = om->originalCols;
	om->colMajor = om->originalColMajor;
	if(om->aliasedPtr != NULL) {
		if(OMX_DEBUG) { omxPrintMatrix(om, "I was");}
		memcpy(om->data, om->aliasedPtr, om->rows*om->cols*sizeof(double));
		if(OMX_DEBUG) { omxPrintMatrix(om, "I am");}
	}
	omxMatrixCompute(om);
}

void omxMatrixCompute(omxMatrix *om) {
	
	if(OMX_DEBUG) { Rprintf("Matrix compute: 0x%0x.\n", om); }
	om->majority = &(omxMatrixMajorityList[(om->colMajor?1:0)]);
	om->minority = &(omxMatrixMajorityList[(om->colMajor?0:1)]);
	om->leading = (om->colMajor?om->rows:om->cols);
	om->lagging = (om->colMajor?om->cols:om->rows);
	om->isDirty = FALSE;
}

double* omxLocationOfMatrixElement(omxMatrix *om, int row, int col) {
	int index = 0;
	if(om->colMajor) {
		index = col * om->rows + row;
	} else {
		index = row * om->cols + col;
	}
	return om->data + index;
}

double omxMatrixElement(omxMatrix *om, int row, int col) {
	int index = 0;
	if(om->colMajor) {
		index = col * om->rows + row;
	} else {
		index = row * om->cols + col;
	}
	return om->data[index];
}

void omxSetMatrixElement(omxMatrix *om, int row, int col, double value) {
	int index = 0;
	if(om->colMajor) {
		index = col * om->rows + row;
	} else {
		index = row * om->cols + col;
	}
	om->data[index] = value;
}

void omxMarkDirty(omxMatrix *om) { om->isDirty = TRUE; }

unsigned short omxMatrixNeedsUpdate(omxMatrix *matrix) { return matrix->isDirty; };

omxMatrix* omxNewMatrixFromMxMatrix(SEXP matrix) {
/* Populates the fields of a omxMatrix with details from an R Matrix. */ 
	
	SEXP className;
	SEXP matrixDims;
	int* dimList;
	
	omxMatrix *om = NULL;
	om = omxInitMatrix(NULL, 0, 0, FALSE);
	
	if(OMX_DEBUG) { Rprintf("Filling omxMatrix from R matrix.\n"); }
	
	if(matrix == NULL) error("Null Matrix detected.\n");
	
	/* Sanity Check */
	if(!isMatrix(matrix) && !isVector(matrix)) {
		SEXP values;
		int *rowVec, *colVec;
		double  *dataVec;
		const char *stringName;
		if(OMX_DEBUG) {Rprintf("R Matrix is an object of some sort.\n");}
		PROTECT(className = getAttrib(matrix, install("class")));
		if(strncmp(CHAR(STRING_ELT(className, 0)), "Symm", 2) != 0) { // Should be "Mx"
			error("omxMatrix::fillFromMatrix--Passed Non-vector, non-matrix SEXP.\n");
		}
		stringName = CHAR(STRING_ELT(className, 0));
		if(strncmp(stringName, "SymmMatrix", 12) == 0) {
			if(OMX_DEBUG) { Rprintf("R matrix is SymmMatrix.  Processing.\n"); }
			PROTECT(values = GET_SLOT(matrix, install("values")));
			om->rows = INTEGER(GET_SLOT(values, install("nrow")))[0];
			om->cols = INTEGER(GET_SLOT(values, install("ncol")))[0];
			
			om->data = (double*) S_alloc(om->rows * om->cols, sizeof(double));		// We can afford to keep through the whole call
			rowVec = INTEGER(GET_SLOT(values, install("rowVector")));
			colVec = INTEGER(GET_SLOT(values, install("colVector")));
			dataVec = REAL(GET_SLOT(values, install("dataVector")));
			for(int j = 0; j < length(GET_SLOT(values, install("dataVector"))); j++) {
				om->data[(rowVec[j]-1) + (colVec[j]-1) * om->rows] = dataVec[j];
				om->data[(rowVec[j]-1) * om->cols + (colVec[j]-1)] = dataVec[j];  // Symmetric, after all.
			}
			UNPROTECT(1); // values
		}
		UNPROTECT(1); // className
	} else {
		if(OMX_DEBUG) { Rprintf("R matrix is Mx Matrix.  Processing.\n"); }
		
		om->data = REAL(matrix);	// TODO: Class-check first?
		
		if(isMatrix(matrix)) {
			PROTECT(matrixDims = getAttrib(matrix, R_DimSymbol));
			dimList = INTEGER(matrixDims);
			om->rows = dimList[0];
			om->cols = dimList[1];
			UNPROTECT(1);	// MatrixDims
		} else if (isVector(matrix)) {			// If it's a vector, assume it's a row vector. BLAS doesn't care.
			if(OMX_DEBUG) { Rprintf("Vector discovered.  Assuming rowity.\n"); }
			om->rows = 1;
			om->cols = length(matrix);
		}
		if(OMX_DEBUG) { Rprintf("Data connected to (%d, %d) matrix.\n", om->rows, om->cols); }
	}	
	
	om->localData = FALSE;
	om->colMajor = TRUE;
	om->originalRows = om->rows;
	om->originalCols = om->cols;
	om->originalColMajor = TRUE;
	om->aliasedPtr = om->data;
	om->algebra = NULL;
	om->objective = NULL;
	
	if(OMX_DEBUG) { Rprintf("Pre-compute call.\n");}
	omxMatrixCompute(om);
	if(OMX_DEBUG) { Rprintf("Post-compute call.\n");}
	
	return om;
}

void omxRemoveRowsAndColumns(omxMatrix *om, int numRowsRemoved, int numColsRemoved, int rowsRemoved[], int colsRemoved[])
{
	if(OMX_DEBUG) { Rprintf("Removing %d rows and %d columns from 0x%0x.\n", numRowsRemoved, numColsRemoved, om);}
	
	if(om->aliasedPtr == NULL) {  // This is meant only for aliased matrices.  Maybe Need a subclass?
		error("removeRowsAndColumns intended only for aliased matrices.\n");
	}
	
	if(numRowsRemoved < 1 || numColsRemoved < 1) { return; }
		
	int numCols = 0;
	int nextCol = 0;
	int nextRow = 0;
	int oldRows = om->rows;
	int oldCols = om->cols;
	int j,k;
	
	om->rows = om->rows - numRowsRemoved;
	om->cols = om->cols - numColsRemoved;
	
	// Note:  This really aught to be done using a matrix multiply.  Why isn't it?
	if(om->colMajor) {
		for(int j = 0; j < oldCols; j++) {
			if(OMX_DEBUG) { Rprintf("Handling %d rows.\n", j);}
			if(colsRemoved[j]) {
				continue;
			} else {
				nextRow = 0;
				for(int k = 0; k <= oldRows; k++) {
					if(OMX_DEBUG) { Rprintf("Examining (%d, %d).\n", j, k, om);}
					if(rowsRemoved[k]) {
						continue;
					} else {
						omxSetMatrixElement(om, nextRow, nextCol, om->aliasedPtr[k + j * oldRows]);
						nextRow++;
					}
				}
				nextCol++;
			}
		}
	} else {
		for(int j = 0; j < oldRows; j++) {
			if(rowsRemoved[j]) {
				continue;
			} else {
				nextCol = 0;
				for(int k = 0; k <= oldCols; k++) {
					if(colsRemoved[k]) {
						continue;
					} else {
						omxSetMatrixElement(om, nextRow, nextCol, om->aliasedPtr[k + j * oldCols]);
						nextCol++;
					}
				}
				nextRow++;
			}
		}
	}

	omxMatrixCompute(om);
}

/* Function wrappers that switch based on inclusion of algebras */
void omxPrintMatrix(omxMatrix *source, char* d) { 					// Pretty-print a (small) matrix
	if(source->algebra != NULL) omxAlgebraPrint(source->algebra, d);
	else if(source->objective != NULL) omxObjectivePrint(source->objective, d);
	else omxMatrixPrint(source, d);
}

unsigned short inline omxNeedsUpdate(omxMatrix *matrix) {
	if(OMX_DEBUG) {Rprintf("MatrixNeedsUpdate?");}
	if(matrix->isDirty) return TRUE;
	if(matrix->algebra != NULL) return omxAlgebraNeedsUpdate(matrix->algebra); 
	else if(matrix->objective != NULL) return omxObjectiveNeedsUpdate(matrix->objective);
	else return omxMatrixNeedsUpdate(matrix);
}

void inline omxRecomputeMatrix(omxMatrix *matrix) {
	if(!omxNeedsUpdate(matrix)) return;
	if(matrix->algebra != NULL) omxAlgebraCompute(matrix->algebra);
	else if(matrix->objective != NULL) omxObjectiveCompute(matrix->objective);
	else omxMatrixCompute(matrix);
}

void inline omxComputeMatrix(omxMatrix *matrix) {
	if(matrix->algebra != NULL) omxAlgebraCompute(matrix->algebra);
	else if(matrix->objective != NULL) omxObjectiveCompute(matrix->objective);
	else omxMatrixCompute(matrix);
}
