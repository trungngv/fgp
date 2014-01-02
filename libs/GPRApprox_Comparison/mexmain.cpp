//-------------------------------------------------------------------
// The code was written by Changjiang Yang and Vikas Raykar
// and is copyrighted under the Lesser GPL: 
//
// Copyright (C) 2006  Changjiang Yang and Vikas Raykar
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; version 2.1 or later.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
// See the GNU Lesser General Public License for more details. 
// You should have received a copy of the GNU Lesser General Public
// License along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, 
// MA 02111-1307, USA.  
//
// The author may be contacted via email at:cyang(at)sarnoff(.)com
// vikas(at)umiacs(.)umd(.)edu
//-------------------------------------------------------------------

//-------------------------------------------------------------------
// File    : mexmain.cpp
// Purpose : Interface between MATLAB and C++
// Author  : Vikas C. Raykar (vikas@cs.umd.edu)
// Date    : April 25 2005
//-------------------------------------------------------------------

#include "mex.h"
#include "KCenterClustering.h"

//The gateway function

void mexFunction(int nlhs,				// Number of left hand side (output) arguments
				 mxArray *plhs[],		// Array of left hand side arguments
				 int nrhs,              // Number of right hand side (input) arguments
				 const mxArray *prhs[])  // Array of right hand side arguments
{

  //check for proper number of arguments 
 
  if(nrhs != 4) mexErrMsgTxt("Four inputs required.");
  if(nlhs != 5) mexErrMsgTxt("Six output required.");

   //////////////////////////////////////////////////////////////
  // Input arguments
  //////////////////////////////////////////////////////////////
  
  //------ the first input argument: Dim ---------------//

  int argu = 0;

  /* check to make sure the input argument is a scalar */
  if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) || mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) 
  {
    mexErrMsgTxt("Input Dim must be a scalar.");
  }

  /*  get the scalar input Dim */
  int Dim = (int) mxGetScalar(prhs[argu]);
  if (Dim <= 0) mexErrMsgTxt("Input Dim must be a positive number.");

  //------ the second input argument: NSources ---------------//

  argu = 1;

  /* check to make sure the input argument is a scalar */
  if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) || mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) 
  {
    mexErrMsgTxt("Input NSources must be a scalar.");
  }

  /*  get the scalar input NSources */
  int NSources = (int) mxGetScalar(prhs[argu]);
  if (NSources <= 0) mexErrMsgTxt("Input NSources must be a positive number.");

  
  //----- the third input argument: pSources--------------//
  //  The 2D array is column-major: each column represents a point.

  argu = 2;

  /*  create a pointer to the input vector pSources */
  double *pSources = mxGetPr(prhs[argu]);
  
  int mrows = (int)mxGetM(prhs[argu]); //mrows
  int ncols = (int)mxGetN(prhs[argu]); //ncols
  if ( mrows != Dim && ncols != NSources)  mexErrMsgTxt("Input PSources must be a Dim x N matrix");

  
  //------ the fourth input argument: NumClusters ---------------//

  argu = 3;

  /* check to make sure the input argument is a scalar */
  if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) || mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) 
  {
    mexErrMsgTxt("Input NumClusters must be a scalar.");
  }


  int NumClusters = (int) mxGetScalar(prhs[argu]);
  if (NumClusters <= 0) mexErrMsgTxt("Input NumClusters must be a positive number.");


  //////////////////////////////////////////////////////////////
  // Output arguments
  //////////////////////////////////////////////////////////////


  //------ the second output argument: pClusterIndex ---------------//
  
  plhs[1]=mxCreateNumericMatrix(1,NSources,mxUINT32_CLASS,mxREAL);
  int *pClusterIndex =(int*) mxGetPr(plhs[1]);


  //////////////////////////////////////////////////////////////
  // function calls;
  //////////////////////////////////////////////////////////////

  //k-center clustering

  KCenterClustering* pKCC = new KCenterClustering(
	  Dim,
	  NSources,
	  pSources,
    pClusterIndex,
	  NumClusters
	);

  pKCC->Cluster();

  //------ the first output argument: MaxClusterRadius ---------------//
  
  plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
  double *MaxClusterRadius = mxGetPr(plhs[0]);
  MaxClusterRadius[0]=pKCC->MaxClusterRadius;
 

  //------ the third output argument: pClusterCenters ---------------//
  
  plhs[2] = mxCreateDoubleMatrix(Dim,NumClusters,mxREAL);
  double *pClusterCenters = mxGetPr(plhs[2]);

  //------ the fourth output argument: pNumPoints ---------------//
  
  plhs[3]=mxCreateNumericMatrix(1,NumClusters,mxUINT32_CLASS,mxREAL);
  int *pNumPoints =(int*) mxGetPr(plhs[3]);

   //------ the fifth output argument: pClusterRadii ---------------//
 
  plhs[4]=mxCreateDoubleMatrix(1,NumClusters,mxREAL);
  double *pClusterRadii = mxGetPr(plhs[4]);
 
  pKCC->ComputeClusterCenters(NumClusters,pClusterCenters,pNumPoints,pClusterRadii);

  delete pKCC;
 
  return;
  
}
