//----------------------------------------------------------------------------
// SurfaceGrowthProto.h - contains definition of Material array of structures,
// function prototypes used in SurfaceGrowth.cpp and other definitions.
// (c) 2010 Mykola Prodanov
// (this code was written in Sumy, Ukraine)
//----------------------------------------------------------------------------

#ifndef __SURFACE_GROWTH_PROTO__
#define __SURFACE_GROWTH_PROTO__

#include "SurfaceGrowth.h"		// Contains necessary CUDA and other headers and wrappers.

int ReadInputFile(TCHAR *szInpFile);
// Computational functions that are invoked on host.
void AllocArrays ();							
void FreeArrays ();
int SetParams();				// If 0 then the system is too large.
int SetupJob();					// If 0 then error in initialization of coordinates.
void InitRand (int randSeedI, SimParams *hparams);
real RandR (SimParams *hparams);
void VRand (float3 *p, SimParams *hparams);
void VRandRfloat4 (float4 *p, SimParams *hparams);
void InitVels();
void InitAccels();
void AccumProps (int icode, SimParams *hparams);
void EamInit();
void GrapheneInit();

// Array of structures with data for materials (metals) (from Zhou et al.).
// Order is important: Cu(0), Ag(1), Au(2), Ni(3), Al(4), Pb(5)
MATERIAL Material [] = 
{
	// copper Cu(0)
	// szName,	re,	       fe,        rhoe,        alpha,     beta,      A,       B,
	TEXT("Cu"), 2.556162, 1.554485,  22.150141,  7.669911,  4.090619,  0.327584, 0.468735,
	// kappa,   lambda,    Fn[0],     Fn[1],       Fn[2],     Fn[3], 
	0.431307,   0.86214,  -2.176490, -0.140035,  0.285621,  -1.750834,
	// F[0],     F[1],     F[2],      F[3],       eta         Fe,       massMe,   density
    -2.19,      0,        0.702991,  0.683705,   0.921150,  -2.191675,  63.54,   8930,

	// silver Ag(1)
	// szName,	re,	       fe,        rhoe,        alpha,     beta,      A,       B,
	TEXT("Ag"), 2.891814, 1.106232,  15.539255,  7.944536,  4.237086,  0.266074, 0.386272,
	// kappa,   lambda,    Fn[0],     Fn[1],       Fn[2],     Fn[3], 
	0.425351,   0.850703, -1.729619, -0.221025,  0.541558,  -0.967036,
	// F[0],     F[1],     F[2],      F[3],       eta         Fe,       massMe,   density
    -1.75,      0,        0.983967,  0.520904,   1.149461,  -1.751274,  107.868,  10490,

	// gold Au(2)
	// szName,	re,	       fe,        rhoe,        alpha,     beta,      A,       B,
	TEXT("Au"), 2.885034, 1.529021,  21.319637,  8.086176,  4.312627,  0.230728, 0.336695,
	// kappa,   lambda,    Fn[0],     Fn[1],       Fn[2],     Fn[3], 
	0.420755,   0.841511, -2.930281, -0.554034,  1.489437,  -0.886809,
	// F[0],     F[1],     F[2],      F[3],       eta         Fe,       massMe,   density
    -2.98,      0,        2.283863,  0.494127,   1.286960,  -2.981365,  196.9665,  19302,

	// nickel Ni(3)
	// szName,	re,	       fe,        rhoe,        alpha,     beta,      A,       B,
	TEXT("Ni"), 2.488746, 2.007018,  27.984706,  8.029633,  4.282471,  0.439664, 0.632771,
	// kappa,   lambda,    Fn[0],     Fn[1],       Fn[2],     Fn[3], 
	0.413436,   0.826873, -2.693996, -0.066073,  0.170482,  -2.457442,
	// F[0],     F[1],     F[2],      F[3],       eta         Fe,       massMe,   density
    -2.70,      0,        0.282257,  0.102879,   0.509860,  -2.700493,  58.71,    8902,

	// aluminium Al(4)
	// szName,	re,	       fe,        rhoe,        alpha,     beta,      A,       B,
	TEXT("Al"), 2.886166, 1.392302,  20.226537,  6.942419,  3.702623,  0.251519, 0.313394,
	// kappa,   lambda,    Fn[0],     Fn[1],       Fn[2],     Fn[3], 
	0.395132,   0.790264, -2.806783, -0.276173,  0.893409,  -1.637201,
	// F[0],     F[1],     F[2],      F[3],       eta         Fe,       massMe,   density
    -2.83,      0,        0.929508,  -0.682320,   0.779208,  -2.829437, 26.98154,  2698.9,	

	// lead Pb(5)
	// szName,	re,	       fe,        rhoe,        alpha,     beta,      A,       B,
	TEXT("Pb"), 3.499723, 0.647872,  8.906840,   8.468412,  4.516486,  0.134878, 0.203093,
	// kappa,   lambda,    Fn[0],     Fn[1],       Fn[2],     Fn[3], 
	0.425877,   0.851753, -1.419644, -0.228622,  0.630069,  -0.560952,
	// F[0],     F[1],     F[2],      F[3],       eta         Fe,       massMe,   density
    -1.44,      0,        0.921049,  0.108847,   1.172361,  -1.440494, 207.19,    11300
};

#endif	// End __SURFACE_GROWTH_PROTO__