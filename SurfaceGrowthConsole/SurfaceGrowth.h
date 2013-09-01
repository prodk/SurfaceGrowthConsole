//-------------------------------------------------------------------------------
// SurfaceGrowth.h - contains necessary headers and prototypes of wrappers 
// from SurfaceGrowth.cu, should be included into the main file SurfaceGrowth.cpp
// (c) 2010 Mykola Prodanov
// (this code was written in Sumy, Ukraine)
//-------------------------------------------------------------------------------

#ifndef __SURFACE_GROWTH__
#define __SURFACE_GROWTH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sm_12_atomic_functions.h>

#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <math.h>
#include <time.h>
#include <tchar.h>
#include <conio.h>

#include "math_constants.h"
#include "vector_functions.h"

#ifndef __CUDACC__
#define __CUDACC__					// For atomic functions.
#endif

#include <sm_11_atomic_functions.h>	
#include <sm_12_atomic_functions.h>	// For atomic functions.

#include "ComputeDefs.h"			// For computations.

#define BLOCK_SIZE 64				// Threads per block.

// For generation of random numbers.
#define IADD   453806245
#define IMUL   314159269
#define MASK   2147483647
#define SCALE  0.4656612873e-9

// Some definitions from windows.h
#ifndef TEXT
#define TEXT(quote) quote
#endif

#define BOOL int
#define FALSE 0
#define TRUE 1
#define INT int

#ifndef MAX_PATH
#define MAX_PATH 264
#endif

#ifndef lstrcpy
#define lstrcpy strcpy
#endif

#ifndef lstrcat
#define lstrcat strcat
#endif

#define RtlZeroMemory(Destination,Length) memset((Destination),0,(Length))
#define ZeroMemory RtlZeroMemory

// Simulation parameters.
struct SimParams 
{
	float3	region;			// simulation region
	float3	vSum;			// total impulse
	uint3	initUcell;		// number of unit cells	in the crystal structure
	real	deltaT,			// time step
			rCutC,			// cutoff distance for graphene
			rrCutC,			// square of cutoff for graphene
			temperature, 
			velMag;			// magnitude of initial velocity
	int		moreCycles,		// continue computing			
			stepAvg,		// average interval
			stepEquil,		// equilibration period
			stepCount,
			stepPdb,		// when to create pdb file
			stepLimit;
	int		bPdb;			// whether to create pdb file
	int		bResult;		// whether to create result file
	int		nMol;			// number of all atoms		
	int		nMolMe;			// number of metallic atoms
	uint	gridSize;		// number of blocks
	uint	blockSize;		// number of threads in a block == maximum atoms per cell
	int3	cells;			// number of cells for building neighbor list
	float3	invWidth;		// inverse width of a cell 
	real	rNebrShell;		// skin thickness for neighbor list
	real	rrNebr;			// rmax * rmax
	int		iNebrMax;		// maximum number of neighbors
	int		maxMol;			// size of array for particles in cells, including virtual
	int		nebrNow;		// update neighbor list
	real	dispHi;			// highest displacement over the time	
	Prop	kinEnergy;		// kinetic energy
	Prop	totEnergy;		// total energy
	Prop	potEnergy;		// potential energy
	Prop	oneStep;		// time of one step

	// EAM parameters
	real	a;				// unit cell in Me, angstrom
	real	re;				// distance between nearest neigbors = a / sqrt(2), angstrom
	real	fe;				// for rho
	real	rhoe;			// distance to the nearest neighbor
	real	alpha, beta;	// dimensionless
	real	A, B;			// eV
	real	kappa, lambda;	// dimensionless
	real	Fn[4], F[4];	// eV
	real	eta;			// dimensionless, exponent
	real	Fe;				// eV
	real	massMe;			// atomic mass	
	real	density;		// density of metal
	real	rCutEam,		// cutoff distance
			rrCutEam,		// square of cutoff
			rei,			// 1 / re
			rhoei;			// 1 / rhoe
	char	szNameMe[3];		// name of the metal
	// units of measurements	
	real	temperatureU;
	real	kB;
	real	enU;
	real	lengthU;
	real	forceU;
	real	massU;
	// graphene potential parameters
	real	mu_r,			// 41.881 eV / angsrom^2
			theta_0,		// 2*M_PI/3, dimensionless
			mu_theta,		// 2.9959 eV / angstrom^2
			mu_p;			// 18.225 eV / angstrom^2
	real	z_0;			// initial z coordinate
	// Berendsen thermostat variables
	int		stepThermostat;	// how often thermostat is applied
	real	gammaBerendsen;	// friction coefficient
	int		stepCool;		// how long thermostat is applied to metal atoms after stepEquil steps
	// regime (bulk == 0, surface growth == 1, shear == 2)
	int		iRegime;
	// Lennard-Jones parameters for metal-substrate interactions
	real	sigmaLJ, epsLJ, rCutLJ, rrCutLJ;
	// deposition
	real	velMagDepos;	// velocity of the deposited atom in z direction = sqrt(2*eDepos / m)
	int		randSeedP;		// initial value (seed) for random numbers
	int		stepDeposit,	// how often to insert atoms
			nMolDeposited,	// atoms have been already deposited
			nMolToDeposit;	// how many atoms are simultaneously deposited
	// measurements of tribology
	Prop	centerOfMass,	// coordinate of center of mass of nanoparticle along shear direction
			cmVel,			// velocity of center of mass of the nanoparticle (x component)
			frictForce;		// friction force acting on the particle from graphene (x comptonent)
	real	deltaF;			// incremental value of the shear force applied to the particle
	float3	particleSize;	// dimensions of the particle
	int		cellShiftZ;		// cells under the graphene layer
	real	shear;			// current value of total shear
	real	totalShear;		// total force acting on the nanoparticle
	// number of metal cells (in shear mode)
	int		initUcellMeX;
	int		initUcellMeY;
	// rdf variables
	real	rangeRdf;			// maximum distance between atoms for rdf
	int		limitRdf;			// number of measurements
	int		sizeHistRdf;		// number of intervals in the histogram
	int		stepRdf;			// how often to make measurements of rdf
	int		countRdf;			// number of measurements
	real	intervalRdf;		// == sizeHistRdf/rangeRdf, to avoid division
	int		bRdf;				// whether to compute rdf
	char	szRdfPath[MAX_PATH];// file path for rdf
	// back up
	TCHAR	szBckup0[20];		// file paths for backup
	TCHAR	szBckup1[20];		// file paths for backup
	BOOL	bBckup;				// whether to use backup
	BOOL	bStartBckup;		// whether to start from backup file
	int		stepBckup;			// how often to create backup file
	real	totalTime;			// duration of the simulation, in seconds
	// material
	int		iMaterial;
	// diffusion variables
	int nValDiffuse;			// number of measurements in the current set (buffer)
	int nBuffDiffuse;			// number of sets (buffers)
	int stepDiffuse;			// how often the measurements are performed
	int limitDiffuseAv;			// number of measurements for averaging
	int countDiffuseAv;			// counter of measurements
	char szDiffusePath[MAX_PATH];// file path for resulting file
};

// Structure for buffer, used in diffusion calcualtion.
typedef struct _TBuf
{
	float3	orgR,	// origin for the set of measurement
			rTrue;	// true displacement without wraparound effects
	real *rrDiffuse;// accumulates the mean-square displacemnets
	int count;		// count
}
TBuf;

// Structure with metal parameters for EAM potential.
typedef struct _MATERIAL
{
	// EAM parameters	
	char*	szName;
	real	re;				// distance between nearest neighbors = a / sqrt(2), angstrom
	real	fe;				// for rho
	real	rhoe;
	real	alpha, beta;	// dimensionless
	real	A, B;			// eV
	real	kappa, lambda;	// dimensionless
	real	Fn[4], F[4];	// eV
	real	eta;			// dimensionless, exponent
	real	Fe;				// eV
	real	massMe;			// atomic mass	(a.u.)
	real	density;		// density of metal	(kg / m^3)
} MATERIAL;

// prototypes of wrappers implemented in SurfaceGrowth.cu
extern "C" 
{
void CudaInitW(int argc, char **argv);
void CudaGLInitW(int argc, char **argv);
void RegisterGLBufferObjectW(uint vbo);
void UnregisterGLBufferObjectW(uint vbo);
void *MapGLBufferObjectW(uint vbo);
void UnmapGLBufferObjectW(uint vbo);
void SetParametersW(SimParams *hostParams);
void SetColorW(float4 *g_dcolor, SimParams* g_hParams);

// Srappers for calculations.
const char* InitCoordsW(float4 *dr, float4 *hr, SimParams* g_hParams);	
char* DoComputationsGLW(float4 *hr, float3 *hv, float3 *ha, float4 *dr, SimParams *hparams, 
					 FILE *fResults, TCHAR *szPdbPath);
char* DoComputationsW(float4 *hr, float3 *hv, float3 *ha, SimParams *hparams, 
					 FILE *fResults, TCHAR *szPdbPath);

}	// extern "C"

// Units of measurement
// Carbon atomic mass = 12.0107 * 1.66053 * 10^(-27) = 19.9441 * 10^(-27) kg				
// Graphite bond length 1.42 angsrom
// time unit = 0.2 ps
// energy unit is chosen to set stepCount == 0.0005 equal to 0.1fs
// energy unit == 6.275049 * 10^(-2)eV == 10.0538208 * 10^(-21) J
// kB == 1.3807*10^(-23) J / K
// dimensionless temperature 1K is 0.00137331158
// 298K dimensionless == 298 K * kB / (10.0538208 * 10^(-21) J) == 0.409246851

#endif		// __SURFACE_GROWTH__