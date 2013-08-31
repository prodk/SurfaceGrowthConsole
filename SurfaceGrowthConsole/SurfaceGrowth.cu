//-------------------------------------------------------------
// SurfaceGrowth.cu - contains kernel functions for computing, 
// C wrappers for some of the CUDA API and wrappers for the
// kernels for convenient calling from "SurfaceGrowth.cpp"
// Copyright (c) 2010 Mykola Prodanov
// (this code has been written in Sumy, Ukraine)
//-------------------------------------------------------------
#include "SurfaceGrowth.h"			// includes cuda headers and prototypes

cudaDeviceProp gDeviceProp;			// device properties

// globals for computing, prefix h - host variable, no prefix or d - device variable
__device__ __constant__ SimParams dparams;		// global device variable in constant memory for GPU
__device__ __constant__ SimParams *pdparams;		// global device variable in constant memory for GPU
__device__	uint count = 0;			// for sum, from Programming guide, p. 111
__shared__	bool isLastBlockDone;	// for sum, from Programming guide, p. 111

__shared__	int K[BLOCK_SIZE];		// BLOCK_SIZE is defined in SurfaceGrowth.h
__shared__	float4 B[BLOCK_SIZE];

//////////////////////////////////////////////////////////
// prototypes of some host functions called from wrappers
//////////////////////////////////////////////////////////

void AccumProps (int icode, SimParams *hparams);	// accumulate properties
void PrintSummary (FILE *fp, SimParams *hparams);	// print results in a file
int CreatePdbFile(char *szFilePath, SimParams *hparams, float4 *r);// create .pdb file
void PrintRdf(SimParams *hparams, uint *hHistRdf);	// prints rdf data in a file
// for random numbers
real RandR (SimParams *hparams);
void VRandRfloat4 (float4 *p, SimParams *hparams);
// host functions for diffusion
void InitDiffusion(TBuf *tBuf, real *rrDiffuseAv, SimParams *hparams);
void ZeroDiffusion(real *rrDiffuseAv, SimParams *hparams);
void PrintDiffusion(real *rrDiffuseAv, FILE *file, SimParams *hparams);
void AccumDiffusion(TBuf *tBuf, real *rrDiffuseAv, FILE *file, SimParams *hparams);
void EvalDiffusion(TBuf *tBuf, real *rrDiffuseAv, FILE *file, SimParams *hparams, float3 centerOfMass);

////////////////////
// device functions
////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
// eam functions (X.W.Zhou, H.N.G.Wadley, R.A.Johnson et al. Acta Materialia 49 (2001) 4005)
////////////////////////////////////////////////////////////////////////////////////////////
// compute phi
__device__ real EamPhi(real r)
{
	real rei = dparams.rei;		// diminish number of calls to constant memory
	real r_rei = r*rei;			// avoid redundant multiplications

	real phi = 
		dparams.A * expf( -dparams.alpha * (r_rei-1.f) ) / 
	( 1.f + powf( (r_rei - dparams.kappa), 20.f) ) -
		dparams.B * expf( -dparams.beta * (r_rei-1.f) ) / 
	( 1.f + powf( (r_rei - dparams.lambda) , 20.f) );

	return phi;
}

// compute f
__device__ real Eamf(real r)
{
	real rei = dparams.rei;
	real r_rei = r*rei;

	return
		dparams.fe*expf( -dparams.beta*(r_rei-1.f) ) / 
		(1.f + powf( (r_rei - dparams.lambda) , 20.f) );
}

// when rho < rhon = 0.85*rhoe
__device__ real EamFrhoSmall(real rho)
{
	//real rhon = 0.85f * dparams.rhoe;
	real rhoni = dparams.rhoei * 1.176471f;	// 1 / (0.85*rhoe)
	real rho_rhoni = rho*rhoni - 1.f;

	return 
		dparams.Fn[0] + 
		dparams.Fn[1]* rho_rhoni + 
		dparams.Fn[2]* rho_rhoni * rho_rhoni +
		dparams.Fn[3]* rho_rhoni * rho_rhoni * rho_rhoni;
}
// when rhon <= rho < rhoo = 1.15*rhoe
__device__ real EamFrhoMedium(real rho)
{	
	real rhoei = dparams.rhoei;
	real rho_rhoei = rho*rhoei - 1.f;

	return 
		dparams.F[0] + 
		dparams.F[1]* rho_rhoei + 
		dparams.F[2]* rho_rhoei * rho_rhoei +
		dparams.F[3]* rho_rhoei * rho_rhoei * rho_rhoei;
}
// when rhoo = 1.15*rhoe <= rho 
__device__ real EamFrhoLarge(real rho)
{
	real rhoei = dparams.rhoei;
	real rho_rhoei = rho*rhoei;

	return
		dparams.Fe * ( 1.f - logf(powf(rho_rhoei, dparams.eta)) ) * powf(rho_rhoei, dparams.eta);
}
// check what embedding functional to use
__device__ real EamF(real rho)
{
	real rhoe = dparams.rhoe;
	real rhon = 0.85f*rhoe;

	if(rho < rhon) return EamFrhoSmall(rho);

	real rhoo = 1.15f*rhoe;
	if( (rho >= rhon) && (rho < rhoo) ) return EamFrhoMedium(rho);

	if( rho >= rhoo ) return EamFrhoLarge(rho);

	return 0.f;
}
////////////////////////////////
// derivatives of eam functions
////////////////////////////////

// dphi / dr
__device__ real EamDPhi(real r)
{
	real rei = dparams.rei;			// diminish number of calls to constant memory
	real alpha = dparams.alpha;
	real beta = dparams.beta;
	real kappa = dparams.kappa;
	real lambda = dparams.lambda;
	real r_rei = r*rei;				// to avoid redundant multiplication

	real denomA = 1.f + powf( (r_rei - kappa), 20.f );
	real denomB = 1.f + powf( (r_rei - lambda) , 20.f );

	return 
		(
		-dparams.A*expf( -alpha*(r_rei-1.f) ) * 
		( alpha + 20.f*powf( (r_rei - kappa),19.f )/denomA ) / denomA +
		dparams.B*expf( -beta*(r_rei-1.f) ) *
		( beta + 20.f*powf( (r_rei - lambda),19.f )/denomB ) / denomB
		)*rei;
}

// df / dr
__device__ real EamDf(real r)
{
	real rei = dparams.rei;
	real beta = dparams.beta;
	real lambda = dparams.lambda;
	real r_rei = r*rei;				// to avoid redundant multiplication

	real denomB = 1.f + powf( (r_rei - lambda) , 20.f);
	
	return
		-dparams.fe * rei * expf(-beta*(r_rei-1.f)) * 
		( beta + 20.f*powf( (r_rei - lambda),19.f )/denomB ) / denomB;
}
// dF / drho when rho < rhon = 0.85*rhoe
__device__ real EamDFrhoSmall(real rho)
{
	real rhoni = dparams.rhoei * 1.176471f;	// 1/rhon = 1 / (0.85*rhoe)
	real rho_rhoni = rho * rhoni - 1.f;			// to avoid redundant multiplication

	return
		rhoni * (
		dparams.Fn[1] + 
		2.f*dparams.Fn[2]* rho_rhoni +
		3.f*dparams.Fn[3]* rho_rhoni * rho_rhoni
		);
}
// dF / drho when rhon <= rho < rhoo = 1.15*rhoe
__device__ real EamDFrhoMedium(real rho)
{
	real rhoei = dparams.rhoei;
	real rho_rhoei = rho*rhoei;

	return
		rhoei * (
		dparams.F[1] + 
		2.f*dparams.F[2]*(rho_rhoei - 1.f) +
		3.f*dparams.F[3]*(rho_rhoei - 1.f)*(rho_rhoei - 1.f)
		);
}
// dF / drho when rhoo = 1.15*rhoe <= rho
__device__ real EamDFrhoLarge(real rho)
{
	real rhoei = dparams.rhoei;
	real eta = dparams.eta;
	real rho_rhoei = rho*rhoei;

	return 
		- dparams.Fe * eta * rhoei * powf(rho_rhoei, (eta-1.f) ) *
		logf(powf(rho_rhoei, eta));
}
// see derivative of what embedding functional to use
__device__ real EamDF(real rho)
{
	real rhoe = dparams.rhoe;
	real rhon = 0.85f*rhoe;

	if(rho < rhon) return EamDFrhoSmall(rho);

	real rhoo = 1.15f*rhoe;
	if( (rho >= rhon) && (rho < rhoo) ) return EamDFrhoMedium(rho);

	if( rho >= rhoo ) return EamDFrhoLarge(rho);

	return 0.f;
}

//////////////////////////////////////////////
// kernels (K at the end of the function name)
//////////////////////////////////////////////

/////////////////////////////////////////
// kernels for generating of coordinates
/////////////////////////////////////////

// initialize coordinates of fcc lattice
// each block is a strip with length of min unit cells, and width and height of 1 cell
// each thread calculates coordinates of atoms in one cell, which contains 4 atoms for fcc
__global__
void InitFccCoordsK(float4 *pos)
{
	float4 gap, c;
	uint j, n, nx, ny, nz;		

	VDiv (gap, dparams.region, dparams.initUcell);		// distance between the cells

	// define atom indexes
	nx = blockIdx.x;
	ny = blockIdx.y;
	nz = threadIdx.z;
	n = 4*(nx + ny * gridDim.x + nz * gridDim.x * gridDim.y);

	// handle threads which do not work
	if(n > dparams.nMol - 4) return;
	VSet (c, nx + 0.25f, ny + 0.25f, nz + 0.25f);	
	
	VMul (c, c, gap);
	VVSAdd (c, -0.5f, dparams.region);
	for (j = 0; j < 4; j ++) {
		pos[n].x = c.x;
		pos[n].y = c.y;
		pos[n].z = c.z;	
		pos[n].w = 0.0f;			
		if (j != 3) {
			if (j != 0) pos[n].x += 0.5f * gap.x;
			if (j != 1) pos[n].y += 0.5f * gap.y;
			if (j != 2) pos[n].z += 0.5f * gap.z;
		}		
		n += 1;
	}	
}

// initialize coordinate of the nanoparticle atoms
// as fcc lattice above the graphene sheet
__global__ void InitSlabCoordsK( float4 *pos )
{
	float4 gap, c;
	uint j, n, nx, ny, nz;
	float3 region;
	real shiftY;
	
	region.x = gridDim.x*dparams.a;
	// number of cells along y * lattice constant
	region.y = gridDim.y*dparams.a;
	// for symmetric coordinates
	if(blockDim.z == 1)	// if one layer	
		shiftY = 0.5f*(region.y - ceilf(0.25f*dparams.nMolMe/gridDim.x) * dparams.a);	
	else	
		shiftY = 0.f;

	region.z = dparams.region.z;// blockDim.z == number of layers of Me

	gap.x = gap.y = gap.z = dparams.a;		// distance between the cells
	
	// define atom indexes
	nx = blockIdx.x;
	ny = blockIdx.y;
	nz = threadIdx.z;
	n = 4*(nx + ny * gridDim.x + nz * gridDim.x * gridDim.y);
	
	// handle threads which do not work
	VSet (c, nx + 0.25f, ny + 0.25f, nz + 0.25f);		
	VMul (c, c, gap);
	VVSAdd (c, -0.5f, region);	
	for (j = 0; j < 4; j ++) {	
		if(n < dparams.nMolMe) 
		{
			pos[n].x = c.x;
			pos[n].y = c.y + shiftY;
			pos[n].z = c.z + 0.5f*dparams.a + dparams.cellShiftZ/dparams.invWidth.z;	
			pos[n].w = 0.0f;			
			if (j != 3) {
				if (j != 0) pos[n].x += 0.5f * gap.x;
				if (j != 1) pos[n].y += 0.5f * gap.y;
				if (j != 2) pos[n].z += 0.5f * gap.z;
			}			
			n += 1;
		}	
	}
}

__global__
void InitGrapheneCoordsK(float4 *pos)
{    
//  6	    7		// this is the numeration of atoms (rotated by 90 anticlockwise)
//	\	   /
//	 \4___/5
//	 /    \  
// 3/      \0
//	\	   /
//	2\____/1		// axes: y from right to left, x from bottom to top in this figure

	real dx, dy, x, y;
	int	n, cx, cy;

	real cos30 = 0.8660254038f;

	// find index of the atom
	n = blockIdx.x*blockDim.x + threadIdx.x;	// blockDim.x is assumed == 32

	if( n < (dparams.nMol-dparams.nMolMe) )
	{
		// find 2D index of the block in the grid of blocks
		cy = blockIdx.x / dparams.initUcell.x;
		cx = blockIdx.x - cy * dparams.initUcell.x;

		// all coordinates in the block are shifted by the following quantities
		dx = cx*8.f*cos30;	// here should be * a = 1.42 angstom, but it is = 1 in our units
		dy = cy*6.f;

		// find index of the thread in a 2D grid of blockDim.x threads
		cy = threadIdx.x / 8.f;	// each cell handles 8 atoms, there are blockDim.x/8 cells
		cx = threadIdx.x - cy*8.f;

		// see what atom we have
		switch(cx)	{
	case 0:
		x = cos30;
		y = 0.f;
		break;
	case 1:
		x = 0.f;
		y = 0.5f;
		break;
	case 2:
		x = 0.f;
		y = 1.5f;
		break;
	case 3:
		x = cos30;
		y = 2.f;
		break;
	case 4:
		x = 2.f*cos30;
		y = 1.5f;
		break;
	case 5:
		x = 2.f*cos30;
		y = 0.5f;
		break;
	case 6:
		x = 3.f*cos30;
		y = 2.f;
		break;
	case 7:
		x = 3.f*cos30;
		y = 0.f;
		break;
		}

		// additional shift dependently on their grid number due to thread
		if( cy == 1 )
			dx += 4.f*cos30;
		else if( cy == 2 )
			dy += 3.f;
		else if( cy == 3 ) {
		dx += 4.f*cos30;
		dy += 3.f;
		}		

		// shift the origin to the center of the layer
		dx = dx + 0.5f*(cos30 - dparams.region.x);
		dy = dy + 0.5f*(1.f - dparams.region.y);

		// save coordinates
		pos[n + dparams.nMolMe].x = x + dx;			// !! note shift in indexes due to metal
		pos[n + dparams.nMolMe].y = y + dy;
		pos[n + dparams.nMolMe].z = dparams.z_0;	
		pos[n + dparams.nMolMe].w = 0.0f;			// zero energy		
	}
}

// integrate equations of motion using Verlet method
__global__ void LeapfrogStepK (int part, float4 *dr, float3 *dv, float3 *da)
{
	// index of molecule for current thread
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( n < dparams.nMolMe )	// for metal process all atoms
	{
		if (part == 1)	{
			VVSAdd (dv[n], 0.5f * dparams.deltaT, da[n]);
			VVSAdd (dr[n], dparams.deltaT, dv[n]);    
		} 
		else {	
			VVSAdd (dv[n], 0.5f * dparams.deltaT, da[n]);
		}
	}
	else if( (n >= dparams.nMolMe ) && 
		( n < dparams.nMol ) )// for graphene don't process boundary atoms
	{
		real xLeft, yBottom, xRight, yTop;
		real cos30 = cos(M_PI / 6.f);

		xLeft = 0.5f*(cos30 - dparams.region.x);
		xRight = 0.5f*(dparams.region.x - cos30);
		yBottom = 0.5f*(1.f - dparams.region.y);
		yTop = 0.5f*(dparams.region.y - 1.f);

		if( (dr[n].x != xLeft) && (dr[n].x < xRight) && 
			(dr[n].y != yTop) && (dr[n].y != yBottom) )
		{
			if (part == 1) {
				VVSAdd (dv[n], 0.5f * dparams.deltaT, da[n]);
				VVSAdd (dr[n], dparams.deltaT, dv[n]);    
			} else {
				VVSAdd (dv[n], 0.5f * dparams.deltaT, da[n]);
			}
		}
	}	// end else if( (n >= dparams.nMolMe ) 
}

__global__ void ApplyBoundaryCondK( float4 *dr )
{
	// index of molecule for current thread
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	// don't perform redundant threads to prevent memory overwriting
	if(n < dparams.nMol) {
		// type manually without macroses to avoid problems
		// x
		if (dr[n].x >= 0.5f * dparams.region.x)      dr[n].x -= dparams.region.x;
		else if (dr[n].x < -0.5f * dparams.region.x) dr[n].x += dparams.region.x;
		// y
		if (dr[n].y >= 0.5f * dparams.region.y)      dr[n].y -= dparams.region.y;
		else if (dr[n].y < -0.5f * dparams.region.y) dr[n].y += dparams.region.y;
		// z
		if (dr[n].z >= 0.5f * dparams.region.z)      dr[n].z -= dparams.region.z;
		else if (dr[n].z < -0.5f * dparams.region.z) dr[n].z += dparams.region.z;
	}
}

__global__ void ApplyBerendsenThermostat( float3 *dv, real *vvSum, int stepCount )
{
	// index of molecule for current thread
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	real beta;

	// !consider only carbon atoms
	if( (n >= dparams.nMolMe) && (n < dparams.nMol) )
	{
		real kinEnergy = (*vvSum)*0.5f / dparams.nMol ;
		beta = 
			sqrtf( 1.f + dparams.gammaBerendsen *
			(dparams.temperature*1.5f*dparams.kB/kinEnergy - 1.f) );
		dv[n].x = beta * dv[n].x;
		dv[n].y = beta * dv[n].y;
		dv[n].z = beta * dv[n].z;
	}
	// for metal atoms
	if( (dparams.iRegime == 2)&&(n < dparams.nMolMe)&&
		(stepCount > dparams.stepEquil)&&		// this is redundant
		(stepCount < dparams.stepEquil + dparams.stepCool) )
	{
		real kinEnergy = (*vvSum)*0.5f / dparams.nMol ;
		beta = 
			sqrtf( 1.f + dparams.gammaBerendsen *
			(dparams.temperature*dparams.kB/kinEnergy - 1.f) );
		dv[n].x = beta * dv[n].x;
		dv[n].y = beta * dv[n].y;
		dv[n].z = beta * dv[n].z;
	}	
}

// deposit atoms (for Surface Growth regime)
__global__ void InsertAtomsK( float4 *dr, float3 *dv, int nMolDeposited, int nMolToDeposit )
{
	real z = 0.49f * dparams.region.z;

	// index of molecule for current thread
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	int iDeposited = nMolDeposited;
	int iInterval = iDeposited + nMolToDeposit;	

	if(n < dparams.nMolMe)		// process only metalic atoms
	{
		if( (n >= iDeposited) && (n < iInterval ) )	//atoms to deposit
		{			
			dr[n].z = z;
			// give initial velocity
			dv[n].z = -dparams.velMagDepos;// !note minus sign
		}
		if(n >= iInterval)		// it is early to deposit these atoms
		{			
			dr[n].z = 1.5f*dparams.region.z + 0.5f*(n+1)*dparams.region.z;
			dv[n].x = 0.f;
			dv[n].y = 0.f;
			dv[n].z = 0.f;
		}
	}
}

// apply shear (for Shear regime)
__global__ void ApplyShearK( float4 *dr, float3 *da, real shear, real centerOfMassX, 
							uint *numOfSharedMols )
{
	// index of molecule for current thread
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( n < dparams.nMolMe)
	{
		real deltaR = centerOfMassX - dr[n].x;
		// apply shear only to atoms that are to the left to the center of mass
		if( deltaR > 0.f)
		{
			da[n].x += shear;
			unsigned int val = 1;
			atomicAdd((unsigned int *)numOfSharedMols, val);
		}
	}	
}

//////////////////////////////////////////////
// kernels assosiated with force calculations
//////////////////////////////////////////////

// build cells
// each block finds particles that belong to the cell with the number equal to blockIdx.x;
// this is done as follows: for each step a thread reads particle coordinates, 
// so each such a step the block reads blockDim.x particles. Then each thread calculates
// index of the cell of the particle, and compares it with the blockIdx.x. If it is =,
// than the thread finds free place in CELL array by atomically enlarging number of
// atoms in the cell, and then stores atomically index at that place
__global__ void BinAtomsIntoCellsK(float4 *dr, int *CELL, uint *molsInCells)
{	
	float4	rShifted, r;					// shifted and initial coordinates
	int3	cc;								// 3D index of the cell
	int		c, whereToWrite, n;				// counters	

	int		count = 0;						// reads have been made
	int		numOfIter = floor( (float)dparams.nMol/blockDim.x );	// total number of reads

	while( 1 )
	{
		n = blockDim.x*count + threadIdx.x;	// get particle index for current thread				
		
		//check whether this is the last cycle
		if( count == numOfIter )
		{
			if( n < dparams.nMol )	r = dr[n];	// process partially filled portions
			else VZero(r);						// virtual particle, fill with 0

			// find index of the cell
			// shift coordinates into 1st quadrant
			rShifted.x = r.x + 0.5f * dparams.region.x;
			rShifted.y = r.y + 0.5f * dparams.region.y;
			rShifted.z = r.z + 0.5f * dparams.region.z;

			// find index of the cell in 3D x, y, z reprezentation
			VMul (cc, rShifted, dparams.invWidth);	// invWidth is placed in dparams to avoid division each time
			c = VLinear (cc, dparams.cells);		// convert index of the cell into 1D			
			
			if( n < dparams.nMol )	// if the particle is not virtual
			{
				if( c == blockIdx.x )	// if the particle is in the desired cell
				{
					// !note here atomicAdd, but not atomicInc !
					// it returns the previous nonincremented value, which is what we need
					whereToWrite = atomicAdd(&molsInCells[c], 1);	// find shift in the array
					CELL[ c * blockDim.x + whereToWrite ] = n;				
				}
			}	// end if( n < dparams.nMol )

			break;	// terminate loop while
		}	// end if( count == numOfIter )

		// if count < numOfIter
		r = dr[n];
		// find index of the cell
		// shift coordinates into 1st quadrant
		rShifted.x = r.x + 0.5f * dparams.region.x;
		rShifted.y = r.y + 0.5f * dparams.region.y;
		rShifted.z = r.z + 0.5f * dparams.region.z;

		// find index of the cell in 3D x, y, z reprezentation
		VMul (cc, rShifted, dparams.invWidth);	// invWidth is placed in params to avoid division each time
		c = VLinear (cc, dparams.cells);		// convert index of the cell into 1D

		if( c == blockIdx.x )	// if the particel is in the desired cell
		{
			whereToWrite = atomicAdd(&molsInCells[c], 1);	// find shift in the array			
			CELL[ c * blockDim.x + whereToWrite ] = n;		// save particle index			
		}
		++count;			
	}	// end while( 1 )	
}

// build neighbor list, algorithm 4 from Anderson et al.
__global__ void BuildNebrListK (float4 *dr,		// coordinates of molecules (global)							  
							   int *CELL,		// indexes of molecules (global)
							   int *NN,			// number of neighbors (global)
							   int *NBL)		// neighbor list (global)							  
{		
	int3 cc, m2v, vOff[] = OFFSET_VALS_LONG;	// 3D indexes of the cells
	int i, j, offset, n;						// counters
	int C;										// 1D cell indexes
	int nNeigh;									// number of neighbors, nebrTabLength in Rapaport
	float3 shift;								// variable for handling boundary conditions
	float4 A, deltaR;							// radus-vectors

	// define 3D index of the current cell, blockIdx.x is 1D index of the cell
	cc.z = blockIdx.x / (dparams.cells.x * dparams.cells.y);
	cc.y = (blockIdx.x - cc.z * dparams.cells.x * dparams.cells.y) / dparams.cells.x;
	cc.x = blockIdx.x - (cc.z*dparams.cells.y + cc.y)*dparams.cells.x;
	
// here begins implementation of the Algorithm 4 from Anderson
	n = blockIdx.x * blockDim.x + threadIdx.x;
	if(n < dparams.maxMol)
		i = CELL[blockIdx.x * blockDim.x + threadIdx.x];// step 1, get particle index
	else i = -1;
	nNeigh = 0;										// step 2
	if(i != -1)	{									// added by me to avoid memory problems
		A = dr[i];									// step 3			
	}
	// begin step 4
	for (offset = 0; offset < N_OFFSET_LONG; offset ++)	// loop over all 27 neighboring cells 
	{
		VAdd (m2v, cc, vOff[offset]);			// find 3D index of the neighboring cell
		
		shift.x = 0.f;							// zero shift for boundary conditions
		shift.y = 0.f;
		shift.z = 0.f;

		// apply boundary conditions
		// !type manually instead of using macroses, because they may not work!
		// x
		if (m2v.x >= dparams.cells.x) { 
			m2v.x = 0; 
			shift.x = dparams.region.x; 
		} else if (m2v.x < 0) {
			m2v.x = dparams.cells.x - 1;
			shift.x = - dparams.region.x;
		}
		// y
		if (m2v.y >= dparams.cells.y) { 
			m2v.y = 0; 
			shift.y = dparams.region.y; 
		} else if (m2v.y < 0) {
			m2v.y = dparams.cells.y - 1;
			shift.y = - dparams.region.y;
		}
		// z
		if (m2v.z >= dparams.cells.z) { 
			m2v.z = 0; 
			shift.z = dparams.region.z; 
		} else if (m2v.z < 0) {
			m2v.z = dparams.cells.z - 1;
			shift.z = - dparams.region.z;
		}

		C = VLinear (m2v, dparams.cells);			// find 1D index of the neighboring cell
	// end step 4

		__syncthreads();							// step 5
		K[threadIdx.x] = CELL[C * blockDim.x + threadIdx.x];	// step 6
		if(K[threadIdx.x] != -1)					// added by me, to avoid memory error
			B[threadIdx.x] = dr[ K[threadIdx.x] ];	// step 7
		else
			VZero(B[threadIdx.x]);					// added by me
		__syncthreads();							// step 8		
		
		if( i != -1)	{							// step 9
			for(j = 0; j < blockDim.x; j++)	
			{
				// step 10, loop over atoms from the current neighboring cell C
				if( K[j] != -1 )	{				// steps 11 - 13,					
					deltaR.x = A.x - B[j].x;		// step 14
					deltaR.y = A.y - B[j].y;
					deltaR.z = A.z - B[j].z;
					
					deltaR.x = deltaR.x - shift.x;	// step 15, boundary conditions
					deltaR.y = deltaR.y - shift.y;
					deltaR.z = deltaR.z - shift.z;
					
					if ( ((deltaR.x*deltaR.x + deltaR.y*deltaR.y + deltaR.z*deltaR.z) < 
						dparams.rrNebr) && (i != K[j]) )	// step 16
					{						
						NBL[nNeigh * dparams.nMol + i] = K[j];	// step 17
						++nNeigh;								// step 18
					}											// step 19
				}	// end if( K[j] != -1 )
			}													// step 20 
		}														// step 21	
	} // step 22 end for (offset = 0; offset < N_OFFSET_LARGE; offset ++) 		

	if(i != -1)
		NN[i] = nNeigh;		// step 23	
}

// compute rho for eam potential
// algorithm is the same as Compute Forces in Anderson;
// the use of separate kernel for this is to avoid thread syncronization
__global__ void EamComputeRhoK(	real	*rho,			// electron density (out)								  
								float4	*dr,			// array of coordinates (global)
								int		*NN,			// array of number of neighbors
								int		*NBL)			// neigbors list	
{	
	float4	A, B, deltaR;					// coordinates
	real	rr, rhoVal, rhoSum;				// scalars
	int		nNeigh;							// number of neighbors
	int		j, k;							// counters

	// zero quntities
	rhoSum = 0.f;	
	rhoVal = 0.f;

// begin Algorithm 2 from Anderson
	// index of molecule for current thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;	// step 1
	// do not perform redundant threads
	// !here nMolMe
	if(i < dparams.nMolMe) 
	{		
		A = dr[i];						// step 3, load coordinates of atom i
		nNeigh = NN[i];					// step 4, get the number of neighbors for atom i

		for(j = 0; j < nNeigh; j++)		// step 5, loop over all neighbors
		{
			k = NBL[j * dparams.nMol + i];	// step 6
			B = dr[k];						// step 7

			// begin step 8		
			deltaR.x = A.x - B.x;
			deltaR.y = A.y - B.y;
			deltaR.z = A.z - B.z;

			// periodic boundaries
			// x
			if (deltaR.x >= 0.5f * dparams.region.x)      deltaR.x -= dparams.region.x; 
			else if (deltaR.x < -0.5f * dparams.region.x) deltaR.x += dparams.region.x;
			// y
			if (deltaR.y >= 0.5f * dparams.region.y)      deltaR.y -= dparams.region.y; 
			else if (deltaR.y < -0.5f * dparams.region.y) deltaR.y += dparams.region.y;
			// z
			if (deltaR.z >= 0.5f * dparams.region.z)      deltaR.z -= dparams.region.z; 
			else if (deltaR.z < -0.5f * dparams.region.z) deltaR.z += dparams.region.z;
			// end step 8

			// begin step 9
			rr = deltaR.x*deltaR.x + deltaR.y*deltaR.y + deltaR.z*deltaR.z;
			if ( (rr < dparams.rrCutEam) && (k < dparams.nMolMe) )
			{
				rhoVal = Eamf( sqrt(rr) );
				rhoSum += rhoVal;
			}			
		}		// step 14
		rho[i] = rhoSum;		
// end Algorithm 2
	}	// end if(i < dparams.nMolMe)
}

// compute all forces based on the Algorithm 2 from Anderson et al.
__global__ void ComputeForcesK (	float3	*a,				// acceleration (out)								
									float4	*dr,			// array of coordinates (global)
									int	*NN,				// array of number of neighbors
									int	*NBL,				// neigbors list	
									real *rho,				// electron density for metal atom
									real *fForce )			// friction force (out)
{
	float3	fSum, C;			// forces
	float4	A, B, deltaR;		// coordinates
	real	rr, rri, rri3, fcVal, 
			r, rhoi, rhok;		// scalars
	real	uSum, uVal;			// potential energy
	int		nNeigh;				// number of neighbors
	int		j, k;				// counters

	float4	rnn[3];				// coordinates of the nearest neighbors of the current atom
	real	absRnn[3],			// modules of dist
			deltaZ[3], dZ,		// changes in applicate
			rjirki,				// scalar product divided by product of modules
			theta,				// angle between bonds
			tmp, dTheta;
	int		counti = 0;			// number of nearest neighbors

	int		nnk;				// neighbors of the nearest neighbor
	int		countk, m, l;		// counts
	float4	rm;					// coordinates of the second neighbor
	float3	frictForceVal;

	int		atomType;			// if == 0 then metal, if == 1, then carbon

	// zero quntities
	uSum = 0.f;	

	// index of molecule for current thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;	// step 1
	// define what atom we have
	if( i < dparams.nMolMe ) atomType = 0;	// metal
	if( (i >= dparams.nMolMe) && (i < dparams.nMol) ) atomType = 1; // carbon

	// do not perform redundant threads
	if(i < dparams.nMol) 
	{
		a[i].x = 0.f;
		a[i].y = 0.f;
		a[i].z = 0.f;

		// zero friction force
		VZero(frictForceVal);

		VZero(fSum);					// step 2
		A = dr[i];						// step 3
		nNeigh = NN[i];					// step 4
		if( atomType == 0 )
			rhoi = rho[i];					// load electron density if metal atom

		for(j = 0; j < nNeigh; j++)		// step 5
		{
			k = NBL[j * dparams.nMol + i];	// step 6
			B = dr[k];						// step 7
			
			deltaR.x = A.x - B.x;
			deltaR.y = A.y - B.y;
			deltaR.z = A.z - B.z;

			// periodic boundaries
			// x
			if (deltaR.x >= 0.5f * dparams.region.x)      deltaR.x -= dparams.region.x; 
			else if (deltaR.x < -0.5f * dparams.region.x) deltaR.x += dparams.region.x;
			// y
			if (deltaR.y >= 0.5f * dparams.region.y)      deltaR.y -= dparams.region.y; 
			else if (deltaR.y < -0.5f * dparams.region.y) deltaR.y += dparams.region.y;
			// z
			if (deltaR.z >= 0.5f * dparams.region.z)      deltaR.z -= dparams.region.z; 
			else if (deltaR.z < -0.5f * dparams.region.z) deltaR.z += dparams.region.z;
			// square of distance between atoms i and k
			rr = deltaR.x*deltaR.x + deltaR.y*deltaR.y + deltaR.z*deltaR.z;

			// compute force for the current atomic type
			switch( atomType )
			{
				case 0:							// metal
					if( k < dparams.nMolMe )	// neighbor is also metal, so compute EAM
					{
						r = sqrt( rr );
						rhok = rho[k];			// load electron density for atom k from global memory
						// compute paiwise interaction and energy
						fcVal = -EamDPhi( r );
						uVal = EamPhi( r );
						// compute interactions due to the embedding contribution
						fcVal = fcVal - EamDf( r ) *( EamDF(rhoi) + EamDF(rhok) );
						// very important! divide by module of the interatomic distance!
						// to normalize radius vector!
						fcVal = fcVal / r;			

						if (rr >= dparams.rrCutEam) fcVal = 0.0f;	// steps 10 - 12

						VSCopy (C, fcVal, deltaR);		// make vector
						VVAdd(fSum, C);		// step 13

						// added by me: compute energy
						if (rr < dparams.rrCutEam) uSum += uVal;
					}
					else						// neighbor is carbon, so compute LJ
					{
						rri = dparams.sigmaLJ*dparams.sigmaLJ / rr;		
						rri3 = Cube (rri);
						fcVal = 48.f * dparams.epsLJ * rri3 * (rri3 - 0.5f) * rri 
								/ (dparams.sigmaLJ*dparams.sigmaLJ);		
						uVal = 4.f * dparams.epsLJ * rri3 * (rri3 - 1.f);		
						// end step 9

						if (rr >= dparams.rrCutLJ) fcVal = 0.0f;	// steps 10 - 12
						// added by me compute energy
						if (rr < dparams.rrCutLJ) uSum += uVal;

						VSCopy (C, fcVal, deltaR);	

						VVAdd(fSum, C);		// step 13
						// save also friction force
						VVAdd(frictForceVal, C);
					}
					break;	// end metal

				case 1:							// carbon
					if( k < dparams.nMolMe )	// neighbor is metal, so compute LJ
					{
						rri = dparams.sigmaLJ*dparams.sigmaLJ / rr;		
						rri3 = Cube (rri);
						fcVal = 48.f * dparams.epsLJ * rri3 * (rri3 - 0.5f) * rri 
								/ (dparams.sigmaLJ*dparams.sigmaLJ);		
						uVal = 4.f * dparams.epsLJ * rri3 * (rri3 - 1.f);		
						// end step 9

						if (rr >= dparams.rrCutLJ) fcVal = 0.0f;	// steps 10 - 12
						// added by me compute energy
						if (rr < dparams.rrCutLJ) uSum += uVal;

						VSCopy (C, fcVal, deltaR);	

						VVAdd(fSum, C);		// step 13
						
					}
					else				// neighbor is also carbon, so compute spring force
					{
						// see whether this is the nearest neighbor
						if ( (rr < dparams.rrCutC) && (counti < 3) )	
						{
							absRnn[counti] = sqrt(rr);
							deltaZ[counti] = B.z - dparams.z_0;
							rnn[counti].x = deltaR.x;	
							rnn[counti].y = deltaR.y;	
							rnn[counti].z = deltaR.z;				
							
							// get number of neighbors for atom k
							nnk = NN[k];	
							countk = 0;		// number of nearest neighbors of atom k
							for(l = 0; l < nnk; l++)	// loop over neighbors of atom k
							{
								m = NBL[l * dparams.nMol + k];
								rm = dr[m];			// get neighbor's coordinate
								// note that for periodic boundaries we use rk - rm
								// but in forces we will use rm - rk and also use ri - rk
								deltaR.x = B.x - rm.x;
								deltaR.y = B.y - rm.y;
								deltaR.z = B.z - rm.z;
								// periodic boundaries
								// x
								if (deltaR.x >= 0.5f * dparams.region.x)      deltaR.x -= dparams.region.x; 
								else if (deltaR.x < -0.5f * dparams.region.x) deltaR.x += dparams.region.x;
								// y
								if (deltaR.y >= 0.5f * dparams.region.y)      deltaR.y -= dparams.region.y; 
								else if (deltaR.y < -0.5f * dparams.region.y) deltaR.y += dparams.region.y;
								// z
								if (deltaR.z >= 0.5f * dparams.region.z)      deltaR.z -= dparams.region.z; 
								else if (deltaR.z < -0.5f * dparams.region.z) deltaR.z += dparams.region.z;
								
								rr = deltaR.x*deltaR.x + deltaR.y*deltaR.y + deltaR.z*deltaR.z;
								if ((rr < dparams.rrCutC)&&(m != i))	// see whether this is the nearest neighbor
								{
									// compute angular part of the force for the neigbor atom
									VScale(deltaR, (-1.f));	// inverse direction of distance
									rjirki = VDot(rnn[counti],deltaR)/(absRnn[counti] * rr);
									theta = acosf(rjirki);
									dTheta = theta - dparams.theta_0;
									// note that there should be r_0*r_0, but in our units it is = 1.
									tmp = dparams.mu_theta * dTheta / sqrtf(1.f-rjirki*rjirki);
									fcVal = tmp/(absRnn[counti] * rr);
									VSCopy (C, fcVal, deltaR);	
									VVAdd(fSum, C);
									fcVal = -tmp*rjirki/(absRnn[counti] * absRnn[counti]);
									VSCopy (C, (fcVal), rnn[counti]);	
									VVAdd(fSum, C);
									// !potential energy is not computed!

									// z contribution						
									fcVal =	-dparams.mu_p*(rm.z - dparams.z_0)/9.f;
									C.x = 0.f;
									C.y = 0.f;
									C.z = fcVal;		
									VVAdd(fSum, C);									

									++ countk;	// enlarge number of nearest neighbors of atom k
								}
								// use goto, because we want terminate only one loop, but break will terminate all loops
								if( countk == 2 ) goto label;	// exit if 2 nn (except i) have been processed								
							} // end loop over neighbors of atom k
label:
							++ counti;	// enlarge number of nearest neighbors of atom i
						}	// end if (rr < dparams.rrCutC) && (counti < 3)
						if (counti == 3) 
						{
							// compute quantities associated with nearest neighbors
							// compute radial part
							for(j = 0; j < counti; j++)
							{
								// note that we use 1, because r_0 = 1.42 angstrom = 1 dimensionless
								fcVal = -dparams.mu_r*(absRnn[j] - 1.f)/absRnn[j];
								VSCopy (C, (fcVal), rnn[j]);	
								VVAdd(fSum, C);
								// potential energy
								uVal = dparams.mu_r*(absRnn[j] - 1.f)*(absRnn[j] - 1.f);
								uSum += uVal;
							}

							// compute contribution assosiated with z coordinate
							dZ = 2.f*(A.z-dparams.z_0)-(deltaZ[0] + deltaZ[1] + deltaZ[2]);
							fcVal =	-0.666666667f*dparams.mu_p*dZ;
							C.x = 0.f;
							C.y = 0.f;
							C.z = fcVal;		
							VVAdd(fSum, C);
							// potential energy
							dZ = (A.z-dparams.z_0)-0.33333333f*(deltaZ[0] + deltaZ[1] + deltaZ[2]);
							uVal = dparams.mu_p*dZ*dZ;
							uSum += uVal;		

							// compute contribution associated with bending bonds and angle	theta
							// vectors 0 and 1							
							rjirki = VDot(rnn[0],rnn[1])/(absRnn[0] * absRnn[1]);
							theta = acosf(rjirki);
							dTheta = theta - dparams.theta_0;
							// note that there should be r_0*r_0, but in our units it is = 1.
							tmp = dparams.mu_theta * dTheta / sqrtf(1.f-rjirki*rjirki);
							fcVal = tmp*(1.f - absRnn[0]*rjirki/absRnn[1])/(absRnn[0]*absRnn[1]);
							VSCopy (C, (fcVal), rnn[1]);	
							VVAdd(fSum, C);
							fcVal = tmp*(1.f - absRnn[1]*rjirki/absRnn[0])/(absRnn[0]*absRnn[1]);
							VSCopy (C, (fcVal), rnn[0]);	// !here rnn[0] 
							VVAdd(fSum, C);
							// potential energy
							uVal = dparams.mu_theta * dTheta * dTheta;
							uSum += uVal;

							// vectors 1 and 2
							rjirki = VDot(rnn[1],rnn[2])/(absRnn[1] * absRnn[2]);
							theta = acosf(rjirki);
							dTheta = theta - dparams.theta_0;
							// note that there should be r_0*r_0, but in our units it is = 1.
							tmp = dparams.mu_theta * dTheta / sqrtf(1.f-rjirki*rjirki);
							fcVal = tmp*(1.f - absRnn[1]*rjirki/absRnn[2])/(absRnn[1]*absRnn[2]);
							VSCopy (C, (fcVal), rnn[2]);	
							VVAdd(fSum, C);
							fcVal = tmp*(1 - absRnn[2]*rjirki/absRnn[1])/(absRnn[1]*absRnn[2]);
							VSCopy (C, (fcVal), rnn[1]);	
							VVAdd(fSum, C);
							// potential energy
							uVal = dparams.mu_theta * dTheta * dTheta;
							uSum += uVal;

							// vectors 2 and 0
							rjirki = VDot(rnn[2],rnn[0])/(absRnn[2] * absRnn[0]);
							theta = acosf(rjirki);
							dTheta = theta - dparams.theta_0;
							// note that there should be r_0*r_0, but in our units it is = 1.
							tmp = dparams.mu_theta * dTheta / sqrtf(1.f-rjirki*rjirki);
							fcVal = tmp*(1.f - absRnn[2]*rjirki/absRnn[0])/(absRnn[2]*absRnn[0]);
							VSCopy (C, (fcVal), rnn[0]);	
							VVAdd(fSum, C);
							fcVal = tmp*(1.f - absRnn[0]*rjirki/absRnn[2])/(absRnn[0]*absRnn[2]);
							VSCopy (C, (fcVal), rnn[2]);	
							VVAdd(fSum, C);
							// potential energy
							uVal = dparams.mu_theta * dTheta * dTheta;
							uSum += uVal;
							++ counti;	// to prevent following computations
						}	// end if (counti == 3)
					}	// end else		// neighbor is also carbon, so spring force
					break;	// end carbon

			}	// end switch( atomType )
		
		}	// end for(j = 0; j < nNeigh; j++)
		
		if( atomType == 0 )
		{
			a[i].x = fSum.x / dparams.massMe;			//! divide by metallic mass
			a[i].y = fSum.y / dparams.massMe;
			a[i].z = fSum.z / dparams.massMe;
			// save potential energy in .w component of coordinate
			// note: we use 0.5 for pairwise energy 
			// but don't use 0.5 for embedded energy in contrast to Rapaport
			// so don't use 0.5 in EvalProps!
			dr[i].w = 0.5f*uSum + EamF(rhoi);	
			// !save friction force only if the correct regime and metal atoms!
			if(fForce != 0)
				fForce[i] = frictForceVal.x;			
		}
		else if( atomType == 1 )
		{
			a[i] = fSum;
			// save potential energy in .w component of coordinate
			dr[i].w = 0.5f*uSum;			// !don't forget about 0.5 			
		}
// end Algorithm 2
	}	// end if(i < dparams.nMol)
}

////////////////////////////////////////////////////////////////////////////////////
// kernels for evaluation of properties (impulse, kinetic energy, etc.) - reduction
////////////////////////////////////////////////////////////////////////////////////

// compute total impulse
// each block first sums a subset of the array and stores the result in global memory;
// when all blocks are done, the last block done reads each of these partial sums 
// and sums them
__global__ void ComputeVSumK( float3 *dv,	// array of velocities
						float3 *hlpArray )	// helper array, size of ceil(nMol / (2*blockDim.x)) is assumed
{
	// we assume that each block at first computes partial sum of 2*blockDim.x elements
	extern __shared__ float3 partialSum[];	// size of 2*blockDim.x is assumed	
	uint stride, t = threadIdx.x;
	uint maxThreadIdx;	

	int maxBlockIdx = floor(dparams.nMol*0.5f/blockDim.x);

	// for partially filled block define the index of the last thread
	if(maxBlockIdx == 0) maxThreadIdx = dparams.nMol;
	else
		maxThreadIdx = (uint) dparams.nMol % (2*blockDim.x);

	// copy data from global memory to shared
	partialSum[t] = dv[2*blockIdx.x*blockDim.x + t];
	partialSum[blockDim.x+t] = dv[(2*blockIdx.x + 1)*blockDim.x + t];

	// if metal scale by mass	
	float m = dparams.massMe;
	// here not threadIdx, but index of molecule!
	if( (2*blockIdx.x*blockDim.x + t) < dparams.nMolMe )
		partialSum[t].x = m*partialSum[t].x;	
	if( ( (2*blockIdx.x+1)*blockDim.x + t) < dparams.nMolMe )
		partialSum[blockDim.x+t].x = m*partialSum[blockDim.x+t].x;

	// check if the thread is above the range then zero the sum
	// this causes divergent warp
	if( (blockIdx.x == maxBlockIdx) && (t >= maxThreadIdx ) ) 
		VZero(partialSum[t]);
	if( (blockIdx.x == maxBlockIdx) && ((blockDim.x+t) >= maxThreadIdx) ) 
		VZero(partialSum[blockDim.x+t]);

	// check if atom is above the range, then zero sums
	if( (2*blockIdx.x*blockDim.x + t) >= dparams.nMol){
		VZero(partialSum[t]);
	}
	if( ((2*blockIdx.x + 1)*blockDim.x + t) >= dparams.nMol){		
		VZero(partialSum[blockDim.x+t]);
	}

	// begin summation, algorithm from lecture 13 Urbana, Illinois
	// !there were two bugs in the lecture: 
	// 1)should be stride >= 1, but not stride > 1;
	// 2) stride = stride >> 1, but not stride >> 1
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);
	}

	// in each block thread t == 0 contains partial sum
	// copy it to global memory (see p. 111 of programming guide)
	if( t == 0)	{
		hlpArray[blockIdx.x] = partialSum[t];		
		__threadfence();	// ensure that the result is visible to all other threads
		// thread 0 of each block signals that it is done
		uint value = atomicInc(&count, gridDim.x);
		// determine if this block is the last block to be done
		isLastBlockDone = (value == (gridDim.x - 1));	// shared memory variable
	}
	// synchronize to make sure that each thread reads the correct value of isLastBlockDone
	__syncthreads();
	// the last block does the final summation
	if(isLastBlockDone)
	{
		VZero(partialSum[t]);		// reuse partial sum, t < blockDim.x
		
		if( t < gridDim.x  )		// max thread index is equal to gridDim.x
			// copy data from global memory to shared
			partialSum[t] = hlpArray[t];		

		VZero(partialSum[blockDim.x+t]);
		if( (blockDim.x+t) < gridDim.x )		
			partialSum[blockDim.x+t] = hlpArray[blockDim.x + t];			

		// begin summation, algorithm from lecture 13 Urbana, Illinois		
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);				
		}
		if(t == 0)	{
			// thread 0 of last block stores total sum to global memory
			// and resets count so next kernel calls work properly
			hlpArray[0] = partialSum[t];
			count = 0;
		}
	}	
}

// compute sum of squares of velocities
// it is stored in .x component of the 0 element of the hlpArray
__global__ void ComputeVvSumK( float3 *dv,	// array of velocities
						float3 *hlpArray,   // helper array, size of ceil(nMol / (2*blockDim.x)) is assumed
						real cmVelX)	// velocity of center of mass of Me should be excluded
{
	// we assume that each block at first computes partial sum of 2*blockDim.x elements
	extern __shared__ float3 partialSum[];	// size of 2*blockDim.x is assumed	
	uint stride, t = threadIdx.x;
	uint maxThreadIdx;

	int maxBlockIdx = floor(dparams.nMol*0.5f/blockDim.x);

	// for partially filled block define the index of the last thread
	if(maxBlockIdx == 0) maxThreadIdx = dparams.nMol;
	else
		maxThreadIdx = (uint) dparams.nMol % (2*blockDim.x);

	// copy data from global memory to shared
	partialSum[t] = dv[2*blockIdx.x*blockDim.x + t];
	partialSum[blockDim.x+t] = dv[(2*blockIdx.x + 1)*blockDim.x + t];

	// check if the thread is above the range then zero the sum
	// this causes divergent warp
	if( (blockIdx.x == maxBlockIdx) && (t >= maxThreadIdx ) ) 
		VZero(partialSum[t]);
	if( (blockIdx.x == maxBlockIdx) && ((blockDim.x+t) >= maxThreadIdx) ) 
		VZero(partialSum[blockDim.x+t]);

	// check if atom is above the range, then zero sums
	if( (2*blockIdx.x*blockDim.x + t) >= dparams.nMol){
		VZero(partialSum[t]);
	}
	if( ((2*blockIdx.x + 1)*blockDim.x + t) >= dparams.nMol){		
		VZero(partialSum[blockDim.x+t]);
	}	

	// if metal subtract velocity of center of mass	
	// here not threadIdx, but index of molecule!
	if( (dparams.iRegime == 2) && ( (2*blockIdx.x*blockDim.x + t) < dparams.nMolMe ) )
		partialSum[t].x = partialSum[t].x - cmVelX;		
	
	if( (dparams.iRegime == 2) && ( ( (2*blockIdx.x+1)*blockDim.x + t) < dparams.nMolMe ) )
		partialSum[blockDim.x+t].x = partialSum[blockDim.x+t].x - cmVelX;	

	// find square of velocity and store it in .x component
	partialSum[t].x = VLenSq (partialSum[t]);
	partialSum[blockDim.x + t].x = VLenSq (partialSum[blockDim.x + t]);

	// if metal scale by mass
	float m = dparams.massMe;
	// here not threadIdx, but index of molecule!
	if( (2*blockIdx.x*blockDim.x + t) < dparams.nMolMe )
		partialSum[t].x = m*partialSum[t].x;	
	if( ( (2*blockIdx.x+1)*blockDim.x + t) < dparams.nMolMe )
		partialSum[blockDim.x+t].x = m*partialSum[blockDim.x+t].x;
	
	// begin summation, algorithm from lecture 13 Urbana, Illinois
	// note that here all components are summed, but we're interested only in .x
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);
	}
	// in each block thread t == 0 contains partial sum
	// copy it to global memory (see p. 111 of programming guide)
	if( t == 0){
		hlpArray[blockIdx.x] = partialSum[t];		
		__threadfence();	// ensure that the result is visible to all other threads
		// thread 0 of each block signals that it is done
		uint value = atomicInc(&count, gridDim.x);
		// determine if this block is the last block to be done
		isLastBlockDone = (value == (gridDim.x - 1));	// shared memory variable
	}
	// synchronize to make sure that each thread reads the correct value of isLastBlockDone
	__syncthreads();
	// the last block does the final summation
	if(isLastBlockDone)
	{
		VZero(partialSum[t]);		// reuse partial sum, t < blockDim.x
		
		if( t < gridDim.x  )		// max thread index is equal to gridDim.x
			// copy data from global memory to shared
			partialSum[t] = hlpArray[t];		

		VZero(partialSum[blockDim.x+t]);
		if( (blockDim.x+t) < gridDim.x )		
			partialSum[blockDim.x+t] = hlpArray[blockDim.x + t];			

		// begin summation, algorithm from lecture 13 Urbana, Illinois		
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);				
		}
		if(t == 0)		{
			// thread 0  of last block stores total sum to global memory
			// and resets count so next kernel calls work properly
			hlpArray[0] = partialSum[t];
			count = 0;
		}
	}	
}

// find maximum of squares of velocities
// it is stored in .x component of the 0 element of the hlpArray
// this is also reduction, so the principle is the same as with summation,
// but instead of summation we compare elements
__global__ void ComputeVvMaxK( float3 *dv,	// array of velocities
						float3 *hlpArray )	// helper array, size of ceil(nMol / (2*blockDim.x)) is assumed
{
	// we assume that each block at first compares 2*blockDim.x elements
	extern __shared__ float3 partialSum[];	// size of 2*blockDim.x is assumed	
	uint stride, t = threadIdx.x;
	uint maxThreadIdx;

	int maxBlockIdx = floor(dparams.nMol*0.5f/blockDim.x);

	// for partially filled block define the index of the last thread
	if(maxBlockIdx == 0) maxThreadIdx = dparams.nMol;
	else
		maxThreadIdx = (uint) dparams.nMol % (2*blockDim.x);

	// copy data from global memory to shared
	partialSum[t] = dv[2*blockIdx.x*blockDim.x + t];
	partialSum[blockDim.x+t] = dv[(2*blockIdx.x + 1)*blockDim.x + t];

	// check if the thread is above the range, then zero the sum
	// this causes divergent warp
	if( (blockIdx.x == maxBlockIdx) && (t >= maxThreadIdx ) ) 
		VZero(partialSum[t]);
	if( (blockIdx.x == maxBlockIdx) && ((blockDim.x+t) >= maxThreadIdx) ) 
		VZero(partialSum[blockDim.x+t]);

	// check if atom is above the range, then zero sums
	if( (2*blockIdx.x*blockDim.x + t) >= dparams.nMol){
		VZero(partialSum[t]);
	}
	if( ((2*blockIdx.x + 1)*blockDim.x + t) >= dparams.nMol){		
		VZero(partialSum[blockDim.x+t]);
	}

	// find square of velocity and store it in .x component
	partialSum[t].x = VLenSq (partialSum[t]);
	partialSum[blockDim.x + t].x = VLenSq (partialSum[blockDim.x + t]);

	// begin comparison, algorithm from lecture 13 Urbana, Illinois
	// we're interested only in .x component
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) 
		{	// !here better to use atomic operation, but atomicMax() exists only for int
			if( partialSum[t].x < partialSum[t + stride].x)
				atomicExch(&partialSum[t].x, partialSum[t + stride].x);
		}
	}

	// in each block thread t == 0 contains maximum vv in .x component
	// copy it to global memory (see p. 111 of programming guide)
	if( t == 0)
	{
		hlpArray[blockIdx.x] = partialSum[t];		
		__threadfence();	// ensure that the result is visible to all other threads
		// thread 0 of each block signals that it is done
		uint value = atomicInc(&count, gridDim.x);
		// determine if this block is the last block to be done
		isLastBlockDone = (value == (gridDim.x - 1));	// shared memory variable
	}
	// synchronize to make sure that each thread reads the correct value of isLastBlockDone
	__syncthreads();
	// the last block does the final summation
	if(isLastBlockDone)
	{
		VZero(partialSum[t]);		// reuse partial sum, t < blockDim.x
		
		if( t < gridDim.x  )		// max thread index is equal to gridDim.x
			// copy data from global memory to shared
			partialSum[t] = hlpArray[t];		

		VZero(partialSum[blockDim.x+t]);
		if( (blockDim.x+t) < gridDim.x )		
			partialSum[blockDim.x+t] = hlpArray[blockDim.x + t];	
		// begin comparison, algorithm from lecture 13 Urbana, Illinois		
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ){
				if( partialSum[t].x < partialSum[t + stride].x)
					atomicExch(&partialSum[t].x, partialSum[t + stride].x);
			}
		}
		if(t == 0){
			// thread 0  of last block stores maximum value to global memory
			// and resets count so next kernel calls work properly
			hlpArray[0] = partialSum[t];
			count = 0;
		}
	}	
}

// compute potential energy usum, it is = sum of .w components of dr elements
__global__ void ComputePotEnergyK( float4 *dr, float3 *hlpArray )	
{
	// we assume that each block at first computes partial sum of 2*blockDim.x elements
	extern __shared__ float3 partialSum[];	// size of 2*blockDim.x is assumed	
	uint stride, t = threadIdx.x;
	uint maxThreadIdx;

	int maxBlockIdx = floor(dparams.nMol*0.5f/blockDim.x);

	// for partially filled block define the index of the last thread
	if(maxBlockIdx == 0) maxThreadIdx = dparams.nMol;
	else
		maxThreadIdx = (uint) dparams.nMol % (2*blockDim.x);

	// copy data from global memory to shared
	// note the use of components, dr.w is copied to .x
	partialSum[t].x = dr[2*blockIdx.x*blockDim.x + t].w;
	partialSum[blockDim.x+t].x = dr[(2*blockIdx.x + 1)*blockDim.x + t].w;

	// check if the thread is above the range then zero the sum
	// this causes divergent warp
	if( (blockIdx.x == maxBlockIdx) && (t >= maxThreadIdx ) ) 
		VZero(partialSum[t]);
	if( (blockIdx.x == maxBlockIdx) && ((blockDim.x+t) >= maxThreadIdx) ) 
		VZero(partialSum[blockDim.x+t]);

	// check if atom is above the range, then zero sums
	if( (2*blockIdx.x*blockDim.x + t) >= dparams.nMol){
		VZero(partialSum[t]);
	}
	if( ((2*blockIdx.x + 1)*blockDim.x + t) >= dparams.nMol){		
		VZero(partialSum[blockDim.x+t]);
	}
	
	// begin summation, algorithm from lecture 13 Urbana, Illinois
	// note that here all components are summed, but we're interested only in .x
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);
	}

	// in each block thread t == 0 contains partial sum
	// copy it to global memory (see p. 111 of programming guide)
	if( t == 0)	{
		hlpArray[blockIdx.x] = partialSum[t];		
		__threadfence();	// ensure that the result is visible to all other threads
		// thread 0 of each block signals that it is done
		uint value = atomicInc(&count, gridDim.x);
		// determine if this block is the last block to be done
		isLastBlockDone = (value == (gridDim.x - 1));	// shared memory variable
	}
	// synchronize to make sure that each thread reads the correct value of isLastBlockDone
	__syncthreads();
	// the last block does the final summation
	if(isLastBlockDone)
	{
		VZero(partialSum[t]);		// reuse partial sum, t < blockDim.x
		
		if( t < gridDim.x  )		// max thread index is equal to gridDim.x
			// copy data from global memory to shared
			partialSum[t] = hlpArray[t];		

		VZero(partialSum[blockDim.x+t]);
		if( (blockDim.x+t) < gridDim.x )		
			partialSum[blockDim.x+t] = hlpArray[blockDim.x + t];			

		// begin summation, algorithm from lecture 13 Urbana, Illinois		
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);				
		}
		if(t == 0){
			// thread 0  of last block stores total sum to global memory
			// and resets count so next kernel calls work properly
			hlpArray[0] = partialSum[t];
			count = 0;
		}
	}	
}

// !note that in contrast to the upper functions
// in below functions we use dparams.nMolMe and not dparams.nMol
// because we are interested only in metal atoms

// compute coordinates of the center of mass of the nanoparticle (reduction)
__global__ void ComputeCenterOfMassK( float4 *dr,	// array of coordinates
						float3 *hlpArray )	// helper array, size of ceil(nMol / (2*blockDim.x)) is assumed
{
	// we assume that each block at first computes partial sum of 2*blockDim.x elements
	extern __shared__ float3 partialSum[];	// size of 2*blockDim.x is assumed	
	uint stride, t = threadIdx.x;
	uint maxThreadIdx;	

	int maxBlockIdx = floor(dparams.nMolMe*0.5f/blockDim.x);

	// for partially filled block define the index of the last thread
	// !note that here we consider only metal atoms!
	if(maxBlockIdx == 0) maxThreadIdx = dparams.nMolMe;
	else
		maxThreadIdx = (uint) dparams.nMolMe % (2*blockDim.x);

	// copy data from global memory to shared
	partialSum[t].x = dr[2*blockIdx.x*blockDim.x + t].x;
	partialSum[t].y = dr[2*blockIdx.x*blockDim.x + t].y;
	partialSum[t].z = dr[2*blockIdx.x*blockDim.x + t].z;
	partialSum[blockDim.x+t].x = dr[(2*blockIdx.x + 1)*blockDim.x + t].x;
	partialSum[blockDim.x+t].y = dr[(2*blockIdx.x + 1)*blockDim.x + t].y;
	partialSum[blockDim.x+t].z = dr[(2*blockIdx.x + 1)*blockDim.x + t].z;
	
	// check if the thread is above the range then zero the sum
	// this causes divergent warp
	if( (blockIdx.x == maxBlockIdx) && (t >= maxThreadIdx ) ) 
		VZero(partialSum[t]);
	if( (blockIdx.x == maxBlockIdx) && ((blockDim.x+t) >= maxThreadIdx) ) 
		VZero(partialSum[blockDim.x+t]);

	// check if this is not metal, then zero sums
	if( (2*blockIdx.x*blockDim.x + t) >= dparams.nMolMe)	{
		VZero(partialSum[t]);
	}
	if( ((2*blockIdx.x + 1)*blockDim.x + t) >= dparams.nMolMe)	{		
		VZero(partialSum[blockDim.x+t]);
	}

	// begin summation, algorithm from lecture 13 Urbana, Illinois	
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);
	}

	// in each block thread t == 0 contains partial sum
	// copy it to global memory (see p. 111 of programming guide)
	if( t == 0)	{
		hlpArray[blockIdx.x] = partialSum[t];		
		__threadfence();	// ensure that the result is visible to all other threads
		// thread 0 of each block signals that it is done
		uint value = atomicInc(&count, gridDim.x);
		// determine if this block is the last block to be done
		isLastBlockDone = (value == (gridDim.x - 1));	// shared memory variable
	}
	// synchronize to make sure that each thread reads the correct value of isLastBlockDone
	__syncthreads();
	// the last block does the final summation
	if(isLastBlockDone)
	{
		VZero(partialSum[t]);		// reuse partial sum, t < blockDim.x
		
		if( t < gridDim.x  )		// max thread index is equal to gridDim.x
			// copy data from global memory to shared
			partialSum[t] = hlpArray[t];		

		VZero(partialSum[blockDim.x+t]);
		if( (blockDim.x+t) < gridDim.x )		
			partialSum[blockDim.x+t] = hlpArray[blockDim.x + t];			

		// begin summation, algorithm from lecture 13 Urbana, Illinois		
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);				
		}
		if(t == 0)	{
			// thread 0  of last block stores total sum to global memory
			// and resets count so next kernel calls work properly
			hlpArray[0] = partialSum[t];
			count = 0;
		}
	}	
}

// compute velocity of the center of mass of the nanoparticle
// (sum of velocities of metal atoms)
__global__ void ComputeCmVelK( float3 *dv,	// array of velocities
						float3 *hlpArray )	// helper array, size of ceil(nMol / (2*blockDim.x)) is assumed
{
	// we assume that each block at first computes partial sum of 2*blockDim.x elements

	extern __shared__ float3 partialSum[];	// size of 2*blockDim.x is assumed	
	uint stride, t = threadIdx.x;
	uint maxThreadIdx;	
	
	int maxBlockIdx = floor(dparams.nMolMe*0.5f/blockDim.x);

	// for partially filled block define the index of the last thread
	if(maxBlockIdx == 0) maxThreadIdx = dparams.nMolMe;
	else
		maxThreadIdx = (uint) dparams.nMolMe % (2*blockDim.x);

	// copy data from global memory to shared
	partialSum[t] = dv[2*blockIdx.x*blockDim.x + t];
	partialSum[blockDim.x+t] = dv[(2*blockIdx.x + 1)*blockDim.x + t];

	// check if the thread is above the range then zero the sum
	// this causes divergent warp
	if( (blockIdx.x == maxBlockIdx) && (t >= maxThreadIdx ) ) 
		VZero(partialSum[t]);
	if( (blockIdx.x == maxBlockIdx) && ((blockDim.x+t) >= maxThreadIdx) ) 
		VZero(partialSum[blockDim.x+t]);

	// check if this is not metal, then zero sums
	if( (2*blockIdx.x*blockDim.x + t) >= dparams.nMolMe)	{
		VZero(partialSum[t]);
	}
	if( ((2*blockIdx.x + 1)*blockDim.x + t) >= dparams.nMolMe)	{		
		VZero(partialSum[blockDim.x+t]);
	}

	// begin summation, algorithm from lecture 13 Urbana, Illinois	
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);
	}

	// in each block thread t == 0 contains partial sum
	// copy it to global memory (see p. 111 of programming guide)
	if( t == 0)	{
		hlpArray[blockIdx.x] = partialSum[t];		
		__threadfence();	// ensure that the result is visible to all other threads
		// thread 0 of each block signals that it is done
		uint value = atomicInc(&count, gridDim.x);
		// determine if this block is the last block to be done
		isLastBlockDone = (value == (gridDim.x - 1));	// shared memory variable
	}
	// synchronize to make sure that each thread reads the correct value of isLastBlockDone
	__syncthreads();
	// the last block does the final summation
	if(isLastBlockDone)
	{
		VZero(partialSum[t]);		// reuse partial sum, t < blockDim.x
		
		if( t < gridDim.x  )		// max thread index is equal to gridDim.x
			// copy data from global memory to shared
			partialSum[t] = hlpArray[t];		

		VZero(partialSum[blockDim.x+t]);
		if( (blockDim.x+t) < gridDim.x )		
			partialSum[blockDim.x+t] = hlpArray[blockDim.x + t];			

		// begin summation, algorithm from lecture 13 Urbana, Illinois		
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);				
		}
		if(t == 0)	{
			// thread 0  of last block stores total sum to global memory
			// and resets count so next kernel calls work properly
			hlpArray[0] = partialSum[t];
			count = 0;
		}
	}	
}

// evaluate dimensions of the nanoparticle
// we find maximum or minimum values of coordinates 
// in corresponding directions and the difference between them on host
__global__ void ComputeParticleSizeK( float4 *dr,	// array of coordinates
								float3 *hlpArray,	// helper array, size of ceil(nMol / (2*blockDim.x)) is assumed
								int		min_max )	// if == 0 find minimum value, 1 - maximum
{
	// we assume that each block at first compares 2*blockDim.x elements
	extern __shared__ float3 partialSum[];	// size of 2*blockDim.x is assumed
	
	uint stride, t = threadIdx.x;
	uint maxThreadIdx;

	// consider only metal atoms
	int maxBlockIdx = floor(dparams.nMolMe*0.5f/blockDim.x);

	// for partially filled block define the index of the last thread
	if(maxBlockIdx == 0) maxThreadIdx = dparams.nMolMe;
	else
		maxThreadIdx = (uint) dparams.nMolMe % (2*blockDim.x);

	// copy data from global memory to shared
	partialSum[t].x = dr[2*blockIdx.x*blockDim.x + t].x;
	partialSum[t].y = dr[2*blockIdx.x*blockDim.x + t].y;
	partialSum[t].z = dr[2*blockIdx.x*blockDim.x + t].z;
	partialSum[blockDim.x+t].x = dr[(2*blockIdx.x + 1)*blockDim.x + t].x;
	partialSum[blockDim.x+t].y = dr[(2*blockIdx.x + 1)*blockDim.x + t].y;
	partialSum[blockDim.x+t].z = dr[(2*blockIdx.x + 1)*blockDim.x + t].z;

	// check if the thread is above the range, then zero the sum
	// this causes divergent warp
	if( (blockIdx.x == maxBlockIdx) && (t >= maxThreadIdx ) ) 
		VZero(partialSum[t]);
	if( (blockIdx.x == maxBlockIdx) && ((blockDim.x+t) >= maxThreadIdx) ) 
		VZero(partialSum[blockDim.x+t]);

	// check if this is not metal, then zero sums
	if( (2*blockIdx.x*blockDim.x + t) >= dparams.nMolMe){
		VZero(partialSum[t]);
	}
	if( ((2*blockIdx.x + 1)*blockDim.x + t) >= dparams.nMolMe){		
		VZero(partialSum[blockDim.x+t]);
	}

	// begin comparison, algorithm from lecture 13 Urbana, Illinois
	// .x component
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) 
		{	
			if(min_max == 0){	// find minimum value
				if( partialSum[t].x > partialSum[t + stride].x)
					atomicExch(&partialSum[t].x, partialSum[t + stride].x);				
			}
			else if(min_max == 1){	// find maximum value
				if( partialSum[t].x < partialSum[t + stride].x)
					atomicExch(&partialSum[t].x, partialSum[t + stride].x);				
			}
		}
	}
	// .y component
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) 
		{	
			if(min_max == 0){	// find minimum value
				if( partialSum[t].y > partialSum[t + stride].y)
					atomicExch(&partialSum[t].y, partialSum[t + stride].y);				
			}
			else if(min_max == 1){	// find maximum value
				if( partialSum[t].y < partialSum[t + stride].y)
					atomicExch(&partialSum[t].y, partialSum[t + stride].y);				
			}
		}
	}
	// .z component
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) 
		{	
			if(min_max == 0){	// find minimum value
				if( partialSum[t].z > partialSum[t + stride].z)
					atomicExch(&partialSum[t].z, partialSum[t + stride].z);				
			}
			else if(min_max == 1){	// find maximum value
				if( partialSum[t].z < partialSum[t + stride].z)
					atomicExch(&partialSum[t].z, partialSum[t + stride].z);				
			}
		}
	}

	// in each block thread t == 0 contains maximum value
	// copy it to global memory (see p. 111 of programming guide)
	if( t == 0)
	{
		hlpArray[blockIdx.x] = partialSum[t];		
		__threadfence();	// ensure that the result is visible to all other threads
		// thread 0 of each block signals that it is done
		uint value = atomicInc(&count, gridDim.x);
		// determine if this block is the last block to be done
		isLastBlockDone = (value == (gridDim.x - 1));	// shared memory variable
	}
	// synchronize to make sure that each thread reads the correct value of isLastBlockDone
	__syncthreads();
	// the last block does the final summation
	if(isLastBlockDone)
	{
		VZero(partialSum[t]);		// reuse partial sum, t < blockDim.x
		
		if( t < gridDim.x  )		// max thread index is equal to gridDim.x
			// copy data from global memory to shared
			partialSum[t] = hlpArray[t];		

		VZero(partialSum[blockDim.x+t]);
		if( (blockDim.x+t) < gridDim.x )		
			partialSum[blockDim.x+t] = hlpArray[blockDim.x + t];	
		// begin comparison, algorithm from lecture 13 Urbana, Illinois
		// .x component
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ){
				if(min_max == 0){	// find minimum value
					if( partialSum[t].x > partialSum[t + stride].x)
						atomicExch(&partialSum[t].x, partialSum[t + stride].x);					
				}
				else if(min_max == 1){	// find maximum value
					if( partialSum[t].x < partialSum[t + stride].x)
						atomicExch(&partialSum[t].x, partialSum[t + stride].x);					
				}
			}
		}
		// .y component
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ) 
			{	
				if(min_max == 0){	// find minimum value
					if( partialSum[t].y > partialSum[t + stride].y)
						atomicExch(&partialSum[t].y, partialSum[t + stride].y);				
				}
				else if(min_max == 1){	// find maximum value
					if( partialSum[t].y < partialSum[t + stride].y)
						atomicExch(&partialSum[t].y, partialSum[t + stride].y);				
				}
			}
		}
		// .z component
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ) 
			{	
				if(min_max == 0){	// find minimum value
					if( partialSum[t].z > partialSum[t + stride].z)
						atomicExch(&partialSum[t].z, partialSum[t + stride].z);				
				}
				else if(min_max == 1){	// find maximum value
					if( partialSum[t].z < partialSum[t + stride].z)
						atomicExch(&partialSum[t].z, partialSum[t + stride].z);				
				}
			}
		}
		if(t == 0){
			// thread 0 of last block stores maximum value to global memory
			// and resets count so next kernel calls work properly
			hlpArray[0] = partialSum[t];
			count = 0;
		}
	}	// end if(isLastBlockDone)
}

// compute total force acting on the nanoparticle (reduction)
__global__ void ComputeNetForceK( real *hlpArray )	// array of accelerations and helper array
{
	// we assume that each block at first computes partial sum of 2*blockDim.x elements
	extern __shared__ float3 partialSum[];	// size of 2*blockDim.x is assumed	
	uint stride, t = threadIdx.x;
	uint maxThreadIdx;	

	// note that we consider only metal atoms
	int maxBlockIdx = floor(dparams.nMolMe*0.5f/blockDim.x);

	// for partially filled block define the index of the last thread
	// !note that here we consider only metal atoms!
	if(maxBlockIdx == 0) maxThreadIdx = dparams.nMolMe;
	else
		maxThreadIdx = (uint) dparams.nMolMe % (2*blockDim.x);

	// copy data from global memory to shared
	// !check index to avoid memory problems!
	if( (2*blockIdx.x*blockDim.x + t) < dparams.nMolMe)
		partialSum[t].x = hlpArray[2*blockIdx.x*blockDim.x + t];
	if( ((2*blockIdx.x + 1)*blockDim.x + t) < dparams.nMolMe)
		partialSum[blockDim.x+t].x = hlpArray[(2*blockIdx.x + 1)*blockDim.x + t];
	
	// check if the thread is above the range then zero the sum
	// this causes divergent warp
	if( (blockIdx.x == maxBlockIdx) && (t >= maxThreadIdx ) ) 
		VZero(partialSum[t]);
	if( (blockIdx.x == maxBlockIdx) && ((blockDim.x+t) >= maxThreadIdx) ) 
		VZero(partialSum[blockDim.x+t]);

	// check if this is not metal, then zero sums
	if( (2*blockIdx.x*blockDim.x + t) >= dparams.nMolMe){
		VZero(partialSum[t]);
	}
	if( ((2*blockIdx.x + 1)*blockDim.x + t) >= dparams.nMolMe){		
		VZero(partialSum[blockDim.x+t]);
	}

	// begin summation, algorithm from lecture 13 Urbana, Illinois	
	for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);
	}

	// in each block thread t == 0 contains partial sum
	// copy it to global memory (see p. 111 of programming guide)
	if( t == 0)	{
		hlpArray[blockIdx.x] = partialSum[t].x;		
		__threadfence();	// ensure that the result is visible to all other threads
		// thread 0 of each block signals that it is done
		uint value = atomicInc(&count, gridDim.x);
		// determine if this block is the last block to be done
		isLastBlockDone = (value == (gridDim.x - 1));	// shared memory variable
	}
	// synchronize to make sure that each thread reads the correct value of isLastBlockDone
	__syncthreads();
	// the last block does the final summation
	if(isLastBlockDone)
	{
		VZero(partialSum[t]);		// reuse partial sum, t < blockDim.x
		
		if( t < gridDim.x  )		// max thread index is equal to gridDim.x
			// copy data from global memory to shared
			partialSum[t].x = hlpArray[t];		

		VZero(partialSum[blockDim.x+t]);
		if( (blockDim.x+t) < gridDim.x )		
			partialSum[blockDim.x+t].x = hlpArray[blockDim.x + t];			

		// begin summation, algorithm from lecture 13 Urbana, Illinois		
		for(stride = blockDim.x; stride >= 1; stride = stride >> 1) {
			__syncthreads();
			if( t < stride ) VVAdd(partialSum[t], partialSum[t + stride]);				
		}
		if(t == 0)	{
			// thread 0  of last block stores total sum to global memory
			// and resets count so next kernel calls work properly
			hlpArray[0] = partialSum[t].x;
			count = 0;
		}
	}	
}

// compute radial distribution function
__global__ void EvalRdfK (float4 *dr,			// coordinates of molecules (global)							  
						  int *CELL,			// indexes of molecules (global)
						  uint *histRdf,			// histogram for rdf
						  int countRdf)			// number of measurements						  
{		
	int3 cc, m2v, vOff[] = OFFSET_VALS_LONG;	// 3D indexes of the cells
	int i, j, offset, n;						// counters
	int C;										// 1D cell indexes	
	float3 shift;								// variable for handling boundary conditions
	float4 A, deltaR;							// radus-vectors
	real rr;				// square of distance between atoms
	int indexRdf;			// index in the histogram

	// define 3D index of the current cell, blockIdx.x is 1D index of the cell
	cc.z = blockIdx.x / (dparams.cells.x * dparams.cells.y);
	cc.y = (blockIdx.x - cc.z * dparams.cells.x * dparams.cells.y) / dparams.cells.x;
	cc.x = blockIdx.x - (cc.z*dparams.cells.y + cc.y)*dparams.cells.x;
	
// here begins implementation of the Algorithm 4 from Anderson
	n = blockIdx.x * blockDim.x + threadIdx.x;

	// zero histogram for rdf if the first time
	if( (countRdf == 0) && (n < dparams.sizeHistRdf) )
		histRdf[n] = 0.f;	

	if(n < dparams.maxMol)
		i = CELL[blockIdx.x * blockDim.x + threadIdx.x];// step 1, get particle index
	else i = -1;	
	if(i != -1)	{									// added by me to avoid memory problems
		A = dr[i];									// step 3			
	}
	// begin step 4
	for (offset = 0; offset < N_OFFSET_LONG; offset ++)	// loop over all 27 neighboring cells 
	{
		VAdd (m2v, cc, vOff[offset]);			// find 3D index of the neighboring cell
		
		shift.x = 0.f;							// zero shift for boundary conditions
		shift.y = 0.f;
		shift.z = 0.f;

		// apply boundary conditions
		// !type manually instead of using macroses, because they do not work!
		// x
		if (m2v.x >= dparams.cells.x) { 
			m2v.x = 0; 
			shift.x = dparams.region.x; 
		} else if (m2v.x < 0) {
			m2v.x = dparams.cells.x - 1;
			shift.x = - dparams.region.x;
		}
		// y
		if (m2v.y >= dparams.cells.y) { 
			m2v.y = 0; 
			shift.y = dparams.region.y; 
		} else if (m2v.y < 0) {
			m2v.y = dparams.cells.y - 1;
			shift.y = - dparams.region.y;
		}
		// z
		if (m2v.z >= dparams.cells.z) { 
			m2v.z = 0; 
			shift.z = dparams.region.z; 
		} else if (m2v.z < 0) {
			m2v.z = dparams.cells.z - 1;
			shift.z = - dparams.region.z;
		}

		C = VLinear (m2v, dparams.cells);			// find 1D index of the neighboring cell
	// end step 4

		__syncthreads();							// step 5
		K[threadIdx.x] = CELL[C * blockDim.x + threadIdx.x];	// step 6
		if(K[threadIdx.x] != -1)					// added by me, to avoid memory error
			B[threadIdx.x] = dr[ K[threadIdx.x] ];	// step 7
		else
			VZero(B[threadIdx.x]);					// added by me
		__syncthreads();							// step 8		
		
		if( i != -1)	{							// step 9
			for(j = 0; j < blockDim.x; j++)	
			{
				// step 10, loop over atoms from the current neighboring cell C
				if( K[j] != -1 )	{				// steps 11 - 13,	
					
					deltaR.x = A.x - B[j].x;		// step 14
					deltaR.y = A.y - B[j].y;
					deltaR.z = A.z - B[j].z;
					
					deltaR.x = deltaR.x - shift.x;	// step 15, boundary conditions
					deltaR.y = deltaR.y - shift.y;
					deltaR.z = deltaR.z - shift.z;

					rr = deltaR.x*deltaR.x + deltaR.y*deltaR.y + deltaR.z*deltaR.z;
					// !consider only metal atoms! step 16
					if ( ( rr < dparams.rangeRdf*dparams.rangeRdf) && (i != K[j]) &&
						(i < dparams.nMolMe) && (K[j]< dparams.nMolMe))	
					{						
						indexRdf = sqrt(rr)*dparams.intervalRdf;// step 17
						atomicAdd(&histRdf[indexRdf], 1);		// step 18
					}											// step 19
				}	// end if( K[j] != -1 )
			}													// step 20 
		}														// step 21	
	} // step 22 end for (offset = 0; offset < N_OFFSET_LARGE; offset ++) 		
}


////////////////////////////////////////////////
// wrappers - W at the end of the function name
////////////////////////////////////////////////
extern "C"	// can be deleted, but then also in SurfaceGrowth.h
{
////////////////////////////
// wrappers calling kernels
////////////////////////////
	

// calls computational kernels when OpenGL is not used
char* DoComputationsW(float4 *hr, float3 *hv, float3 *ha, SimParams *hparams, 
					 FILE *fResults, TCHAR *szPdbPath)
{	
	// pointers to host memory
	uint	*hHistRdf = 0;
	if(hparams->bRdf != 0)
		AllocMem(hHistRdf, hparams->sizeHistRdf, uint);

	// host variables for diffusion
	int nb;
	TBuf *tBuf;
	real *rrDiffuseAv;
	FILE *fileDiffuse = NULL;
	if(hparams->iRegime == 2)	// if shear then measure diffusion
	{
		AllocMem(tBuf, hparams->nBuffDiffuse, TBuf);
		AllocMem(rrDiffuseAv, hparams->nValDiffuse, real);
		for(nb = 0; nb < hparams->nBuffDiffuse; nb++)
			AllocMem(tBuf[nb].rrDiffuse, hparams->nValDiffuse, real);
		InitDiffusion(tBuf, rrDiffuseAv, hparams);
		// open file for diffusion
		if( (fileDiffuse = _tfopen(hparams->szDiffusePath, TEXT("w"))) == NULL ){
			lstrcpy(szPdbPath, "Cannot open diffuse file!");
			return szPdbPath;
		}
	}

	// pointers to device memory
	float3	*dv,			// velocities
			*da,			// accelerations			
			*hlpArray;		// helper array
	float4	*dr;			// positions

	real	*dcarbonForce = 0;	// forces acting on metal atoms from carbon atoms (in x direction)

	real	*rho;			// electron density for eam

	int		*CELL;			// indexes of particles for cells
	int		*NN,			// number of neighbors for each particle		
			*NBL;			// neighbors list, i.e. indeces of neighboring particles
	uint	*molsInCells;	// number of atoms in each cell, size is = number of cells

	uint	*histRdf = 0;	// histogram for rdf

	// number of blocks for summing and the size of the array hlpArray;
	// each block contains 512 threads and processes 1024 elements;
	// because the last block computes total sum of grid elements,
	// so grid could not be greater than 1024, and hence nMol <= 1024 * 1024 = 1048576
	uint grid = (uint) ceil((float)hparams->nMol / (1024) );
	uint block = 512;		// threads per block for summation

	dim3 dimBlock(hparams->blockSize, 1, 1);	// number of threads	
	// define number of blocks as in Anderson
	dim3 dimGrid(hparams->gridSize);	

	float3	vSum;			// total impulse
	real	vvSum = 0.f, vvMax = 0.f, uSum = 0.f;

	// tribological properties
	float3	centerOfMass, frictForce, cmVel;
	float3	particleSize, particleSizeMin;	// dimensions of the nanoparticle

	int iBckup = 1;		// for choosing of the backup file
	
	// for error handling
	cudaError_t error;

	float hTime, hTimeTotal;	// time of one time step and of the complete run
	cudaEvent_t start, stop, totalStart, totalStop;

// begin memory allocation	
	cudaMalloc(&dr, hparams->nMol * sizeof(float4));
	// also allocate memory for velocities and accelerations	
	cudaMalloc(&dv, hparams->nMol * sizeof(float3));	
	cudaMalloc(&da, hparams->nMol * sizeof(float3));

	// helper array for computing system properties, total impulse, energy, etc.
	// its size is the size of the grid of blocks for summing
	cudaMalloc(&hlpArray, grid * sizeof(float3));

	// for eam
	cudaMalloc(&rho, hparams->nMolMe * sizeof(real));

	// allocate memory for particle indexes and neighbor list
	// !note (hparams->maxMol+1) to avoid subtle bug!
	cudaMalloc(&CELL, (hparams->maxMol+1) * sizeof(int));
	// number of neighbors
	cudaMalloc(&NN, hparams->nMol * sizeof(int));
	// indexes of neighboring atoms
	cudaMalloc(&NBL, hparams->nMol * hparams->iNebrMax * sizeof(int));
	// number of atoms in each cell
	cudaMalloc(&molsInCells, VProd(hparams->cells) * sizeof(uint));	

	if( (hparams->iRegime != 0) && (hparams->nMolMe != 0) )	// allocate memory for friction force
		cudaMalloc(&dcarbonForce, hparams->nMolMe * sizeof(real));
	// if needed allocate memory for rdf
	if( hparams->bRdf != 0 )
		cudaMalloc(&histRdf, hparams->sizeHistRdf * sizeof(int));

	// check errors	
	error = cudaGetLastError();
	if( error != cudaSuccess) {
		cudaThreadExit();
		const char* errorString = cudaGetErrorString (error);
		lstrcpy(szPdbPath, "Problems with memory allocation! Exception: ");
		lstrcat(szPdbPath, errorString);		
		return szPdbPath;
	}
// end memory allocation

	// copy data from host to device
	cudaMemcpy(dr, hr, hparams->nMol*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(dv, hv, hparams->nMol*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(da, ha, hparams->nMol*sizeof(float3), cudaMemcpyHostToDevice);

	// if needed, make the first coordinate snapshot
	if( hparams->bPdb != 0 ) {		
		CreatePdbFile(szPdbPath, hparams, hr);			
	}		
	
	// initialize variables for computing of one timestep
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// for total time
	cudaEventCreate(&totalStart);
	cudaEventCreate(&totalStop);

// begin computation of one time step
	while(hparams->moreCycles)
	{		
		cudaEventRecord(start, 0);	// record start time
		cudaEventRecord(totalStart, 0);
		
		++ hparams->stepCount;	// increment step count

// code for insertion of atoms (for surface growth regime)
	// before equilibration atoms are not deposited
	if( (hparams->iRegime == 1) && (hparams->stepCount <= hparams->stepEquil) )
	{	
		int nMolToDeposit = 0;		
		InsertAtomsK<<< dimGrid, dimBlock >>>
			( dr, dv, hparams->nMolDeposited, nMolToDeposit );		
	}
	if( (hparams->iRegime == 1) && (hparams->stepCount > hparams->stepEquil) )
	{
		if(hparams->nMolDeposited < hparams->nMolMe)
		{
		int nMolToDeposit = 0;
		if((hparams->stepCount-hparams->stepEquil) % hparams->stepDeposit == 0)
				nMolToDeposit = hparams->nMolToDeposit;
			InsertAtomsK<<< dimGrid, dimBlock >>>
				( dr, dv, hparams->nMolDeposited, nMolToDeposit );
			// enlarge the number of deposited atoms
			hparams->nMolDeposited += nMolToDeposit;			
		}
	}
	// check errors	
	error = cudaGetLastError();
	if( error != cudaSuccess) {
		cudaThreadExit();
		const char* errorString = cudaGetErrorString (error);
		lstrcpy(szPdbPath, "Problems with insertion of atoms! Exception: ");
		lstrcat(szPdbPath, errorString);		
		return szPdbPath;
	}
// end code for insertion of atoms
		
		LeapfrogStepK<<< dimGrid, dimBlock >>> ( 1, dr, dv, da );
		ApplyBoundaryCondK<<< dimGrid, dimBlock >>> ( dr );

		// check errors	
		error = cudaGetLastError();
		if( error != cudaSuccess) {
			cudaThreadExit();
			const char* errorString = cudaGetErrorString (error);
			lstrcpy(szPdbPath, "Problems with the 1st part of Verlet! Exception: ");
			lstrcat(szPdbPath, errorString);		
			return szPdbPath;
		}

		if (hparams->nebrNow) {
			hparams->nebrNow = 0;
			hparams->dispHi = 0.f;
			// fill cells with -1 (empty particles)
			// !note (hparams->maxMol+1) to avoid subtle bug with index 0
			cudaMemset(CELL, -1, (hparams->maxMol+1) * sizeof(int));
			// fill number of neighbors with 0
			cudaMemset(NN, 0, hparams->nMol * sizeof(int));
			// fill neighbor list with -1
			cudaMemset(NBL, -1, hparams->nMol * hparams->iNebrMax * sizeof(int)); 
			// fill number of atoms in each cell by 0
			cudaMemset(molsInCells, 0, VProd(hparams->cells) * sizeof(uint));
			// define cells of atoms
			BinAtomsIntoCellsK<<< dimGrid, dimBlock >>> (dr, CELL, molsInCells);
			// check errors	
			error = cudaGetLastError();
			if( error != cudaSuccess) {
				cudaThreadExit();
				const char* errorString = cudaGetErrorString (error);
				lstrcpy(szPdbPath, "Problems with building of cells! Exception: ");
				lstrcat(szPdbPath, errorString);		
				return szPdbPath;
			}

			BuildNebrListK<<< dimGrid, dimBlock >>>	(dr, CELL, NN, NBL);
			// check errors
			error = cudaGetLastError();
			if( error != cudaSuccess) {
				cudaThreadExit();
				const char* errorString = cudaGetErrorString (error);
				lstrcpy(szPdbPath, "Problems with building of neigbor list! Exception: ");
				lstrcat(szPdbPath, errorString);		
				return szPdbPath;
			}
		}		
		if( hparams->nMolMe != 0)
			EamComputeRhoK<<< dimGrid, dimBlock >>>(rho, dr, NN, NBL);
		
		// compute and save in the dcarbonForce forces acting from C on Me
		ComputeForcesK<<< dimGrid, dimBlock >>> (da, dr, NN, NBL, rho, dcarbonForce);
		// check errors
		if( error != cudaSuccess) {
			cudaThreadExit();
			const char* errorString = cudaGetErrorString (error);
			lstrcpy(szPdbPath, "Problems with foce evaluation! Exception: ");
			lstrcat(szPdbPath, errorString);		
			return szPdbPath;
		}

		if( dcarbonForce != 0 )
		{
			ComputeNetForceK<<< grid,		// number of blocks <= 1024
				block,						// number of threads
				2*block*sizeof(float3)		// memory for dynamic array in shared memory
				>>>(dcarbonForce);
			// copy force on host
			cudaMemcpy(&frictForce, dcarbonForce, sizeof(real), cudaMemcpyDeviceToHost);
		}
		else frictForce.x = 0.f;

		LeapfrogStepK<<< dimGrid, dimBlock >>> ( 2, dr, dv, da );	

// kernels, that form EvalProps - evaluate properties
// begin compute tribological properties
		if(hparams->nMolMe != 0)			// avoid bad values without metal atoms
		{
			// compute coordinate of center of mass of the nanoparticle
			ComputeCenterOfMassK<<< grid,	// number of blocks <= 1024
				block,						// number of threads
				2*block*sizeof(float3)		// memory for dynamic array in shared memory
				>>>(dr, hlpArray);
			// copy center of mass on host
			cudaMemcpy(&centerOfMass, hlpArray, sizeof(float3), cudaMemcpyDeviceToHost);

			// compute velocity of center of mass of the nanoparticle
			ComputeCmVelK<<< grid,			// number of blocks <= 1024
				block,						// number of threads
				2*block*sizeof(float3)		// memory for dynamic array in shared memory
				>>>(dv, hlpArray);
			// copy velocity on host
			cudaMemcpy(&cmVel, hlpArray, sizeof(float3), cudaMemcpyDeviceToHost);

			// compute dimensions of the nanoparticle
			// compute minimum radius-vector
			ComputeParticleSizeK<<< grid,	// number of blocks <= 1024
				block,						// number of threads
				2*block*sizeof(float3)		// memory for dynamic array in shared memory
				>>>(dr, hlpArray, 0);
			// copy minimum radius vector on host
			cudaMemcpy(&particleSizeMin, hlpArray, sizeof(float3), cudaMemcpyDeviceToHost);
			// compute maximum radius-vector
			ComputeParticleSizeK<<< grid,				// number of blocks <= 1024
				block,						// number of threads
				2*block*sizeof(float3)		// memory for dynamic array in shared memory
				>>>(dr, hlpArray, 1);
			// copy maximum radius-vector on host
			cudaMemcpy(&particleSize, hlpArray, sizeof(float3), cudaMemcpyDeviceToHost);
			// find difference particleSize = particleSize - particleSizeMin;
			particleSize.x = particleSize.x - particleSizeMin.x;
			particleSize.y = particleSize.y - particleSizeMin.y;
			particleSize.z = particleSize.z - particleSizeMin.z;

			// check errors
			error = cudaGetLastError();
			if( error != cudaSuccess) {
				cudaThreadExit();
				const char* errorString = cudaGetErrorString (error);
				lstrcpy(szPdbPath, "Problems with tribological properties! Exception: ");
				lstrcat(szPdbPath, errorString);		
				return szPdbPath;
			}

			// save properties
			// consider only x component because shear is in x direction
			hparams->centerOfMass.val = centerOfMass.x / hparams->nMolMe;
			hparams->cmVel.val = cmVel.x / hparams->nMolMe;
			hparams->frictForce.val = frictForce.x;
			hparams->particleSize.x = particleSize.x;
			hparams->particleSize.y = particleSize.y;
			hparams->particleSize.z = particleSize.z;

			// calculate diffusion if shear regime
			if(hparams->iRegime == 2) {
				if( (hparams->stepCount > hparams->stepEquil) &&
					( (hparams->stepCount - hparams->stepEquil) % hparams->stepDiffuse == 0) )
				{
					centerOfMass.x /= hparams->nMolMe;
					centerOfMass.y /= hparams->nMolMe;
					centerOfMass.z /= hparams->nMolMe;
					EvalDiffusion(tBuf, rrDiffuseAv, fileDiffuse, hparams, centerOfMass);
				}
			}

		}	// end if(hparams->nMolMe != 0)
		else
		{
			hparams->centerOfMass.val = 0.f;
			hparams->cmVel.val = 0.f;
			hparams->frictForce.val = 0.f;
			hparams->particleSize.x = 0.f;
			hparams->particleSize.y = 0.f;
			hparams->particleSize.z = 0.f;
		}
// end compute tribological properties

		// compute total impulse
		ComputeVSumK<<< grid,						// number of blocks <= 1024
			block,						// number of threads
			2*block*sizeof(float3)		// memory for dynamic array in shared memory
			>>>(dv, hlpArray);
		// copy total impulse on host
		cudaMemcpy(&vSum, hlpArray, sizeof(float3), cudaMemcpyDeviceToHost);

		// find maximum of squares of velocities
		ComputeVvMaxK<<< grid,						// number of blocks <= 1024
						block,						// number of threads
						2*block*sizeof(float3)		// memory for dynamic array in shared memory
					>>>(dv, hlpArray);
		// copy maximum of squares of velocities on host
		cudaMemcpy(&vvMax, &hlpArray[0].x, sizeof(float), cudaMemcpyDeviceToHost);

		// compute sum of squares of velocities
		ComputeVvSumK<<< grid,						// number of blocks <= 1024
						block,						// number of threads
						2*block*sizeof(float3)		// memory for dynamic array in shared memory
						>>>(dv, hlpArray, hparams->cmVel.val);
		// copy sum of squares of velocities on host
		cudaMemcpy(&vvSum, &hlpArray[0].x, sizeof(float), cudaMemcpyDeviceToHost);

		// apply Berendsen thermostat
		// for shear apply thermostat to all atoms after stepEquil,
		// for metal only during step cool, for carbon - till the end
		if( (hparams->iRegime == 2) && (hparams->stepCount > hparams->stepEquil)
			&& (hparams->stepCount % hparams->stepThermostat == 0))
			ApplyBerendsenThermostat<<< dimGrid, dimBlock >>> 
			( dv, &hlpArray[0].x, hparams->stepCount );	// here if step > cool Me is not thermostatted	

		// for SG apply thermostat to carbon atoms during all simulation
		if( (hparams->iRegime == 1)	&& (hparams->stepCount % hparams->stepThermostat == 0))
			ApplyBerendsenThermostat<<< dimGrid, dimBlock >>> 
			( dv, &hlpArray[0].x, hparams->stepCount );
		
		// compute potential energy
		ComputePotEnergyK<<< grid, block, 2*block*sizeof(float3) >>>(dr, hlpArray);
		// copy potential energy on host
		cudaMemcpy(&uSum, &hlpArray[0].x, sizeof(float), cudaMemcpyDeviceToHost);
		// check errors
		error = cudaGetLastError();
		if( error != cudaSuccess) {
			cudaThreadExit();
			const char* errorString = cudaGetErrorString (error);
			lstrcpy(szPdbPath, "Problems with evaluation of properties! Exception: ");
			lstrcat(szPdbPath, errorString);		
			return szPdbPath;
		}

		// see whether the building of the list is ripe
		hparams->dispHi += sqrt (vvMax) * hparams->deltaT;	
		if (hparams->dispHi > 0.5f * hparams->rNebrShell) hparams->nebrNow = 1;
		hparams->kinEnergy.val = 0.5f * vvSum /	hparams->nMol;	// compute kinetic energy
		hparams->potEnergy.val = uSum / hparams->nMol;			// potential energy
		hparams->totEnergy.val = hparams->kinEnergy.val + hparams->potEnergy.val;
		VCopy(hparams->vSum, vSum);	// copy impulse

// begin apply shear
	if( (hparams->iRegime == 2) && (hparams->nMolMe != 0) )
	{
		if( hparams->stepCount > (hparams->stepEquil + hparams->stepCool) )
		{
			if( hparams->cmVel.val < 0.005f )	// !adjust value of cmVel !
				hparams->shear += hparams->deltaF;

			// zero number of atoms to which shear is applied
			cudaMemset(&molsInCells[0], 0, sizeof(uint));			
			ApplyShearK<<< dimGrid, dimBlock >>> ( dr, da, hparams->shear, hparams->centerOfMass.val, &molsInCells[0] );
			// copy number of sheared atoms
			uint hnOfShearedMol = 0;
			cudaMemcpy(&hnOfShearedMol, &molsInCells[0], sizeof(uint), cudaMemcpyDeviceToHost);
			// !compute total shear force			
			hparams->totalShear = hnOfShearedMol*hparams->shear;
		}
	}
// end apply shear

// begin compute rdf
	if( (hparams->bRdf != 0) && ( hparams->stepCount % hparams->stepRdf == 0 ) )
	{		
		EvalRdfK<<< dimGrid, dimBlock >>> (dr, CELL, histRdf, hparams->countRdf);		
		++hparams->countRdf;
		if(hparams->countRdf == hparams->limitRdf)
		{
			// copy rdf on host
			cudaMemcpy(hHistRdf, histRdf, hparams->sizeHistRdf*sizeof(uint), cudaMemcpyDeviceToHost);			
			PrintRdf(hparams, hHistRdf);
			hparams->countRdf = 0;
		}
	}
// end compute rdf

	cudaEventRecord(stop, 0);		// record end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&hTime, start, stop);		

	hparams->oneStep.val = hTime;
// end EvalProps

	AccumProps (1, hparams);
	if (hparams->stepCount % hparams->stepAvg == 0) {
		AccumProps (2, hparams);
		if( hparams->bResult != 0 )
			PrintSummary (fResults, hparams);
		AccumProps (0, hparams);
	}		

	if( (hparams->stepCount % hparams->stepPdb == 0) && (hparams->bPdb!=0)) {
		// copy memory from device to host
		cudaMemcpy(hr, dr, hparams->nMol*sizeof(float4), cudaMemcpyDeviceToHost);
		CreatePdbFile(szPdbPath, hparams, hr);			
	}		
	
	if( hparams->stepCount >= hparams->stepLimit )		
		hparams->moreCycles = 0;	

// make backup if needed
	if( (hparams->bBckup) && (hparams->stepCount % hparams->stepBckup == 0) )
	{
		FILE *file = NULL;		
		// copy data from device to host
		// if stepBckup is dividable by stepPdb, then hr has already been copied
		if(hparams->stepBckup % hparams->stepPdb != 0)
			cudaMemcpy(hr, dr, hparams->nMol*sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemcpy(hv, dv, hparams->nMol*sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(ha, da, hparams->nMol*sizeof(float3), cudaMemcpyDeviceToHost);
		if(iBckup) {
			iBckup = 0;
			file = fopen(hparams->szBckup0, "w+b");
		}
		else {
			iBckup = 1;
			file = fopen(hparams->szBckup1, "w+b");
		}
		// write data to file		
		fwrite((void*)hparams, sizeof(SimParams), 1, file);
		fwrite((void*)hr, sizeof(float4), hparams->nMol, file);
		fwrite((void*)hv, sizeof(float3), hparams->nMol, file);
		fwrite((void*)ha, sizeof(float3), hparams->nMol, file);
		fclose(file);			
	}	// end if( (hparams->bBckup) &&	...
// end make backup
	
	// record total one step
	cudaEventRecord(totalStop, 0);		// record end time
	cudaEventSynchronize(totalStop);
	cudaEventElapsedTime(&hTimeTotal, totalStart, totalStop);
	hparams->totalTime += hTimeTotal*0.001f;

	}	// end while(hparams->moreCycles)
// end computation of one time step	

	// print total time
	if( hparams->bResult != 0 )
		fprintf (fResults, "\nDuration of the simulation = %f s", hparams->totalTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaEventDestroy(totalStart);
	cudaEventDestroy(totalStop);

	// cleanup
	cudaFree(dr);	
	cudaFree(dv);
	cudaFree(da);
	cudaFree(hlpArray);
	cudaFree(rho);
	cudaFree(CELL);
	cudaFree(NN);
	cudaFree(NBL);	
	cudaFree(molsInCells);
	if( dcarbonForce != 0)	// if regime != 0
		cudaFree(dcarbonForce);
	if((hparams->bRdf != 0) && (histRdf != 0) )
		cudaFree(histRdf);
	// free host memory
	if( (hparams->bRdf != 0) && (hHistRdf != 0) )
		free(hHistRdf);
	// free buffers for diffusion
	if(hparams->iRegime == 2) {
		for(nb = 0; nb < hparams->nBuffDiffuse; nb++)
			if(tBuf[nb].rrDiffuse) free(tBuf[nb].rrDiffuse);
	}
	if(tBuf) free(tBuf);
	if(rrDiffuseAv) free(rrDiffuseAv);
	if(fileDiffuse != NULL)
		fclose(fileDiffuse);
		
	return 0;
}

// wrapper that initializes coordinates
const char* InitCoordsW(float4 *dr, float4 *hr, SimParams* hparams)	
{
	int i;
	cudaMalloc(&dr, hparams->nMol * sizeof(float4));	// allocate device memory 
	if(hparams->iRegime == 0)	// if bulk, then only fcc lattice
	{
		uint max, middle, min;	// numbers of unit cells
		max = hparams->initUcell.x;
		middle = hparams->initUcell.y;
		min = hparams->initUcell.z;		
		
		// each thread handles one unit cell, and 4 atoms for fcc lattice
		dim3 dimBlock(1, 1, min);		// number of threads in the block (3D), min <= 512 cells
		// each block handles a strip of height min cells, length and width is 1 cell
		dim3 dimGrid(max, middle);		// grid size (number of blocks in 2D grid)
		// execute the kernel    
		InitFccCoordsK<<< dimGrid, dimBlock >>>(dr);
		// copy memory from device to host
		cudaMemcpy(hr, dr, hparams->nMol*sizeof(float4), cudaMemcpyDeviceToHost);
	}
	else if( hparams->iRegime == 1)	// if surface growth, generate random Me coordinates for Me
	{
		dim3 dimBlockCarbon(32, 1, 1);
		int numBlocks = (hparams->nMol-hparams->nMolMe) / dimBlockCarbon.x;
		dim3 dimGridCarbon(numBlocks, 1, 1);
		InitGrapheneCoordsK<<< dimGridCarbon, dimBlockCarbon >>>(dr);
		// copy memory from device to host here to avoid overwriting memory for metal
		cudaMemcpy(hr, dr, hparams->nMol*sizeof(float4), cudaMemcpyDeviceToHost);

		// generate random coordinates for metal atoms on host in advance
		for(i = 0; i < hparams->nMolMe; i++)
		{
			VRandRfloat4(&hr[i], hparams);
			hr[i].x = 0.5f*hparams->region.x*hr[i].x;
			hr[i].y = 0.5f*hparams->region.y*hr[i].y;
			hr[i].z = 1.5f*hparams->region.z + 0.5f*(i+1)*hparams->region.z;
			hr[i].w = 0.f;
		}
	} 
	else if( hparams->iRegime == 2)	// if shear, use Slab coords for metal
	{
		dim3 dimBlockCarbon(32, 1, 1);
		int numBlocks = (hparams->nMol-hparams->nMolMe) / dimBlockCarbon.x;
		dim3 dimGridCarbon(numBlocks, 1, 1);
		InitGrapheneCoordsK<<< dimGridCarbon, dimBlockCarbon >>>(dr);
		// copy memory from device to host
		cudaMemcpy(hr, dr, hparams->nMol*sizeof(float4), cudaMemcpyDeviceToHost);

		// define number of unit cells of metal and number of layers
		if(hparams->nMolMe != 0)
		{
			int initUcellMeX, initUcellMeY, numOfLayers;		
			initUcellMeX = ( hparams->region.x - 2*hparams->a ) / hparams->a;
			initUcellMeY = ( hparams->region.y - 2*hparams->a ) / hparams->a;
			numOfLayers = ceil(hparams->nMolMe*0.25 /(initUcellMeX*initUcellMeY) );

			hparams->initUcellMeX = initUcellMeX;
			hparams->initUcellMeY = initUcellMeY;
			
			dim3 blockMe(1, 1, numOfLayers);
			dim3 gridMe(initUcellMeX, initUcellMeY, 1);

			InitSlabCoordsK<<<gridMe, blockMe>>>( dr );
			// copy memory from device to host
			cudaMemcpy(hr, dr, hparams->nMolMe*sizeof(float4), cudaMemcpyDeviceToHost);
		}		
	} 	
	cudaFree(dr);
	// check errors
	cudaError_t error;
	error = cudaGetLastError();
	if( error != cudaSuccess)
	{
		const char* errorString = cudaGetErrorString (error);
		return errorString;
	}
	else return 0;
}

// define color for atoms
//void SetColorW(float4 *color, SimParams* hparams)
//{
	//dim3 dimGrid(hparams->gridSize, 1, 1);		// define number of blocks as in Anderson		
	//dim3 dimBlock(hparams->blockSize, 1, 1);	// number of threads

	//SetColorK <<< dimGrid, dimBlock >>> (color);	
//}

/////////////////////////
// wrappers for CUDA apis
/////////////////////////
void CudaInitW(int argc, char **argv)
{   
//#if __DEVICE_EMULATION__
   //cutilDeviceInit(argc, argv);
//#else
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
   // if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
       // cutilDeviceInit(argc, argv);
    //} else {
        //cudaSetDevice( cutGetMaxGflopsDeviceId() );
    //}
	// get cuda device properties
	 int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
	// Set the first GPU as the working one.
	int gpuId = 0;
	cudaSetDevice( gpuId );
	cudaGetDeviceProperties(&gDeviceProp, 0);	
//#endif
}

/*
void CudaGLInitW(int argc, char **argv)
{   
#if __DEVICE_EMULATION__
   cutilDeviceInit(argc, argv);
#else
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
        cutilDeviceInit(argc, argv);
    } else {
        cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    }
#endif
	// get cuda device properties
	cudaGetDeviceProperties(&gDeviceProp, 0);
}

void RegisterGLBufferObjectW(uint vbo)
{
    cutilSafeCall(cudaGLRegisterBufferObject(vbo));
}

void UnregisterGLBufferObjectW(uint vbo)
{
    cutilSafeCall(cudaGLUnregisterBufferObject(vbo));
}

void *MapGLBufferObjectW(uint vbo)
{
    void *ptr;
    cutilSafeCall(cudaGLMapBufferObject(&ptr, vbo));
    return ptr;
}

void UnmapGLBufferObjectW(uint vbo)
{
    cutilSafeCall(cudaGLUnmapBufferObject(vbo));
}
*/

void SetParametersW(SimParams *hostParams)
{
	// Allocate device memory.
	//SimParams *deviceParams;
	cudaError_t error_id;// = 
		//cudaMalloc((void**)&deviceParams,sizeof(SimParams));
	//if (error_id != cudaSuccess){
       // printf("Failed to allocate memory %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
       // exit(EXIT_FAILURE);
    //}
	// Copy host data to the device.
	//error_id = cudaMemcpy(deviceParams,hostParams,sizeof(SimParams),cudaMemcpyHostToDevice);
   // if (error_id != cudaSuccess){
      //  printf("Failed to copy memory %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
      //  exit(EXIT_FAILURE);
   // }

    // copy parameters to constant memory
    //cutilSafeCall( cudaMemcpyToSymbol(dparams, hostParams, sizeof(SimParams)) );
	error_id = 
		cudaMemcpyToSymbol((const void*)&dparams, hostParams, sizeof(SimParams));
		//cudaMemcpyToSymbol(pdparams, deviceParams, sizeof(SimParams));
		//cudaMemcpyFromSymbolAsync(dparams, hostParams, sizeof(SimParams), 0, cudaMemcpyHostToDevice, 0);

    if (error_id != cudaSuccess)
    {
        printf("Memory error while copying %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

	//cudaFree(deviceParams);
	//cudaError_t cudaMemcpyFromSymbolAsync(
	//cudaMemcpyToSymbol(dparams, hostParams, sizeof(SimParams), 0, cudaMemcpyHostToDevice);
}

}	// end extern "C"

////////////////////////////////////////////
// some host functions called from wrappers
////////////////////////////////////////////

// accumulate properties
void AccumProps (int icode, SimParams *hparams)
{
  if (icode == 0) {
    PropZero (hparams->totEnergy);
	PropZero (hparams->potEnergy);
    PropZero (hparams->kinEnergy);
	PropZero (hparams->oneStep); 
	PropZero (hparams->centerOfMass);
	PropZero (hparams->cmVel);
	PropZero (hparams->frictForce);
  } else if (icode == 1) {
    PropAccum (hparams->totEnergy);
	PropAccum (hparams->potEnergy);
    PropAccum (hparams->kinEnergy);
	PropAccum (hparams->oneStep);
	PropAccum (hparams->centerOfMass);
	PropAccum (hparams->cmVel);
	PropAccum (hparams->frictForce);  
  } else if (icode == 2) {
    PropAvg (hparams->totEnergy, hparams->stepAvg);
	PropAvg (hparams->potEnergy, hparams->stepAvg);
    PropAvg (hparams->kinEnergy, hparams->stepAvg);
	PropAvg (hparams->oneStep, hparams->stepAvg); 
	PropAvg (hparams->centerOfMass, hparams->stepAvg);
	PropAvg (hparams->cmVel, hparams->stepAvg);
	PropAvg (hparams->frictForce, hparams->stepAvg);
  }
}

// print results in a file
void PrintSummary (FILE *fp, SimParams *hparams)
{
	real totEn, totEnRms, potEn, potEnRms, T, TRms;
	
	// compute values in physical units
	// total energy in eV
	totEn = hparams->totEnergy.sum * hparams->enU;
	totEnRms = hparams->totEnergy.sum2 * hparams->enU;
	// potential energy in eV
	potEn = hparams->potEnergy.sum * hparams->enU;
	potEnRms = hparams->potEnergy.sum2 * hparams->enU;
	// temperature in Kelvin
	T = 2.f*hparams->kinEnergy.sum * hparams->temperatureU / 
		( NDIM*hparams->kB/*(1-1/hparams->nMol)*/ );
	TRms = 2.f*hparams->kinEnergy.sum2 * hparams->temperatureU
		/ ( NDIM*hparams->kB/**(1-1/hparams->nMol)*/ );
	
	// print values in the file
	fprintf (fp,
		"%5d\t %7.7f\t %7.7f\t %7.7f\t %7.7f\t %7.7f\t %7.7f\t %7.7f\t %7.7f\t",
		hparams->stepCount, VCSum (hparams->vSum) / hparams->nMol, 
		totEn,	 totEnRms,
		potEn,	 potEnRms,
		T, TRms,
		hparams->oneStep.sum);

	// tribological properties
	real centerOfMass, frictForce, xParticleSize, yParticleSize, zParticleSize, totalShear;
	centerOfMass = hparams->centerOfMass.sum * hparams->lengthU;
	// here is not force unit but 1 nN dimensionless, so divide, not multiply
	frictForce = hparams->frictForce.sum / hparams->forceU;	
	xParticleSize = hparams->particleSize.x * hparams->lengthU;
	yParticleSize = hparams->particleSize.y * hparams->lengthU;
	zParticleSize = hparams->particleSize.z * hparams->lengthU;
	totalShear = hparams->totalShear / hparams->forceU;

	fprintf (fp,
		"%7.7f\t %7.7f\t %7.6f\t %7.7f\t %7.7f\t %7.7f\t %7.7f\n",
		hparams->cmVel.sum, centerOfMass, frictForce, xParticleSize, yParticleSize,
		zParticleSize, totalShear);

	fflush (fp);		// reset buffer
}

// write coordinates to .pdb (protein data bank) file for future use with VMD
int CreatePdbFile(char *szPdb, SimParams *hparams, float4 *r)
{	
	int i;
	int n = 0;
	real m = 0.f;
	TCHAR szFileName[MAX_PATH],szBuf[MAX_PATH];

	ZeroMemory(szBuf, MAX_PATH);

	// to avoid problems with file exstentions
	i = hparams->stepCount / hparams->stepPdb;
	if(i < 100)
		sprintf(szBuf, TEXT("_%i"), i);
	else if(i < 1000)
		sprintf(szBuf, TEXT("_%-3i"), i);
	else if(i < 10000)
		sprintf(szBuf, TEXT("_%-4i"), i);

	lstrcat(szBuf, TEXT(".pdb"));
	lstrcpy(szFileName, szPdb);
	lstrcat(szFileName, szBuf);	

	FILE *pdb = fopen(szFileName, "w");		// Use standard function to open the file	
	if( !pdb )
	{
		//printf("Failed to create output file!");
		return 0;
	}

	// Print information to pdb file accordingly to its format	
	for(i = 0; i < hparams->nMol; i++)
	{   		
										//		Name				   Position in a file
		fprintf(pdb, "ATOM  ");			// Record name colums			1 - 6
		fprintf(pdb, "%-7i", i);		// Atom serial number colums	7 - 11
		if(i >= hparams->nMolMe)
			fprintf(pdb, "C  ");		// Atom name					13 - 16
		else 
			fprintf(pdb, "%2s ", hparams->szNameMe);				
		fprintf(pdb, " ");				// Alternate location indicator 17
		if(i >= hparams->nMolMe)
			fprintf(pdb, " C   ");		// Residue name					18 - 20 ? 21
		else
			fprintf(pdb, " %2s  ", hparams->szNameMe);
		fprintf(pdb, " ");				// Chain identifier				22
		fprintf(pdb, "    ");			// Residue sequence number		23 - 26
		
		fprintf(pdb, "%-4i", n);    	// Code for insertion of residues 27 - 30
		fprintf(pdb, "%-8.2lf", 1.42 *	// Output is in angstoms, so multiply by unit 
					r[i].x);			// Orthogonal coordinates for X  
										// in Angstroms					31 - 38
		fprintf(pdb, "%-8.2lf",	1.42 * 
					r[i].y);			// Orthogonal coordinates for Y  
										// in Angstroms					39 - 46
		// for SG print smaller coordinate for non deposited atoms
		if( (hparams->iRegime == 1) && (i < hparams->nMolMe) && (i >= hparams->nMolDeposited) )
			fprintf(pdb, "%-8.2lf",	1.42 * 2*
			hparams->region.z  );	// Orthogonal coordinates for Z  
										// in Angstroms					47 - 54
		else
			fprintf(pdb, "%-8.2lf",	1.42 * 
					r[i].z);			// Orthogonal coordinates for Z  
										// in Angstroms					47 - 54
		
		fprintf(pdb, " %4.1lf",	m);		// Occupancy					55 - 60		
		fprintf(pdb, "  1.00");		    // Temperature factor			61 - 66
		fprintf(pdb, "          ");		// Not documented				67 - 76
		if(i >= hparams->nMolMe)
			fprintf(pdb, " C");			// Element symbol, right-justified	77 - 78
		else
			fprintf(pdb, "%2s", hparams->szNameMe);

		fprintf(pdb, "  \n");			// Charge on the atom			79 - 80
	}

	fclose(pdb);

	return 1;
}

void PrintRdf(SimParams *hparams, uint *hHistRdf)
{
	real rb;
	int n;
	TCHAR szFileName[MAX_PATH], szBuf[MAX_PATH];
	ZeroMemory(szFileName, MAX_PATH);
	// define filename
	sprintf(szBuf, TEXT("_stepCount_%i_"), hparams->stepCount);	
	lstrcpy(szFileName, hparams->szRdfPath);
	lstrcat(szBuf, hparams->szNameMe);		// add metal name
	lstrcat(szFileName,szBuf);
	lstrcat(szFileName,TEXT(".txt"));
	// this code in Rapaport is outside of PrintRdf
	// but to avoid using real array for histRdf I put this code inside PrintRdf
	real normFac = VProd(hparams->particleSize)*Cube(hparams->intervalRdf) / 
				(2.f * M_PI * hparams->countRdf);
	// to avoid negative values, divide sequentially
	normFac = normFac/hparams->nMolMe;
	normFac = normFac/hparams->nMolMe;

	FILE *rdf = fopen(szFileName,TEXT("w+"));		// Use standard function to open the file

	real histRdf = 0.f;	
	
	for(n = 0; n < hparams->sizeHistRdf; n++)
	{
		rb = (n + 0.5f)*hparams->rangeRdf*hparams->lengthU / hparams->sizeHistRdf;
		histRdf = (real) hHistRdf[n]*normFac / ((n - 0.5f)*(n - 0.5f));
		fprintf(rdf, TEXT("%8.4f\t %8.4f\n"), rb, histRdf);
	}

	fclose(rdf);
}

// generates uniformly distributed random number as VRandR
void VRandRfloat4 (float4 *p, SimParams *hparams)
{
  real s, x, y;

  s = 2.f;
  while (s > 1.f) {
    x = 2.f * RandR (hparams) - 1.f;
    y = 2.f * RandR (hparams) - 1.f;
    s = Sqr (x) + Sqr (y);
  }
  p->z = 1.f - 2.f * s;
  s = 2.f * sqrt (1.f - s);
  p->x = s * x;
  p->y = s * y;
}

// host functions for diffusion
// initialize parameters for diffusion coefficient
void InitDiffusion(TBuf *tBuf, real *rrDiffuseAv, SimParams *hparams)
{
	int nb;
	// assign negative values to count as the initial values
	for(nb = 0; nb < hparams->nBuffDiffuse; nb++)
		tBuf[nb].count = -nb * hparams->nValDiffuse / hparams->nBuffDiffuse;
	ZeroDiffusion(rrDiffuseAv, hparams);
}

// reset parameters
void ZeroDiffusion(real *rrDiffuseAv, SimParams *hparams)
{
	int j;

	hparams->countDiffuseAv = 0;
	for(j = 0; j < hparams->nValDiffuse; j++) rrDiffuseAv[j] = 0.;
}

// print diffusion coefficient in a file
void PrintDiffusion(real *rrDiffuseAv, FILE *file, SimParams *hparams)
{
	real tVal;
	int j;

	fprintf(file, TEXT("diffusion\n"));
	for(j = 0; j < hparams->nValDiffuse; j++)
	{
		tVal = j * hparams->stepDiffuse * hparams->deltaT;
		fprintf(file, TEXT("%8.4f %8.4f\n"), tVal, rrDiffuseAv[j]);
	}
}

// accumulate data for diffusion
void AccumDiffusion(TBuf *tBuf, real *rrDiffuseAv, FILE *file, SimParams *hparams)
{
	real fac;
	int j, nb;

	for(nb = 0; nb < hparams->nBuffDiffuse; nb++)	{
		if(tBuf[nb].count == hparams->nValDiffuse)	{
			for(j = 0; j < hparams->nValDiffuse; j++)
				rrDiffuseAv[j] += tBuf[nb].rrDiffuse[j];
			tBuf[nb].count = 0;
			++ hparams->countDiffuseAv;
			if(hparams->countDiffuseAv == hparams->limitDiffuseAv)
			{
				fac = 1. / (NDIM* 2 * hparams->stepDiffuse * 
					hparams->deltaT * hparams->limitDiffuseAv);
				for(j = 1; j < hparams->nValDiffuse; j++)
					rrDiffuseAv[j] *= fac / j;
				PrintDiffusion(rrDiffuseAv, file, hparams);
				ZeroDiffusion(rrDiffuseAv, hparams);
			}
		}
	}
}

// compute diffusion constant
void EvalDiffusion(TBuf *tBuf, real *rrDiffuseAv, FILE *file, SimParams *hparams,
				   float3 centerOfMass)
{
	float3 dr;
	int nb, ni;

	// loop over all measurement sets
	for(nb = 0; nb < hparams->nBuffDiffuse; nb++) {
		if(tBuf[nb].count == 0){
			tBuf[nb].orgR = centerOfMass;
			tBuf[nb].rTrue = centerOfMass;
		}

		if(tBuf[nb].count >= 0) {
			ni = tBuf[nb].count;
			tBuf[nb].rrDiffuse[ni] = 0.;
			VSub(dr, tBuf[nb].rTrue, centerOfMass);			
			VDiv(dr, dr, hparams->region);
			dr.x = Nint(dr.x);
			dr.y = Nint(dr.y);
			dr.z = Nint(dr.z);
			VMul(dr, dr, hparams->region);
			VAdd(tBuf[nb].rTrue, centerOfMass, dr);			
			VSub(dr, tBuf[nb].rTrue, tBuf[nb].orgR);
			tBuf[nb].rrDiffuse[ni] += VLenSq(dr);
		}// end if(tBuf[nb].count >= 0)
		++ tBuf[nb].count;
	} // end of loop over all sets
	AccumDiffusion(tBuf, rrDiffuseAv, file, hparams);
}