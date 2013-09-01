//----------------------------------------------
// ComputeDefs.h - definitions for computations
// (c) 2010 Mykola Prodanov
// (this code was written in Sumy, Ukraine)
//----------------------------------------------
//-------------------------------------------------------------------
// Most of the macroses are from the code accompanying the book
// "The Art of Molecular Dynamics Simulation", 2nd edition,
// by D. C. Rapaport, published by Cambridge University Press (2004).
// (C) 2004  D. C. Rapaport
//-------------------------------------------------------------------

#ifndef __COMPUTE_DEFS__
#define __COMPUTE_DEFS__

#include "vector_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef float real;

// from cutil_math.h
typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef NDIM
#define NDIM  3
#endif

// structures
typedef struct _VecR2
{real x, y;} 
VecR2;

typedef struct _VecR3
{real x, y, z;} 
VecR3;

typedef struct _VecI2
{int x, y;} 
VecI2;

typedef struct _VecI3
{int x, y, z;} 
VecI3;

typedef VecR3 VecR;
typedef VecI3 VecI;

// for properties
typedef struct _Prop { //__align__(16)
  real val, sum, sum2;
} Prop;

#define PropZero(v)  v.sum = v.sum2 = 0.
#define PropAccum(v)  v.sum += v.val, v.sum2 += Sqr (v.val)
#define PropAvg(v, n) \
   v.sum /= n, v.sum2 = sqrt (Max (v.sum2 / n - Sqr (v.sum), 0.))
#define PropEst(v)  v.sum, v.sum2

#define AllocMem(a, n, t)  a = (t *) malloc ((n) * sizeof (t))

#define Sqr(x)     ((x) * (x))
#define Cube(x)    ((x) * (x) * (x))
#define Sgn(x, y)  (((y) >= 0) ? (x) : (- (x)))
#define IsEven(x)  ((x) & ~1)
#define IsOdd(x)   ((x) & 1)
#define Nint(x)                                             \
   (((x) < 0.) ? (- (int) (0.5 - (x))): ((int) (0.5 + (x))))
#define Min(x1, x2)  (((x1) < (x2)) ? (x1) : (x2))
#define Max(x1, x2)  (((x1) > (x2)) ? (x1) : (x2))
#define Min3(x1, x2, x3) \
   (((x1) < (x2)) ? (((x1) < (x3)) ? (x1) : (x3)) :         \
                    (((x2) < (x3)) ? (x2) : (x3)))
#define Max3(x1, x2, x3) \
   (((x1) > (x2)) ? (((x1) > (x3)) ? (x1) : (x3)) :         \
                    (((x2) > (x3)) ? (x2) : (x3)))
#define Clamp(x, lo, hi)                                    \
   (((x) >= (lo) && (x) <= (hi)) ? (x) :                    \
   (((x) < (lo)) ? (lo) : (hi)))

// macroses
#if NDIM == 3

typedef VecR3 VecR;
typedef VecI3 VecI;

#define VSet(v, sx, sy, sz)                                 \
   (v).x = sx,                                              \
   (v).y = sy,                                              \
   (v).z = sz
#define VCopy(v1, v2)                                       \
   (v1).x = (v2).x,                                         \
   (v1).y = (v2).y,                                         \
   (v1).z = (v2).z
#define VScale(v, s)                                        \
   (v).x *= s,                                              \
   (v).y *= s,                                              \
   (v).z *= s
#define VSCopy(v2, s1, v1)                                  \
   (v2).x = (s1) * (v1).x,                                  \
   (v2).y = (s1) * (v1).y,                                  \
   (v2).z = (s1) * (v1).z
#define VAdd(v1, v2, v3)                                    \
   (v1).x = (v2).x + (v3).x,                                \
   (v1).y = (v2).y + (v3).y,                                \
   (v1).z = (v2).z + (v3).z
#define VSub(v1, v2, v3)                                    \
   (v1).x = (v2).x - (v3).x,                                \
   (v1).y = (v2).y - (v3).y,                                \
   (v1).z = (v2).z - (v3).z
#define VMul(v1, v2, v3)                                    \
   (v1).x = (v2).x * (v3).x,                                \
   (v1).y = (v2).y * (v3).y,                                \
   (v1).z = (v2).z * (v3).z
#define VDiv(v1, v2, v3)                                    \
   (v1).x = (v2).x / (v3).x,                                \
   (v1).y = (v2).y / (v3).y,                                \
   (v1).z = (v2).z / (v3).z
#define VSAdd(v1, v2, s3, v3)                               \
   (v1).x = (v2).x + (s3) * (v3).x,                         \
   (v1).y = (v2).y + (s3) * (v3).y,                         \
   (v1).z = (v2).z + (s3) * (v3).z
#define VSSAdd(v1, s2, v2, s3, v3)                          \
   (v1).x = (s2) * (v2).x + (s3) * (v3).x,                  \
   (v1).y = (s2) * (v2).y + (s3) * (v3).y,                  \
   (v1).z = (s2) * (v2).z + (s3) * (v3).z
#define VDot(v1, v2)                                        \
   ((v1).x * (v2).x + (v1).y * (v2).y + (v1).z * (v2).z)
#define VWDot(v1, v2, v3)                                   \
   ((v1).x * (v2).x * (v3).x + (v1).y * (v2).y * (v3).y +   \
   (v1).z * (v2).z * (v3).z)
#define VCross(v1, v2, v3)                                  \
   (v1).x = (v2).y * (v3).z - (v2).z * (v3).y,              \
   (v1).y = (v2).z * (v3).x - (v2).x * (v3).z,              \
   (v1).z = (v2).x * (v3).y - (v2).y * (v3).x
#define MVMul(v1, m, v2)                                         \
   (v1).x = (m)[0] * (v2).x + (m)[3] * (v2).y + (m)[6] * (v2).z, \
   (v1).y = (m)[1] * (v2).x + (m)[4] * (v2).y + (m)[7] * (v2).z, \
   (v1).z = (m)[2] * (v2).x + (m)[5] * (v2).y + (m)[8] * (v2).z
#define MVMulT(v1, m, v2)                                        \
   (v1).x = (m)[0] * (v2).x + (m)[1] * (v2).y + (m)[2] * (v2).z, \
   (v1).y = (m)[3] * (v2).x + (m)[4] * (v2).y + (m)[5] * (v2).z, \
   (v1).z = (m)[6] * (v2).x + (m)[7] * (v2).y + (m)[8] * (v2).z
#define VProd(v)                                            \
   ((v).x * (v).y * (v).z)
#define VGe(v1, v2)                                         \
   ((v1).x >= (v2).x && (v1).y >= (v2).y && (v1).z >= (v2).z)
#define VLt(v1, v2)                                         \
   ((v1).x < (v2).x && (v1).y < (v2).y && (v1).z < (v2).z)
#define VLinear(p, s)                                       \
   (((p).z * (s).y + (p).y) * (s).x + (p).x)
#define VSetAll(v, s)                                       \
   VSet (v, s, s, s)
#define VAddCon(v1, v2, s)                                  \
   (v1).x = (v2).x + (s),                                   \
   (v1).y = (v2).y + (s),                                   \
   (v1).z = (v2).z + (s)
#define VComp(v, k)                                         \
   *((k == 0) ? &(v).x : ((k == 1) ? &(v).y : &(v).z))
#define VToLin(a, n, v)                                     \
   a[(n) + 0] = (v).x,                                      \
   a[(n) + 1] = (v).y,                                      \
   a[(n) + 2] = (v).z
#define VFromLin(v, a, n)                                   \
   VSet (v, a[(n) + 0], a[(n) + 1], a[(n) + 2])
#define VCSum(v)                                            \
   ((v).x + (v).y + (v).z)
#endif

#define VZero(v)  VSetAll (v, 0)
#define VLenSq(v)  VDot (v, v)
#define VWLenSq(v1, v2)  VWDot(v1, v2, v2)
#define VLen(v)  sqrt (VDot (v, v))
#define VVAdd(v1, v2)  VAdd (v1, v1, v2)
#define VVSub(v1, v2)  VSub (v1, v1, v2)
#define VVSAdd(v1, s2, v2) VSAdd (v1, v1, s2, v2)
#define VInterp(v1, s2, v2, v3)                             \
   VSSAdd (v1, s2, v2, 1. - (s2), v3)

// periodic boundaries macroses
// ! note, here device region, not host
#define VWrap(v, t)                                         \
   if ((v).t >= 0.5f * dparams.region.t)      (v).t -= dparams.region.t;         \
   else if ((v).t < -0.5f * dparams.region.t) (v).t += dparams.region.t

#define VShiftWrap(v, t)                                    \
   if (v.t >= 0.5f * region.t) {                             \
     shift.t -= region.t;                                   \
     v.t -= region.t;                                       \
   } else if (v.t < -0.5f * region.t) {                      \
     shift.t += region.t;                                   \
     v.t += region.t;                                       \
   }

#define VCellWrap(t)                                        \
   if (m2v.t >= dparams.cells.t) {                                  \
     m2v.t = 0;                                             \
     shift.t = dparams.region.t;                                    \
   } else if (m2v.t < 0) {                                  \
     m2v.t = dparams.cells.t - 1;                                   \
     shift.t = - dparams.region.t;                                  \
   }

#define VWrapAll(v)                                         \
   {VWrap (v, x);                                           \
   VWrap (v, y);                                            \
   VWrap (v, z);}
#define VShiftAll(v)                                        \
   {VShift (v, x);                                          \
   VShift (v, y);                                           \
   VShift (v, z);}
#define VCellWrapAll()                                      \
   {VCellWrap (x);                                          \
   VCellWrap (y);                                           \
   VCellWrap (z);}

// array for looping over all 27 neighbouring cells
#define OFFSET_VALS_LONG                                 \
{   {0,0,0},  {0,0,1},  {0,0,-1},                        \
    {0,1,0},  {0,1,1},  {0,1,-1},                        \
    {0,-1,0}, {0,-1,1}, {0,-1,-1},                       \
    {1,0,0},  {1,0,1},  {1,0,-1},                        \
    {1,1,0},  {1,1,1},  {1,1,-1},                        \
    {1,-1,0}, {1,-1,1}, {1,-1,-1},                       \
    {-1,0,0},  {-1,0,1},  {-1,0,-1},                     \
    {-1,1,0},  {-1,1,1},  {-1,1,-1},                     \
    {-1,-1,0}, {-1,-1,1}, {-1,-1,-1}                     \
}

// number of cells in long path
#define N_OFFSET_LONG  27	

#endif		// end __COMPUTE_DEFS__