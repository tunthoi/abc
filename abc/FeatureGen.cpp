/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "FeatureGen.h"

// cl
#include "clImgProc/ImageBuf.h"
#include "clUtils/utils.h"

// platform
#include <smmintrin.h>
#include <cmath>
#include <omp.h>

using namespace comed::abc;

#ifdef _DEBUG
#define new DEBUG_NEW 
#endif 

// macro
#define HISTSIZE		256


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// only public methods
bool CFeatureGen::CalcFeatures( 
				IN		const cl::img::CImageBuf & img, 
				OUT		double** adbFeatures )
//				OUT		double adbFeatures[ ABC_REGION_DIVIDE_2 ][ ABC_FEATURE_COUNT ] )
{
	if ( ! img.IsValid() )
		return false;

	// asserting
	ASSERT( img.GetType() == cl::img::EIT_Gray16bit );
	ASSERT( adbFeatures != nullptr );

	// image dimension
	const int nImgW = img.GetWidth(), nImgH = img.GetHeight();

	ASSERT( nImgW > ABC_REGION_DIVIDE );
	ASSERT( nImgH > ABC_REGION_DIVIDE );

	// block
	const int nBlkW = nImgW / ABC_REGION_DIVIDE, nBlkH = nImgH / ABC_REGION_DIVIDE;

	ASSERT( nBlkW > 0 );
	ASSERT( nBlkH > 0 );

	// pixel buffer
	const WORD* pwSrc = img.GetPixelDataWord();
	ASSERT( pwSrc != nullptr );

	// first, we have to local statistics
	WORD awLocalMax[ ABC_REGION_DIVIDE_2 ], awLocalMin[ ABC_REGION_DIVIDE_2 ];
	UINT anLocalSum[ ABC_REGION_DIVIDE_2 ];
	double adbLocalSoS[ ABC_REGION_DIVIDE_2 ];

	_calcLocalStatistics( pwSrc, nImgW, nImgH, nBlkW, nBlkH, awLocalMax, awLocalMin, anLocalSum, adbLocalSoS );

	// global statistics
	WORD wGlobalMax = 0x0, wGlobalMin = 0xffff;
	UINT nGlobalSum = 0;
	double dbGlobalSoS = 0.0;

	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		wGlobalMax = CLU_MAX( wGlobalMax, awLocalMax[ bi ] );
		wGlobalMin = CLU_MIN( wGlobalMin, awLocalMin[ bi ] );

		nGlobalSum += anLocalSum[ bi ];
		dbGlobalSoS += adbLocalSoS[ bi ];
	}

	// global otsu
	double dbGlobalOtsu = 0., dbGlobalInner = 0., dbGlobalInter = 0.;
	_calcOtsu( pwSrc, nImgW, nImgW, nImgH, wGlobalMin, wGlobalMax, 
					&dbGlobalOtsu, &dbGlobalInner, &dbGlobalInter );

	// local otsu
	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		const int bx = bi % ABC_REGION_DIVIDE;
		const int by = bi / ABC_REGION_DIVIDE;

		const WORD* pwBlk = pwSrc + ( bx * nBlkW ) + ( by * nBlkH ) * nImgW;

		double dbLocalOtsu = 0., dbLocalInner = 0., dbLocalInter = 0.;
		_calcOtsu( pwBlk, nImgW, nBlkW, nBlkH, awLocalMin[ bi ], awLocalMax[ bi ], 
						&dbLocalOtsu, &dbLocalInner, &dbLocalInter );
		
		double* pdbFeatures = adbFeatures[ bi ];
		ASSERT( pdbFeatures );

		pdbFeatures[ kABCFeatureId_Local_Otsu	] = dbLocalOtsu;
		pdbFeatures[ kABCFeatureId_Local_Inner	] = dbLocalInner;
		pdbFeatures[ kABCFeatureId_Local_Inter	] = dbLocalInter;
	}

	// block population
	const double dbBlkPopulation = (double)( nBlkW * nBlkH );
	const double dbGlobalPopulation = dbBlkPopulation * ABC_REGION_DIVIDE_2;

	// output
	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		double* pdbFeatures = adbFeatures[ bi ];
		ASSERT( pdbFeatures );

		pdbFeatures[ kABCFeatureId_Global_Otsu	]	= dbGlobalOtsu;
		pdbFeatures[ kABCFeatureId_Global_Inner	]	= dbGlobalInner;
		pdbFeatures[ kABCFeatureId_Global_Inter	]	= dbGlobalInter;

		pdbFeatures[ kABCFeatureId_Global_Max   ]	= (double) wGlobalMax;
		pdbFeatures[ kABCFeatureId_Global_Min   ]	= (double) wGlobalMin;
		pdbFeatures[ kABCFeatureId_Global_Mean  ]	= (double) nGlobalSum / dbGlobalPopulation;
		pdbFeatures[ kABCFeatureId_Global_Std   ]	= 
			sqrt( CLU_SQUARE( pdbFeatures[ kABCFeatureId_Global_Mean ] ) - dbGlobalSoS / dbGlobalPopulation );

		pdbFeatures[ kABCFeatureId_Local_Max   ]	= (double) awLocalMax[ bi ];
		pdbFeatures[ kABCFeatureId_Local_Min   ]	= (double) awLocalMin[ bi ];
		pdbFeatures[ kABCFeatureId_Local_Mean  ]	= (double) anLocalSum[ bi ] / dbBlkPopulation;
		pdbFeatures[ kABCFeatureId_Local_Std   ]	=
			sqrt( CLU_SQUARE( pdbFeatures[ kABCFeatureId_Local_Mean ] ) - adbLocalSoS[ bi ] / dbBlkPopulation );
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// local
void CFeatureGen::_calcLocalStatistics( 
			const WORD* pwSrc, int nImgW, int nImgH, int nBlkW, int nBlkH, 
			WORD awLocalMax[], WORD awLocalMin[], UINT anLocalSum[], double adbLocalSoS[] )
{
	UNREFERENCED_PARAMETER( nImgH );

	// loop for blocks
	#pragma omp parallel for schedule(static)		\
				firstprivate( pwSrc, nBlkW, nBlkH, nImgW, awLocalMax, awLocalMin, anLocalSum, adbLocalSoS )

	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		const int bx = bi % ABC_REGION_DIVIDE;
		const int by = bi / ABC_REGION_DIVIDE;

		const WORD* pwBlk = pwSrc + ( bx * nBlkW ) + ( by * nBlkH ) * nImgW;

		_calcBlockStatistics( pwBlk, nImgW, nBlkW, nBlkH, 
			awLocalMax + bi, awLocalMin + bi, anLocalSum + bi, adbLocalSoS + bi );
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// block
void CFeatureGen::_calcBlockStatistics( const WORD* pwBlk, int nStrider, int nBlkW, int nBlkH, 
									    WORD* pwLocalMax, WORD* pwLocalMin, UINT* pnLocalSum, double* pdbLocalSoS )
{
	// local constant
	const int nBlkW_8 = ( nBlkW / 8 ) * 8;
	const __m128i mZeros = _mm_set1_epi16( 0 );
	const __m128i mOnes  = _mm_set1_epi16( -1 );

	// initialize
	*pwLocalMax = 0x0000;
	*pwLocalMin = 0xffff;
	*pnLocalSum = 0;
	*pdbLocalSoS = 0.;

	// loop
	for ( int y=0; y<nBlkH; y++ )
	{
		int x = 0;

		__m128i mMax = mZeros;
		__m128i mMin = mOnes;
		__m128i mSum = mZeros;
		__m128  mfSoS = _mm_set1_ps( 0.f );

		// SSE operation을 사용해서 8 pixels씩 처리한다.
		for ( ; x<nBlkW_8; x+=8 )
		{
			__m128i mValue = _mm_loadu_si128( reinterpret_cast<const __m128i*>( pwBlk ) );

			// min, max
			mMax = _mm_max_epu16( mMax, mValue );
			mMin = _mm_min_epi16( mMin, mValue );

			// convert to 32bit 
			__m128i mValue0 = _mm_unpacklo_epi16( mValue, mZeros );
			__m128i mValue1 = _mm_unpackhi_epi16( mValue, mZeros );

			// sum
			mSum = _mm_add_epi32( mSum, mValue0 );
			mSum = _mm_add_epi32( mSum, mValue1 );

			// sum of square
			__m128 mfValue0 = _mm_cvtepi32_ps( mValue0 );
			__m128 mfValue1 = _mm_cvtepi32_ps( mValue1 );

			mfValue0 = _mm_mul_ps( mfValue0, mfValue0 );
			mfValue1 = _mm_mul_ps( mfValue1, mfValue1 );

			mfSoS = _mm_add_ps( mfSoS, mfValue0 );
			mfSoS = _mm_add_ps( mfSoS, mfValue1 );
		}

		// merge 
		for ( int i=0; i<8; i++ )
		{
			*pwLocalMax   = CLU_MAX( *pwLocalMax, mMax.m128i_u16[ i ] );
			*pwLocalMin   = CLU_MIN( *pwLocalMin, mMin.m128i_u16[ i ] );
		}
		for ( int i=0; i<4; i++ )
		{
			*pnLocalSum  += static_cast<UINT>( mSum.m128i_u32[ i ] );
			*pdbLocalSoS += CLU_SQUARE( static_cast<double>( mfSoS.m128_f32[ i ] ) );
		}

		// remaining pixels
		for ( ; x<nBlkW; x++ )
		{
			*pwLocalMax = CLU_MAX( *pwLocalMax, pwBlk[ x ] );
			*pwLocalMin = CLU_MIN( *pwLocalMin, pwBlk[ x ] );
			*pnLocalSum += static_cast<UINT>( pwBlk[ x ] );
			*pdbLocalSoS += CLU_SQUARE( static_cast<double>( pwBlk[ x ] ) );
		}

		// next line
		pwBlk += nStrider;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CalOtsu
void CFeatureGen::_calcOtsu( 
					const WORD* pwSrc, int nStrider, int nBlkW, int nBlkH, WORD wMin, WORD wMax,
					double* pdOtsu, double* pdInner, double* pdInter ) 
{

	// histogram
	int iHist[ HISTSIZE + 1 ];		// +1 simple bound
	{
		const double dFactor = HISTSIZE / (double)( wMax - wMin + 1 );
		const WORD* pwLine = pwSrc;

		::ZeroMemory( iHist, sizeof(int) * ( HISTSIZE + 1 ) );

		for ( int y=0; y<nBlkH; y++, pwLine += nStrider )
		for ( int x=0; x<nBlkW; x++ )
		{
			const register int nBin = (int)( ( pwLine[ x ] - wMin ) * dFactor );
			ASSERT( nBin >= 0 );
			ASSERT( nBin < HISTSIZE );

			iHist[ nBin ] ++;
		}
	}

	// cumulative histogram, mul-sum
	int iCumHist[ HISTSIZE ];
	int iCumMulHist[ HISTSIZE ];
	int iHistSum = 0, iHistMulSum = 0;
	{
		for ( int i=0; i<HISTSIZE; i++ )
		{
			iHistSum	+= iHist[ i ];
			iHistMulSum += iHist[ i ] * i;

			iCumHist[ i ] = iHistSum;
			iCumMulHist[ i ] = iHistMulSum;
		}
	}

	// just for simplicity
	const double dblHistSum = (double) iHistSum;
	double dInner[ HISTSIZE ];
	double dInter[ HISTSIZE ];

	for ( int i=0; i<HISTSIZE; i++ )
	{
		double dCumVarB = 0.0;
		double dCumVarF = 0.0;

		const double dMeanB = (double) iCumMulHist[i] / iCumHist[i];
		const double dMeanF = (double)( iHistMulSum - iCumMulHist[i] ) / ( iHistSum - iCumHist[i] );

		for ( int j=0; j<i; j++ )
		{
			dCumVarB = dCumVarB + CLU_SQUARE( j - dMeanB ) * iHist[j];
		}
		for ( int k=i; k<HISTSIZE; k++ )
		{
			dCumVarF = dCumVarF + CLU_SQUARE( k - dMeanF ) * iHist[k];
		}		

		const double dVarB = dCumVarB / iCumHist[i];
		const double dVarF = dCumVarF / ( iHistSum - iCumHist[i] );

		const double dWeightB = iCumHist[ i ] / dblHistSum;
		const double dWeightF = 1.0 - dWeightB;

		dInner[i] = dWeightB * dVarB + dWeightF * dVarF;
		dInter[i] = dWeightB * dWeightF * ( ( dMeanB - dMeanF ) * ( dMeanB - dMeanF ) );
	}	

	// find maximum for otsu
	double dTempMax = 0.0;
	int nMaxIndex = 0;

	for ( int i=0; i<HISTSIZE; i++ )
	{
		const double rst = dInter[i] / dInner[i];

		if ( rst > dTempMax )
		{
			dTempMax = rst;
			nMaxIndex = i;
		}
	}

	*pdOtsu = (double) nMaxIndex / HISTSIZE;
	*pdInner = dInner[ nMaxIndex ];
	*pdInter = dInter[ nMaxIndex ];
}