/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "FeatureGen.h"

// cl
#include "clImgProc/ImageBuf.h"
#include "clImgProc/resize.h"
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
#define HISTSIZE					256
#define THRESHOLD_IMG_SIZE			( 1024 * 1024 )


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// only public methods
bool CFeatureGen::CalcFeatures( IN const cl::img::CImageBuf & img__, OUT double** adbFeatures )
{
	if ( ! img__.IsValid() )
		return false;

	// make half image if too large
	const cl::img::CImageBuf* pImg = &img__;
	cl::img::CImageBuf imgHalf;

	if ( img__.GetWidth() * img__.GetHeight() >= THRESHOLD_IMG_SIZE )
	{
		VERIFY( cl::img::utils::ResizeWholeImage( img__, img__.GetWidth() / 2, img__.GetHeight() / 2, 
													cl::img::utils::EINTP_Nearest, &imgHalf ) );
		pImg = &imgHalf;
	}
	const cl::img::CImageBuf& img = *pImg;

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
	double adLocalSum[ ABC_REGION_DIVIDE_2 ];
	double adLocalSoS[ ABC_REGION_DIVIDE_2 ];

	_calcLocalStatistics( pwSrc, nImgW, nImgH, nBlkW, nBlkH, awLocalMax, awLocalMin, adLocalSum, adLocalSoS );

	// global statistics using local statistics
	WORD wGlobalMax = 0x0, wGlobalMin = 0xffff;
	double dGlobalSum = 0.0;
	double dGlobalSoS = 0.0;

	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		wGlobalMax = CLU_MAX( wGlobalMax, awLocalMax[ bi ] );
		wGlobalMin = CLU_MIN( wGlobalMin, awLocalMin[ bi ] );

		dGlobalSum += adLocalSum[ bi ];
		dGlobalSoS += adLocalSoS[ bi ];
	}

	// global otsu
	double dGlobalOtsu = 0.5, dGlobalInner = 0.5, dGlobalInter = 0.5, dGlobalMode = 0.5;

	_calcOtsu( 
		pwSrc + nBlkW + nBlkH * nImgW,		// exclude boundary blocks
		nImgW,								// strider 
		nBlkW * ( ABC_REGION_DIVIDE - 2 ),
		nBlkH * ( ABC_REGION_DIVIDE - 2 ) , 
		wGlobalMin, wGlobalMax, &dGlobalOtsu, &dGlobalInner, &dGlobalInter, &dGlobalMode );

	// local otsu
	double adLocalOtsu[ ABC_REGION_DIVIDE_2 ];
	double adLocalMode[ ABC_REGION_DIVIDE_2 ];

	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		const int bx = bi % ABC_REGION_DIVIDE;
		const int by = bi / ABC_REGION_DIVIDE;

		double dLocalOtsu = 0.5, dLocalInner = 0.5, dLocalInter = 0.5, dLocalMode = 0.5;

		if ( bx == 0 || bx == ABC_REGION_DIVIDE - 1 || by == 0 || by == ABC_REGION_DIVIDE - 1 )
		{
			// do nothing
		}
		else 
		{
			const WORD* pwBlk = pwSrc + ( bx * nBlkW ) + ( by * nBlkH ) * nImgW;

			_calcOtsu( pwBlk, nImgW, nBlkW, nBlkH, awLocalMin[ bi ], awLocalMax[ bi ], 
							&dLocalOtsu, &dLocalInner, &dLocalInter, &dLocalMode );
		}
		
		adLocalOtsu[ bi ] = dLocalOtsu;
		adLocalMode[ bi ] = dLocalMode;
	}

	// block population
	const double dBlkPopulation = (double)( nBlkW * nBlkH );
	const double dGlobalPopulation = dBlkPopulation * ( ABC_REGION_DIVIDE - 2 ) * ( ABC_REGION_DIVIDE - 2 );

	const double dGlobalMax  = (double) wGlobalMax;
	const double dGlobalMin  = (double) wGlobalMin;
	const double dGlobalMean = dGlobalSum / dGlobalPopulation;

	// FIX: When an image having the same value is input, it may have a value smaller than 0 due to an error.
	const double dGlobalStd  = sqrt( CLU_LBOUND( dGlobalSoS / dGlobalPopulation - CLU_SQUARE( dGlobalMean ), 0. ) );

	// output
	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		double* pdFeatures = adbFeatures[ bi ];
		ASSERT( pdFeatures );

		pdFeatures[ kABCFeatureId_Global_Otsu	]	= dGlobalOtsu;
		pdFeatures[ kABCFeatureId_Global_Max	]	= dGlobalMax	/ 65535.;
		pdFeatures[ kABCFeatureId_Global_Min	]	= dGlobalMin	/ 65535.;
		pdFeatures[ kABCFeatureId_Global_Mean	]	= dGlobalMean	/ 65535.;
		pdFeatures[ kABCFeatureId_Global_Std	]	= dGlobalStd	/ 65535.;
		pdFeatures[ kABCFeatureId_Global_Mode	]	= dGlobalMode;

		const double dLocalMean = adLocalSum[ bi ] / dBlkPopulation;
		const double dLocalStd  = sqrt( CLU_LBOUND( adLocalSoS[ bi ] / dBlkPopulation - CLU_SQUARE( dLocalMean ), 0. ) );

		pdFeatures[ kABCFeatureId_Local_Otsu	]	= adLocalOtsu[ bi ];
		pdFeatures[ kABCFeatureId_Local_Max		]	= (double) awLocalMax[ bi ] / 65535.;
		pdFeatures[ kABCFeatureId_Local_Min		]	= (double) awLocalMin[ bi ] / 65535.;
		pdFeatures[ kABCFeatureId_Local_Mean	]	= dLocalMean				/ 65535.;
		pdFeatures[ kABCFeatureId_Local_Std		]	= dLocalStd					/ 65535.;
		pdFeatures[ kABCFeatureId_Local_Mode	]	= adLocalMode[ bi ];
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// local
void CFeatureGen::_calcLocalStatistics( 
			const WORD* pwSrc, int nImgW, int nImgH, int nBlkW, int nBlkH, 
			WORD awLocalMax[], WORD awLocalMin[], double adLocalSum[], double adLocalSoS[] )
{
	UNREFERENCED_PARAMETER( nImgH );

	// Using openmp does not get much faster. Removed.
	// loop for blocks
//	#pragma omp parallel for schedule(static)		\
//				firstprivate( pwSrc, nBlkW, nBlkH, nImgW, awLocalMax, awLocalMin, anLocalSum, adbLocalSoS )

	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		const int bx = bi % ABC_REGION_DIVIDE;
		const int by = bi / ABC_REGION_DIVIDE;

		// boundary blocks
		if ( bx == 0 || bx == ABC_REGION_DIVIDE - 1 || by == 0 || by == ABC_REGION_DIVIDE - 1 )
		{
			awLocalMax[ bi ] = 0x0000;
			awLocalMin[ bi ] = 0xffff;
			adLocalSum[ bi ] = 0.;
			adLocalSoS[ bi ] = 0.;
		}
		// non-boundary blocks
		else 
		{
			const WORD* pwBlk = pwSrc + ( bx * nBlkW ) + ( by * nBlkH ) * nImgW;

			_calcBlockStatistics( pwBlk, nImgW, nBlkW, nBlkH, 
				awLocalMax + bi, awLocalMin + bi, adLocalSum + bi, adLocalSoS + bi );
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// block
void CFeatureGen::_calcBlockStatistics( const WORD* pwBlk, int nStrider, int nBlkW, int nBlkH, 
									    WORD* pwLocalMax, WORD* pwLocalMin, double* pdLocalSum, double* pdLocalSoS )
{
	// local constant
	const int nBlkW_8 = ( nBlkW / 8 ) * 8;
	const __m128i mZeros = _mm_set1_epi16( 0 );
	const __m128i mOnes  = _mm_set1_epi16( -1 );

	// initialize
	*pwLocalMax = 0x0000;
	*pwLocalMin = 0xffff;
	*pdLocalSum = 0.;
	*pdLocalSoS = 0.;

	__m128i mMax = mZeros;
	__m128i mMin = mOnes;

	// loop
	for ( int y=0; y<nBlkH; y++ )
	{
		int x = 0;

		__m128i mSum = mZeros;
		__m128  mfSoS = _mm_set1_ps( 0.f );

		// Use SSE operation to process 8 pixels each.
		for ( ; x<nBlkW_8; x+=8 )
		{
			__m128i mValue = _mm_loadu_si128( reinterpret_cast<const __m128i*>( pwBlk + x ) );

			// min, max
			mMax = _mm_max_epu16( mMax, mValue );
			mMin = _mm_min_epu16( mMin, mValue );

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
			*pdLocalSum += static_cast<double>( mSum.m128i_u32[ i ] );
			*pdLocalSoS += static_cast<double>( mfSoS.m128_f32[ i ] );
		}

		// remaining pixels
		for ( ; x<nBlkW; x++ )
		{
			*pwLocalMax  = CLU_MAX( *pwLocalMax, pwBlk[ x ] );
			*pwLocalMin  = CLU_MIN( *pwLocalMin, pwBlk[ x ] );
			*pdLocalSum += static_cast<double>( pwBlk[ x ] );
			*pdLocalSoS += static_cast<double>( pwBlk[ x ] );
		}

		// next line
		pwBlk += nStrider;
	}

	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CalOtsu
void CFeatureGen::_calcOtsu( 
					const WORD* pwSrc, int nStrider, int nBlkW, int nBlkH, WORD wMin, WORD wMax,
					double* pdOtsu, double* pdInner, double* pdInter, double* pdMode ) 
{
	if ( wMin == wMax )
	{
		*pdOtsu  = 0.5; *pdInner = 0.5; *pdInter = 0.5; *pdMode = 0.5;
		return;
	}

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
		int nMode = 0, nModeValue = iHist[ 0 ];

		for ( int i=0; i<HISTSIZE; i++ )
		{
			iHistSum	+= iHist[ i ];
			iHistMulSum += iHist[ i ] * i;

			iCumHist[ i ] = iHistSum;
			iCumMulHist[ i ] = iHistMulSum;

			if ( nModeValue < iHist[ i ] )
			{
				nModeValue = iHist[ i ];
				nMode = i;
			}
		}

		// normalized mode
		*pdMode = (double) nMode / HISTSIZE;
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