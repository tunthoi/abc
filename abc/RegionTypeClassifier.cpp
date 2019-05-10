/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "abc/RegionTypeClassifier.h"
#include "FeatureGenerator.h"
#include "FeatureGen.h"
#include "opencv2/opencv.hpp"

// cl
#include "clImgProc/ImageBuf.h"
#include "clUtils/path_utils.h"

// logger
#include "abc.logger.h"

using namespace comed::abc;

#ifdef _DEBUG
#define new DEBUG_NEW 
#endif 


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor
CRegionTypeClassifier::CRegionTypeClassifier(void)
{
	_pMLP = new CvANN_MLP;
	ASSERT( _pMLP );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor
CRegionTypeClassifier::~CRegionTypeClassifier(void)
{
	delete _pMLP;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// load trained data for ABC algorithm
bool CRegionTypeClassifier::Initialize( LPCTSTR lpszPath )
{
	if ( CLU_IsPathExist( lpszPath ) )
	{
		LOG_DEBUG( _T("Classifier tries to load data from [%s]"), lpszPath );

		const int anLayerInfo[] = { 14, 28, 2 };
		const int nLayerInfoCount = sizeof( anLayerInfo ) / sizeof(int) ;

		cv::Mat layers( nLayerInfoCount, 1, CV_32SC1 );

		for ( int i=0; i<nLayerInfoCount; i++ )
		{
			layers.row( i ) = cv::Scalar( anLayerInfo[i] );
		}

		// create layers
		_pMLP->create( layers );

		// load the data
		_pMLP->load( CT2A( lpszPath ) );
		return true;
	}

	LOG_ERROR( _T("No trained data exists [%s]"), lpszPath );

	return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// method used in ClassifyRegion(...)
static inline float _calcTime( const LARGE_INTEGER& llFreq, const LARGE_INTEGER& ll2, const LARGE_INTEGER& ll1 )
{
	return (float)( ( (double)( ll2.QuadPart - ll1.QuadPart ) / (double) llFreq.QuadPart ) * 1000. );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// do classfy 
bool CRegionTypeClassifier::ClassfyRegion( 
							const cl::img::CImageBuf& img, RegionType arrResult[ABC_REGION_DIVIDE] ) const
{
	ASSERT( _pMLP );
	ASSERT( arrResult != nullptr );

	if ( ! img.IsValid() )
		return false;

	// to profile time
	LARGE_INTEGER llLap1, llLap2;
	LARGE_INTEGER llFreq;

	::QueryPerformanceCounter( &llLap1 );

	// local constants
	const int nNumBlocks = ABC_REGION_DIVIDE * ABC_REGION_DIVIDE;

	// allocate temp buffers
	double **ppDblFeatures = new double* [ nNumBlocks ];
	ASSERT( ppDblFeatures );

	for ( int i=0; i<nNumBlocks; i++ )
	{
		ppDblFeatures[i] = new double [ ABC_FEATURE_COUNT ];
		ASSERT( ppDblFeatures[i] );
	}

	// feature generation
	VERIFY( CFeatureGen::CalcFeatures( img, ppDblFeatures ) );

	// do predict
	{
		cv::Mat feature	( nNumBlocks, ABC_FEATURE_COUNT, cv::DataType<double>::type );
		cv::Mat result	( nNumBlocks, ABC_RESULT_COUNT,  cv::DataType<double>::type );

		for ( int by=0; by<ABC_REGION_DIVIDE; by++ )
		for ( int bx=0; bx<ABC_REGION_DIVIDE; bx++ )
		{
			const int bi = bx + by * ABC_REGION_DIVIDE;

			for ( int i=0; i<ABC_FEATURE_COUNT; i++ )
				feature.at<double>( bi,  i ) = ppDblFeatures[ bi ][ i ];

			for ( int i=0; i<ABC_RESULT_COUNT; i++ )
				result.at<double>( bi,  i ) = 0.;
		}

		const float s = _pMLP->predict( feature, result );
		UNREFERENCED_PARAMETER( s );

		// make output
		for ( int i=0; i<nNumBlocks; i++ )
		{
			const bool bMetal = ( result.at<double>(i, 0) > 0.0 );
			const bool bBackg = ( result.at<double>(i, 1) > 0.0 );

			arrResult[ i ].bMetal = bMetal;
			arrResult[ i ].bBackground = bBackg;
		}
	}

	// delete temp buffers
	for ( int i=0; i<nNumBlocks; i++ )
		delete [] ppDblFeatures[ i ];
	delete [] ppDblFeatures;

	::QueryPerformanceCounter( &llLap2 );
	::QueryPerformanceFrequency( &llFreq );

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// get global features, optimized 
void CRegionTypeClassifier::_calcGlobalFeatures( 
			const cl::img::CImageBuf& img, double* pdbGMax, double* pdbGMin, double* pdbGMean, double* pdbGStd )
{
	ASSERT( img.IsValid() );
	ASSERT( pdbGMax );
	ASSERT( pdbGMin );
	ASSERT( pdbGMean );
	ASSERT( pdbGStd );

	const int nImgW = img.GetWidth(), nImgH = img.GetHeight();
	const int nArea = nImgW * nImgH;
	const int nArea_16 = ( nArea / 16 ) * 16;

	const WORD* pwSrc = img.GetPixelDataWord();
	ASSERT( pwSrc != nullptr );

	WORD wMin = 0xffff, wMax = 0x0000;
	unsigned int nSum = 0;
	float fSS = 0.f;

	int i = 0;
	{
		// initial values
		const __m128i mZeros = _mm_setzero_si128();
		__m128i mMin  = _mm_set1_epi16( wMin );
		__m128i mMax  = _mm_set1_epi16( wMax );
		__m128i mSum0 = _mm_setzero_si128();
		__m128i mSum1 = _mm_setzero_si128();
		__m128  mfSS0 = _mm_setzero_ps();		// sum of squares
		__m128  mfSS1 = _mm_setzero_ps();

		for ( ; i<nArea_16; i+=16 )
		{
			// load 16 pixels
			__m128i mValue0 = _mm_load_si128( (const __m128i*)( pwSrc + i ) + 0 );
			__m128i mValue1 = _mm_load_si128( (const __m128i*)( pwSrc + i ) + 1 );

			// min, max
			mMin = _mm_min_epi16( mValue0, mMin );
			mMax = _mm_min_epi16( mValue0, mMax );
			mMin = _mm_min_epi16( mValue1, mMin );
			mMax = _mm_min_epi16( mValue1, mMax );

			// convert to 32-bit integers
			__m128i mValue00 = _mm_unpacklo_epi16( mValue0, mZeros );
			__m128i mValue01 = _mm_unpackhi_epi16( mValue0, mZeros );
			__m128i mValue10 = _mm_unpacklo_epi16( mValue1, mZeros );
			__m128i mValue11 = _mm_unpackhi_epi16( mValue1, mZeros );

			mSum0 = _mm_add_epi32( mSum0, mValue00 );
			mSum1 = _mm_add_epi32( mSum1, mValue01 );
			mSum0 = _mm_add_epi32( mSum0, mValue10 );
			mSum1 = _mm_add_epi32( mSum1, mValue11 );

			// convert to floats
			__m128 mfValue00 = _mm_cvtepi32_ps( mValue00 );
			__m128 mfValue01 = _mm_cvtepi32_ps( mValue01 );
			__m128 mfValue10 = _mm_cvtepi32_ps( mValue10 );
			__m128 mfValue11 = _mm_cvtepi32_ps( mValue11 );

			// squares
			mfValue00 = _mm_mul_ps( mfValue00, mfValue00 );
			mfValue01 = _mm_mul_ps( mfValue01, mfValue01 );
			mfValue10 = _mm_mul_ps( mfValue10, mfValue10 );
			mfValue11 = _mm_mul_ps( mfValue11, mfValue11 );

			// sum of squares
			mfSS0 = _mm_add_ps( mfSS0, mfValue00 );
			mfSS1 = _mm_add_ps( mfSS1, mfValue01 );
			mfSS0 = _mm_add_ps( mfSS0, mfValue10 );
			mfSS1 = _mm_add_ps( mfSS1, mfValue11 );
		}

		// min, max
		for ( int k=0; k<8; k++ )
		{
			wMin = std::min( wMin, mMin.m128i_u16[k] );
			wMax = std::max( wMax, mMax.m128i_u16[k] );
		}

		// sum
		mSum0 = _mm_add_epi32( mSum0, mSum1 );
		for ( int k=0; k<4; k++ )
		{
			nSum += mSum0.m128i_u32[k];
		}

		// sum of squares
		mfSS0 = _mm_add_ps( mfSS0, mfSS1 );
		for ( int k=0; k<4; k++ )
		{
			fSS += mfSS0.m128_f32[k];
		}
	}

	// remaining pixels
	for ( ; i<nArea; i++ )
	{
		register const WORD wValue = pwSrc[ i ];

		wMin = std::min( wMin, wValue );
		wMax = std::max( wMax, wValue );
		nSum += wValue;
		fSS += static_cast<float>( wValue ) * wValue;
	}

	// store the values
	*pdbGMax  = static_cast<double>( wMax );
	*pdbGMin  = static_cast<double>( wMin );
	*pdbGMean = static_cast<double>( nSum ) / nArea;
	*pdbGStd  = sqrt( static_cast<double>( fSS ) / nArea - (*pdbGMean) * (*pdbGMean) );

	return;
}

