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
		for ( int bi=0; bi<nNumBlocks; bi++ )
		{
			const int bx = bi % ABC_REGION_DIVIDE;
			const int by = bi / ABC_REGION_DIVIDE;

			if ( bx == 0 || bx == ABC_REGION_DIVIDE - 1 || by == 0 || by == ABC_REGION_DIVIDE - 1 )
			{
				arrResult[ bi ].bMetal = false;
				arrResult[ bi ].bBackground = true;
			}
			else 
			{
				const bool bMetal = ( result.at<double>(bi, 0) > 0.0 );
				const bool bBackg = ( result.at<double>(bi, 1) > 0.0 );

				arrResult[ bi ].bMetal = bMetal;
				arrResult[ bi ].bBackground = bBackg;
			}
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

