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
	_pMLP_Objec = new CvANN_MLP;
	ASSERT( _pMLP_Objec );

	_pMLP_Metal = new CvANN_MLP;
	ASSERT( _pMLP_Metal );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor
CRegionTypeClassifier::~CRegionTypeClassifier(void)
{
	delete _pMLP_Objec;
	delete _pMLP_Metal;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// load trained data for ABC algorithm
bool CRegionTypeClassifier::Initialize( LPCTSTR lpszPath_Objec, LPCTSTR lpszPath_Metal )
{
	if ( CLU_IsPathExist( lpszPath_Objec ) && CLU_IsPathExist( lpszPath_Metal ) )
	{
		LOG_DEBUG( _T("Classifier tries to load data from [%s] and [%s]"), lpszPath_Objec, lpszPath_Metal );

		const int anLayerInfo[] = { ABC_FEATURE_COUNT, ABC_FEATURE_COUNT * 2, 1 };
		const int nLayerInfoCount = sizeof( anLayerInfo ) / sizeof(int) ;

		cv::Mat layers( nLayerInfoCount, 1, CV_32SC1 );

		for ( int i=0; i<nLayerInfoCount; i++ )
		{
			layers.row( i ) = cv::Scalar( anLayerInfo[i] );
		}

		// create layers
		_pMLP_Objec->create( layers );
		_pMLP_Metal->create( layers );

		// load the data
		_pMLP_Objec->load( CT2A( lpszPath_Objec ) );
		_pMLP_Metal->load( CT2A( lpszPath_Metal ) );

		return true;
	}

	LOG_ERROR( _T("No trained data exists - [%s] and [%s]"), lpszPath_Objec, lpszPath_Metal );

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
				IN		const cl::img::CImageBuf& img, 
				OUT		RegionType arrResult[ ABC_REGION_DIVIDE ],
				OUT		int* pnNumObjBlocks,
				OUT		int* pnMeanObjBlocks
			) const
{
	ASSERT( _pMLP_Objec );
	ASSERT( _pMLP_Metal );

	ASSERT( arrResult != nullptr );
	ASSERT( pnNumObjBlocks );
	ASSERT( pnMeanObjBlocks );

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
		cv::Mat resultObjec	( nNumBlocks, 1,  cv::DataType<double>::type );
		cv::Mat resultMetal ( nNumBlocks, 1,  cv::DataType<double>::type );

		for ( int by=0; by<ABC_REGION_DIVIDE; by++ )
		for ( int bx=0; bx<ABC_REGION_DIVIDE; bx++ )
		{
			const int bi = bx + by * ABC_REGION_DIVIDE;

			for ( int i=0; i<ABC_FEATURE_COUNT; i++ )
				feature.at<double>( bi,  i ) = ppDblFeatures[ bi ][ i ];

			resultObjec.at<double>( bi ) = 0.;
			resultMetal.at<double>( bi ) = 0.;
		}

		const float s1 = _pMLP_Objec->predict( feature, resultObjec );
		const float s2 = _pMLP_Metal->predict( feature, resultMetal );
		UNREFERENCED_PARAMETER( s1 );
		UNREFERENCED_PARAMETER( s2 );

		// make output
		int nNumObjBlocks = 0;
		double dSumObjBlocks = 0.;

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
				const bool bMetal = ( resultMetal.at<double>(bi, 0) > 0.0 );
				const bool bBackg = ( resultObjec.at<double>(bi, 1) > 0.0 );

				arrResult[ bi ].bMetal = bMetal;
				arrResult[ bi ].bBackground = bBackg;

				if ( ! bBackg && ! bMetal )
				{
					nNumObjBlocks ++;
					dSumObjBlocks += ppDblFeatures[ bi ][ kABCFeatureId_Global_Mean ];
				}
			}
		}

		// output value
		*pnNumObjBlocks = nNumObjBlocks;
		*pnMeanObjBlocks = ( nNumObjBlocks != 0 ) ? (int)( dSumObjBlocks / nNumObjBlocks ) : 0;
	}

	// delete temp buffers
	for ( int i=0; i<nNumBlocks; i++ )
		delete [] ppDblFeatures[ i ];
	delete [] ppDblFeatures;

	::QueryPerformanceCounter( &llLap2 );
	::QueryPerformanceFrequency( &llFreq );

	return true;
}

