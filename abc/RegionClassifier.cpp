/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "abc/RegionClassifier.h"
#include "AlgABCFeatureGenerator.h"

// openCV2
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
CRegionClassifier::CRegionClassifier(void)
{
	_pMLP = new CvANN_MLP;
	ASSERT( _pMLP );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor
CRegionClassifier::~CRegionClassifier(void)
{
	delete _pMLP;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// load trained data for ABC algorithm
bool CRegionClassifier::Initialize( LPCTSTR lpszPath )
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
// do classfy 
bool CRegionClassifier::ClassfyRegion( 
							const cl::img::CImageBuf& img, RegionType arrResult[ABC_REGION_DIVIDE] ) const
{
	ASSERT( _pMLP );
	ASSERT( arrResult != nullptr );

	if ( ! img.IsValid() )
		return false;

	// local constants
	const int h_divide = ABC_REGION_DIVIDE;
	const int v_divide = ABC_REGION_DIVIDE;

	// feature generator 
	CAlgABCFeatureGenerator featureGenerator( img, h_divide, v_divide );

	cv::Mat feature	( h_divide * v_divide, 14, cv::DataType<double>::type );
	cv::Mat result	( h_divide * v_divide,  2, cv::DataType<double>::type );

	double dbGOtsu, dbGInner, dbGInter;
	featureGenerator.GetGOtsu( dbGOtsu, dbGInner, dbGInter );

	const double dbGMax = featureGenerator.GetGMax();
	const double dbGMin = featureGenerator.GetGMin();
	const double dbGMean = featureGenerator.GetGMean();
	const double dbGStd = featureGenerator.GetGStd();

	for ( int i=0; i<v_divide; i++ )
	for ( int j=0; j<h_divide; j++ )
	{
		const int nIndex = i * h_divide + j;

		double dbLOtsu, dbLInner, dbLInter;
		featureGenerator.GetLOtsu( j, i, dbLOtsu, dbLInner, dbLInter );

		const double dbLMax  = featureGenerator.GetLMax(  j, i );
		const double dbLMin  = featureGenerator.GetLMin(  j, i );
		const double dbLMean = featureGenerator.GetLMean( j, i );
		const double dbLStd  = featureGenerator.GetLStd(  j, i );

		feature.at<double>( nIndex,  0 ) = dbGOtsu;
		feature.at<double>( nIndex,  1 ) = dbGInner;
		feature.at<double>( nIndex,  2 ) = dbGInter;
		feature.at<double>( nIndex,  3 ) = dbGMax;
		feature.at<double>( nIndex,  4 ) = dbGMin;
		feature.at<double>( nIndex,  5 ) = dbGMean;
		feature.at<double>( nIndex,  6 ) = dbGStd;
		feature.at<double>( nIndex,  7 ) = dbLOtsu;
		feature.at<double>( nIndex,  8 ) = dbLInner;
		feature.at<double>( nIndex,  9 ) = dbLInter;
		feature.at<double>( nIndex, 10 ) = dbLMax;
		feature.at<double>( nIndex, 11 ) = dbLMin;
		feature.at<double>( nIndex, 12 ) = dbLMean;
		feature.at<double>( nIndex, 13 ) = dbLStd;

		result.at<double>( nIndex, 0 )  = 0.f;
		result.at<double>( nIndex, 1 )  = 0.f;
	}

	// do predict
	const float s = _pMLP->predict( feature, result );
	UNREFERENCED_PARAMETER( s );

	// output
	const int nNums = h_divide * v_divide;

	for ( int i=0; i<nNums; i++ )
	{
		const bool bMetal = ( result.at<double>(i, 0) > 0.0 );
		const bool bBackg = ( result.at<double>(i, 1) > 0.0 );

		arrResult[ i ].bMetal = bMetal;
		arrResult[ i ].bBackground = bBackg;
	}

	return true;
}

