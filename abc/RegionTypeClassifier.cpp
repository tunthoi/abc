/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "abc/RegionTypeClassifier.h"
#include "FeatureGenerator.h"

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
	LARGE_INTEGER llLap1, llLap2, llLap3, llLap4, llLap5, llLap6;
	LARGE_INTEGER llFreq;

	::QueryPerformanceCounter( &llLap1 );

	// local constants
	const int h_divide = ABC_REGION_DIVIDE;
	const int v_divide = ABC_REGION_DIVIDE;

	// feature generator 
	CFeatureGenerator featureGenerator( img, h_divide, v_divide );

	::QueryPerformanceCounter( &llLap2 );

	cv::Mat feature	( h_divide * v_divide, 14, cv::DataType<double>::type );
	cv::Mat result	( h_divide * v_divide,  2, cv::DataType<double>::type );

	double dbGOtsu, dbGInner, dbGInter;
	featureGenerator.GetGOtsu( dbGOtsu, dbGInner, dbGInter );

	const double dbGMax = featureGenerator.GetGMax();
	const double dbGMin = featureGenerator.GetGMin();
	const double dbGMean = featureGenerator.GetGMean();
	const double dbGStd = featureGenerator.GetGStd();

	::QueryPerformanceCounter( &llLap3 );

	for ( int i=0; i<v_divide; i++ )
	for ( int j=0; j<h_divide; j++ )
	{
		const int nIndex = i * h_divide + j;

		// 30 ms
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

	::QueryPerformanceCounter( &llLap4 );

	// do predict
	const float s = _pMLP->predict( feature, result );
	UNREFERENCED_PARAMETER( s );

	::QueryPerformanceCounter( &llLap5 );

	// output
	const int nNums = h_divide * v_divide;

	for ( int i=0; i<nNums; i++ )
	{
		const bool bMetal = ( result.at<double>(i, 0) > 0.0 );
		const bool bBackg = ( result.at<double>(i, 1) > 0.0 );

		arrResult[ i ].bMetal = bMetal;
		arrResult[ i ].bBackground = bBackg;
	}

	::QueryPerformanceCounter( &llLap6 );
	::QueryPerformanceFrequency( &llFreq );

	/*
	CString strTime;
	strTime.Format( 
		_T("Feature: %.3f ms, ")
		_T("Global : %.3f ms, ")
		_T("Local  : %.3f ms, ")
		_T("Predict: %.3f ms, ")
		_T("Output : %.3f ms"),
		_calcTime( llFreq, llLap2, llLap1 ),
		_calcTime( llFreq, llLap3, llLap2 ),
		_calcTime( llFreq, llLap4, llLap3 ),
		_calcTime( llFreq, llLap5, llLap4 ),
		_calcTime( llFreq, llLap6, llLap5 ) );
	LOG_ERROR( strTime );
	*/

	return true;
}

