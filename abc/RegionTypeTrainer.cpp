/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "abc/RegionTypeTrainer.h"
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

// macro constants
#define TRAINING_PARAM_COUNT			16


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// internal data types
struct CRegionTypeTrainer::DataTrainingParams
{
	int nRow, nCol;
	double adFeatures[ ABC_FEATURE_COUNT ];
	double adResults[ ABC_RESULT_COUNT ];
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// default constructor
CRegionTypeTrainer::CRegionTypeTrainer(void)
{
	_pMLP = new CvANN_MLP;
	ASSERT( _pMLP );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor
CRegionTypeTrainer::~CRegionTypeTrainer(void)
{
	delete _pMLP;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// initialize
bool CRegionTypeTrainer::Initialize(void)
{
	ASSERT( _pMLP );

	const int anLayerInfo[] = { 14, 28, 2 };
	const int nLayerInfoCount = sizeof( anLayerInfo ) / sizeof(int) ;

	cv::Mat layers( nLayerInfoCount, 1, CV_32SC1 );

	for ( int i=0; i<nLayerInfoCount; i++ )
	{
		layers.row( i ) = cv::Scalar( anLayerInfo[i] );
	}

	// create 
	CvANN_MLP* pInstance = reinterpret_cast<CvANN_MLP*>( _pMLP );
	ASSERT( pInstance != nullptr );

	pInstance->create( layers );

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// add training data
bool CRegionTypeTrainer::AddTrainingData( 
	const cl::img::CImageBuf& img, const RegionType arrTypes[ ABC_REGION_DIVIDE_2 ] )
{
	ASSERT( arrTypes != nullptr );

	if ( ! img.IsValid() )
	{
		LOG_ERROR( _T("Invalid image to train.") );
		return false;
	}

	// local constants
	const int h_divide = ABC_REGION_DIVIDE, v_divide = ABC_REGION_DIVIDE;

	// feature generator
	CAlgABCFeatureGenerator featureGenerator( img, h_divide, v_divide );

	double dbGOtsu, dbGInner, dbGInter;
	featureGenerator.GetGOtsu( dbGOtsu, dbGInner, dbGInter );

	const double dbGMax  = featureGenerator.GetGMax();
	const double dbGMin  = featureGenerator.GetGMin();
	const double dbGMean = featureGenerator.GetGMean();
	const double dbGStd  = featureGenerator.GetGStd();

	int nIndex = 0;
	for ( int i=0; i<v_divide; i++ )
	for ( int j=0; j<h_divide; j++, nIndex++ )
	{
		double dbLOtsu, dbLInner, dbLInter;
		featureGenerator.GetLOtsu( j, i, dbLOtsu, dbLInner, dbLInter );

		const double dbLMax  = featureGenerator.GetLMax(  j, i );
		const double dbLMin  = featureGenerator.GetLMin(  j, i );
		const double dbLMean = featureGenerator.GetLMean( j, i );
		const double dbLStd  = featureGenerator.GetLStd(  j, i );

		DataTrainingParams * pTraningData = new DataTrainingParams;
		ASSERT( pTraningData );

		pTraningData->nRow = i;
		pTraningData->nCol = j;
		pTraningData->adFeatures[ 0 ] = dbGOtsu;
		pTraningData->adFeatures[ 1 ] = dbGInner;
		pTraningData->adFeatures[ 2 ] = dbGInter;
		pTraningData->adFeatures[ 3 ] = dbGMax;
		pTraningData->adFeatures[ 4 ] = dbGMin;
		pTraningData->adFeatures[ 5 ] = dbGMean;
		pTraningData->adFeatures[ 6 ] = dbGStd;
		pTraningData->adFeatures[ 7 ] = dbLOtsu;
		pTraningData->adFeatures[ 8 ] = dbLInner;
		pTraningData->adFeatures[ 9 ] = dbLInter;
		pTraningData->adFeatures[ 10 ] = dbLMax;
		pTraningData->adFeatures[ 11 ] = dbLMin;
		pTraningData->adFeatures[ 12 ] = dbLMean;
		pTraningData->adFeatures[ 13 ] = dbLStd;

		pTraningData->adResults[ 0 ] = arrTypes[ nIndex ].bMetal      ? 1. : -1.;
		pTraningData->adResults[ 1 ] = arrTypes[ nIndex ].bBackground ? 1. : -1.;

		_listData.AddTail( pTraningData );
	}

	return true;
}
