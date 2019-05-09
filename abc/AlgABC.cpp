/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "abc/AlgABC.h"
#include "AlgABCFeatureGenerator.h"

#include "opencv2/opencv.hpp"

// cl
#include "clUtils/path_utils.h"

using namespace comed::alg;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif 

// macro
#define TRAINING_PARAM_COUNT			16

// static member
int CAlgABC::s_nDevicedCount = 8;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TrainingData
struct CAlgABC::TrainingData
{
	int nRow;
	int nCol;
	double pData[TRAINING_PARAM_COUNT];
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor
CAlgABC::CAlgABC(void) 
	: _pMLP( nullptr )
{
	_pMLP = new CvANN_MLP;
	ASSERT( _pMLP );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor
CAlgABC::~CAlgABC(void)
{
	clearTrainingData();

	CvANN_MLP* pInstance = reinterpret_cast<CvANN_MLP*>(  _pMLP );

	if ( pInstance )
		delete pInstance;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// intialize 
bool CAlgABC::Initialize(void)
{
	CvANN_MLP* pInstance = reinterpret_cast<CvANN_MLP*>(  _pMLP );

	if ( pInstance == nullptr )
		return false;

	const int anLayerInfo[] = { 14, 28, 2 };
	const int nLayerInfoCount = sizeof( anLayerInfo ) / sizeof(int) ;

	cv::Mat layers( nLayerInfoCount, 1, CV_32SC1 );

	for ( int i=0; i<nLayerInfoCount; i++ )
	{
		layers.row( i ) = cv::Scalar( anLayerInfo[i] );
	}

	pInstance->create( layers );

	// [TODO] error check
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// clean up
void CAlgABC::CleanUp(void)
{
	CvANN_MLP* pInstance = reinterpret_cast<CvANN_MLP*>( _pMLP );

	if ( pInstance == nullptr )
		return;

	pInstance->clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// add training data
bool CAlgABC::AddTrainingData( const cl::img::CImageBuf& img, const BYTE* pMetalDetected, const BYTE* pBGDetected )
{
	if ( pMetalDetected == nullptr || pBGDetected == nullptr )
		return false;

	int nDivide = s_nDevicedCount;
	CAlgABCFeatureGenerator featureGenerator( img, nDivide, nDivide );

	double dbGOtsu, dbGInner, dbGInter;
	featureGenerator.GetGOtsu( dbGOtsu, dbGInner, dbGInter );

	double dbGMax = featureGenerator.GetGMax();
	double dbGMin = featureGenerator.GetGMin();
	double dbGMean = featureGenerator.GetGMean();
	double dbGStd = featureGenerator.GetGStd();

	for( int i = 0; i < nDivide; i++ )
	for( int j = 0; j < nDivide; j++ )
	{
		double dbLOtsu, dbLInner, dbLInter;
		featureGenerator.GetLOtsu( j, i, dbLOtsu, dbLInner, dbLInter );

		double dbLMax = featureGenerator.GetLMax( j, i );
		double dbLMin = featureGenerator.GetLMin( j, i );
		double dbLMean = featureGenerator.GetLMean( j, i );
		double dbLStd = featureGenerator.GetLStd( j, i );

		TrainingData* pTraningData = new TrainingData;
		pTraningData->nRow = i;
		pTraningData->nCol = j;
		pTraningData->pData[ 0 ] = dbGOtsu;
		pTraningData->pData[ 1 ] = dbGInner;
		pTraningData->pData[ 2 ] = dbGInter;
		pTraningData->pData[ 3 ] = dbGMax;
		pTraningData->pData[ 4 ] = dbGMin;
		pTraningData->pData[ 5 ] = dbGMean;
		pTraningData->pData[ 6 ] = dbGStd;
		pTraningData->pData[ 7 ] = dbLOtsu;
		pTraningData->pData[ 8 ] = dbLInner;
		pTraningData->pData[ 9 ] = dbLInter;
		pTraningData->pData[ 10 ] = dbLMax;
		pTraningData->pData[ 11 ] = dbLMin;
		pTraningData->pData[ 12 ] = dbLMean;
		pTraningData->pData[ 13 ] = dbLStd;

		pTraningData->pData[ 14 ] = pMetalDetected[ j + i * nDivide ] == 0x01 ? 1 : -1;
		pTraningData->pData[ 15 ] = pBGDetected[ j + i * nDivide ] == 0x01 ? 1 : -1;

		_listTrainingData.AddTail( pTraningData );
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// train and save trained data to lpszPath
bool CAlgABC::Train( LPCTSTR lpszPath )
{
	CvANN_MLP* pInstance = reinterpret_cast<CvANN_MLP*>(  _pMLP );

	if ( pInstance == nullptr )
		return false;

	int nDivide = s_nDevicedCount;
	const int nTrainingDataCount = static_cast<int>( _listTrainingData.GetCount() );

	const int nFeatureCount = 14;
	const int nResultCount = 2;

	cv::Mat feature	( nTrainingDataCount, nFeatureCount, cv::DataType<double>::type );
	cv::Mat result	( nTrainingDataCount, nResultCount,  cv::DataType<double>::type );
	
	POSITION pos = _listTrainingData.GetHeadPosition();
	while( pos )
	{
		TrainingData* pData = _listTrainingData.GetNext( pos );
		if ( pData )
		{
			const int nIndex = pData->nCol + pData->nRow * nDivide;
			
			// feature
			for ( int i = 0 ; i < nFeatureCount ; i++ )
			{
				feature.at<double>( nIndex, i ) = pData->pData[ i ];
			}

			// result
			for ( int i = 0 ; i < nResultCount ; i++ )
			{
				result.at<double>( nIndex, i ) = pData->pData[ i + nFeatureCount ];
			}
		}
	}

	CvTermCriteria criteria;
	criteria.max_iter = 1000;
	criteria.epsilon = 0.0002f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::RPROP;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	int nCount = pInstance->train( feature, result, cv::Mat(), cv::Mat(), params );

	if ( nCount > 0 )
	{
		if ( lpszPath != nullptr )
		{
			USES_CONVERSION;
			pInstance->save( T2A( lpszPath ) );

			//saveTrainingData( lpszPath );
		}

		// remove training data
		clearTrainingData();

		return true;
	}
	
	return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// clear training data
void CAlgABC::clearTrainingData(void)
{
	POSITION pos = _listTrainingData.GetHeadPosition();
	while( pos )
	{
		TrainingData* pData = _listTrainingData.GetNext( pos );
		delete pData;
	}

	_listTrainingData.RemoveAll();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// load trained data for ABC algorithm
bool CAlgABC::LoadTrainedData( LPCTSTR lpszPath )
{
	CvANN_MLP* pInstance = reinterpret_cast<CvANN_MLP*>( _pMLP );

	if ( pInstance == nullptr )
		return false;

	if ( CLU_IsPathExist( lpszPath ) )
	{
		USES_CONVERSION;
		pInstance->load( T2A( lpszPath ) );
	}

	// [TODO] error check
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// execute
bool CAlgABC::Execute( const cl::img::CImageBuf& img, CArray<ABCDetectedResult>& arrResult ) const
{
	CvANN_MLP* pInstance = reinterpret_cast<CvANN_MLP*>(  _pMLP );
	
	if ( pInstance == nullptr )
		return false;

	int nDivide = s_nDevicedCount;

	arrResult.RemoveAll();

	CAlgABCFeatureGenerator featureGenerator( img, nDivide, nDivide );
	
	cv::Mat feature	( nDivide * nDivide, 14, cv::DataType<double>::type );
	cv::Mat result	( nDivide * nDivide, 2,  cv::DataType<double>::type );

	double dbGOtsu, dbGInner, dbGInter;
	featureGenerator.GetGOtsu( dbGOtsu, dbGInner, dbGInter );

	double dbGMax = featureGenerator.GetGMax();
	double dbGMin = featureGenerator.GetGMin();
	double dbGMean = featureGenerator.GetGMean();
	double dbGStd = featureGenerator.GetGStd();

	for( int i = 0; i < nDivide; i++ )
	for( int j = 0; j < nDivide; j++ )
	{
		int nIndex = i * nDivide + j;

		double dbLOtsu, dbLInner, dbLInter;
		featureGenerator.GetLOtsu( j, i, dbLOtsu, dbLInner, dbLInter );

		double dbLMax = featureGenerator.GetLMax( j, i );
		double dbLMin = featureGenerator.GetLMin( j, i );
		double dbLMean = featureGenerator.GetLMean( j, i );
		double dbLStd = featureGenerator.GetLStd( j, i );

		feature.at<double>(nIndex,0) = dbGOtsu;
		feature.at<double>(nIndex,1) = dbGInner;
		feature.at<double>(nIndex,2) = dbGInter;
		feature.at<double>(nIndex,3) = dbGMax;
		feature.at<double>(nIndex,4) = dbGMin;
		feature.at<double>(nIndex,5) = dbGMean;
		feature.at<double>(nIndex,6) = dbGStd;

		feature.at<double>(nIndex,7) = dbLOtsu;
		feature.at<double>(nIndex,8) = dbLInner;
		feature.at<double>(nIndex,9) = dbLInter;
		feature.at<double>(nIndex,10) = dbLMax;
		feature.at<double>(nIndex,11) = dbLMin;
		feature.at<double>(nIndex,12) = dbLMean;
		feature.at<double>(nIndex,13) = dbLStd;

		result.at<double>( nIndex, 0 )  = 0.f;
		result.at<double>( nIndex, 1 )  = 0.f;
	}

	float s = pInstance->predict( feature, result );
	UNREFERENCED_PARAMETER( s );

	int nMetalDetected = 0;
	int nBackGroundDetected = 0;

	for ( int i = 0 ; i < nDivide*nDivide ; i++ )
	{
		if ( result.at<double>(i, 0) > 0.0 )
			nMetalDetected++;

		if ( result.at<double>(i, 1) > 0.0 )
			nBackGroundDetected++;

		ABCDetectedResult abcResult;

		abcResult.m_bIsMetalDetected = result.at<double>(i, 0) > 0.0;
		abcResult.m_bIsBackgroundDetected = result.at<double>(i, 1) > 0.0;
		arrResult.Add( abcResult );
	}

	TRACE( _T("Metal Detected = %d, Background Detected = %d\n"), nMetalDetected, nBackGroundDetected );

	// [TODO] error check

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// get trainind data count
int CAlgABC::GetTrainingDataCount(void) const
{
	return static_cast<int>( _listTrainingData.GetCount() );
}
