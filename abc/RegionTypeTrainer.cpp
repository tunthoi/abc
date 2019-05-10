/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "abc/RegionTypeTrainer.h"
#include "FeatureGen.h"

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
// internal data types
struct CRegionTypeTrainer::DataTrainingParams
{
	int nCol, nRow;
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
	ClearTrainingData();

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
	_pMLP->create( layers );

	// clear previous data
	ClearTrainingData();

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

	// temp buffers
	double** ppdbFeatures = new double* [ ABC_REGION_DIVIDE_2 ];
	ASSERT( ppdbFeatures );

	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		ppdbFeatures[ bi ] = new double [ ABC_FEATURE_COUNT ];
		ASSERT( ppdbFeatures[ bi ] );
	}

	// feature generation
	CFeatureGen::CalcFeatures( img, ppdbFeatures ); 

	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
	{
		const int bx = bi % ABC_REGION_DIVIDE;
		const int by = bi / ABC_REGION_DIVIDE;

		if ( bx == 0 || bx == ABC_REGION_DIVIDE - 1 || by == 0 || by == ABC_REGION_DIVIDE - 1 )
		{
			// ignore the boundary blocks
		}
		else 
		{
			DataTrainingParams * pTraningData = new DataTrainingParams;
			ASSERT( pTraningData );

			pTraningData->nCol = bx;
			pTraningData->nRow = by;

			for ( int j=0; j<ABC_FEATURE_COUNT; j++ )
				pTraningData->adFeatures[ j ] = ppdbFeatures[ bi ][ j ];

			pTraningData->adResults[ 0 ] = arrTypes[ bi ].bMetal      ? 1. : -1.;
			pTraningData->adResults[ 1 ] = arrTypes[ bi ].bBackground ? 1. : -1.;

			_listData.AddTail( pTraningData );
		}
	}

	// delete 
	for ( int bi=0; bi<ABC_REGION_DIVIDE_2; bi++ )
		delete [] ppdbFeatures[ bi ];
	delete [] ppdbFeatures;

	return true;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// train and save trained data to lpszPath
bool CRegionTypeTrainer::SaveTrainingResult( LPCTSTR lpszResultPath ) const
{
	ASSERT( _pMLP );
	ASSERT( lpszResultPath != nullptr );

	// training data count
	const int nTrainingDataCount = static_cast<int>( _listData.GetCount() );

	if ( nTrainingDataCount < ABC_MIN_TRAININGDATACOUNT )
	{
		LOG_ERROR( _T("Too small number of training data. - %d"), nTrainingDataCount );
		return false;
	}

	// open cv
	cv::Mat feature( nTrainingDataCount, ABC_FEATURE_COUNT, cv::DataType<double>::type );
	cv::Mat result ( nTrainingDataCount, ABC_RESULT_COUNT,  cv::DataType<double>::type );

	int nIndex = 0;
	POSITION pos = _listData.GetHeadPosition();

	while( pos )
	{
		DataTrainingParams* pData = _listData.GetNext( pos );
		ASSERT( pData );

		// feature
		for ( int i=0 ; i<ABC_FEATURE_COUNT; i++ )
		{
			feature.at<double>( nIndex, i ) = pData->adFeatures[ i ];
		}

		// result
		for ( int i=0; i<ABC_RESULT_COUNT; i++ )
		{
			result.at<double>( nIndex, i ) = pData->adResults[ i ];
		}

		nIndex ++;
	}

	// criteria
	CvTermCriteria criteria;
	criteria.max_iter = 1000;
	criteria.epsilon = 0.0002f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::RPROP;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	// do train
	const int nCount = _pMLP->train( feature, result, cv::Mat(), cv::Mat(), params );

	if ( nCount > 0 )
	{
		// TODO: how to check ?
		_pMLP->save( CT2A( lpszResultPath ) );
		return true;
	}

	return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// clear all the training data
void CRegionTypeTrainer::ClearTrainingData(void)
{
	POSITION pos = _listData.GetHeadPosition();
	while ( pos )
	{
		delete _listData.GetNext( pos );
	}
	_listData.RemoveAll();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// get number of data instances
int CRegionTypeTrainer::GetTrainingDataCount(void) const
{
	return (int) _listData.GetCount();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extract parameters
int CRegionTypeTrainer::_extractData( const CString strLine, DataTrainingParams* pParam )
{
	ASSERT( pParam != nullptr );
	
	int nCount = 0;

	int nCurPos = 0;
	CString strToken = strLine.Tokenize( _T("\t "), nCurPos );

	while ( ! strToken.IsEmpty() )
	{
		if ( nCount == 0 )
		{
			pParam->nCol = ::_tstoi( strToken );
		}
		else if ( nCount == 1 )
		{
			pParam->nRow = ::_tstoi( strToken );
		}
		else if ( nCount < ABC_FEATURE_COUNT + 2 )
		{
			pParam->adFeatures[ nCount - 2 ] = ::_tstof( strToken );
		}
		else if ( nCount < ABC_RESULT_COUNT + ABC_FEATURE_COUNT + 2 )
		{
			pParam->adResults[ nCount - ( ABC_FEATURE_COUNT + 2 ) ] = ::_tstof( strToken );
		}
		else 
		{
			break;
		}

		nCount ++;
		strToken = strLine.Tokenize( _T("\t "), nCurPos );
	}

	return nCount;

	// ¾Æ·¡°¡ µ¿ÀÛÇÏÁö ¾ÊÀ½....
	/*
	return ::_stscanf_s( strLine, strFormat, 
								&( pData->nCol ),
								&( pData->nRow ),
								&( pData->adFeatures[  0 ] ),
								&( pData->adFeatures[  1 ] ),
								&( pData->adFeatures[  2 ] ),
								&( pData->adFeatures[  3 ] ),
								&( pData->adFeatures[  4 ] ),
								&( pData->adFeatures[  5 ] ),
								&( pData->adFeatures[  6 ] ),
								&( pData->adFeatures[  7 ] ),
								&( pData->adFeatures[  8 ] ),
								&( pData->adFeatures[  9 ] ),
								&( pData->adFeatures[ 10 ] ),
								&( pData->adFeatures[ 11 ] ),
								&( pData->adFeatures[ 12 ] ),
								&( pData->adFeatures[ 13 ] ),
								&( pData->adResults [  0 ] ),
								&( pData->adResults [  1 ] ) );
	*/
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// load data from file
bool CRegionTypeTrainer::AddTrainingDataFrom( LPCTSTR lpszFilePath )
{
	bool bOk = true;

	TRY 
	{
		// format 
		CString strFormat( _T("%d\t%d\t") );

		for ( int i=0; i<ABC_FEATURE_COUNT + ABC_RESULT_COUNT; i++ )
			strFormat += _T("%f\t");

		strFormat.TrimRight();

		// open file
		CStdioFile file( lpszFilePath, CFile::modeRead );
		CString strLine;
		int nLineCount = 0;

		while ( bOk && file.ReadString( strLine ) )
		{
			nLineCount ++;

			strLine.Trim();
			if ( strLine.IsEmpty() )
				continue;

			DataTrainingParams *pData = new DataTrainingParams;
			ASSERT( pData );

			const int nExtracted = _extractData( strLine, pData );

			if ( nExtracted != 2 + ABC_FEATURE_COUNT + ABC_RESULT_COUNT )
			{
				LOG_ERROR( _T("Can't read line %d from [%s]"), nLineCount, lpszFilePath );

				delete pData;
				bOk = false;
			}
			else if ( pData->nCol == 0 || pData->nCol == ABC_REGION_DIVIDE - 1 ||
					  pData->nRow == 0 || pData->nRow == ABC_REGION_DIVIDE - 1 )
			{
				// ignore boundary blocks
				delete pData;
			}
			else 
			{
				_listData.AddTail( pData );
			}
		}

		file.Close();
	}
	CATCH ( CException, e )
	{
		LOG_ERROR( _T("Can't read training data from [%s] - %s"), lpszFilePath, CLU_GetErrorMessageFromException( e ) );
		bOk = false;
	}
	END_CATCH;

	return bOk;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// save the data to the file
bool CRegionTypeTrainer::SaveTrainingData( LPCTSTR lpszFilePath )
{
	if ( GetTrainingDataCount() == 0 )
	{
		LOG_ERROR( _T("No data to save.") );
		return false;
	}

	bool bOk = true;

	TRY 
	{
		CStdioFile file( lpszFilePath, CFile::modeWrite | CFile::modeCreate );

		POSITION pos = _listData.GetHeadPosition();
		while ( pos )
		{
			DataTrainingParams* pParam = _listData.GetNext( pos );
			ASSERT( pParam != nullptr );

			CString strLine;
			strLine.Format( _T("%d\t%d\t"), pParam->nCol, pParam->nRow );

			for ( int i=0; i<ABC_FEATURE_COUNT; i++ )
				strLine += CLU_FormatText( _T("%g\t"), pParam->adFeatures[ i ] );

			for ( int i=0; i<ABC_RESULT_COUNT; i++ )
				strLine += CLU_FormatText( _T("%g\t"), pParam->adResults[ i ] );

			strLine.TrimRight();
			strLine += _T("\n");

			file.WriteString( strLine );
		}

		file.Close();

		LOG_DEBUG( _T("Successfully write %d data to [%s]"), GetTrainingDataCount(), lpszFilePath );
	}
	CATCH ( CException, e )
	{
		LOG_ERROR( _T("Can't write training data to [%s] - %s"), lpszFilePath, CLU_GetErrorMessageFromException( e ) );
		bOk = false;
	}
	END_CATCH;

	return bOk;
}