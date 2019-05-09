/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "AlgABCFeatureGenerator.h"

// cl
#include "clImgProc/ImageBuf.h"

using namespace comed::alg;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif 

#define HISTSIZE		256
#define HISTMAX			255


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor
CAlgABCFeatureGenerator::CAlgABCFeatureGenerator( const cl::img::CImageBuf& img, int divideW, int divideH )
{
	ASSERT( img.IsValid() );
	ASSERT( img.GetType() == cl::img::EIT_Gray16bit );

	const int nWidth = img.GetWidth();
	const int nHeight = img.GetHeight();

	const WORD* pData = img.GetPixelDataWord();
	ASSERT( pData != nullptr );

	_imgBuf.create( nWidth, nHeight, cv::DataType<WORD>::type );

	for( int i = 0; i < nHeight; i++ )
	for( int j = 0; j < nWidth; j++ )
	{
		_imgBuf.at<WORD>(i, j) = pData[ i * nWidth + j ];
	}

	_nGlobalWidth = nWidth;
	_nGlobalHeight = nHeight;

	_nCols = divideW;
	_nRows = divideH;

	// _nRows 와 _nCols 로 나누어 정수만 취함. 나머지는 버림.
	_nLocalHeight = _nGlobalHeight / _nRows;
	_nLocalWidth = _nGlobalWidth / _nCols;

	// [TODO] how to check error ?
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor
CAlgABCFeatureGenerator::~CAlgABCFeatureGenerator(void)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetGOtsu
void CAlgABCFeatureGenerator::GetGOtsu( double& otsu, double& inner, double& inter ) const
{
	this->CalOtsu( _imgBuf, otsu, inner, inter );	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetGOtsu
double CAlgABCFeatureGenerator::GetGMax(void) const
{
	double dmax, dmin;
	cv::minMaxIdx( this->_imgBuf, &dmin, &dmax );

	return dmax;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetGOtsu
double CAlgABCFeatureGenerator::GetGMin(void) const
{
	double dmax, dmin;
	cv::minMaxIdx( this->_imgBuf, &dmin, &dmax );

	return dmin;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetGOtsu
double CAlgABCFeatureGenerator::GetGMean(void) const
{
	cv::Mat avg(  1, 1, CV_32F );
	cv::Mat sstd( 1, 1, CV_32F );

	cv::meanStdDev( _imgBuf, avg, sstd );

	return avg.at<double>( 0, 0 );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetGStd
double CAlgABCFeatureGenerator::GetGStd(void) const
{
	cv::Mat avg(  1, 1, CV_32F );
	cv::Mat sstd( 1, 1, CV_32F );

	cv::meanStdDev( _imgBuf, avg, sstd );

	return sstd.at<double>( 0, 0 );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetLOtsu
void CAlgABCFeatureGenerator::GetLOtsu( int col, int row, double& otsu, double& inner, double& inter ) const
{
	cv::Rect rect( col * _nLocalWidth, row * _nLocalHeight, _nLocalWidth, _nLocalHeight );
	cv::Mat tile( _imgBuf, rect );

	this->CalOtsu( tile, otsu, inner, inter );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetLMax
double CAlgABCFeatureGenerator::GetLMax( int col, int row ) const
{
	cv::Rect rect( col * _nLocalWidth, row * _nLocalHeight, _nLocalWidth, _nLocalHeight );
	cv::Mat tile( _imgBuf, rect );
	
	double dmin, dmax;
	cv::minMaxIdx( tile, &dmin, &dmax );

	return dmax;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetLMin
double CAlgABCFeatureGenerator::GetLMin( int col, int row ) const
{
	cv::Rect rect( col*_nLocalWidth, row*_nLocalHeight, _nLocalWidth, _nLocalHeight );
	cv::Mat tile( _imgBuf, rect );

	double dmin, dmax;
	cv::minMaxIdx( tile, &dmin, &dmax );

	return dmin;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetGOtsu
double CAlgABCFeatureGenerator::GetLMean( int col, int row ) const
{
	cv::Rect rect( col*_nLocalWidth, row*_nLocalHeight, _nLocalWidth, _nLocalHeight );
	cv::Mat tile( _imgBuf, rect );

	cv::Mat avg(1,1, cv::DataType<double>::type );
	cv::Mat sstd( 1,1, cv::DataType<double>::type );

	cv::meanStdDev( tile, avg, sstd );

	return avg.at<double>(0,0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GetLStd
double CAlgABCFeatureGenerator::GetLStd( int col, int row ) const
{
	cv::Rect rect( col*_nLocalWidth, row*_nLocalHeight, _nLocalWidth, _nLocalHeight );
	cv::Mat tile( _imgBuf, rect );
		
	cv::Mat avg(1,1, CV_32F );
	cv::Mat sstd( 1,1, CV_32F );

	cv::meanStdDev( tile, avg, sstd );

	return sstd.at<double>(0,0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CalOtsu
void CAlgABCFeatureGenerator::CalOtsu( const cv::Mat& buf, double& otsu, double& inner, double& inter ) const
{
	int iHist[HISTSIZE];
	int iCumHist[HISTSIZE];;
	int iCumMulHist[HISTSIZE];

	double dWeightB[HISTSIZE], dWeightF[HISTSIZE];

	int iHistSum = 0;
	int iHistMulSum = 0;

	double dMeanB[HISTSIZE], dMeanF[HISTSIZE];
	double dVarB[HISTSIZE], dVarF[HISTSIZE];

	double dInner[HISTSIZE];
	double dInter[HISTSIZE];

	double min, max;	

	cv::minMaxIdx( buf, &min, &max );

	memset( iHist, 0, sizeof(int)*HISTSIZE );

	for( int r = 0; r < buf.rows; r++ )
	for( int c = 0; c < buf.cols; c++ )
	{
		iHist[ (int)( (buf.at<unsigned short>( r, c ) - min ) / ( ( max - min ) / HISTMAX ) ) ] ++;
	}

	for( int i = 0; i < HISTSIZE; i++ )
	{
		if ( i == 0 ) 
			iCumHist[i] = iHist[i];
		else  		 
			iCumHist[i] = iCumHist[i-1] + iHist[i];
		
		iHistSum    = iHistSum + iHist[i];
		iHistMulSum = iHistMulSum + ( iHist[i] * i );
	}

	for( int i = 0; i < HISTSIZE; i ++ )
	{
		dWeightB[i] = iCumHist[i] / (double)iHistSum;
		dWeightF[i] = 1.0 - dWeightB[i];
	} 

	for( int i = 0; i < HISTSIZE; i++ )
	{
		double dCumVarB = 0.0;
		double dCumVarF = 0.0;

		if ( i == 0 ) 
			iCumMulHist[i] = 0;
		else		 
			iCumMulHist[i] = iCumMulHist[i-1] + ( iHist[i] * i );

		dMeanB[i] = (double)iCumMulHist[i] / iCumHist[i];
		dMeanF[i] = (double)( iHistMulSum - iCumMulHist[i] ) / ( iHistSum - iCumHist[i] );

		for( int j = 0; j < i; j++ )
		{
			dCumVarB = dCumVarB + ( ( ( j - dMeanB[i] ) * ( j - dMeanB[i] ) ) * iHist[j] );
		}
		for( int k = i; k < 256; k++ )
		{
			dCumVarF = dCumVarF + ( ( ( k - dMeanF[i] ) * ( k - dMeanF[i] ) ) * iHist[k] );		
		}		

		dVarB[i] = dCumVarB / iCumHist[i];
		dVarF[i] = dCumVarF / ( iHistSum - iCumHist[i] );

		dInner[i] = dWeightB[i] * dVarB[i] + dWeightF[i] * dVarF[i];
		dInter[i] = dWeightB[i] * dWeightF[i] * ( ( dMeanB[i] - dMeanF[i] ) * ( dMeanB[i] - dMeanF[i] ) );
	}	

	// find maximum for otsu
	double tempMax = 0.0;
	int maxIndex = 0;

	for( int i = 0; i < HISTSIZE; i++ )
	{
		const double rst = dInter[i] / dInner[i];
		if( rst > tempMax )
		{
			tempMax = rst;
			maxIndex = i;
		}
	}

	otsu = (double)maxIndex / HISTMAX;
	inner = dInner[maxIndex];
	inter = dInter[maxIndex];
}