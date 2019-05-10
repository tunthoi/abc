/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "opencv2/opencv.hpp"

// forward declaration
namespace cl { namespace img { class CImageBuf; }}

namespace comed { namespace abc
{
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// abc feature generator
	class CFeatureGenerator
	{
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// contructor & destructor
	public:
		CFeatureGenerator( const cl::img::CImageBuf& img, int divideW, int divideH );
		virtual ~CFeatureGenerator(void);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// public methods
	public:
		void   GetGOtsu( double& otsu, double& inner, double& inter ) const;	
		double GetGMax(void) const;
		double GetGMin(void) const;
		double GetGMean(void) const;
		double GetGStd(void) const;

		void   GetLOtsu( int col, int row, double& otsu, double& inner, double& inter ) const;	
		double GetLMax( int col, int row ) const;
		double GetLMin( int col, int row ) const;
		double GetLMean( int col, int row ) const;
		double GetLStd( int col, int row ) const;	

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// protected methods
	protected:
		void CalOtsu( const cv::Mat& buf, double& otsu, double& inner, double& inter ) const;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// private data
	private:
		cv::Mat _imgBuf;

		// 전체 이미지의 가로 : Width 와 세로 : Height
		int _nGlobalWidth, _nGlobalHeight;

		// 로컬 이미지의 _nRows : 가로축 갯수, _nCols : 세로축 갯수
		int _nRows, _nCols;

		// 전체 이미지의 가로 : Width 와 세로 : Height
		int _nLocalWidth, _nLocalHeight;
	};
}}