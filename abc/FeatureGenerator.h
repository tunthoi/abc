/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "opencv2/opencv.hpp"
#include "clUtils/defines.h"

// forward declaration
namespace cl { namespace img { class CImageBuf; }}

namespace comed { namespace abc
{
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// abc feature generator
	class CFeatureGenerator
	{
		CL_NO_DEFAULT_CONSTRUCTOR( CFeatureGenerator )

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// contructor & destructor
	public:
		// constructor
		CFeatureGenerator( const cl::img::CImageBuf& img, int divideW, int divideH );

		// copy constructor
		CFeatureGenerator( const CFeatureGenerator& other );

		// destructor
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

		// AuA¨ù AI©öIAoAC ¡Æ¢®¡¤I : Width ¢¯I ¨ù¨ù¡¤I : Height
		int _nGlobalWidth, _nGlobalHeight;

		// ¡¤IAA AI©öIAoAC _nRows : ¡Æ¢®¡¤IAa ¡Æ©ö¨ùo, _nCols : ¨ù¨ù¡¤IAa ¡Æ©ö¨ùo
		int _nRows, _nCols;

		// AuA¨ù AI©öIAoAC ¡Æ¢®¡¤I : Width ¢¯I ¨ù¨ù¡¤I : Height
		int _nLocalWidth, _nLocalHeight;
	};
}}