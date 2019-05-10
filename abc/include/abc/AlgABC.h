/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "clImgProc/ImageBuf.h"

namespace comed { namespace abc 
{
	struct ABCDetectedResult
	{
		bool m_bIsMetalDetected;
		bool m_bIsBackgroundDetected;
	};

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// ABC Algorithm
	class AFX_EXT_CLASS CAlgABC
	{
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// contructor & destructor
	public:
		/// <summary>
		/// constructor
		/// </summary>
		CAlgABC(void);

		/// <summary>
		/// destructor
		/// </summary>
		virtual ~CAlgABC(void);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// public methods
	public:

		// intialize 
		bool Initialize(void);

		// clean up
		void CleanUp(void);

		// add training data
		bool AddTrainingData( const cl::img::CImageBuf& img, const BYTE* pMetalDetected, const BYTE* pBGDetected );

		// get trainind data count
		int GetTrainingDataCount(void) const;

		// train and save trained data to lpszPath
		bool Train( LPCTSTR lpszPath );

		// load trained data for ABC algorithm
		bool LoadTrainedData( LPCTSTR lpszPath );

		// execute
		bool Execute( const cl::img::CImageBuf& img, CArray<ABCDetectedResult>& arrResult ) const;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// private methods
	private:

		// clear training data
		void clearTrainingData(void);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// private data
	private:
		LPVOID _pMLP;
		static int s_nDevicedCount;
		
		struct TrainingData;
		CList<TrainingData*> _listTrainingData;
	};
}}