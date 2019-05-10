/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "clUtils/defines.h"
#include "abc/abc_types.h"

// forward declaration
class CvANN_MLP;
namespace cl { namespace img { class CImageBuf; }}

namespace comed { namespace abc 
{
	/// <summary>
	/// region classifer - training helper
	/// </summary>
	class AFX_EXT_CLASS CRegionTypeTrainer
	{
		CL_NO_ASSIGNMENT_OPERATOR( CRegionTypeTrainer )
		CL_NO_COPY_CONSTRUCTOR( CRegionTypeTrainer )

		// internal data types
		struct DataTrainingParams;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// constructor and destrucrtor
	public:
		/// <summary>
		/// constructor
		/// </summary>
		CRegionTypeTrainer(void);

		/// <summary>
		/// destructor
		/// </summary>
		virtual ~CRegionTypeTrainer(void);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// public methods
	public:

		/// <summary>
		/// initialize
		/// </summary>
		bool Initialize(void);

		/// <summary>
		/// add training data
		/// </summary>
		bool AddTrainingData( 
				int nKv, float fMa,
				const cl::img::CImageBuf& img, const RegionType arrTypes[ ABC_REGION_DIVIDE_2 ] );

		/// <summary>
		/// clear all the training data
		/// </summary>
		void ClearTrainingData(void);

		/// <summary>
		/// get number of data instances
		/// </summary>
		int GetTrainingDataCount(void) const;

		/// <summary>
		/// load data from file
		/// </summary>
		bool AddTrainingDataFrom( LPCTSTR lpszFilePath );

		/// <summary>
		/// get current data
		/// </summary>
		bool GetCurrentData( RegionType arrTypes[ ABC_REGION_DIVIDE_2 ] ) const;

		/// <summary>
		/// save the data to the file
		/// </summary>
		bool SaveTrainingData( LPCTSTR lpszFilePath );

		/// <summary>
		/// train and store the result.
		/// </summary>
		bool SaveTrainingResult( LPCTSTR lpszResultPath_Objec, LPCTSTR lpszResultPath_Metal ) const;


		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// private methods
	private:
		static int _extractData( const CString strLine, DataTrainingParams* pParam );

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// private data 
	private:
		CvANN_MLP *_pMLP_Objec, *_pMLP_Metal;
		CList< DataTrainingParams* > _listData;
	};
}} // comed::abc
