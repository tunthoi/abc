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

		// initialize
		bool Initialize(void);

		/// <summary>
		/// add training data
		/// </summary>
		bool AddTrainingData( const cl::img::CImageBuf& img, const RegionType arrTypes[ ABC_REGION_DIVIDE_2 ] );

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// private data 
	private:
		CvANN_MLP* _pMLP;
		CList< DataTrainingParams* > _listData;


	};
}} // comed::abc
