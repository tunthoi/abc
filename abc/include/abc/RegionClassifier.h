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
	/// region classifier with pre-trained data
	/// </summary>
	class AFX_EXT_CLASS CRegionClassifier
	{
		CL_NO_COPY_CONSTRUCTOR( CRegionClassifier )
		CL_NO_ASSIGNMENT_OPERATOR( CRegionClassifier )

	public:
		/// <summary>
		/// default constructor
		/// </summary>
		CRegionClassifier(void);

		// desrtructor
		virtual ~CRegionClassifier(void);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// public methods 
	public:

		/// <summary>
		/// initialize the classfier with the pre-trained data
		/// </summary>
		bool Initialize( LPCTSTR lpszDataPath );

		/// <summary>
		/// classfy the region
		/// </summary>
		bool ClassfyRegion( const cl::img::CImageBuf& img, RegionType arrResult[ ABC_REGION_DIVIDE ] ) const;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// private data
	private:
		CvANN_MLP* _pMLP;
	};

}} // comed::abc 
