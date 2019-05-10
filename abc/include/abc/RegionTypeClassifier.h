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
	class AFX_EXT_CLASS CRegionTypeClassifier
	{
		CL_NO_COPY_CONSTRUCTOR( CRegionTypeClassifier )
		CL_NO_ASSIGNMENT_OPERATOR( CRegionTypeClassifier )

	public:
		/// <summary>
		/// default constructor
		/// </summary>
		CRegionTypeClassifier(void);

		// desrtructor
		virtual ~CRegionTypeClassifier(void);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// public methods 
	public:

		/// <summary>
		/// initialize the classfier with the pre-trained data
		/// </summary>
		bool Initialize( LPCTSTR lpszResultPath_Objec, LPCTSTR lpszResultPath_Metal );

		/// <summary>
		/// classfy the region
		/// </summary>
		bool ClassfyRegion( 
				IN		const cl::img::CImageBuf& img, 
				OUT		RegionType arrResult[ ABC_REGION_DIVIDE ],
				OUT		int* pnNumObjBlocks,
				OUT		int* pnMeanObjBlocks,
				OUT		int* pnMinObj,
				OUT		int* pnMaxObj
			) const;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// private data
	private:
		CvANN_MLP* _pMLP_Objec;
		CvANN_MLP* _pMLP_Metal;
	};

}} // comed::abc 
