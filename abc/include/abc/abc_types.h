/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once 

// macro constant 
// 16x16 Slice Area
#define ABC_REGION_DIVIDE				16
#define ABC_REGION_DIVIDE_2				( ABC_REGION_DIVIDE * ABC_REGION_DIVIDE )

#define ABC_FEATURE_COUNT				14
#define ABC_RESULT_COUNT				2


namespace comed { namespace abc 
{
	/// <summary>
	/// classfier result
	/// </summary>
	struct RegionType
	{
		bool bMetal, bBackground;
	};

}} // comed::abc