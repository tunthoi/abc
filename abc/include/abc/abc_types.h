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

#define ABC_MIN_TRAININGDATACOUNT		10

namespace comed { namespace abc 
{
	/// <summary>
	/// ABC feature id
	/// </summary>
	enum E_ABCFeatureId 
	{
		kABCFeatureId_Global_Otsu = 0,
		kABCFeatureId_Global_Inner,
		kABCFeatureId_Global_Inter,
		kABCFeatureId_Global_Max,
		kABCFeatureId_Global_Min,
		kABCFeatureId_Global_Mean,
		kABCFeatureId_Global_Std,

		kABCFeatureId_Local_Otsu,
		kABCFeatureId_Local_Inner,
		kABCFeatureId_Local_Inter,
		kABCFeatureId_Local_Max,
		kABCFeatureId_Local_Min,
		kABCFeatureId_Local_Mean,
		kABCFeatureId_Local_Std,
	};

	/// <summary>
	/// ABC result id
	/// </summary>
	enum E_ABCResultId
	{
		kABCResultId_Metal = 0,
		kABCResultId_Background,
	};

	/// <summary>
	/// classfier result
	/// </summary>
	struct RegionType
	{
		bool bMetal, bBackground;
	};

}} // comed::abc