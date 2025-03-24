/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		int2* rects,
		float3 boxmin,
		float3 boxmax,
		bool _to_ortho,
		float ortho_scale);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		int* is_surface);

	void compute3dfeatures(
		const int N, int D, int M,
		const int S, const int P,
		const float* samples_pos,
		const int* sample_neighbours,
		const int* sample_idx_itselves,
		const float* pos_cuda,
		float* rot_cuda,
		const float* scale_cuda,
		const float* opacity_cuda,
		const float* shs_cuda,
		const float* half_length_cuda,
		float* sigma_inv_cuda,
		float* result,
		float* feature_opacity_cuda
	);

	void compute3dfeatures_grid(
		const int valid_grid_num, int D, int M,
		const int P, int S_PerGird,
		const int* valid_grid_cuda,
		const int* grid_gs_prefix_sum_cuda,
		const float* samples_pos,
		const float* pos_cuda,
		float* rot_cuda,
		const float* scale_cuda,
		const float* opacity_cuda,
		const float* shs_cuda,
		const float* half_length_cuda,
		float* sigma_cuda,
		float* sigma_damp_cuda,
		float* result,
		float* feature_opacity_cuda,
		const int* grided_gs_idx_cuda,
		bool* grid_is_converged_cuda,
		bool* grid_nearly_converged_cuda,
		bool* opt_options_cuda,
		float low_pass_param,
		float* ada_lpf_ratio,
		float3 min_xyz,
		float grid_step,
		int grid_num,
		int* gs_init_grid_idx_cuda,
		int* empty_grid_cuda,
		int* current_static_grids_cuda,
		bool has_soup
	);

	void computeL1loss3d(
		const int valid_grid_num, int D, int M,
		const int P, int S_PerGird,
		const float* aim_feature_cuda,
		const float* cur_feature_cuda,
		const float* aim_opacity_cuda,
		const float* cur_opacity_cuda,
		float* opacity_grad_cuda,
		float* feature_grad_cuda,
		float* total_feature_loss,
		float* total_shape_loss,
		bool* grid_is_converged_cuda,
		bool* grid_nearly_converged_cuda,
		float* grid_loss_sums_cuda,
		bool* opt_options_cuda,
		int* empty_grid_cuda,
		bool adjust_op_range,
		bool has_soup
	);
}


#endif