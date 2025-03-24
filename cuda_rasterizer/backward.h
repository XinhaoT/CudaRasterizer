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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);

	void compute3dgrads_grid(
		const int valid_grid_num, int D, int M,
		const int P, int S_PerGird,
		const int* valid_grid_cuda,
		const int* grid_gs_prefix_sum_cuda,
		const float* samples_pos,
		const float* pos_cuda,
		const float* rot_cuda,
		const float* scale_cuda,
		const float* opacity_cuda,
		const float* shs_cuda,
		const float* half_length_cuda,
		const float* sigma_cuda,
		float* sigma_damp_cuda,
		const float* opacity_grad_cuda,
		const float* feature_grad_cuda,
		float* dF_dopacity,
		float* dF_dshs,
		float* dF_dpos,
		float* dF_drot,
		float* dF_dscale,
		float* dF_dcov3D,
		const int* grided_gs_idx_cuda,
		bool* grid_is_converged_cuda,
		bool* opt_options_cuda,
		float3 min_xyz,
		float grid_step,
		int grid_num,
		float* ada_lpf_ratio,
		int* empty_grid_cuda,
		int* current_static_grids_cuda,
		int* moved_gaussians_cuda,
		bool has_soup
	);

	void compute3dgrads(
		const int N, int D, int M,
		const int P,
		const float* samples_pos,
		const int* sample_neighbours,
		const int* sample_idx_itselves,
		const float* pos_cuda,
		const float* rot_cuda,
		const float* scale_cuda,
		const float* opacity_cuda,
		const float* shs_cuda,
		const float* half_length_cuda,
		const float* sigma_cuda,
		const float* feature_grad_cuda,
		float* dF_dopacity,
		float* dF_dshs,
		float* dF_dpos,
		float* dF_drot,
		float* dF_dscale,
		float* dF_dcov3D
	);

	void updatefeature3d(
		const int P, int D, int M,
		const float* dF_dopacity,
		const float* dF_dshs,
		const float* dF_dpos,
		const float* dF_drot,
		const float* dF_dscale,
		float* opacity_cuda,
		float* shs_cuda,
		float* pos_cuda,
		float* rot_cuda,
		float* scale_cuda,
		float* m_opacity_cuda,
		float* v_opacity_cuda,
		float* m_shs_cuda,
		float* v_shs_cuda,
		float* m_pos_cuda,
		float* v_pos_cuda,
		float* m_rot_cuda,
		float* v_rot_cuda,
		float* m_scale_cuda,
		float* v_scale_cuda,
		float* max_scale_cuda,
		int* step,
		bool* opt_options_cuda,
		float* learning_rate_cuda,
		int _optimize_steps,
		int* moved_gaussians_cuda
	);
}

#endif